#!/usr/bin/env python3
"""Run an upstream PEFT-Arena tool with Orbit-side environment fixes applied.

The PEFT-Arena checkout is a pristine clone we do not patch, but a couple of its
weight-analysis tools trip over the local environment before they get to do any
work. This shim applies the fixes in-process and then executes the target script
unchanged, so ``../PEFT-Arena`` stays exactly as cloned.

Usage:
    python tools/run_peft_arena_tool.py <script.py> [script args...]
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _bypass_incompatible_torchao() -> None:
    """Neutralize PEFT's optional TorchAO dispatcher when the installed torchao
    is too old for the installed peft.

    ``peft.import_utils.is_torchao_available()`` *raises* rather than returning
    False when it finds an incompatible torchao, and the LoRA dispatch chain
    calls it while building every adapter layer. Orbit checkpoints never use
    TorchAO quantized layers, so the correct outcome is "not available" and a
    fall-through to the vanilla Linear dispatcher.

    Same fix as ``_bypass_incompatible_torchao_for_lora()`` in the vendored
    ``examples/peft_arena/backend/tools/merge_peft.py``; kept here so it also
    covers upstream tools we do not vendor. Orbit's own stack uses torchao
    through SGLang's quantization path, so upgrading or removing the package to
    work around this is not an option.
    """
    try:
        import peft.import_utils as peft_imports
    except ImportError:
        return

    try:
        peft_imports.is_torchao_available()
    except ImportError as exc:
        if "incompatible version of torchao" not in str(exc):
            raise
        peft_imports.is_torchao_available = lambda: False
        for module_name in ("peft.tuners.lora.torchao", "peft.tuners.oft.torchao"):
            try:
                __import__(module_name)
            except ImportError:
                continue
            sys.modules[module_name].is_torchao_available = lambda: False
        print(
            "[run_peft_arena_tool] Disabled PEFT's TorchAO dispatcher: the "
            "installed torchao is too old for this peft. Orbit adapters do not "
            "use TorchAO layers, so the vanilla backend is the right path.",
            file=sys.stderr,
        )


def _force_cuda_svd() -> None:
    """Route every ``torch.linalg.svd`` call through the GPU.

    ``spectral_analysis.py`` moves the operand to CUDA in its non-OFT branch
    (``safe_svd(W_pre.cuda())``) but not in the OFT one, where both
    ``safe_svd(W_ft)`` and ``safe_svd(W_pre)`` run full-size SVDs on the CPU --
    hours per checkpoint instead of minutes. Patching the call site from here
    keeps ../PEFT-Arena an unpatched clone and does not depend on upstream line
    numbers. ``safe_svd`` already returns ``.cpu()`` tensors, so callers see no
    device change.

    Set ORBIT_PEFT_ARENA_CUDA_SVD=0 to leave upstream's device placement alone.
    """
    if os.environ.get("ORBIT_PEFT_ARENA_CUDA_SVD", "1") == "0":
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return

    original_svd = torch.linalg.svd

    def svd_on_cuda(A, *args, **kwargs):
        if isinstance(A, torch.Tensor) and not A.is_cuda:
            A = A.cuda()
        return original_svd(A, *args, **kwargs)

    torch.linalg.svd = svd_on_cuda
    print(
        "[run_peft_arena_tool] Routing torch.linalg.svd through the GPU "
        "(upstream's OFT branch would otherwise run full-size SVDs on the CPU).",
        file=sys.stderr,
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: run_peft_arena_tool.py <script.py> [args...]")

    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        sys.exit(f"run_peft_arena_tool.py: no such script: {script}")

    _bypass_incompatible_torchao()
    _force_cuda_svd()

    # Upstream tools import their siblings flatly (e.g. `from diagnostic_utils
    # import ...`), which only works when the script's own directory is on the
    # path -- normally supplied by running it directly.
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
