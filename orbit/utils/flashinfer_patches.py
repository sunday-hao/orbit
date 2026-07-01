import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_PATCHED: set[str] = set()

# flashinfer's find_loaded_library() substring-matches "libcudart" against every
# shared library mapped into the process (via /proc/self/maps) and returns the
# first hit. tilelang ships libcudart_stub.so -- a link-time-only stub missing
# most of the real cudart symbol table (e.g. cudaDeviceReset) -- whose filename
# also matches that substring. If tilelang gets loaded into the process before
# the real cudart does (e.g. via fla's tilelang backend, imported transitively
# through sglang), CudaRTLibrary() binds to the stub and crashes with
# "undefined symbol: cudaDeviceReset" the first time flashinfer touches CUDA IPC.
#
# Fix mirrors https://github.com/flashinfer-ai/flashinfer/pull/3677 (open,
# unmerged as of 2026-07): only accept an exact "libcudart" stem or a
# hash-suffixed "libcudart-<hash>" (as PyTorch ships), never a look-alike like
# "libcudart_stub".
_BUGGY_ASSERT = 'assert filename.rpartition(".so")[0].startswith(lib_name), ('

_OLD_BODY = '''    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), (
        f"Unexpected filename: {filename} for library {lib_name}"
    )
    return path'''

_NEW_BODY = '''    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name not in line or "/" not in line:
                continue
            path = line[line.index("/"):].strip()
            stem = path.split("/")[-1].rpartition(".so")[0]
            if stem == lib_name or stem.startswith(lib_name + "-"):
                return path
    # the library is not loaded in the current process
    return None'''


def apply_flashinfer_cuda_ipc_patch() -> None:
    """Idempotently rewrite flashinfer's buggy find_loaded_library() on disk.

    Must run before anything in this process imports `flashinfer` (or a
    package that imports it, like sglang): the crash happens as a side effect
    of `import flashinfer.comm.cuda_ipc` itself (`cudart = CudaRTLibrary()`
    runs at module scope), so patching after import is too late. We locate
    the installed .py file via a plain filesystem scan of sys.path rather
    than importlib, so locating it never triggers the crash we're defusing.
    """
    if "flashinfer.comm.cuda_ipc" in _PATCHED:
        return

    if any(name in sys.modules for name in ("flashinfer", "flashinfer.comm", "flashinfer.comm.cuda_ipc")):
        logger.warning(
            "flashinfer cuda_ipc patch skipped: flashinfer already imported in this "
            "process, too late to patch the on-disk source safely."
        )
        return

    path = _find_cuda_ipc_file()
    if path is None:
        return  # flashinfer not installed, nothing to patch

    try:
        source = path.read_text()
    except OSError as e:
        logger.warning("flashinfer cuda_ipc patch skipped: could not read %s (%s)", path, e)
        return

    if _BUGGY_ASSERT not in source:
        _PATCHED.add("flashinfer.comm.cuda_ipc")
        return  # already fixed upstream, or code shape changed -- don't touch it

    if _OLD_BODY not in source:
        logger.warning(
            "flashinfer cuda_ipc patch skipped: %s has the buggy assert but its "
            "surrounding code doesn't match the expected shape, refusing to risk a bad patch.",
            path,
        )
        return

    try:
        path.write_text(source.replace(_OLD_BODY, _NEW_BODY))
    except OSError as e:
        logger.warning("flashinfer cuda_ipc patch skipped: could not write %s (%s)", path, e)
        return

    logger.info("Patched %s: find_loaded_library() no longer mistakes tilelang's libcudart_stub.so for libcudart.", path)
    _PATCHED.add("flashinfer.comm.cuda_ipc")


def _find_cuda_ipc_file() -> Path | None:
    for entry in sys.path:
        candidate = Path(entry) / "flashinfer" / "comm" / "cuda_ipc.py"
        if candidate.is_file():
            return candidate
    return None
