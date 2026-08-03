#!/usr/bin/env python3
"""Derive ΔW rank/energy metrics from PEFT-Arena spectral analysis artifacts.

``spectral_analysis.py`` keeps only ``delta_sv_top5`` in ``spectral_summary.json``
but saves the *full* singular value spectrum of ΔW into each per-layer ``.pt``.
Every metric below is a function of that spectrum, so this runs post-hoc in
seconds with no GPU and no model loading.

Per analyzed weight:

- ``frobenius``              ‖ΔW‖_F = sqrt(Σ σ_i²)
- ``spectral_norm``          σ_1
- ``stable_rank``            ‖ΔW‖_F² / σ_1²   (matches upstream ``effective_rank``)
- ``energy_effective_rank``  exp(H) with p_i = σ_i² / Σ σ_j²   (energy-normalized)
- ``entropy_effective_rank`` exp(H) with p_i = σ_i / Σ σ_j     (Roy & Vetterli)
- ``coverage@k``             Σ_{i≤k} σ_i² / Σ_i σ_i²

Usage:
    python tools/spectral_rank_metrics.py --run-dir eval_results/<run>
    python tools/spectral_rank_metrics.py --spectral-dir eval_results/<run>/iter_0000050/spectral
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import torch

DEFAULT_COVERAGE_RANKS = (8, 16, 32, 64)


def _iter_sort_key(iter_name: str) -> int:
    match = re.search(r"(\d+)$", iter_name)
    return int(match.group(1)) if match else -1


def _split_weight_name(name: str) -> tuple[str, str]:
    layer_match = re.search(r"layers\.(\d+)\.", name)
    layer = layer_match.group(1) if layer_match else ""
    stripped = name.removesuffix(".weight")
    module = stripped.rsplit(".", 1)[-1] if "." in stripped else stripped
    return layer, module


def _unmangle(stem: str) -> str:
    """Undo ``spectral_analysis.py``'s ``name.replace(".", "_")`` well enough to
    recover layer index and module name. Underscores inside module names (e.g.
    ``q_proj``) make this lossy, so only the parts we report are reconstructed."""
    layer_match = re.search(r"layers_(\d+)_", stem)
    layer = layer_match.group(1) if layer_match else ""
    module = stem.removesuffix("_weight").rsplit("_", 2)[-2:]
    return f"layers.{layer}." + "_".join(module) if layer else stem


def _shannon_exp(probabilities: torch.Tensor) -> float:
    """exp of Shannon entropy over a probability vector (natural log)."""
    positive = probabilities[probabilities > 0]
    if positive.numel() == 0:
        return 0.0
    entropy = -(positive * positive.log()).sum().item()
    return math.exp(entropy)


def spectrum_metrics(singular_values: torch.Tensor, coverage_ranks: tuple[int, ...]) -> dict[str, float]:
    sv = singular_values.detach().float().flatten()
    sv = sv[sv.isfinite()]
    sv, _ = sv.sort(descending=True)

    energy = sv.pow(2)
    total_energy = float(energy.sum())
    metrics: dict[str, float] = {"num_sv": float(sv.numel())}

    if sv.numel() == 0 or total_energy <= 0.0:
        metrics.update(
            frobenius=0.0,
            spectral_norm=0.0,
            stable_rank=0.0,
            energy_effective_rank=0.0,
            entropy_effective_rank=0.0,
        )
        for k in coverage_ranks:
            metrics[f"coverage@{k}"] = 0.0
        return metrics

    sigma_1 = float(sv[0])
    metrics["frobenius"] = math.sqrt(total_energy)
    metrics["spectral_norm"] = sigma_1
    metrics["stable_rank"] = total_energy / (sigma_1**2)
    metrics["energy_effective_rank"] = _shannon_exp(energy / total_energy)
    metrics["entropy_effective_rank"] = _shannon_exp(sv / float(sv.sum()))

    cumulative = energy.cumsum(0)
    for k in coverage_ranks:
        index = min(k, sv.numel()) - 1
        metrics[f"coverage@{k}"] = float(cumulative[index]) / total_energy
    return metrics


def analyze_spectral_dir(spectral_dir: Path, coverage_ranks: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tensor_path in sorted(spectral_dir.glob("*.pt")):
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "delta_sv" not in payload:
            continue
        weight = _unmangle(tensor_path.stem)
        layer, module = _split_weight_name(weight)
        row: dict[str, object] = {"weight": weight, "layer": layer, "module": module}
        row.update(spectrum_metrics(payload["delta_sv"], coverage_ranks))
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _round(value: object) -> object:
    return round(value, 6) if isinstance(value, float) else value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _round(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="eval_results/<run>; sweeps every iter_*/spectral under it")
    source.add_argument("--spectral-dir", help="A single eval_results/<run>/<iter>/spectral directory")
    parser.add_argument(
        "--coverage-ranks",
        default=",".join(str(k) for k in DEFAULT_COVERAGE_RANKS),
        help="Comma-separated k values for energy coverage@k",
    )
    parser.add_argument("--output", default=None, help="Per-layer CSV path")
    parser.add_argument("--summary-output", default=None, help="Per-checkpoint CSV path (--run-dir only)")
    args = parser.parse_args()

    coverage_ranks = tuple(int(k.strip()) for k in args.coverage_ranks.split(",") if k.strip())
    metric_columns = [
        "num_sv",
        "frobenius",
        "spectral_norm",
        "stable_rank",
        "energy_effective_rank",
        "entropy_effective_rank",
        *(f"coverage@{k}" for k in coverage_ranks),
    ]

    if args.spectral_dir:
        spectral_dir = Path(args.spectral_dir).resolve()
        rows = analyze_spectral_dir(spectral_dir, coverage_ranks)
        output = Path(args.output).resolve() if args.output else spectral_dir / "rank_metrics.csv"
        _write_csv(output, ["weight", "layer", "module", *metric_columns], rows)
        print(output)
        return

    run_dir = Path(args.run_dir).resolve()
    layer_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spectral_dir in sorted(run_dir.glob("iter_*/**/spectral")):
        iter_name = next((part for part in spectral_dir.relative_to(run_dir).parts if part.startswith("iter_")), None)
        if iter_name is None:
            continue
        rows = analyze_spectral_dir(spectral_dir, coverage_ranks)
        if not rows:
            continue
        iter_index = _iter_sort_key(iter_name)
        for row in rows:
            layer_rows.append({"iter": iter_index, "iter_name": iter_name, **row})

        summary: dict[str, object] = {
            "iter": iter_index,
            "iter_name": iter_name,
            "num_weights": len(rows),
            # Model-level ‖ΔW‖_F is the root of the summed squares, not a mean.
            "frobenius_total": math.sqrt(sum(float(row["frobenius"]) ** 2 for row in rows)),
        }
        for column in metric_columns:
            summary[f"{column}_mean"] = _mean([float(row[column]) for row in rows])
        summary_rows.append(summary)

    layer_rows.sort(key=lambda row: (int(row["iter"]), str(row["weight"])))
    summary_rows.sort(key=lambda row: int(row["iter"]))

    layer_output = Path(args.output).resolve() if args.output else run_dir / "rank_metrics_per_layer.csv"
    summary_output = (
        Path(args.summary_output).resolve() if args.summary_output else run_dir / "rank_metrics_summary.csv"
    )
    _write_csv(layer_output, ["iter", "iter_name", "weight", "layer", "module", *metric_columns], layer_rows)
    _write_csv(
        summary_output,
        ["iter", "iter_name", "num_weights", "frobenius_total", *(f"{c}_mean" for c in metric_columns)],
        summary_rows,
    )
    print(summary_output)


if __name__ == "__main__":
    main()
