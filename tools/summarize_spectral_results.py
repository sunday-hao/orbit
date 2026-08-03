#!/usr/bin/env python3
"""Summarize PEFT-Arena spectral analysis output into curve-friendly CSVs.

Reads every ``iter_*/spectral/spectral_summary.json`` under a run directory and
writes two files:

- ``spectral_summary.csv``   one row per checkpoint (layer-aggregated)
- ``spectral_per_layer.csv`` one row per checkpoint x analyzed weight

The per-checkpoint aggregates are unweighted means/medians over the analyzed
weights, which is only meaningful when every checkpoint in the run was analyzed
with the same ``--modules``/``--layers`` filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

LAYER_METRICS = (
    "effective_rank",
    "smoothness_retention",
    "smoothness_adaptation",
    "delta_P_diag_mean",
    "delta_P_diag_max",
    "E_delta_mean",
    "E_delta_max",
    "sva_mean",
    "sva_min",
)
AGGREGATED = (
    "effective_rank",
    "smoothness_retention",
    "smoothness_adaptation",
    "delta_P_diag_mean",
    "E_delta_mean",
    "sva_mean",
)
def _agg_column(name: str) -> str:
    """Run-level column for a per-layer metric, without doubling ``_mean``."""
    return name if name.endswith("_mean") else f"{name}_mean"


SUMMARY_FIELDNAMES = (
    "iter",
    "iter_name",
    "peft_type",
    "num_layers",
    "erank_mean",
    "erank_median",
    "erank_max",
    *(_agg_column(name) for name in AGGREGATED if name != "effective_rank"),
    "summary_file",
)
PER_LAYER_FIELDNAMES = (
    "iter",
    "iter_name",
    "weight",
    "layer",
    "module",
    *LAYER_METRICS,
    "delta_sv_top1",
)


def _iter_sort_key(iter_name: str) -> int:
    match = re.search(r"(\d+)$", iter_name)
    return int(match.group(1)) if match else -1


def _find_iter_name(path: Path, run_dir: Path) -> str | None:
    try:
        rel_parts = path.relative_to(run_dir).parts
    except ValueError:
        return None
    for part in rel_parts:
        if part.startswith("iter_"):
            return part
    return None


def _split_weight_name(name: str) -> tuple[str, str]:
    """``model.layers.5.self_attn.q_proj.weight`` -> ``("5", "q_proj")``."""
    layer_match = re.search(r"layers\.(\d+)\.", name)
    layer = layer_match.group(1) if layer_match else ""
    stripped = name.removesuffix(".weight")
    module = stripped.rsplit(".", 1)[-1] if "." in stripped else stripped
    return layer, module


def _stringify_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def collect(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_rows: list[dict[str, str]] = []
    layer_rows: list[dict[str, str]] = []

    for summary_path in sorted(run_dir.glob("iter_*/**/spectral_summary.json")):
        iter_name = _find_iter_name(summary_path, run_dir)
        if iter_name is None:
            continue
        with summary_path.open(encoding="utf-8") as f:
            summary: dict[str, Any] = json.load(f)

        per_layer: dict[str, dict[str, Any]] = summary.get("per_layer") or {}
        iter_index = _iter_sort_key(iter_name)
        collected: dict[str, list[float]] = {name: [] for name in AGGREGATED}

        for weight_name, metrics in per_layer.items():
            layer, module = _split_weight_name(weight_name)
            delta_sv = metrics.get("delta_sv_top5") or []
            row = {
                "iter": str(iter_index),
                "iter_name": iter_name,
                "weight": weight_name,
                "layer": layer,
                "module": module,
                "delta_sv_top1": _stringify_metric(delta_sv[0] if delta_sv else None),
            }
            for name in LAYER_METRICS:
                row[name] = _stringify_metric(metrics.get(name))
            layer_rows.append(row)

            for name in AGGREGATED:
                value = metrics.get(name)
                if isinstance(value, (int, float)):
                    collected[name].append(float(value))

        eranks = collected["effective_rank"]
        summary_row = {
            "iter": str(iter_index),
            "iter_name": iter_name,
            "peft_type": _stringify_metric(summary.get("peft_type")),
            "num_layers": _stringify_metric(summary.get("num_layers_analyzed", len(per_layer))),
            "erank_mean": _stringify_metric(_mean(eranks)),
            "erank_median": _stringify_metric(statistics.median(eranks) if eranks else None),
            "erank_max": _stringify_metric(max(eranks) if eranks else None),
            "summary_file": str(summary_path),
        }
        for name in AGGREGATED:
            if name == "effective_rank":
                continue
            summary_row[_agg_column(name)] = _stringify_metric(_mean(collected[name]))
        summary_rows.append(summary_row)

    summary_rows.sort(key=lambda row: int(row["iter"]))
    layer_rows.sort(key=lambda row: (int(row["iter"]), row["weight"]))
    return summary_rows, layer_rows


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PEFT-Arena spectral analysis output.")
    parser.add_argument("--run-dir", required=True, help="Eval results run directory, e.g. eval_results/<run>")
    parser.add_argument("--output", default=None, help="Summary CSV path. Defaults to <run-dir>/spectral_summary.csv")
    parser.add_argument(
        "--per-layer-output",
        default=None,
        help="Per-layer CSV path. Defaults to <run-dir>/spectral_per_layer.csv",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_path = Path(args.output).resolve() if args.output else run_dir / "spectral_summary.csv"
    per_layer_path = (
        Path(args.per_layer_output).resolve() if args.per_layer_output else run_dir / "spectral_per_layer.csv"
    )

    summary_rows, layer_rows = collect(run_dir)
    _write_csv(summary_path, SUMMARY_FIELDNAMES, summary_rows)
    _write_csv(per_layer_path, PER_LAYER_FIELDNAMES, layer_rows)
    print(summary_path)


if __name__ == "__main__":
    main()
