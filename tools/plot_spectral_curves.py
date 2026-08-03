#!/usr/bin/env python3
"""Plot ΔW geometry metrics against training step, one subplot per metric.

Reads the ``rank_metrics_summary.csv`` written by
``examples/peft_arena/eval/spectral-peft-arena.sh`` for one or more runs and
draws a shared-legend grid: x = training step, y = metric, one line per run.

Usage:
    python tools/plot_spectral_curves.py \\
        --run "Full-vocab FKL=eval_results/<run_a>" \\
        --run "Sampled-token=eval_results/<run_b>" \\
        --output figures/opd_lora_geometry.png

Each --run is LABEL=PATH, where PATH is either a run directory or the CSV
itself. --dry-run prints the series instead of plotting, so the data path can be
checked without a display or matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

SUMMARY_BASENAME = "rank_metrics_summary.csv"
DEFAULT_METRICS = (
    "frobenius_total",
    "stable_rank_mean",
    "energy_effective_rank_mean",
    "coverage@16_mean",
)
# Axis labels for the columns worth plotting; anything else falls back to a
# prettified column name.
AXIS_LABELS = {
    "frobenius_total": "Global Frobenius norm",
    "frobenius_mean": "Mean Frobenius norm",
    "spectral_norm_mean": "Mean spectral norm",
    "stable_rank_mean": "Mean stable rank",
    "energy_effective_rank_mean": "Mean energy-effective rank",
    "entropy_effective_rank_mean": "Mean entropy-effective rank",
    "num_sv_mean": "Mean singular values per weight",
}


def _axis_label(column: str) -> str:
    if column in AXIS_LABELS:
        return AXIS_LABELS[column]
    if column.startswith("coverage@") and column.endswith("_mean"):
        k = column[len("coverage@") : -len("_mean")]
        return f"Mean rank-{k} energy coverage"
    return column.replace("_", " ").capitalize()


def parse_metric_floats(specs: list[str] | None, option: str) -> dict[str, float]:
    """``METRIC=VALUE`` pairs -> {metric: value}."""
    parsed: dict[str, float] = {}
    for spec in specs or []:
        metric, sep, value = spec.rpartition("=")
        if not sep:
            raise SystemExit(f"{option} must be METRIC=VALUE, got {spec!r}")
        try:
            parsed[metric.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"{option}: bad number in {spec!r}") from None
    return parsed


def parse_metric_ranges(specs: list[str] | None, option: str) -> dict[str, tuple[float, float]]:
    """``METRIC=MIN:MAX`` pairs -> {metric: (min, max)}."""
    parsed: dict[str, tuple[float, float]] = {}
    for spec in specs or []:
        metric, sep, rest = spec.rpartition("=")
        low, colon, high = rest.partition(":")
        if not sep or not colon:
            raise SystemExit(f"{option} must be METRIC=MIN:MAX, got {spec!r}")
        try:
            parsed[metric.strip()] = (float(low), float(high))
        except ValueError:
            raise SystemExit(f"{option}: bad number in {spec!r}") from None
    return parsed


def apply_axis_overrides(ax, metric: str, yticks: dict[str, float], ylims: dict[str, tuple[float, float]]) -> None:
    """Fixed y range and/or tick spacing for one subplot.

    Tick spacing alone does not change how dramatic a curve looks -- matplotlib
    still autoscales to the data range. Widening the range with --ylim is what
    makes a small absolute variation read as small.
    """
    if metric in ylims:
        ax.set_ylim(*ylims[metric])
    if metric in yticks:
        from matplotlib.ticker import MultipleLocator

        ax.yaxis.set_major_locator(MultipleLocator(yticks[metric]))


def resolve_log_metrics(logy: str | None, metrics: list[str]) -> set[str]:
    """Which metrics get a log y axis. ``None`` -> none, ``"all"`` -> every one,
    otherwise a comma-separated subset."""
    if not logy:
        return set()
    if logy == "all":
        return set(metrics)
    return {m.strip() for m in logy.split(",") if m.strip()}


def _resolve_csv(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_dir():
        path = path / SUMMARY_BASENAME
    if not path.is_file():
        raise SystemExit(f"plot_spectral_curves: no such summary CSV: {path}")
    return path


def load_run(path_str: str, step_offset: int) -> dict[str, list[float]]:
    """Return {"step": [...], <metric>: [...]} sorted by step."""
    csv_path = _resolve_csv(path_str)
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("iter"):
                rows.append(row)
    if not rows:
        raise SystemExit(f"plot_spectral_curves: {csv_path} has no data rows")

    rows.sort(key=lambda r: int(r["iter"]))
    series: dict[str, list[float]] = {"step": [int(r["iter"]) + step_offset for r in rows]}
    for column in rows[0]:
        if column in ("iter", "iter_name"):
            continue
        values: list[float] = []
        for row in rows:
            raw = row.get(column, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(math.nan)
        series[column] = values
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run to plot; PATH is a run dir or a rank_metrics_summary.csv. Repeatable.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated columns, one subplot each",
    )
    parser.add_argument("--output", default="spectral_curves.png", help="Output image path")
    parser.add_argument("--title", default="Cumulative update geometry")
    parser.add_argument(
        "--step-offset",
        type=int,
        default=1,
        help="Added to the 0-indexed iter to get the training step (iter_0000009 -> step 10)",
    )
    parser.add_argument("--ncols", type=int, default=2)
    parser.add_argument("--title-y", type=float, default=0.99, help="Suptitle height in figure coords")
    parser.add_argument("--title-size", type=float, default=14.0)
    parser.add_argument(
        "--legend-y",
        type=float,
        default=0.945,
        help="Top edge of the legend box in figure coords; keep it below --title-y",
    )
    parser.add_argument("--legend-ncol", type=int, default=None, help="Legend columns (default: min(#runs, 3))")
    parser.add_argument(
        "--top",
        type=float,
        default=None,
        help="Top of the axes area in figure coords; default leaves room for the legend",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", default=None, help="WxH in inches, e.g. 12x8. Default scales with the grid.")
    parser.add_argument(
        "--logy",
        nargs="?",
        const="all",
        default=None,
        metavar="all|METRICS",
        help="Log-scale the y axis. Bare --logy applies to every subplot; pass a comma-separated "
        "metric list to restrict it (coverage@k lives in (0,1] and rarely benefits).",
    )
    parser.add_argument(
        "--ytick",
        action="append",
        metavar="METRIC=STEP",
        help="Y tick spacing for one subplot, e.g. stable_rank_mean=0.2. Repeatable.",
    )
    parser.add_argument(
        "--ylim",
        action="append",
        metavar="METRIC=MIN:MAX",
        help="Fixed y range for one subplot, e.g. stable_rank_mean=0:3. Repeatable. "
        "Use this (not --ytick) to stop a small variation from looking dramatic.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the series instead of plotting")
    args = parser.parse_args()

    runs: list[tuple[str, dict[str, list[float]]]] = []
    for spec in args.run:
        # rpartition, not partition: labels may contain '=' ("LoRA r=32"),
        # paths essentially never do.
        label, sep, path_str = spec.rpartition("=")
        if not sep:
            raise SystemExit(f"plot_spectral_curves: --run must be LABEL=PATH, got {spec!r}")
        runs.append((label, load_run(path_str, args.step_offset)))

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    missing = [m for m in metrics if not any(m in series for _, series in runs)]
    if missing:
        available = sorted(set().union(*(set(s) for _, s in runs)) - {"step"})
        raise SystemExit(
            f"plot_spectral_curves: no run has {missing}.\n"
            f"  Available columns: {', '.join(available)}\n"
            "  Coverage columns depend on the COVERAGE_RANKS used for the sweep."
        )

    if args.dry_run:
        for label, series in runs:
            print(f"[{label}] steps: {series['step']}")
            for metric in metrics:
                values = series.get(metric)
                print(f"  {metric}: {'<absent>' if values is None else values}")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = min(args.ncols, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    if args.figsize:
        width, _, height = args.figsize.partition("x")
        figsize = (float(width), float(height))
    else:
        figsize = (6.0 * ncols, 4.0 * nrows)

    log_metrics = resolve_log_metrics(args.logy, metrics)
    yticks = parse_metric_floats(args.ytick, "--ytick")
    ylims = parse_metric_ranges(args.ylim, "--ylim")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    for ax, metric in zip(flat_axes, metrics):
        if metric in log_metrics:
            ax.set_yscale("log")
        for label, series in runs:
            values = series.get(metric)
            if values is None:
                # A run swept with different COVERAGE_RANKS simply has no such
                # column; plot the runs that do rather than failing the figure.
                print(f"[plot_spectral_curves] {label!r} has no {metric}, skipping that line", file=sys.stderr)
                continue
            ax.plot(series["step"], values, marker="o", markersize=4, label=label)
        apply_axis_overrides(ax, metric, yticks, ylims)
        ax.set_xlabel("Training step")
        ax.set_ylabel(_axis_label(metric))
        ax.grid(True, alpha=0.3)

    for ax in flat_axes[len(metrics) :]:
        ax.set_visible(False)

    # Title on top, legend anchored below it, axes below that. Anchoring the
    # legend explicitly matters: loc="upper center" alone parks it at the very
    # top of the figure, right on top of the suptitle.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    legend_ncol = args.legend_ncol or min(len(runs), 3)
    legend_rows = math.ceil(len(labels) / legend_ncol) if labels else 1

    fig.suptitle(args.title, y=args.title_y, fontsize=args.title_size)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, args.legend_y),
        ncol=legend_ncol,
        frameon=True,
    )
    # Reserve the band the legend actually occupies; a taller legend (more runs,
    # fewer columns) pushes the axes further down.
    top = args.top if args.top is not None else args.legend_y - 0.045 * legend_rows - 0.01
    fig.tight_layout(rect=(0, 0, 1, top))

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
