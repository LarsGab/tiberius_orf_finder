"""Slide-ready average-only vertebrate ORF benchmark plots.

Produces two PDFs (one per plot):

  benchmark_avg.pdf
      Locus-level Sensitivity vs Precision for all ORF tools, averaged across
      species.  "tiberius" is labelled "Tiberius-like ORFfinder".
      No StringTie upper-bound line.

  accuracy_avg.pdf
      Gene-level Sensitivity vs Precision for orf_prediction / tiberius /
      merged / braker3, averaged across species.
      "orf_prediction" is labelled "Tiberius-like ORFfinder".

Both plots: large fonts and markers suitable for a 16:9 slide.

Usage:
    python scripts/plot_slide_vertebrates.py \\
        --benchmark-table results/vertebrates_test/benchmark_orf_tools_filt_tpm1cov3len300/accuracy_table.tsv \\
        --accuracy-table  results/vertebrates_test/eval_accuracy_epoch_74_filt/accuracy_table.tsv \\
        --out-dir         results/slides/vertebrates
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Slide style constants
# ---------------------------------------------------------------------------
FS_TITLE  = 22
FS_LABEL  = 18
FS_TICK   = 16
FS_LEGEND = 15
FS_F1     = 11
MARKER_SIZE = 260   # scatter s= parameter


# ---------------------------------------------------------------------------
# Benchmark plot (ORF tool comparison)
# ---------------------------------------------------------------------------
BENCH_TOOL_LABELS = {
    "tiberius":              "Tiberius-like ORFfinder",
    "transdecoder1":         "TransDecoder v1",
    "transdecoder2":         "TransDecoder2",
    "transdecoder2_precise": "TransDecoder2 (--precise)",
    "gmst":                  "GeneMarkS-T",
}
BENCH_TOOL_ORDER = list(BENCH_TOOL_LABELS.keys())
BENCH_TOOL_COLORS = {
    "tiberius":              "#e41a1c",
    "transdecoder1":         "#377eb8",
    "transdecoder2":         "#4daf4a",
    "transdecoder2_precise": "#984ea3",
    "gmst":                  "#ff7f00",
}
BENCH_TOOL_MARKERS = {
    "tiberius":              "o",
    "transdecoder1":         "s",
    "transdecoder2":         "^",
    "transdecoder2_precise": "D",
    "gmst":                  "P",
}

# ---------------------------------------------------------------------------
# Combination/accuracy plot (orf_prediction vs tiberius vs merged vs braker3)
# ---------------------------------------------------------------------------
ACC_GS_KEEP = ["orf_prediction", "tiberius_filtered", "merged", "braker3"]
ACC_GS_LABELS = {
    "orf_prediction":    "Tiberius-like ORFfinder",
    "tiberius_filtered": "Tiberius (filtered)",
    "merged":            "Merged (Tiberius filtered + ORF)",
    "braker3":           "BRAKER3",
}
ACC_GS_MARKERS = {
    "orf_prediction":    "o",
    "tiberius_filtered": "s",
    "merged":            "^",
    "braker3":           "D",
}
ACC_GS_COLORS = {
    "orf_prediction":    "#e41a1c",
    "tiberius_filtered": "#377eb8",
    "merged":            "#4daf4a",
    "braker3":           "#984ea3",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _read_table(path: Path) -> list[dict]:
    text = path.read_text().splitlines()
    header = text[0].split("\t")
    rows: list[dict] = []
    for line in text[1:]:
        if not line.strip():
            continue
        vals = line.split("\t")
        d: dict = {}
        for k, v in zip(header, vals):
            if k in ("species", "tool", "gene_set"):
                d[k] = v
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = float("nan")
        rows.append(d)
    return rows


def _f1_isocontours(ax: plt.Axes) -> None:
    for f1 in (40, 60, 80):
        xs, ys = [], []
        for p in range(f1 // 2, 101):
            denom = 2 * p - f1
            if denom <= 0:
                continue
            s = f1 * p / denom
            if 0 <= s <= 100:
                xs.append(p)
                ys.append(s)
        if xs:
            ax.plot(xs, ys, "--", color="lightgray", linewidth=1.0, zorder=1)
            ax.text(xs[-1] - 1.5, ys[-1] + 1.0, f"F1={f1}%",
                    fontsize=FS_F1, color="darkgray", va="bottom", ha="right")


def _style_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=FS_TITLE, pad=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Precision (%)", fontsize=FS_LABEL)
    ax.set_ylabel("Sensitivity (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3, linewidth=0.8)
    _f1_isocontours(ax)


def _average_by_key(rows: list[dict], key: str,
                    s_col: str, p_col: str) -> dict[str, tuple[float, float]]:
    """Return {key_val: (mean_S, mean_P)} across all rows."""
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        kv = r.get(key)
        s, p = r.get(s_col, float("nan")), r.get(p_col, float("nan"))
        if kv and not (math.isnan(s) or math.isnan(p)):
            buckets[kv].append((s, p))
    return {
        kv: (sum(v[0] for v in pts) / len(pts),
             sum(v[1] for v in pts) / len(pts))
        for kv, pts in buckets.items()
    }


# ---------------------------------------------------------------------------
# Plot 1: ORF tool benchmark (average)
# ---------------------------------------------------------------------------
def plot_benchmark_avg(table_path: Path, out_path: Path,
                       level: str = "locus") -> None:
    rows = _read_table(table_path)
    avg = _average_by_key(rows, "tool", f"{level}_S", f"{level}_P")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    present = [t for t in BENCH_TOOL_ORDER if t in avg]
    for tool in present:
        s, p = avg[tool]
        ax.scatter(p, s,
                   color=BENCH_TOOL_COLORS.get(tool, "gray"),
                   marker=BENCH_TOOL_MARKERS.get(tool, "x"),
                   s=MARKER_SIZE, edgecolors="k", linewidths=1.2, zorder=3)

    _style_ax(ax, f"ORF-tool accuracy — {level.capitalize()} level")

    handles = [
        Line2D([0], [0], marker=BENCH_TOOL_MARKERS[t], color="w",
               markerfacecolor=BENCH_TOOL_COLORS[t], markeredgecolor="k",
               markersize=14, label=BENCH_TOOL_LABELS[t])
        for t in present
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND,
              title="Tool", title_fontsize=FS_LEGEND,
              loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: combination accuracy (average)
# ---------------------------------------------------------------------------
def plot_accuracy_avg(table_path: Path, out_path: Path,
                      level: str = "gene") -> None:
    rows = _read_table(table_path)
    avg = _average_by_key(rows, "gene_set", f"{level}_S", f"{level}_P")

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    present = [gs for gs in ACC_GS_KEEP if gs in avg]
    for gs in present:
        s, p = avg[gs]
        ax.scatter(p, s,
                   color=ACC_GS_COLORS.get(gs, "gray"),
                   marker=ACC_GS_MARKERS.get(gs, "x"),
                   s=MARKER_SIZE, edgecolors="k", linewidths=1.2, zorder=3)

    level_label = {"gene": "Gene (locus)", "transcript": "Transcript",
                   "exon": "Exon"}.get(level, level.capitalize())
    _style_ax(ax, f"ORF prediction accuracy — {level_label} level")

    handles = [
        Line2D([0], [0], marker=ACC_GS_MARKERS[gs], color="w",
               markerfacecolor=ACC_GS_COLORS[gs], markeredgecolor="k",
               markersize=14, label=ACC_GS_LABELS[gs])
        for gs in present
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND,
              title="Gene set", title_fontsize=FS_LEGEND,
              loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Wrote {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Slide-ready average-only vertebrate ORF benchmark plots."
    )
    ap.add_argument("--benchmark-table", type=Path, default=None,
                    help="accuracy_table.tsv from slurm_eval_orf_tool_benchmarks_*.sh "
                         "(columns: species tool base_S base_P exon_S exon_P "
                         "transcript_S transcript_P locus_S locus_P). "
                         "Generates benchmark_avg.pdf.")
    ap.add_argument("--accuracy-table", type=Path, default=None,
                    help="accuracy_table.tsv from evaluate_accuracy.py "
                         "(columns: species gene_set gene_S gene_P gene_F1 "
                         "transcript_S transcript_P transcript_F1 exon_S exon_P exon_F1). "
                         "Generates accuracy_avg.pdf.")
    ap.add_argument("--out-dir", type=Path, default=Path("results/slides/vertebrates"),
                    help="Output directory (default: results/slides/vertebrates).")
    ap.add_argument("--bench-level",
                    choices=["base", "exon", "transcript", "locus"],
                    default="locus",
                    help="Accuracy level for benchmark plot (default: locus).")
    ap.add_argument("--acc-level",
                    choices=["gene", "transcript", "exon"],
                    default="gene",
                    help="Accuracy level for combination plot (default: gene).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.benchmark_table is None and args.accuracy_table is None:
        raise SystemExit("Provide at least one of --benchmark-table or --accuracy-table.")

    if args.benchmark_table is not None:
        plot_benchmark_avg(
            args.benchmark_table,
            args.out_dir / "benchmark_avg.pdf",
            level=args.bench_level,
        )

    if args.accuracy_table is not None:
        plot_accuracy_avg(
            args.accuracy_table,
            args.out_dir / "accuracy_avg.pdf",
            level=args.acc_level,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
