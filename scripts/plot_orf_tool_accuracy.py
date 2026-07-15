"""Plot Sensitivity vs Precision (at a chosen gffcompare level) from an
ORF-tool accuracy table.

Reads the TSV produced by
``slurm_eval_orf_tool_benchmarks_vertebrates_test[_filt_tpm1cov3len300].sh``
(one row per (species, tool) with base/exon/transcript/locus S/P
columns) and writes a scatter plot at the requested level — one
subplot per species plus an average.

Usage::

    python scripts/plot_orf_tool_accuracy.py \\
        --table results/vertebrates_test/benchmark_orf_tools_filt_tpm1cov3len300/accuracy_table.tsv \\
        --level locus \\
        --out   results/vertebrates_test/benchmark_orf_tools_filt_tpm1cov3len300/locus_accuracy.pdf
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


TOOL_LABELS = {
    "tiberius":              "Tiberius (subseq-filt)",
    "transdecoder1":         "TransDecoder v1",
    "transdecoder2":         "TransDecoder2",
    "transdecoder2_precise": "TransDecoder2 (--precise)",
    "gmst":                  "GeneMarkS-T",
}
TOOL_ORDER = list(TOOL_LABELS.keys())
TOOL_COLORS = {
    "tiberius":              "#e41a1c",   # red
    "transdecoder1":         "#377eb8",   # blue
    "transdecoder2":         "#4daf4a",   # green
    "transdecoder2_precise": "#984ea3",   # purple
    "gmst":                  "#ff7f00",   # orange
}
TOOL_MARKERS = {
    "tiberius":              "o",
    "transdecoder1":         "s",
    "transdecoder2":         "^",
    "transdecoder2_precise": "D",
    "gmst":                  "P",
}


_LEVEL_LABELS = {
    "base":       "Base",
    "exon":       "Exon",
    "transcript": "Transcript",
    "locus":      "Locus",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot Sensitivity vs Precision (chosen gffcompare "
                    "level) from an ORF-tool accuracy TSV.",
    )
    ap.add_argument("--table", type=Path, required=True,
                    help="accuracy_table.tsv (columns: species tool "
                         "base_S base_P exon_S exon_P transcript_S "
                         "transcript_P locus_S locus_P).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output plot (any matplotlib extension; PDF "
                         "recommended).")
    ap.add_argument("--level", choices=list(_LEVEL_LABELS.keys()),
                    default="locus",
                    help="Which gffcompare accuracy level to plot "
                         "(default: locus).")
    ap.add_argument("--title", type=str, default=None,
                    help="Figure title (default derived from --level).")
    ap.add_argument("--upper-bound-tsv", type=Path, default=None,
                    help="Optional per-species upper-bound TSV (columns "
                         "species base_S exon_S transcript_S locus_S) "
                         "produced by compute_stringtie_upper_bound.py. "
                         "Draws a horizontal dashed line at the "
                         "corresponding sensitivity in each subplot.")
    return ap.parse_args(argv)


def _read_table(path: Path) -> list[dict]:
    """Parse the TSV into a list of dicts, one per row."""
    text = path.read_text().splitlines()
    header = text[0].split("\t")
    rows: list[dict] = []
    for line in text[1:]:
        if not line.strip():
            continue
        vals = line.split("\t")
        d: dict = {}
        for k, v in zip(header, vals):
            if k in ("species", "tool"):
                d[k] = v
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = float("nan")
        rows.append(d)
    return rows


def _plot_species(ax: plt.Axes, rows: list[dict], title: str,
                  level: str = "locus",
                  upper_bound: float | None = None) -> None:
    """Draw one subplot: <level>_P (x) vs <level>_S (y).

    If ``upper_bound`` is given (0-100), overlay a horizontal dashed
    line at that sensitivity (per-species StringTie ceiling).
    """
    p_key = f"{level}_P"
    s_key = f"{level}_S"
    for row in rows:
        tool = row["tool"]
        p = row.get(p_key, float("nan"))
        s = row.get(s_key, float("nan"))
        if math.isnan(p) or math.isnan(s):
            continue
        ax.scatter(
            p, s,
            color=TOOL_COLORS.get(tool, "gray"),
            marker=TOOL_MARKERS.get(tool, "x"),
            s=90, edgecolors="k", linewidths=0.6, zorder=3,
        )
    if upper_bound is not None and not math.isnan(upper_bound):
        ax.axhline(
            upper_bound, linestyle="--", color="#555555",
            linewidth=1.1, zorder=2,
            label=f"StringTie upper bound ({upper_bound:.1f}%)",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Precision (%)", fontsize=8)
    ax.set_ylabel("Sensitivity (%)", fontsize=8)
    ax.grid(alpha=0.35, linewidth=0.6)
    ax.tick_params(labelsize=7)
    # F1 iso-lines (dashed) at 40/60/80 to give visual anchors.
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
            ax.plot(xs, ys, "--", color="lightgray", linewidth=0.7, zorder=1)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _read_table(args.table)
    if not rows:
        raise SystemExit(f"No rows in {args.table}")

    # Group per species; preserve first-seen order.
    rows_by_sp: dict[str, list[dict]] = defaultdict(list)
    species_order: list[str] = []
    for r in rows:
        sp = r["species"]
        if sp not in rows_by_sp:
            species_order.append(sp)
        rows_by_sp[sp].append(r)

    level = args.level
    s_key = f"{level}_S"
    p_key = f"{level}_P"

    # Average row per tool: mean of <level>_S and <level>_P across species.
    tool_vals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        tool = r["tool"]
        s, p = r.get(s_key), r.get(p_key)
        if s is not None and p is not None and not math.isnan(s) and not math.isnan(p):
            tool_vals[tool].append((s, p))
    avg_rows: list[dict] = []
    for tool, vals in tool_vals.items():
        if vals:
            avg_rows.append({
                "species": "Average",
                "tool": tool,
                s_key: sum(v[0] for v in vals) / len(vals),
                p_key: sum(v[1] for v in vals) / len(vals),
            })

    n_sp = len(species_order)
    n_plots = n_sp + 1
    ncols = min(3, n_plots)
    nrows = math.ceil(n_plots / ncols)

    fig, axes_arr = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.6, nrows * 3.6),
        squeeze=False,
    )
    axes_flat = [axes_arr[r][c] for r in range(nrows) for c in range(ncols)]

    # Optional per-species upper-bound sensitivity for horizontal line.
    ub_by_sp: dict[str, float] = {}
    if args.upper_bound_tsv and args.upper_bound_tsv.exists():
        ub_rows = _read_table(args.upper_bound_tsv)
        for r in ub_rows:
            v = r.get(s_key)
            if isinstance(v, float) and not math.isnan(v):
                ub_by_sp[r["species"]] = v

    for i, sp in enumerate(species_order):
        _plot_species(axes_flat[i], rows_by_sp[sp],
                      sp.replace("_", " "), level=level,
                      upper_bound=ub_by_sp.get(sp))
    avg_ub = (sum(ub_by_sp[sp] for sp in species_order if sp in ub_by_sp)
              / len(ub_by_sp)) if ub_by_sp else None
    _plot_species(axes_flat[n_sp], avg_rows, "Average", level=level,
                  upper_bound=avg_ub)
    for j in range(n_plots, nrows * ncols):
        axes_flat[j].set_visible(False)

    # Which tools actually appear in the data (in canonical order).
    present_tools = [t for t in TOOL_ORDER if t in {r["tool"] for r in rows}]
    legend_handles = [
        Line2D(
            [0], [0],
            marker=TOOL_MARKERS[t], color="w",
            markerfacecolor=TOOL_COLORS[t],
            markeredgecolor="k", markersize=9,
            label=TOOL_LABELS[t],
        )
        for t in present_tools
    ]
    if ub_by_sp:
        legend_handles.append(Line2D(
            [0], [0], color="#555555", linestyle="--", linewidth=1.1,
            label="StringTie upper bound",
        ))
    fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.01, 0.5), loc="center left",
        frameon=True, fontsize=9, title="Tool",
    )
    title = args.title or (
        f"ORF-tool {_LEVEL_LABELS[level].lower()}-level accuracy "
        f"(gffcompare -e 3)"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.85, 0.97))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
