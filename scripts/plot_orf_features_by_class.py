"""Visualize ORF feature distributions by gffcompare class across species.

Loads orf_features.tsv files from all vertebrate test species and produces a
multi-page PDF with:
  - Page 1 : Class frequency table and bar chart (per species + pooled)
  - Page 2 : Numeric feature distributions (violin/box) per class
  - Page 3 : Binary / categorical feature rates per class
  - Page 4 : 2-D scatter panels for feature pairs, coloured by class
  - Page 5 : Per-species class breakdown heatmap (fraction of each class)

Usage
-----
python scripts/plot_orf_features_by_class.py \\
  --base-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \\
  --out-pdf   results/figures/orf_features_by_class.pdf

Optional
--------
--classes  Space-separated subset of gffcompare codes to include (default: all)
--min-count Minimum number of transcripts a class must have to be plotted (default: 20)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgba
from scipy import stats

# ─── gffcompare class metadata ─────────────────────────────────────────────
CLASS_LABELS = {
    "=": "= exact match",
    "~": "~ single-exon match",
    "j": "j junction overlap",
    "c": "c contained in ref",
    "k": "k contains ref",
    "o": "o generic overlap",
    "e": "e exon/intron overlap",
    "m": "m multi-overlap",
    "n": "n partial overlap",
    "p": "p run-on",
    "i": "i within intron",
    "u": "u intergenic",
    "x": "x antisense",
    "s": "s shadow overlap",
    "y": "y contains ref intron",
}
# Rough quality grouping for colour coding
CLASS_GROUP = {
    "=": "correct", "~": "correct",
    "j": "partial", "c": "partial", "k": "partial",
    "o": "other", "e": "other", "m": "other", "n": "other", "p": "other",
    "i": "wrong", "u": "wrong", "x": "wrong", "s": "wrong", "y": "wrong",
}
GROUP_PALETTE = {"correct": "#2ca02c", "partial": "#ff7f0e", "other": "#1f77b4", "wrong": "#d62728"}

NUMERIC_FEATURES = [
    ("cds_length_nt",           "CDS length (nt)"),
    ("n_exons",                 "# exons"),
    ("dist_upstream_stop_nt",   "Upstream stop dist (nt)"),
    ("n_upstream_atgs",         "# upstream ATGs"),
    ("frac_introns_supported",  "Fraction introns supported"),
    ("best_identity",           "Best protein identity"),
    ("best_norm_bitscore",      "Best norm bitscore"),
    ("best_target_coverage",    "Best target coverage"),
    ("cds_length_pct",          "CDS / transcript length"),
    ("n_overlapping_alignments","# overlapping alignments"),
]
BINARY_FEATURES = [
    ("has_protein_support",     "Has protein support"),
    ("has_start_hint",          "Has start hint"),
    ("has_stop_hint",           "Has stop hint"),
    ("has_conflict",            "Has conflict"),
    ("has_upstream_partner",    "Has upstream partner"),
    ("has_downstream_partner",  "Has downstream partner"),
]
CATEGORICAL_FEATURES = [
    ("lorf_class",   "LORF class"),
    ("support_level","Support level"),
]


# ─── helpers ───────────────────────────────────────────────────────────────

def load_data(base_dir: Path, annot_tag: str) -> pd.DataFrame:
    frames = []
    for sp_dir in sorted(base_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        tsv = sp_dir / annot_tag / "orf_features.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        df["species"] = sp_dir.name
        frames.append(df)
    if not frames:
        sys.exit(f"No orf_features.tsv files found under {base_dir}/{annot_tag}")
    data = pd.concat(frames, ignore_index=True)
    # coerce numeric columns silently
    for col, _ in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col, _ in BINARY_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def class_colour(code: str) -> str:
    group = CLASS_GROUP.get(code, "other")
    return GROUP_PALETTE[group]


def ordered_classes(df: pd.DataFrame, min_count: int) -> list[str]:
    counts = df["gffcompare_class"].value_counts()
    ordered = ["=", "~", "j", "c", "k", "o", "e", "m", "n", "p", "i", "u", "x", "s", "y"]
    return [c for c in ordered if c in counts.index and counts[c] >= min_count]


# ─── page builders ─────────────────────────────────────────────────────────

def page_class_frequency(df: pd.DataFrame, classes: list[str], pdf: PdfPages):
    species_list = sorted(df["species"].unique())
    n_sp = len(species_list)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("gffcompare class frequency", fontsize=14, fontweight="bold")

    # Pooled bar
    ax = axes[0]
    counts = df[df["gffcompare_class"].isin(classes)]["gffcompare_class"].value_counts().reindex(classes, fill_value=0)
    colours = [class_colour(c) for c in classes]
    bars = ax.bar(classes, counts.values, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("gffcompare class")
    ax.set_ylabel("# transcripts (all species)")
    ax.set_title(f"Pooled  (N={len(df):,})")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Per-species stacked bar (fraction)
    ax2 = axes[1]
    sp_frac = (
        df[df["gffcompare_class"].isin(classes)]
        .groupby(["species", "gffcompare_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=classes, fill_value=0)
    )
    sp_frac = sp_frac.div(sp_frac.sum(axis=1), axis=0)
    bottom = np.zeros(n_sp)
    sp_names = [s.replace("_", " ") for s in sp_frac.index]
    for cls in classes:
        vals = sp_frac[cls].values if cls in sp_frac.columns else np.zeros(n_sp)
        ax2.bar(range(n_sp), vals, bottom=bottom, color=class_colour(cls),
                edgecolor="black", linewidth=0.3, label=cls)
        bottom += vals
    ax2.set_xticks(range(n_sp))
    ax2.set_xticklabels(sp_names, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Fraction of transcripts")
    ax2.set_title("Per-species composition")
    ax2.legend(title="class", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_numeric_violins(df: pd.DataFrame, classes: list[str], pdf: PdfPages):
    n_feat = len(NUMERIC_FEATURES)
    n_cols = 2
    n_rows = (n_feat + n_cols - 1) // n_cols
    colours = [class_colour(c) for c in classes]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    fig.suptitle("Numeric features by gffcompare class", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    sub = df[df["gffcompare_class"].isin(classes)].copy()

    for idx, (col, label) in enumerate(NUMERIC_FEATURES):
        ax = axes_flat[idx]
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        data_by_class = [sub.loc[sub["gffcompare_class"] == c, col].dropna().values for c in classes]
        # only pass non-empty arrays to violinplot
        valid_pos = [i for i, d in enumerate(data_by_class) if len(d) > 0]
        valid_data = [data_by_class[i] for i in valid_pos]
        if not valid_data:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(valid_data, positions=valid_pos,
                              showmedians=True, showextrema=False)
        for pc, pos in zip(parts["bodies"], valid_pos):
            pc.set_facecolor(colours[pos])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label, fontsize=8)
        # annotate median value
        for i, d in enumerate(data_by_class):
            if len(d) > 0:
                med = np.median(d)
                ax.text(i, ax.get_ylim()[1] * 0.97, f"{med:.2g}",
                        ha="center", va="top", fontsize=6, color="black")

    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_binary_rates(df: pd.DataFrame, classes: list[str], pdf: PdfPages):
    sub = df[df["gffcompare_class"].isin(classes)].copy()
    n_feat = len(BINARY_FEATURES) + len(CATEGORICAL_FEATURES)
    n_cols = 2
    n_rows = (n_feat + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    fig.suptitle("Binary / categorical features by gffcompare class", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()
    colours = [class_colour(c) for c in classes]

    idx = 0
    # Binary: fraction True per class
    for col, label in BINARY_FEATURES:
        ax = axes_flat[idx]
        if col not in sub.columns:
            ax.set_visible(False)
            idx += 1
            continue
        rates = [sub.loc[sub["gffcompare_class"] == c, col].mean() for c in classes]
        ax.bar(classes, rates, color=colours, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Fraction True")
        ax.set_xlabel("gffcompare class")
        for i, r in enumerate(rates):
            ax.text(i, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
        idx += 1

    # Categorical: grouped stacked bars
    for col, label in CATEGORICAL_FEATURES:
        ax = axes_flat[idx]
        if col not in sub.columns:
            ax.set_visible(False)
            idx += 1
            continue
        cat_vals = sorted(sub[col].dropna().unique())
        frac_mat = (
            sub.groupby(["gffcompare_class", col])
            .size()
            .unstack(fill_value=0)
            .reindex(index=classes, columns=cat_vals, fill_value=0)
        )
        frac_mat = frac_mat.div(frac_mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        cmap = plt.cm.get_cmap("tab10", len(cat_vals))
        bottom = np.zeros(len(classes))
        for j, cat in enumerate(cat_vals):
            vals = frac_mat[cat].values
            ax.bar(range(len(classes)), vals, bottom=bottom,
                   color=cmap(j), edgecolor="black", linewidth=0.3, label=str(cat))
            bottom += vals
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Fraction")
        ax.legend(fontsize=7, loc="upper right")
        idx += 1

    for j in range(idx, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_scatter_pairs(df: pd.DataFrame, classes: list[str], pdf: PdfPages):
    sub = df[df["gffcompare_class"].isin(classes)].copy()
    pairs = [
        ("best_norm_bitscore", "frac_introns_supported"),
        ("cds_length_pct",     "best_identity"),
        ("dist_upstream_stop_nt", "n_upstream_atgs"),
        ("best_norm_bitscore", "cds_length_pct"),
        ("frac_introns_supported", "cds_length_pct"),
        ("best_identity",      "frac_introns_supported"),
    ]
    pairs = [(x, y) for x, y in pairs if x in sub.columns and y in sub.columns]
    n_cols = 2
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    fig.suptitle("Feature pairs coloured by gffcompare class", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten()

    # subsample to avoid overplotting
    max_pts = 5000
    for idx, (xcol, ycol) in enumerate(pairs):
        ax = axes_flat[idx]
        for cls in reversed(classes):  # plot "correct" on top
            mask = sub["gffcompare_class"] == cls
            pts = sub[mask][[xcol, ycol]].dropna()
            if len(pts) > max_pts:
                pts = pts.sample(max_pts, random_state=42)
            ax.scatter(pts[xcol], pts[ycol], s=4, alpha=0.4,
                       color=class_colour(cls), label=cls, rasterized=True)
        ax.set_xlabel(xcol, fontsize=9)
        ax.set_ylabel(ycol, fontsize=9)
        ax.set_title(f"{xcol} vs {ycol}", fontsize=9)
        ax.legend(markerscale=2, fontsize=7, loc="best")

    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_species_heatmap(df: pd.DataFrame, classes: list[str], pdf: PdfPages):
    species_list = sorted(df["species"].unique())
    sub = df[df["gffcompare_class"].isin(classes)]
    mat = (
        sub.groupby(["species", "gffcompare_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=species_list, columns=classes, fill_value=0)
    )
    mat_frac = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(species_list) * 0.7 + 2)))
    fig.suptitle("Per-species class breakdown", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], mat,      "# transcripts",        "{:.0f}"),
        (axes[1], mat_frac, "Fraction per species",  "{:.2f}"),
    ]:
        im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(range(len(species_list)))
        ax.set_yticklabels([s.replace("_", " ") for s in species_list], fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data.values[r, c]
                ax.text(c, r, fmt.format(val), ha="center", va="center",
                        fontsize=6, color="black" if val < data.values.max() * 0.7 else "white")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_rescue_candidates(df: pd.DataFrame, classes: list[str], pdf: PdfPages):
    """Feature distributions focusing on partial-match classes vs exact match."""
    rescue_classes = [c for c in ["j", "c", "k"] if c in classes]
    correct_classes = [c for c in ["=", "~"] if c in classes]
    if not rescue_classes:
        return

    sub = df[df["gffcompare_class"].isin(rescue_classes + correct_classes)].copy()
    sub["group"] = sub["gffcompare_class"].map(
        lambda x: "correct (=,~)" if x in correct_classes else f"partial ({','.join(rescue_classes)})"
    )

    n_feat = len(NUMERIC_FEATURES)
    n_cols = 2
    n_rows = (n_feat + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    fig.suptitle("Rescue candidate analysis: partial-match vs exact-match features",
                 fontsize=12, fontweight="bold")
    axes_flat = axes.flatten()
    groups = sub["group"].unique()
    gcolours = {"correct (=,~)": "#2ca02c"}
    for cls in rescue_classes:
        gcolours[f"partial ({','.join(rescue_classes)})"] = "#ff7f0e"

    for idx, (col, label) in enumerate(NUMERIC_FEATURES):
        ax = axes_flat[idx]
        if col not in sub.columns:
            ax.set_visible(False)
            continue
        data_by_group = [sub.loc[sub["group"] == g, col].dropna().values for g in groups]
        valid_pos = [i for i, d in enumerate(data_by_group) if len(d) > 0]
        valid_data = [data_by_group[i] for i in valid_pos]
        if not valid_data:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(valid_data, positions=valid_pos,
                              showmedians=True, showextrema=False)
        for pc, pos in zip(parts["bodies"], valid_pos):
            pc.set_facecolor(gcolours.get(list(groups)[pos], "#1f77b4"))
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[:30] for g in groups], fontsize=8)
        ax.set_title(label, fontsize=9)

        # Mann-Whitney p-value if two groups
        if len(data_by_group) == 2 and len(data_by_group[0]) > 0 and len(data_by_group[1]) > 0:
            _, p = stats.mannwhitneyu(data_by_group[0], data_by_group[1], alternative="two-sided")
            ax.set_xlabel(f"p={p:.2e}", fontsize=7)

    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir",  required=True, type=Path,
                   help="Root of vertebrates_test results (contains one dir per species)")
    p.add_argument("--annot-tag", default="annotate_epoch_74_filt_tpm1cov3len300_lorf",
                   help="Subdirectory name that contains orf_features.tsv")
    p.add_argument("--out-pdf",   default="results/figures/orf_features_by_class.pdf", type=Path)
    p.add_argument("--classes",   nargs="+", default=None,
                   help="gffcompare codes to include (default: all with >= --min-count)")
    p.add_argument("--min-count", type=int, default=20,
                   help="Min transcripts a class needs to appear in plots")
    args = p.parse_args()

    print(f"Loading data from {args.base_dir} / {args.annot_tag} …", flush=True)
    df = load_data(args.base_dir, args.annot_tag)
    print(f"  Loaded {len(df):,} transcripts from {df['species'].nunique()} species", flush=True)

    classes = args.classes if args.classes else ordered_classes(df, args.min_count)
    print(f"  Classes to plot: {classes}", flush=True)

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        print("Page 1: class frequency …", flush=True)
        page_class_frequency(df, classes, pdf)
        print("Page 2: numeric feature violins …", flush=True)
        page_numeric_violins(df, classes, pdf)
        print("Page 3: binary / categorical rates …", flush=True)
        page_binary_rates(df, classes, pdf)
        print("Page 4: feature scatter pairs …", flush=True)
        page_scatter_pairs(df, classes, pdf)
        print("Page 5: per-species heatmap …", flush=True)
        page_species_heatmap(df, classes, pdf)
        print("Page 6: rescue candidate analysis …", flush=True)
        page_rescue_candidates(df, classes, pdf)

    print(f"Done. PDF written to {args.out_pdf}", flush=True)


if __name__ == "__main__":
    main()
