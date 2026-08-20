"""Plot ORFfinder scores for Tiberius ab initio predictions.

Loads per-species score TSVs produced by score_tiberius.py and generates
a multi-panel PDF. When reference annotations are provided, each Tiberius
transcript is classified as TP (class_code '=' in gffcompare) or FP, and
the output includes TP/FP violin plots, ROC curves, and an AUC summary.

Usage
-----
# Score distributions only (no TP/FP):
python scripts/plot_tiberius_scores.py \\
  --score-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --score-tag score_tiberius_epoch_74_up500 \\
  --out-pdf   results/figures/tiberius_orf_scores.pdf

# With TP/FP classification:
python scripts/plot_tiberius_scores.py \\
  --score-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --score-tag score_tiberius_epoch_74_up500 \\
  --tib-tmpl  '/home/gabriell/tiberius_benchmarking/paper/Vertebrata/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf' \\
  --ref-tmpl  '/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/assembly/annot_cds.gff' \\
  --out-pdf   results/figures/tiberius_orf_scores_tp_fp.pdf
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SPECIES = [
    "Gallus_gallus",
    "Pristiophorus_japonicus",
    "Bos_taurus",
    "Delphinapterus_leucas",
    "Homo_sapiens",
]

_SCORE_COLS = [
    "mean_coding_prob",
    "start_prob",
    "stop_prob",
    "mean_ir_prob",
    "frac_argmax_coding",
]

_COL_LABELS = {
    "mean_coding_prob":    "Mean P(coding) over CDS",
    "start_prob":          "P(START) at ATG",
    "stop_prob":           "P(STOP) at last CDS base",
    "mean_ir_prob":        "Mean P(IR) over CDS",
    "frac_argmax_coding":  "Frac. positions argmax=coding",
}

_SP_SHORT = {
    "Gallus_gallus":           "G.gal",
    "Pristiophorus_japonicus": "P.jap",
    "Bos_taurus":              "B.tau",
    "Delphinapterus_leucas":   "D.leu",
    "Homo_sapiens":            "H.sap",
}

_TP_COLOR = "#1976D2"   # blue
_FP_COLOR = "#D32F2F"   # red


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_scores(score_dir: Path, score_tag: str) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for sp in _SPECIES:
        tsv = score_dir / sp / score_tag / "scores.tsv"
        if not tsv.exists():
            print(f"[plot] missing {tsv}, skipping {sp}")
            continue
        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        df["species"] = sp
        dfs[sp] = df
        print(f"[plot] {sp}: {len(df)} transcripts")
    return dfs


# ---------------------------------------------------------------------------
# gffcompare TP/FP classification
# ---------------------------------------------------------------------------

def _run_gffcompare(tib_gtf: Path, ref_gff: Path, tmpdir: Path) -> set[str]:
    """Return set of qry transcript_ids with class_code '=' (exact CDS match).

    With a single query file gffcompare writes a .tracking file, not .tmap.
    Tracking format (tab-separated):
      col0  TCONS_ID
      col1  XLOC_ID
      col2  ref_gene|ref_id
      col3  class_code          ← '=' means exact match
      col4  q1:<gene>|<tid>|num_exons|FPKM|TPM|cov|len
    Transcript id is the second pipe-field after stripping the 'q1:' prefix.
    """
    prefix = tmpdir / "gc"
    cmd = [
        "gffcompare", "--strict-match", "-e", "3",
        "-r", str(ref_gff), "-o", str(prefix), str(tib_gtf),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[plot]   gffcompare failed:\n{result.stderr[:500]}")
        return set()

    tracking = Path(str(prefix) + ".tracking")
    if not tracking.exists():
        print(f"[plot]   WARNING: no .tracking file found in {tmpdir}")
        return set()

    tp_ids: set[str] = set()
    for line in tracking.read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        # class_code at col 3; query info at col 4 as "q1:<gene>|<tid>|..."
        if len(parts) < 5 or parts[3] != "=":
            continue
        try:
            qry_part = parts[4].split(":", 1)[1]   # strip "q1:"
            tp_ids.add(qry_part.split("|")[1])      # transcript_id
        except (IndexError, ValueError):
            continue
    return tp_ids


def _add_tp_labels(
    dfs: dict[str, pd.DataFrame],
    tib_tmpl: str,
    ref_tmpl: str,
) -> dict[str, pd.DataFrame]:
    for sp, df in dfs.items():
        tib_gtf = Path(tib_tmpl.format(sp=sp))
        ref_gff = Path(ref_tmpl.format(sp=sp))
        if not tib_gtf.exists() or not ref_gff.exists():
            print(f"[plot] {sp}: missing GTF or reference — skipping TP labels")
            df["tp"] = pd.NA
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            tp_ids = _run_gffcompare(tib_gtf, ref_gff, Path(tmpdir))
        df["tp"] = df["transcript_id"].isin(tp_ids)
        n_tp = int(df["tp"].sum())
        n_fp = int((~df["tp"]).sum())
        print(f"[plot] {sp}: TP={n_tp:,}  FP={n_fp:,}  total={len(df):,}")
    return dfs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal AUC without sklearn. labels: 0/1, scores: higher = more likely TP."""
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.cumsum(labels) / n_pos
    fpr = np.cumsum(1 - labels) / n_neg
    return float(np.trapz(tpr, fpr))


def _plot_violins(dfs: dict[str, pd.DataFrame], fig, n_rows, row_offset) -> None:
    """One row of violin plots per score metric, split TP/FP per species."""
    species_list = sorted(dfs.keys())
    n_sp   = len(species_list)
    n_cols = len(_SCORE_COLS)

    for ci, score in enumerate(_SCORE_COLS):
        ax = fig.add_subplot(n_rows, n_cols, row_offset * n_cols + ci + 1)
        positions_tp, positions_fp = [], []
        data_tp,      data_fp      = [], []
        xticks, xlabels = [], []

        for si, sp in enumerate(species_list):
            df = dfs[sp].dropna(subset=[score])
            has_tp = "tp" in df.columns and df["tp"].notna().any()
            x_base = si * 3
            xticks.append(x_base + 1)
            xlabels.append(_SP_SHORT.get(sp, sp))

            if has_tp:
                tp_vals = df.loc[df["tp"] == True,  score].dropna().values
                fp_vals = df.loc[df["tp"] == False, score].dropna().values
                positions_tp.append(x_base)
                positions_fp.append(x_base + 2)
                data_tp.append(tp_vals)
                data_fp.append(fp_vals)
            else:
                all_vals = df[score].dropna().values
                vp = ax.violinplot([all_vals], positions=[x_base + 1],
                                   showmedians=True, widths=1.8)
                for pc in vp["bodies"]:
                    pc.set_facecolor("#888888")
                    pc.set_alpha(0.6)

        if data_tp:
            # Filter out empty arrays (species where one class has 0 members)
            tp_pairs = [(d, p) for d, p in zip(data_tp, positions_tp) if len(d) > 0]
            fp_pairs = [(d, p) for d, p in zip(data_fp, positions_fp) if len(d) > 0]
            if tp_pairs:
                d_tp, p_tp = zip(*tp_pairs)
                vp_tp = ax.violinplot(list(d_tp), positions=list(p_tp),
                                      showmedians=True, widths=1.2)
                for pc in vp_tp["bodies"]:
                    pc.set_facecolor(_TP_COLOR); pc.set_alpha(0.7)
            if fp_pairs:
                d_fp, p_fp = zip(*fp_pairs)
                vp_fp = ax.violinplot(list(d_fp), positions=list(p_fp),
                                      showmedians=True, widths=1.2)
                for pc in vp_fp["bodies"]:
                    pc.set_facecolor(_FP_COLOR); pc.set_alpha(0.7)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(score.replace("_", "\n"), fontsize=7)
        ax.set_title(_COL_LABELS[score], fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        if ci == 0 and data_tp:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=_TP_COLOR, label="TP"),
                                Patch(color=_FP_COLOR, label="FP")],
                      fontsize=7, loc="upper left")


def _plot_roc_curves(dfs: dict[str, pd.DataFrame], fig, n_rows, row_offset) -> None:
    """One ROC curve per species, for the three most discriminative scores."""
    species_list = sorted(dfs.keys())
    n_sp   = len(species_list)
    scores_to_plot = ["mean_coding_prob", "start_prob", "frac_argmax_coding"]
    colors = ["#1565C0", "#E65100", "#2E7D32"]

    for si, sp in enumerate(species_list):
        ax = fig.add_subplot(n_rows, n_sp, row_offset * n_sp + si + 1)
        df = dfs[sp].dropna(subset=["tp"])
        if df.empty or df["tp"].nunique() < 2:
            ax.text(0.5, 0.5, "no TP/FP data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_title(_SP_SHORT.get(sp, sp), fontsize=8)
            continue

        labels = df["tp"].astype(int).values
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)

        for score, color in zip(scores_to_plot, colors):
            vals = df[score].fillna(0).values
            # flip mean_ir_prob (lower = better)
            if score == "mean_ir_prob":
                vals = -vals
            order  = np.argsort(vals)[::-1]
            labs_s = labels[order]
            n_pos  = labs_s.sum()
            n_neg  = len(labs_s) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            tpr = np.cumsum(labs_s) / n_pos
            fpr = np.cumsum(1 - labs_s) / n_neg
            auc = float(np.trapz(tpr, fpr))
            ax.plot(fpr, tpr, color=color, lw=1.2,
                    label=f"{_COL_LABELS[score].split()[0]} ({auc:.2f})")

        ax.set_xlabel("FPR", fontsize=7)
        if si == 0:
            ax.set_ylabel("TPR", fontsize=7)
        ax.set_title(_SP_SHORT.get(sp, sp), fontsize=8)
        ax.legend(fontsize=6, loc="lower right")
        ax.tick_params(labelsize=7)


def _plot_auc_summary(dfs: dict[str, pd.DataFrame], fig, n_rows, row_offset) -> None:
    """Bar chart: AUC per (score, species)."""
    species_list = sorted(dfs.keys())
    n_sp   = len(species_list)
    n_sc   = len(_SCORE_COLS)
    width  = 0.8 / n_sc
    cmap   = plt.cm.get_cmap("tab10", n_sc)

    ax = fig.add_subplot(n_rows, 1, row_offset + 1)

    for si, score in enumerate(_SCORE_COLS):
        aucs = []
        for sp in species_list:
            df = dfs[sp].dropna(subset=[score, "tp"])
            if df.empty or df["tp"].nunique() < 2:
                aucs.append(float("nan"))
                continue
            labels = df["tp"].astype(int).values
            vals   = df[score].fillna(0).values
            if score == "mean_ir_prob":
                vals = -vals  # lower IR = better TP predictor
            aucs.append(_auc_from_scores(labels, vals))
        xs = np.arange(n_sp) + si * width
        ax.bar(xs, aucs, width=width * 0.9, color=cmap(si),
               label=_COL_LABELS[score], alpha=0.85)

    ax.axhline(0.5, color="black", lw=0.8, linestyle="--", label="random")
    ax.set_xticks(np.arange(n_sp) + (n_sc - 1) * width / 2)
    ax.set_xticklabels([_SP_SHORT.get(s, s) for s in species_list], fontsize=9)
    ax.set_ylabel("AUC  (TP vs FP classification)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_title("ORFfinder score AUC — Tiberius TP vs FP", fontsize=10)


def _plot_scatter(dfs: dict[str, pd.DataFrame], fig, n_rows, row_offset) -> None:
    """P(START) vs P(STOP) scatter coloured by TP/FP."""
    ax = fig.add_subplot(n_rows, 1, row_offset + 1)
    species_list = sorted(dfs.keys())
    markers = ["o", "s", "^", "D", "v"]
    colors_sp = plt.cm.tab10.colors

    has_tp_any = False
    for ci, sp in enumerate(species_list):
        df = dfs[sp].dropna(subset=["start_prob", "stop_prob"])
        has_tp = "tp" in df.columns and df["tp"].notna().any()
        if has_tp:
            has_tp_any = True
            tp = df[df["tp"] == True]
            fp = df[df["tp"] == False]
            ax.scatter(tp["start_prob"], tp["stop_prob"],
                       s=4, alpha=0.35, color=_TP_COLOR,
                       marker=markers[ci % len(markers)],
                       label=f"{_SP_SHORT.get(sp,sp)} TP")
            ax.scatter(fp["start_prob"], fp["stop_prob"],
                       s=4, alpha=0.12, color=_FP_COLOR,
                       marker=markers[ci % len(markers)])
        else:
            ax.scatter(df["start_prob"], df["stop_prob"],
                       s=4, alpha=0.2, color=colors_sp[ci],
                       marker=markers[ci % len(markers)],
                       label=_SP_SHORT.get(sp, sp))

    if has_tp_any:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(color=_TP_COLOR, label="TP"), Patch(color=_FP_COLOR, label="FP")],
            fontsize=8, loc="upper left",
        )
    else:
        ax.legend(fontsize=8, markerscale=3, loc="upper left")

    ax.set_xlabel("P(START) at ATG", fontsize=9)
    ax.set_ylabel("P(STOP) at last CDS base", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("START vs STOP probability per transcript", fontsize=10)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot ORFfinder scores for Tiberius ab initio predictions."
    )
    ap.add_argument("--score-dir", type=Path, required=True,
                    help="Root results dir containing <species>/<score-tag>/scores.tsv.")
    ap.add_argument("--score-tag", default="score_tiberius_epoch_74_up500",
                    help="Subdirectory name under each species dir.")
    ap.add_argument("--out-pdf",   type=Path, required=True)
    ap.add_argument("--tib-tmpl",  default=None,
                    help="Path template for Tiberius GTF with {sp} placeholder.")
    ap.add_argument("--ref-tmpl",  default=None,
                    help="Path template for reference GFF with {sp} placeholder.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    dfs = _load_scores(args.score_dir, args.score_tag)
    if not dfs:
        print("[plot] No score TSVs found.")
        return 1

    has_tp = False
    if args.tib_tmpl and args.ref_tmpl:
        print("[plot] Running gffcompare for TP/FP labels …", flush=True)
        dfs    = _add_tp_labels(dfs, args.tib_tmpl, args.ref_tmpl)
        has_tp = any("tp" in df.columns and df["tp"].notna().any()
                     for df in dfs.values())

    # Layout:
    #  row 0 : violin plots (1 row, len(_SCORE_COLS) cols)
    #  row 1 : scatter (full width)     <- if has_tp, else last row
    # [row 2]: ROC curves               <- only if has_tp
    # [row 3]: AUC summary              <- only if has_tp
    n_extra = 2 if has_tp else 0      # ROC + AUC rows
    n_rows  = 2 + n_extra

    fig = plt.figure(figsize=(4 * len(_SCORE_COLS), 4 * n_rows))

    _plot_violins(dfs, fig, n_rows=n_rows, row_offset=0)

    _plot_scatter(dfs, fig, n_rows=n_rows, row_offset=1)

    if has_tp:
        n_sp = len(dfs)
        _plot_roc_curves(dfs, fig, n_rows=n_rows, row_offset=2)
        _plot_auc_summary(dfs, fig, n_rows=n_rows, row_offset=3)

    tag = args.score_tag
    fig.suptitle(
        f"ORFfinder model scores on Tiberius ab initio predictions  [{tag}]",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[plot] Saved -> {args.out_pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
