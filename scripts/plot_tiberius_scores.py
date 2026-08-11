"""Plot ORFfinder scores for Tiberius ab initio predictions.

Loads per-species score TSVs produced by score_tiberius.py and generates
a multi-panel PDF with score distributions.

Optionally, if --ref-dir is given, gffcompare is used to classify each
Tiberius transcript as TP (matched reference CDS) or FP, enabling
TP-vs-FP score distributions and AUC computation.

Usage
-----
# Without reference (distribution plots only):
python scripts/plot_tiberius_scores.py \\
  --score-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --score-tag score_tiberius_epoch_74 \\
  --out-pdf   results/figures/tiberius_orf_scores_epoch74.pdf

# With reference annotation for TP/FP classification:
python scripts/plot_tiberius_scores.py \\
  --score-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --score-tag score_tiberius_epoch_74 \\
  --tib-tmpl  '/home/gabriell/tiberius_benchmarking/paper/Vertebrata/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf' \\
  --ref-tmpl  '/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/assembly/annot_cds.gff' \\
  --gffcompare gffcompare \\
  --out-pdf   results/figures/tiberius_orf_scores_epoch74.pdf
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
    "start_prob":          "P(START) at CDS pos 0",
    "stop_prob":           "P(STOP) at CDS last pos",
    "mean_ir_prob":        "Mean P(IR) over CDS",
    "frac_argmax_coding":  "Frac. positions argmax=coding",
}

_SP_SHORT = {
    "Gallus_gallus":           "Gga",
    "Pristiophorus_japonicus": "Pja",
    "Bos_taurus":              "Bta",
    "Delphinapterus_leucas":   "Dle",
    "Homo_sapiens":            "Hsa",
}


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
    return dfs


def _run_gffcompare(tib_gtf: Path, ref_gff: Path, tmpdir: Path) -> set[str]:
    """Return set of transcript_ids classified as matching (TP) by gffcompare.

    Runs: gffcompare --strict-match -e 3 -T -r <ref> -o <pfx> <query>
    Parses the .tmap file to find transcripts with class_code '='.
    """
    prefix = tmpdir / "gc"
    cmd = [
        "gffcompare", "--strict-match", "-e", "3", "-T",
        "-r", str(ref_gff), "-o", str(prefix), str(tib_gtf),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmap = Path(str(prefix) + ".tib_seqlen.gtf.tmap")
    if not tmap.exists():
        # gffcompare may name it differently; find any .tmap
        tmaps = list(tmpdir.glob("*.tmap"))
        tmap = tmaps[0] if tmaps else None
    if tmap is None or not tmap.exists():
        return set()
    tp_ids: set[str] = set()
    for line in tmap.read_text().splitlines():
        if line.startswith("ref_gene_id"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "=":
            tp_ids.add(parts[1])  # qry_id column
    return tp_ids


def _add_tp_labels(
    dfs: dict[str, pd.DataFrame],
    tib_tmpl: str,
    ref_tmpl: str,
    gffcompare_bin: str,
) -> dict[str, pd.DataFrame]:
    """Add 'tp' bool column to each species DataFrame."""
    for sp, df in dfs.items():
        tib_gtf = Path(tib_tmpl.format(sp=sp))
        ref_gff = Path(ref_tmpl.format(sp=sp))
        if not tib_gtf.exists() or not ref_gff.exists():
            print(f"[plot] skipping TP labels for {sp}: missing files")
            df["tp"] = pd.NA
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            tp_ids = _run_gffcompare(tib_gtf, ref_gff, Path(tmpdir))
        df["tp"] = df["transcript_id"].isin(tp_ids)
        n_tp = df["tp"].sum()
        print(f"[plot] {sp}: {n_tp}/{len(df)} transcripts classified as TP")
    return dfs


def _plot_distributions(dfs: dict[str, pd.DataFrame], ax_grid, has_tp: bool) -> None:
    """Fill ax_grid[row][col] with per-score histograms."""
    species_list = sorted(dfs.keys())
    colors_tp = {"TP": "#2196F3", "FP": "#F44336", "all": "#555555"}
    alpha = 0.65

    for col_i, score in enumerate(_SCORE_COLS):
        for row_i, sp in enumerate(species_list):
            ax = ax_grid[row_i][col_i]
            df = dfs[sp].dropna(subset=[score])
            if has_tp and "tp" in df.columns and df["tp"].notna().any():
                tp_vals = df.loc[df["tp"] == True, score]
                fp_vals = df.loc[df["tp"] == False, score]
                bins = np.linspace(0, 1, 51)
                ax.hist(tp_vals, bins=bins, alpha=alpha, color=colors_tp["TP"],
                        label=f"TP (n={len(tp_vals):,})", density=True)
                ax.hist(fp_vals, bins=bins, alpha=alpha, color=colors_tp["FP"],
                        label=f"FP (n={len(fp_vals):,})", density=True)
                ax.legend(fontsize=6, loc="upper left")
            else:
                bins = np.linspace(0, 1, 51)
                ax.hist(df[score], bins=bins, alpha=0.8, color=colors_tp["all"],
                        density=True)
                ax.text(0.97, 0.95, f"n={len(df):,}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7)
            if row_i == 0:
                ax.set_title(_COL_LABELS[score], fontsize=8)
            if col_i == 0:
                ax.set_ylabel(_SP_SHORT.get(sp, sp), fontsize=8)
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=7)


def _plot_auc(dfs: dict[str, pd.DataFrame], ax) -> None:
    """AUC bar chart: one bar per (species, score_column)."""
    from sklearn.metrics import roc_auc_score

    species_list = sorted(dfs.keys())
    n_sp = len(species_list)
    n_sc = len(_SCORE_COLS)
    width = 0.8 / n_sc
    cmap = plt.cm.get_cmap("tab10", n_sc)

    for si, score in enumerate(_SCORE_COLS):
        aucs = []
        for sp in species_list:
            df = dfs[sp].dropna(subset=[score, "tp"])
            if df.empty or df["tp"].nunique() < 2:
                aucs.append(float("nan"))
                continue
            try:
                auc = roc_auc_score(df["tp"].astype(int), df[score])
            except Exception:
                auc = float("nan")
            aucs.append(auc)
        xs = np.arange(n_sp) + si * width
        ax.bar(xs, aucs, width=width * 0.9, color=cmap(si),
               label=_COL_LABELS[score], alpha=0.85)

    ax.axhline(0.5, color="black", lw=0.8, linestyle="--", label="random")
    ax.set_xticks(np.arange(n_sp) + (n_sc - 1) * width / 2)
    ax.set_xticklabels([_SP_SHORT.get(s, s) for s in species_list], fontsize=9)
    ax.set_ylabel("AUC (TP vs FP)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("ORFfinder score AUC for Tiberius TP vs FP predictions", fontsize=10)


def _plot_scatter(dfs: dict[str, pd.DataFrame], ax, has_tp: bool) -> None:
    """start_prob vs stop_prob scatter, one point per transcript (all species)."""
    colors = plt.cm.tab10.colors
    species_list = sorted(dfs.keys())
    for ci, sp in enumerate(species_list):
        df = dfs[sp].dropna(subset=["start_prob", "stop_prob"])
        if has_tp and "tp" in df.columns and df["tp"].notna().any():
            tp = df[df["tp"] == True]
            fp = df[df["tp"] == False]
            ax.scatter(tp["start_prob"], tp["stop_prob"], s=3, alpha=0.3,
                       color=colors[ci], marker="o", label=f"{_SP_SHORT.get(sp,sp)} TP")
            ax.scatter(fp["start_prob"], fp["stop_prob"], s=3, alpha=0.15,
                       color=colors[ci], marker="x")
        else:
            ax.scatter(df["start_prob"], df["stop_prob"], s=3, alpha=0.2,
                       color=colors[ci], label=_SP_SHORT.get(sp, sp))
    ax.set_xlabel("P(START) at ATG position", fontsize=9)
    ax.set_ylabel("P(STOP) at last CDS position", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, markerscale=3, loc="upper left")
    ax.set_title("START vs STOP probability per transcript", fontsize=10)


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot ORFfinder scores for Tiberius predictions.")
    ap.add_argument("--score-dir", type=Path, required=True,
                    help="Root of vertebrates_test results dir "
                         "(contains <species>/score_tiberius_*/scores.tsv).")
    ap.add_argument("--score-tag", default="score_tiberius_epoch_74",
                    help="Subdirectory name under each species dir (default: "
                         "score_tiberius_epoch_74).")
    ap.add_argument("--out-pdf", type=Path, required=True,
                    help="Output PDF path.")
    ap.add_argument("--tib-tmpl", default=None,
                    help="Path template for Tiberius GTF with {sp} placeholder. "
                         "Required for TP/FP classification.")
    ap.add_argument("--ref-tmpl", default=None,
                    help="Path template for reference GFF with {sp} placeholder. "
                         "Required for TP/FP classification.")
    ap.add_argument("--gffcompare", default="gffcompare",
                    help="Path to gffcompare binary (default: 'gffcompare').")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    print(f"[plot] Loading scores from {args.score_dir} / {args.score_tag}", flush=True)
    dfs = _load_scores(args.score_dir, args.score_tag)
    if not dfs:
        print("[plot] No score TSVs found. Exiting.")
        return 1

    has_tp = False
    if args.tib_tmpl and args.ref_tmpl:
        print("[plot] Running gffcompare for TP/FP labels ...", flush=True)
        dfs = _add_tp_labels(dfs, args.tib_tmpl, args.ref_tmpl, args.gffcompare)
        has_tp = any("tp" in df.columns and df["tp"].notna().any()
                     for df in dfs.values())

    species_list = sorted(dfs.keys())
    n_sp = len(species_list)

    # --- Figure layout -------------------------------------------------------
    # Row 0..n_sp-1 : histograms (one row per species, one col per score)
    # Row n_sp      : scatter (start_prob vs stop_prob)
    # Row n_sp+1    : AUC bars (only if has_tp)
    n_hist_rows = n_sp
    n_extra = 1 + int(has_tp)
    n_rows = n_hist_rows + n_extra
    n_cols = len(_SCORE_COLS)

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))

    # Build axes grid for histograms
    ax_hist = [
        [fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
         for col in range(n_cols)]
        for row in range(n_hist_rows)
    ]
    _plot_distributions(dfs, ax_hist, has_tp)

    # Scatter spanning full width
    ax_scatter = fig.add_subplot(n_rows, 1, n_hist_rows + 1)
    _plot_scatter(dfs, ax_scatter, has_tp)

    # AUC bars
    if has_tp:
        try:
            ax_auc = fig.add_subplot(n_rows, 1, n_hist_rows + 2)
            _plot_auc(dfs, ax_auc)
        except ImportError:
            print("[plot] sklearn not available — skipping AUC panel", flush=True)

    fig.suptitle(
        "ORFfinder model scores on Tiberius ab initio predictions "
        f"(epoch_74, vertebrates test)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[plot] Saved -> {args.out_pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
