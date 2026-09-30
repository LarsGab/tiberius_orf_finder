"""Analyse why the embryophyta LGB filters too many Tiberius predictions.

Outputs (stdout + <out-dir>/analyse_tib_lgb_errors.pdf):
  1. Feature importances from trained model
  2. Feature medians: Tib "correct" vs "wrong" by LGB
  3. lorf_class distribution shift (Tiberius=all NA vs training ORFs)
  4. Threshold sensitivity: fraction kept at 0.2–0.7 for Tib vs ORF preds
  5. Training score calibration: gffcompare class survival rates at threshold 0.5

Usage
-----
python scripts/analyse_tib_lgb_errors.py \\
    --out-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/training_embryophyta_test_v2/lgb_error_analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import numpy as np
import pandas as pd

PROJDIR   = Path("/projects/AI-GUSTUS/tiberius_orf_finder")
TEST_DIR  = PROJDIR / "results/training_embryophyta_test_v2"
TRAIN_DIR = PROJDIR / "results/training_embryophyta_v2"
MODEL     = PROJDIR / "results/filter_analysis/lgb_embryophyta/lgb_3class_model.pkl"
ANNOT_TAG = "annotate_run001_e300"

SPECIES = [
    "Arabidopsis_thaliana",
    "Eschscholzia_californica",
    "Freycinetia_multiflora",
    "Medicago_truncatula",
    "Mimulus_guttatus",
    "Urochloa_brizantha",
]

NUMERIC_FEATURES = [
    "n_exons", "cds_length_nt", "dist_upstream_stop_nt", "n_upstream_atgs",
    "n_overlapping_alignments", "best_identity", "best_norm_bitscore",
    "best_target_coverage", "frac_introns_supported", "cds_length_pct",
    "n_overlapping_alignments_pct",
    "protein_extends_5prime_codons", "protein_extends_3prime_codons",
]
BINARY_FEATURES = [
    "has_protein_support", "has_start_hint", "has_stop_hint",
    "has_conflict", "has_upstream_partner", "has_downstream_partner",
]


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_tib(species: list[str]) -> pd.DataFrame:
    frames = []
    for sp in species:
        feat = TEST_DIR / sp / "tiberius_lgb_filtered/tiberius_features.tsv"
        scores = TEST_DIR / sp / "tiberius_lgb_filtered/tiberius_lgb_filtered.scores.tsv"
        if not feat.exists() or not scores.exists():
            print(f"  [skip] {sp}: missing tib features/scores", file=sys.stderr)
            continue
        df = pd.read_csv(feat, sep="\t", low_memory=False)
        sc = pd.read_csv(scores, sep="\t", low_memory=False)[
            ["transcript_id", "prob_wrong", "prob_partial", "prob_correct",
             "prob_not_wrong", "lgb_class"]
        ]
        df = df.merge(sc, on="transcript_id", how="left")
        df["species"] = sp
        df["source"]  = "tiberius"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_orf_test(species: list[str]) -> pd.DataFrame:
    frames = []
    for sp in species:
        feat = TEST_DIR / sp / ANNOT_TAG / "orf_features.tsv"
        scores = TEST_DIR / sp / ANNOT_TAG / "orfs_lgb3_filtered.scores.tsv"
        if not feat.exists() or not scores.exists():
            continue
        df = pd.read_csv(feat, sep="\t", low_memory=False)
        sc = pd.read_csv(scores, sep="\t", low_memory=False)[
            ["transcript_id", "prob_wrong", "prob_partial", "prob_correct",
             "prob_not_wrong", "lgb_class"]
        ]
        df = df.merge(sc, on="transcript_id", how="left")
        df["species"] = sp
        df["source"]  = "orf_test"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_orf_train_scores() -> pd.DataFrame:
    """Training-set scores (has gffcompare_class column)."""
    p = PROJDIR / "results/filter_analysis/lgb_embryophyta/lgb_3class_scores.tsv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, sep="\t", low_memory=False)


# ---------------------------------------------------------------------------
# analysis helpers
# ---------------------------------------------------------------------------

def feature_medians_by_class(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for cls in ["correct", "partial", "wrong"]:
        sub = df[df["lgb_class"] == cls]
        row = {"lgb_class": cls, "n": len(sub)}
        for c in cols:
            if c in sub.columns:
                row[c] = pd.to_numeric(sub[c], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).set_index("lgb_class")


def threshold_retention(df: pd.DataFrame, col: str = "prob_not_wrong") -> dict[float, float]:
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    return {t: float(np.mean(vals >= t)) for t in np.arange(0.2, 0.75, 0.05)}


def gffcmp_survival(train_scores: pd.DataFrame,
                    thresholds: list[float] = [0.3, 0.4, 0.5, 0.6]) -> pd.DataFrame:
    """Fraction of each gffcompare class surviving each threshold."""
    col = "prob_not_wrong"
    classes_order = ["=", "~", "j", "c", "k", "o", "e", "m", "n", "p", "i", "u", "x"]
    present = [c for c in classes_order if c in train_scores["gffcompare_class"].values]
    rows = []
    for cls in present:
        sub = train_scores[train_scores["gffcompare_class"] == cls][col]
        row = {"gffcompare_class": cls, "n": len(sub)}
        for t in thresholds:
            row[f"thr_{t:.1f}"] = float(np.mean(sub >= t))
        rows.append(row)
    return pd.DataFrame(rows).set_index("gffcompare_class")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_feature_importances(clf, feat_names: list[str], pdf, topn: int = 20):
    imps = clf.feature_importances_
    idx  = np.argsort(imps)[::-1][:topn]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(topn), imps[idx][::-1], color="#1f77b4")
    ax.set_yticks(range(topn))
    ax.set_yticklabels([feat_names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Importance (split gain)")
    ax.set_title(f"Top {topn} LGB feature importances (embryophyta model)")
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_prob_distribution(tib: pd.DataFrame, orf: pd.DataFrame, pdf):
    """prob_correct distribution: Tib (all lorf_class=NA) vs ORF test preds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (df, label, color) in zip(axes, [
        (tib, "Tiberius predictions", "#d62728"),
        (orf, "ORF test predictions", "#2ca02c"),
    ]):
        vals = pd.to_numeric(df["prob_correct"], errors="coerce").dropna()
        ax.hist(vals, bins=50, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("P(correct)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\nn={len(vals):,}, median={vals.median():.3f}")

    fig.suptitle("P(correct) distribution — Tiberius vs ORF predictions", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_feature_by_class(df: pd.DataFrame, cols: list[str], title: str, pdf):
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)] if nrows > 1 \
                else list(axes)

    colors = {"correct": "#2ca02c", "partial": "#ff7f0e", "wrong": "#d62728"}
    for ax, col in zip(axes_flat, cols):
        for cls in ["correct", "partial", "wrong"]:
            vals = pd.to_numeric(df[df["lgb_class"] == cls][col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.55, color=colors[cls], label=cls, density=True)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(cols):]:
        ax.set_visible(False)

    axes_flat[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_threshold_sensitivity(tib: pd.DataFrame, orf: pd.DataFrame, pdf):
    thresholds = np.arange(0.2, 0.75, 0.05)
    tib_ret = [np.mean(pd.to_numeric(tib["prob_not_wrong"], errors="coerce").dropna() >= t)
               for t in thresholds]
    orf_ret = [np.mean(pd.to_numeric(orf["prob_not_wrong"], errors="coerce").dropna() >= t)
               for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, [r * 100 for r in tib_ret], "o-", color="#d62728", label="Tiberius")
    ax.plot(thresholds, [r * 100 for r in orf_ret], "s-", color="#2ca02c", label="ORF preds")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="current threshold")
    ax.set_xlabel("Threshold  P(partial)+P(correct) ≥ thr")
    ax.set_ylabel("% transcripts retained")
    ax.set_title("Threshold sensitivity: fraction retained")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def plot_gffcmp_survival(surv: pd.DataFrame, pdf):
    if surv.empty:
        return
    thr_cols = [c for c in surv.columns if c.startswith("thr_")]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(surv))
    width = 0.2
    palette = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for i, col in enumerate(thr_cols):
        ax.bar(x + i * width, surv[col] * 100, width,
               label=col.replace("thr_", "thr="), alpha=0.8, color=palette[i % len(palette)])
    ax.set_xticks(x + width * (len(thr_cols) - 1) / 2)
    ax.set_xticklabels(surv.index, rotation=45, ha="right")
    ax.set_ylabel("% kept")
    ax.set_title("Training ORF survival rate by gffcompare class\n"
                 "(= and ~ are true positives; u/i/x are wrong)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model …", flush=True)
    clf, feat_names = joblib.load(MODEL)

    print("Loading Tiberius test features+scores …", flush=True)
    tib = load_tib(SPECIES)
    print(f"  {len(tib):,} Tiberius transcripts across {tib['species'].nunique()} species")

    print("Loading ORF test features+scores …", flush=True)
    orf = load_orf_test(SPECIES)
    print(f"  {len(orf):,} ORF transcripts across {orf['species'].nunique()} species")

    print("Loading training scores (gffcompare labels) …", flush=True)
    train_scores = load_orf_train_scores()
    print(f"  {len(train_scores):,} training transcripts")

    # ── 1. Feature importances ──────────────────────────────────────────────
    print("\n--- Feature importances (top 15) ---")
    imps = clf.feature_importances_
    for i in np.argsort(imps)[::-1][:15]:
        print(f"  {feat_names[i]:<45} {imps[i]:.1f}")

    # ── 2. lorf_class distribution in Tib vs ORF ────────────────────────────
    print("\n--- lorf_class distribution ---")
    for label, df in [("Tiberius", tib), ("ORF test", orf)]:
        if "lorf_class" in df.columns:
            vc = df["lorf_class"].value_counts(normalize=True) * 100
            print(f"  {label}: {vc.to_dict()}")

    # ── 3. lgb_class breakdown ───────────────────────────────────────────────
    print("\n--- lgb_class distribution ---")
    for label, df in [("Tiberius", tib), ("ORF test", orf)]:
        if "lgb_class" in df.columns:
            vc = df["lgb_class"].value_counts()
            tot = len(df)
            print(f"  {label}: " +
                  "  ".join(f"{k}={v} ({100*v/tot:.1f}%)" for k, v in vc.items()))

    # ── 4. Feature medians by lgb_class ─────────────────────────────────────
    print("\n--- Tiberius feature medians by lgb_class ---")
    if not tib.empty:
        med = feature_medians_by_class(tib, NUMERIC_FEATURES + BINARY_FEATURES)
        print(med.T.to_string())

    # ── 5. Threshold retention ───────────────────────────────────────────────
    print("\n--- Threshold retention (prob_not_wrong) ---")
    print(f"  {'threshold':>12}  {'Tib %kept':>12}  {'ORF %kept':>12}")
    for t in np.arange(0.2, 0.75, 0.05):
        tib_r = np.mean(pd.to_numeric(tib["prob_not_wrong"], errors="coerce").dropna() >= t) * 100
        orf_r = np.mean(pd.to_numeric(orf["prob_not_wrong"], errors="coerce").dropna() >= t) * 100
        print(f"  {t:>12.2f}  {tib_r:>12.1f}  {orf_r:>12.1f}")

    # ── 6. gffcompare survival on training data ──────────────────────────────
    if not train_scores.empty and "gffcompare_class" in train_scores.columns:
        surv = gffcmp_survival(train_scores)
        print("\n--- Training gffcompare class survival at different thresholds ---")
        print(surv.to_string())
    else:
        surv = pd.DataFrame()

    # ── 7. Per-species threshold table for Tiberius ───────────────────────────
    print("\n--- Per-species Tiberius retention at threshold 0.5 ---")
    if not tib.empty:
        for sp in SPECIES:
            sub = tib[tib["species"] == sp]
            if sub.empty:
                continue
            kept  = (sub["lgb_class"] == "correct").sum() + (sub["lgb_class"] == "partial").sum()
            total = len(sub)
            wrong = (sub["lgb_class"] == "wrong").sum()
            print(f"  {sp:<35} kept={kept}/{total} ({100*kept/total:.1f}%)  "
                  f"dropped_as_wrong={wrong} ({100*wrong/total:.1f}%)")

    # ── PDF ─────────────────────────────────────────────────────────────────
    pdf_path = args.out_dir / "analyse_tib_lgb_errors.pdf"
    print(f"\nWriting PDF → {pdf_path}", flush=True)
    with mpdf.PdfPages(pdf_path) as pdf:
        plot_feature_importances(clf, feat_names, pdf)
        if not tib.empty and not orf.empty:
            plot_prob_distribution(tib, orf, pdf)
            plot_feature_by_class(tib, NUMERIC_FEATURES + BINARY_FEATURES,
                                  "Tiberius predictions: feature distributions by lgb_class", pdf)
            plot_threshold_sensitivity(tib, orf, pdf)
        if not surv.empty:
            plot_gffcmp_survival(surv, pdf)

    print("Done.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
