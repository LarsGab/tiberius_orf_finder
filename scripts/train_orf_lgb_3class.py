"""Train a 3-class LightGBM on ORF features and save the model for later inference.

Classes
-------
  0  wrong       gffcompare: u, i, x
  1  partial     gffcompare: j, c, k, o, e, m, n, p
  2  correct     gffcompare: =, ~

Using partial as a real training class (rather than holding it out entirely)
lets the model learn what separates partial matches from outright wrong, and
from perfect matches, without the binary-only bias.

Filtering logic (applied by apply_lgb_model_gtf.py)
----------------------------------------------------
A transcript is KEPT when:
    P(correct) + P(partial) >= threshold   [default 0.5]
i.e. the model does not primarily classify it as wrong.
P(correct) and P(partial) are both written to the scores TSV for inspection.

Outputs
-------
  <out-dir>/lgb_3class_model.pkl     trained model (joblib)
  <out-dir>/lgb_3class_scores.tsv   per-transcript probabilities (all input rows)
  <out-dir>/lgb_3class.pdf          evaluation plots + SHAP

Usage
-----
python scripts/train_orf_lgb_3class.py \\
  --base-dir  /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \\
  --out-dir   /projects/AI-GUSTUS/tiberius_orf_finder/results/filter_analysis
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import lightgbm as lgb

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─── constants ─────────────────────────────────────────────────────────────

LABEL_MAP = {
    "u": 0, "i": 0, "x": 0,
    "j": 1, "c": 1, "k": 1, "o": 1, "e": 1, "m": 1, "n": 1, "p": 1,
    "=": 2, "~": 2,
}
CLASS_NAMES = ["wrong (u/i/x)", "partial (j/c/k/…)", "correct (=/~)"]
CLASS_COLOURS = ["#d62728", "#ff7f0e", "#2ca02c"]

CLASS_ORDER = ["=", "~", "j", "c", "k", "o", "e", "m", "n", "p", "i", "u", "x"]
GFF_COLOUR = {
    "=": "#2ca02c", "~": "#2ca02c",
    "j": "#ff7f0e", "c": "#ff7f0e", "k": "#ff7f0e",
    "o": "#1f77b4", "e": "#1f77b4", "m": "#1f77b4", "n": "#1f77b4", "p": "#1f77b4",
    "i": "#d62728", "u": "#d62728", "x": "#d62728",
}

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
CATEGORICAL_FEATURES = {
    "lorf_class":    ["LORF_UPSTOP", "LORF_NOUPSTOP", "sORF_UPSTOP", "sORF_NOUPSTOP", "upLORF"],
    "support_level": ["fullSupport", "anySupport", "noSupport"],
}


# ─── data helpers ──────────────────────────────────────────────────────────

def load_data(base_dir: Path, annot_tag: str) -> pd.DataFrame:
    frames = []
    for sp_dir in sorted(base_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        tsv = sp_dir / annot_tag / "orf_features.tsv"
        if tsv.exists():
            df = pd.read_csv(tsv, sep="\t", low_memory=False)
            df["species"] = sp_dir.name
            frames.append(df)
    if not frames:
        sys.exit(f"No orf_features.tsv found under {base_dir}")
    return pd.concat(frames, ignore_index=True)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce").fillna(0).rename(col))
            names.append(col)
    for col in BINARY_FEATURES:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce").fillna(0).rename(col))
            names.append(col)
    for col, cats in CATEGORICAL_FEATURES.items():
        if col not in df.columns:
            continue
        for cat in cats:
            name = f"{col}__{cat}"
            parts.append((df[col] == cat).astype(float).rename(name))
            names.append(name)
    return pd.concat(parts, axis=1).values, names


# ─── plots ─────────────────────────────────────────────────────────────────

def page_confusion(clf, X_test, y_test, pdf):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, ax=ax,
        display_labels=CLASS_NAMES, colorbar=False, normalize="true",
    )
    ax.set_title("Confusion matrix (row-normalised, test split)", fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_score_distributions(scores_df: pd.DataFrame, pdf: PdfPages):
    classes = [c for c in CLASS_ORDER if c in scores_df["gffcompare_class"].values]
    score_cols = [
        ("prob_wrong",   "P(wrong)",   "#d62728"),
        ("prob_partial", "P(partial)", "#ff7f0e"),
        ("prob_correct", "P(correct)", "#2ca02c"),
        ("prob_not_wrong", "P(partial)+P(correct)", "#1f77b4"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("3-class LGB score distributions by gffcompare class",
                 fontsize=13, fontweight="bold")

    for ax, (col, label, colour) in zip(axes.flatten(), score_cols):
        data = [scores_df.loc[scores_df["gffcompare_class"] == c, col].values
                for c in classes]
        valid = [(i, d) for i, d in enumerate(data) if len(d) > 0]
        if valid:
            vpos, vdata = zip(*valid)
            parts = ax.violinplot(list(vdata), positions=list(vpos),
                                  showmedians=True, showextrema=False)
            for pc, pos in zip(parts["bodies"], vpos):
                pc.set_facecolor(GFF_COLOUR.get(classes[pos], "#999"))
                pc.set_alpha(0.75)
            parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_ylabel(label)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(label)

    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_rescue_rates(scores_df: pd.DataFrame, threshold: float, pdf: PdfPages):
    classes = [c for c in CLASS_ORDER if c in scores_df["gffcompare_class"].values]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Fraction kept at threshold {threshold}  (P(not wrong) ≥ {threshold})",
                 fontsize=12, fontweight="bold")

    for ax, (col, label) in zip(axes, [
        ("prob_correct",   "P(correct) ≥ thr"),
        ("prob_not_wrong", "P(partial)+P(correct) ≥ thr"),
        ("prob_partial",   "P(partial) ≥ thr"),
    ]):
        fracs = [
            np.mean(scores_df.loc[scores_df["gffcompare_class"] == c, col].values >= threshold)
            for c in classes
        ]
        colours = [GFF_COLOUR.get(c, "#999") for c in classes]
        bars = ax.bar(classes, fracs, color=colours, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Fraction kept")
        ax.set_title(label)
        for bar, f in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width() / 2, f + 0.01,
                    f"{f:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_feature_importance(clf, feat_names: list[str], pdf: PdfPages):
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:25]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(order)), imp[order[::-1]],
            color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=9)
    ax.set_xlabel("Feature importance (split gain)")
    ax.set_title("Top-25 features — 3-class LGB", fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def page_shap(clf, X_all, feat_names, pdf: PdfPages):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_all), min(5000, len(X_all)), replace=False)
    X_shap = X_all[idx]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_shap)
    # multiclass: sv is list of [class0, class1, class2] arrays
    # show SHAP for class 2 (correct) and class 0 (wrong) side by side
    for class_idx, class_label in [(2, "correct (=/~)"), (0, "wrong (u/i/x)")]:
        sv_cls = sv[class_idx] if isinstance(sv, list) else sv[:, :, class_idx]
        fig, _ = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv_cls, X_shap, feature_names=feat_names,
                          show=False, max_display=20, plot_type="dot")
        plt.title(f"SHAP beeswarm — class: {class_label}",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close("all")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir",   required=True, type=Path)
    p.add_argument("--annot-tag",  default="annotate_epoch_74_filt_tpm1cov3len300_lorf")
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--max-depth",  type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="P(not wrong) threshold used in rescue rate plots")
    p.add_argument("--test-frac",  type=float, default=0.2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("Loading data …", flush=True)
    df = load_data(args.base_dir, args.annot_tag)
    print(f"  {len(df):,} transcripts, {df['species'].nunique()} species", flush=True)

    X_all, feat_names = build_feature_matrix(df)
    y_all = df["gffcompare_class"].map(LABEL_MAP)

    labelled = y_all.notna()
    X_lab = X_all[labelled]
    y_lab = y_all[labelled].values.astype(int)
    counts = np.bincount(y_lab, minlength=3)
    print(f"  Labelled: {labelled.sum():,}  wrong={counts[0]}  partial={counts[1]}  correct={counts[2]}",
          flush=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=42)
    train_idx, test_idx = next(sss.split(X_lab, y_lab))
    X_train, X_test = X_lab[train_idx], X_lab[test_idx]
    y_train, y_test = y_lab[train_idx], y_lab[test_idx]

    print(f"Training 3-class LightGBM ({args.n_estimators} estimators) …", flush=True)
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    print(f"  Done. Depth={clf.get_params()['max_depth']}, "
          f"leaves={clf.booster_.num_trees() // 3} trees/class", flush=True)

    # evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # score all transcripts
    proba_all = clf.predict_proba(X_all)   # shape (n, 3)
    scores_df = df[["transcript_id", "species", "gffcompare_class",
                    "lorf_class", "support_level"]].copy()
    scores_df["prob_wrong"]    = proba_all[:, 0]
    scores_df["prob_partial"]  = proba_all[:, 1]
    scores_df["prob_correct"]  = proba_all[:, 2]
    scores_df["prob_not_wrong"] = proba_all[:, 1] + proba_all[:, 2]
    scores_df["predicted_class"] = clf.predict(X_all)

    scores_path = args.out_dir / "lgb_3class_scores.tsv"
    scores_df.to_csv(scores_path, sep="\t", index=False)
    print(f"\nScores written to {scores_path}", flush=True)

    # save model
    model_path = args.out_dir / "lgb_3class_model.pkl"
    joblib.dump((clf, feat_names), model_path)
    print(f"Model saved to {model_path}", flush=True)

    # per-class rescue summary
    print(f"\nFraction kept per gffcompare class at P(not wrong) >= {args.threshold}:")
    print(f"  {'class':<5} {'N':>6}  {'P(correct)':>10}  {'P(not wrong)':>12}  {'P(partial)':>10}")
    for cls in CLASS_ORDER:
        sub = scores_df[scores_df["gffcompare_class"] == cls]
        if len(sub) == 0:
            continue
        rc = (sub["prob_correct"].values  >= args.threshold).mean()
        rn = (sub["prob_not_wrong"].values >= args.threshold).mean()
        rp = (sub["prob_partial"].values  >= args.threshold).mean()
        print(f"  {cls:<5} {len(sub):>6}  {rc:>10.3f}  {rn:>12.3f}  {rp:>10.3f}")

    # plots
    pdf_path = args.out_dir / "lgb_3class.pdf"
    print(f"\nWriting PDF to {pdf_path} …", flush=True)
    with PdfPages(pdf_path) as pdf:
        page_confusion(clf, X_test, y_test, pdf)
        page_feature_importance(clf, feat_names, pdf)
        page_score_distributions(scores_df, pdf)
        page_rescue_rates(scores_df, args.threshold, pdf)
        if HAS_SHAP:
            print("  Computing SHAP values …", flush=True)
            page_shap(clf, X_all, feat_names, pdf)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
