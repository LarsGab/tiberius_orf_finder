"""Decision-tree classifier to separate correct ORF predictions from wrong ones.

Training signal
---------------
Positive (label=1): gffcompare class  =  or  ~  (complete / single-exon match)
Negative (label=0): gffcompare class  u, i, x   (intergenic / intronic / antisense)

Intermediate classes (j, c, k, o, e, m, n, p) are held out from training
and scored by the fitted model — the resulting probability is the rescue score.

Outputs
-------
  <out-dir>/decision_tree.pdf          Tree diagram + feature importances + score distributions
  <out-dir>/orf_scores.tsv            Per-transcript probability (all classes)
  <out-dir>/tree_rules.txt            Human-readable rule dump of the tree

Usage
-----
python scripts/train_orf_classifier.py \\
  --base-dir  /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \\
  --out-dir   /projects/AI-GUSTUS/tiberius_orf_finder/results/filter_analysis \\
  [--max-depth 4]  [--kept-only]
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ─── feature setup ─────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "n_exons",
    "cds_length_nt",
    "dist_upstream_stop_nt",
    "n_upstream_atgs",
    "n_overlapping_alignments",
    "best_identity",
    "best_norm_bitscore",
    "best_target_coverage",
    "frac_introns_supported",
    "cds_length_pct",
    "protein_extends_5prime_codons",
    "protein_extends_3prime_codons",
]

BINARY_FEATURES = [
    "has_protein_support",
    "has_start_hint",
    "has_stop_hint",
    "has_conflict",
    "has_upstream_partner",
    "has_downstream_partner",
]

CATEGORICAL_FEATURES = {
    "lorf_class": ["LORF_UPSTOP", "LORF_NOUPSTOP", "sORF_UPSTOP", "sORF_NOUPSTOP", "upLORF"],
    "support_level": ["fullSupport", "anySupport", "noSupport"],
}

POSITIVE_CLASSES = {"=", "~"}
NEGATIVE_CLASSES = {"u", "i", "x"}
INTERMEDIATE_CLASSES = {"j", "c", "k", "o", "e", "m", "n", "p"}

CLASS_ORDER = ["=", "~", "j", "c", "k", "o", "e", "m", "n", "p", "i", "u", "x"]
CLASS_COLOUR = {
    "=": "#2ca02c", "~": "#2ca02c",
    "j": "#ff7f0e", "c": "#ff7f0e", "k": "#ff7f0e",
    "o": "#1f77b4", "e": "#1f77b4", "m": "#1f77b4", "n": "#1f77b4", "p": "#1f77b4",
    "i": "#d62728", "u": "#d62728", "x": "#d62728",
}


# ─── data loading ──────────────────────────────────────────────────────────

def load_data(base_dir: Path, annot_tag: str) -> pd.DataFrame:
    frames = []
    for sp_dir in sorted(base_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        # allow pooled TSV directly
        for candidate in [
            sp_dir / annot_tag / "orf_features.tsv",
            sp_dir / annot_tag / "orf_features_kept.tsv",
        ]:
            if candidate.exists():
                df = pd.read_csv(candidate, sep="\t", low_memory=False)
                df["species"] = sp_dir.name
                frames.append(df)
                break
    if not frames:
        sys.exit(f"No orf_features.tsv found under {base_dir}")
    return pd.concat(frames, ignore_index=True)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_names). Fills NA with 0 for binary/numeric, uses OHE for categorical."""
    parts = []
    feat_names = []

    # numeric
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(s.rename(col))
            feat_names.append(col)

    # binary
    for col in BINARY_FEATURES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            parts.append(s.rename(col))
            feat_names.append(col)

    # categorical → one-hot
    for col, cats in CATEGORICAL_FEATURES.items():
        if col not in df.columns:
            continue
        for cat in cats:
            name = f"{col}__{cat}"
            parts.append((df[col] == cat).astype(int).rename(name))
            feat_names.append(name)

    X = pd.concat(parts, axis=1)
    return X, feat_names


# ─── plotting helpers ───────────────────────────────────────────────────────

def page_tree(clf: DecisionTreeClassifier, feature_names: list[str], pdf: PdfPages):
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        clf, feature_names=feature_names,
        class_names=["wrong (u/i/x)", "correct (=/~)"],
        filled=True, rounded=True, impurity=True,
        max_depth=clf.get_depth(), ax=ax, fontsize=8,
    )
    ax.set_title(f"Decision tree  (depth={clf.get_depth()}, "
                 f"leaves={clf.get_n_leaves()})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, dpi=100)
    plt.close(fig)


def page_feature_importance(clf: DecisionTreeClassifier, feature_names: list[str],
                            pdf: PdfPages):
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    top = order[:25]  # top 25

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = ["#2ca02c" if imp[i] > 0 else "#cccccc" for i in top]
    ax.barh(range(len(top)), imp[top][::-1], color=colours[::-1], edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feature_names[i] for i in top[::-1]], fontsize=9)
    ax.set_xlabel("Gini importance")
    ax.set_title("Feature importances (top 25)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_score_distributions(scores_df: pd.DataFrame, pdf: PdfPages):
    classes_present = [c for c in CLASS_ORDER if c in scores_df["gffcompare_class"].values]
    data_by_cls = [scores_df.loc[scores_df["gffcompare_class"] == c, "correct_prob"].values
                   for c in classes_present]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Predicted 'correct' probability by gffcompare class",
                 fontsize=13, fontweight="bold")

    # violin
    ax = axes[0]
    valid = [(i, d) for i, d in enumerate(data_by_cls) if len(d) > 0]
    if valid:
        vpos, vdata = zip(*valid)
        parts = ax.violinplot(list(vdata), positions=list(vpos),
                              showmedians=True, showextrema=False)
        for pc, pos in zip(parts["bodies"], vpos):
            pc.set_facecolor(CLASS_COLOUR.get(classes_present[pos], "#999999"))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(classes_present)))
    ax.set_xticklabels(classes_present)
    ax.set_ylabel("P(correct)")
    ax.set_title("Score distribution per class")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="threshold=0.5")
    ax.legend()

    # fraction above 0.5 per class
    ax2 = axes[1]
    fracs = [np.mean(d >= 0.5) if len(d) > 0 else 0 for d in data_by_cls]
    colours = [CLASS_COLOUR.get(c, "#999999") for c in classes_present]
    bars = ax2.bar(classes_present, fracs, color=colours, edgecolor="black", linewidth=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fraction with P >= 0.5")
    ax2.set_title("Rescue rate per class at threshold 0.5")
    for bar, f in zip(bars, fracs):
        ax2.text(bar.get_x() + bar.get_width() / 2, f + 0.01,
                 f"{f:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_roc_confusion(clf, X_test, y_test, pdf: PdfPages):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Classifier evaluation on held-out test split",
                 fontsize=12, fontweight="bold")

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0],
                                     name=f"Decision tree (AUC={auc:.3f})")
    axes[0].set_title("ROC curve")

    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, ax=axes[1],
        display_labels=["wrong (u/i/x)", "correct (=/~)"],
        colorbar=False,
    )
    axes[1].set_title("Confusion matrix")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_score_vs_features(scores_df: pd.DataFrame, pdf: PdfPages):
    """Score vs individual feature, coloured by class, for intermediate classes only."""
    inter = scores_df[scores_df["gffcompare_class"].isin(INTERMEDIATE_CLASSES)].copy()
    if len(inter) == 0:
        return

    feats = ["best_norm_bitscore", "frac_introns_supported",
             "best_target_coverage", "cds_length_pct"]
    feats = [f for f in feats if f in inter.columns]
    if not feats:
        return

    fig, axes = plt.subplots(1, len(feats), figsize=(5 * len(feats), 5))
    fig.suptitle("Rescue score vs features (intermediate classes only)",
                 fontsize=11, fontweight="bold")
    if len(feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, feats):
        for cls in [c for c in CLASS_ORDER if c in inter["gffcompare_class"].values]:
            sub = inter[inter["gffcompare_class"] == cls]
            x = pd.to_numeric(sub[feat], errors="coerce").fillna(0)
            ax.scatter(x, sub["correct_prob"], s=3, alpha=0.3,
                       color=CLASS_COLOUR.get(cls, "#999999"), label=cls, rasterized=True)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("P(correct)")
        ax.set_title(feat, fontsize=9)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax.legend(markerscale=4, fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir",   required=True, type=Path)
    p.add_argument("--annot-tag",  default="annotate_epoch_74_filt_tpm1cov3len300_lorf")
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--max-depth",  type=int, default=4,
                   help="Max tree depth (default 4)")
    p.add_argument("--test-frac",  type=float, default=0.2,
                   help="Fraction of labelled data held out for evaluation")
    p.add_argument("--min-weight", type=float, default=0.0001,
                   help="min_weight_fraction_leaf (avoids tiny leaves)")
    p.add_argument("--kept-only",  action="store_true",
                   help="Load orf_features_kept.tsv instead of orf_features.tsv")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("Loading data …", flush=True)
    df = load_data(args.base_dir, args.annot_tag)
    print(f"  {len(df):,} transcripts, {df['species'].nunique()} species", flush=True)

    X_all, feat_names = build_feature_matrix(df)
    y_all = df["gffcompare_class"].map(
        lambda c: 1 if c in POSITIVE_CLASSES else (0 if c in NEGATIVE_CLASSES else np.nan)
    )

    # ── training set: only labelled rows ────────────────────────────────────
    labelled = y_all.notna()
    X_lab = X_all[labelled].values
    y_lab = y_all[labelled].values.astype(int)
    print(f"  Labelled: {labelled.sum():,}  (pos={int(y_lab.sum())}, neg={int((y_lab==0).sum())})",
          flush=True)

    # stratified train/test split by species to avoid leakage
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=42)
    train_idx, test_idx = next(sss.split(X_lab, y_lab))
    X_train, X_test = X_lab[train_idx], X_lab[test_idx]
    y_train, y_test = y_lab[train_idx], y_lab[test_idx]

    # ── fit ─────────────────────────────────────────────────────────────────
    print(f"Training decision tree (max_depth={args.max_depth}) …", flush=True)
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_weight_fraction_leaf=args.min_weight,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    print(f"  Depth={clf.get_depth()}, leaves={clf.get_n_leaves()}", flush=True)

    # ── evaluate ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred,
                                target_names=["wrong (u/i/x)", "correct (=/~)"]))

    # ── score all transcripts ────────────────────────────────────────────────
    proba_all = clf.predict_proba(X_all.values)[:, 1]
    scores_df = df[["transcript_id", "species", "gffcompare_class",
                    "lorf_class", "support_level"]].copy()
    scores_df["correct_prob"] = proba_all
    scores_df["predicted_label"] = (proba_all >= 0.5).astype(int)
    scores_out = args.out_dir / "orf_scores.tsv"
    scores_df.to_csv(scores_out, sep="\t", index=False)
    print(f"\nScores written to {scores_out}", flush=True)

    # per-class score summary
    print("\nMedian P(correct) per gffcompare class:")
    for cls in [c for c in CLASS_ORDER if c in scores_df["gffcompare_class"].values]:
        sub = scores_df[scores_df["gffcompare_class"] == cls]["correct_prob"]
        above = (sub >= 0.5).mean()
        print(f"  {cls}  median={sub.median():.3f}  frac>=0.5={above:.3f}  n={len(sub)}")

    # ── tree rules text dump ─────────────────────────────────────────────────
    rules_txt = export_text(clf, feature_names=feat_names)
    rules_path = args.out_dir / "tree_rules.txt"
    rules_path.write_text(rules_txt)
    print(f"Tree rules written to {rules_path}", flush=True)

    # ── plots ────────────────────────────────────────────────────────────────
    pdf_path = args.out_dir / "decision_tree.pdf"
    print(f"Writing PDF to {pdf_path} …", flush=True)
    with PdfPages(pdf_path) as pdf:
        page_tree(clf, feat_names, pdf)
        page_feature_importance(clf, feat_names, pdf)
        page_roc_confusion(clf, X_test, y_test, pdf)
        page_score_distributions(scores_df, pdf)
        page_score_vs_features(scores_df, pdf)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
