"""Logistic regression (with interaction terms) and LightGBM + SHAP for ORF rescue scoring.

Training signal
---------------
Positive (label=1): gffcompare class  =  or  ~
Negative (label=0): gffcompare class  u, i, x
Intermediate (j, c, k, o, e, m, n, p): scored but not trained on

Outputs
-------
  <out-dir>/advanced_classifiers.pdf   all plots
  <out-dir>/orf_scores_lr.tsv          logistic regression probability
  <out-dir>/orf_scores_lgb.tsv         LightGBM probability
  <out-dir>/shap_values.npy            SHAP matrix (n_all × n_features)

Usage
-----
python scripts/train_orf_advanced_classifiers.py \\
  --base-dir  /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \\
  --out-dir   /projects/AI-GUSTUS/tiberius_orf_finder/results/filter_analysis
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False
    print("lightgbm not found, falling back to HistGradientBoostingClassifier "
          "(no SHAP available)", file=sys.stderr)

try:
    import shap
    HAS_SHAP = True and HAS_LGB
except ImportError:
    HAS_SHAP = False


# ─── constants ─────────────────────────────────────────────────────────────

POSITIVE_CLASSES = {"=", "~"}
NEGATIVE_CLASSES = {"u", "i", "x"}
CLASS_ORDER = ["=", "~", "j", "c", "k", "o", "e", "m", "n", "p", "i", "u", "x"]
CLASS_COLOUR = {
    "=": "#2ca02c", "~": "#2ca02c",
    "j": "#ff7f0e", "c": "#ff7f0e", "k": "#ff7f0e",
    "o": "#1f77b4", "e": "#1f77b4", "m": "#1f77b4", "n": "#1f77b4", "p": "#1f77b4",
    "i": "#d62728", "u": "#d62728", "x": "#d62728",
}

NUMERIC_FEATURES = [
    "n_exons", "cds_length_nt", "dist_upstream_stop_nt", "n_upstream_atgs",
    "n_overlapping_alignments", "best_identity", "best_norm_bitscore",
    "best_target_coverage", "frac_introns_supported", "cds_length_pct",
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

# ─── data loading ──────────────────────────────────────────────────────────

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


def build_base_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns (X_base, feat_names) with NaNs filled."""
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
    return pd.concat(parts, axis=1), names


def add_interactions(X: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Hand-picked biologically motivated interaction terms for logistic regression."""
    def _get(col: str) -> np.ndarray:
        return X[col].values if col in X.columns else np.zeros(len(X))

    bitscore  = _get("best_norm_bitscore")
    frac_int  = _get("frac_introns_supported")
    target_cov= _get("best_target_coverage")
    cds_pct   = _get("cds_length_pct")
    identity  = _get("best_identity")
    n_exons   = _get("n_exons")
    has_prot  = _get("has_protein_support")
    no_supp   = _get("support_level__noSupport")
    lorf_up   = _get("lorf_class__LORF_UPSTOP")
    lorf_no   = _get("lorf_class__LORF_NOUPSTOP")

    interaction_terms = [
        (bitscore  * frac_int,   "bitscore×frac_introns"),
        (bitscore  * target_cov, "bitscore×target_cov"),
        (bitscore  * cds_pct,    "bitscore×cds_pct"),
        (frac_int  * cds_pct,    "frac_introns×cds_pct"),
        (has_prot  * identity,   "has_prot×identity"),
        (n_exons   * frac_int,   "n_exons×frac_introns"),
        (no_supp   * bitscore,   "no_support×bitscore"),
        (lorf_up   * frac_int,   "LORF_UPSTOP×frac_introns"),
        (lorf_no   * bitscore,   "LORF_NOUPSTOP×bitscore"),
        (target_cov* frac_int,   "target_cov×frac_introns"),
    ]

    Xarr = X.values
    int_arrays = np.column_stack([v for v, _ in interaction_terms])
    int_names  = [n for _, n in interaction_terms]

    X_out   = np.hstack([Xarr, int_arrays])
    names_out = names + int_names
    return X_out, names_out


# ─── plotting helpers ───────────────────────────────────────────────────────

def _violin_scores(ax, scores_df: pd.DataFrame, score_col: str, title: str):
    classes = [c for c in CLASS_ORDER if c in scores_df["gffcompare_class"].values]
    data = [scores_df.loc[scores_df["gffcompare_class"] == c, score_col].values for c in classes]
    valid = [(i, d) for i, d in enumerate(data) if len(d) > 0]
    if valid:
        vpos, vdata = zip(*valid)
        parts = ax.violinplot(list(vdata), positions=list(vpos),
                              showmedians=True, showextrema=False)
        for pc, pos in zip(parts["bodies"], vpos):
            pc.set_facecolor(CLASS_COLOUR.get(classes[pos], "#999"))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel("P(correct)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax.set_title(title)


def _rescue_bar(ax, scores_df: pd.DataFrame, score_col: str, title: str):
    classes = [c for c in CLASS_ORDER if c in scores_df["gffcompare_class"].values]
    fracs = [
        np.mean(scores_df.loc[scores_df["gffcompare_class"] == c, score_col].values >= 0.5)
        for c in classes
    ]
    colours = [CLASS_COLOUR.get(c, "#999") for c in classes]
    bars = ax.bar(classes, fracs, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fraction P ≥ 0.5")
    ax.set_title(title)
    for bar, f in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, f + 0.01,
                f"{f:.2f}", ha="center", va="bottom", fontsize=7)


# ─── LR section ─────────────────────────────────────────────────────────────

def run_lr(X_train, y_train, X_test, y_test, X_all,
           feat_names, df, out_dir, pdf):
    print("Fitting logistic regression …", flush=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5, class_weight="balanced",
            max_iter=2000, solver="lbfgs", random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    print(f"  LR AUC (test) = {auc:.4f}", flush=True)
    print(classification_report(y_test, (proba_test >= 0.5).astype(int),
                                target_names=["wrong", "correct"]))

    proba_all = pipe.predict_proba(X_all)[:, 1]
    scores_df = df[["transcript_id", "species", "gffcompare_class",
                    "lorf_class", "support_level"]].copy()
    scores_df["correct_prob_lr"] = proba_all
    scores_df.to_csv(out_dir / "orf_scores_lr.tsv", sep="\t", index=False)

    # ── coefficient plot ──
    lr = pipe.named_steps["lr"]
    coefs = lr.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:30]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Logistic Regression  (AUC={auc:.4f})", fontsize=13, fontweight="bold")

    ax = axes[0]
    colours = ["#2ca02c" if coefs[i] > 0 else "#d62728" for i in order[::-1]]
    ax.barh(range(len(order)), coefs[order[::-1]], color=colours, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=8)
    ax.set_xlabel("Coefficient (positive = more likely correct)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top-30 coefficients (green=correct, red=wrong)")

    _violin_scores(axes[1], scores_df, "correct_prob_lr", "Score distribution per class")
    _rescue_bar(axes[2], scores_df, "correct_prob_lr", "Fraction rescued at P≥0.5")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax,
                                     name=f"LR (AUC={auc:.3f})")
    ax.set_title("LR ROC — test split")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return scores_df


# ─── LightGBM + SHAP section ────────────────────────────────────────────────

def run_lgb(X_train, y_train, X_test, y_test, X_all,
            feat_names_base, df, out_dir, pdf):
    print("Fitting LightGBM …", flush=True)
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(n_pos, 1)

    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            scale_pos_weight=spw,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, verbose=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=20,
            class_weight="balanced", random_state=42,
        )

    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    print(f"  LGB AUC (test) = {auc:.4f}", flush=True)
    print(classification_report(y_test, (proba_test >= 0.5).astype(int),
                                target_names=["wrong", "correct"]))

    proba_all = model.predict_proba(X_all)[:, 1]
    scores_df = df[["transcript_id", "species", "gffcompare_class",
                    "lorf_class", "support_level"]].copy()
    scores_df["correct_prob_lgb"] = proba_all
    scores_df.to_csv(out_dir / "orf_scores_lgb.tsv", sep="\t", index=False)

    # ── feature importance ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"LightGBM  (AUC={auc:.4f})", fontsize=13, fontweight="bold")

    ax = axes[0]
    if HAS_LGB:
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1][:25]
        ax.barh(range(len(order)), imp[order[::-1]],
                color="#1f77b4", edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_names_base[i] for i in order[::-1]], fontsize=8)
        ax.set_xlabel("Feature importance (split gain)")
        ax.set_title("Top-25 features")
    else:
        ax.text(0.5, 0.5, "Feature importance\nnot available\n(no LightGBM)",
                ha="center", va="center", transform=ax.transAxes)

    _violin_scores(axes[1], scores_df, "correct_prob_lgb", "Score distribution per class")
    _rescue_bar(axes[2], scores_df, "correct_prob_lgb", "Fraction rescued at P≥0.5")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax,
                                     name=f"LGB (AUC={auc:.3f})")
    ax.set_title("LightGBM ROC — test split")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── SHAP ──
    if HAS_SHAP:
        print("  Computing SHAP values …", flush=True)
        # use a subsample for speed (max 5000 rows)
        rng = np.random.default_rng(42)
        n_shap = min(5000, len(X_all))
        idx = rng.choice(len(X_all), n_shap, replace=False)
        X_shap = X_all[idx]

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_shap)
        # for binary lgb: sv is list of [neg_class, pos_class] or just array
        if isinstance(sv, list):
            sv = sv[1]

        np.save(out_dir / "shap_values.npy", sv)

        # beeswarm
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, feature_names=feat_names_base,
                          show=False, max_display=20, plot_type="dot")
        plt.title("SHAP beeswarm — positive class (correct prediction)",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

        # bar summary
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(sv, X_shap, feature_names=feat_names_base,
                          show=False, max_display=20, plot_type="bar")
        plt.title("SHAP mean |value| — feature importance",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

        # SHAP score vs gffcompare class: top-3 features
        mean_abs = np.abs(sv).mean(axis=0)
        top3 = np.argsort(mean_abs)[::-1][:3]
        shap_df = pd.DataFrame(sv[:, top3],
                               columns=[feat_names_base[i] for i in top3])
        shap_df["gffcompare_class"] = df["gffcompare_class"].iloc[idx].values

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("SHAP value distributions per gffcompare class (top-3 features)",
                     fontsize=11, fontweight="bold")
        classes = [c for c in CLASS_ORDER if c in shap_df["gffcompare_class"].values]
        for ax, col in zip(axes, shap_df.columns[:3]):
            data = [shap_df.loc[shap_df["gffcompare_class"] == c, col].values for c in classes]
            valid = [(i, d) for i, d in enumerate(data) if len(d) > 0]
            if valid:
                vpos, vdata = zip(*valid)
                parts = ax.violinplot(list(vdata), positions=list(vpos),
                                      showmedians=True, showextrema=False)
                for pc, pos in zip(parts["bodies"], vpos):
                    pc.set_facecolor(CLASS_COLOUR.get(classes[pos], "#999"))
                    pc.set_alpha(0.7)
                parts["cmedians"].set_color("black")
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, fontsize=8)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_title(col, fontsize=9)
            ax.set_ylabel("SHAP value")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return scores_df


# ─── comparison page ─────────────────────────────────────────────────────────

def page_comparison(lr_scores: pd.DataFrame, lgb_scores: pd.DataFrame,
                    y_test, proba_lr_test, proba_lgb_test, pdf: PdfPages):
    merged = lr_scores[["transcript_id", "gffcompare_class", "correct_prob_lr"]].merge(
        lgb_scores[["transcript_id", "correct_prob_lgb"]], on="transcript_id", how="inner"
    )

    classes = [c for c in CLASS_ORDER if c in merged["gffcompare_class"].values]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Model comparison: Logistic Regression vs LightGBM",
                 fontsize=13, fontweight="bold")

    # ROC overlay
    ax = axes[0, 0]
    auc_lr  = roc_auc_score(y_test, proba_lr_test)
    auc_lgb = roc_auc_score(y_test, proba_lgb_test)
    RocCurveDisplay.from_predictions(y_test, proba_lr_test,  ax=ax, name=f"LR  (AUC={auc_lr:.3f})")
    RocCurveDisplay.from_predictions(y_test, proba_lgb_test, ax=ax, name=f"LGB (AUC={auc_lgb:.3f})")
    ax.set_title("ROC — test split")

    # LR rescue fracs
    _rescue_bar(axes[0, 1], merged, "correct_prob_lr",  "LR: fraction P≥0.5 per class")
    _rescue_bar(axes[0, 2], merged, "correct_prob_lgb", "LGB: fraction P≥0.5 per class")

    # LR vs LGB scatter for intermediate classes only
    inter = merged[merged["gffcompare_class"].isin({"j", "c", "k", "o", "e", "m", "n", "p"})]
    ax = axes[1, 0]
    for cls in [c for c in ["j", "c", "k", "o", "e", "m"] if c in inter["gffcompare_class"].values]:
        sub = inter[inter["gffcompare_class"] == cls].sample(
            min(500, len(inter[inter["gffcompare_class"] == cls])), random_state=42)
        ax.scatter(sub["correct_prob_lr"], sub["correct_prob_lgb"],
                   s=5, alpha=0.4, color=CLASS_COLOUR.get(cls, "#999"), label=cls, rasterized=True)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("LR P(correct)")
    ax.set_ylabel("LGB P(correct)")
    ax.set_title("LR vs LGB score (intermediate classes)")
    ax.legend(markerscale=3, fontsize=8)

    # Agreement table
    ax = axes[1, 1]
    summary_rows = []
    for cls in classes:
        sub = merged[merged["gffcompare_class"] == cls]
        n = len(sub)
        lr_rescue  = (sub["correct_prob_lr"]  >= 0.5).mean()
        lgb_rescue = (sub["correct_prob_lgb"] >= 0.5).mean()
        both_rescue = ((sub["correct_prob_lr"] >= 0.5) & (sub["correct_prob_lgb"] >= 0.5)).mean()
        summary_rows.append([cls, n, f"{lr_rescue:.2f}", f"{lgb_rescue:.2f}", f"{both_rescue:.2f}"])

    ax.axis("off")
    tbl = ax.table(
        cellText=summary_rows,
        colLabels=["class", "N", "LR≥0.5", "LGB≥0.5", "both≥0.5"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.4)
    ax.set_title("Rescue rate summary per class", fontsize=10, fontweight="bold")

    # Violin comparison for j class (most important)
    ax = axes[1, 2]
    j_lr  = merged.loc[merged["gffcompare_class"] == "j", "correct_prob_lr"].values
    j_lgb = merged.loc[merged["gffcompare_class"] == "j", "correct_prob_lgb"].values
    parts = ax.violinplot([j_lr, j_lgb], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], ["#1f77b4", "#ff7f0e"]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LR", "LGB"])
    ax.set_ylabel("P(correct)")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Score distribution for class 'j'")

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
    p.add_argument("--test-frac",  type=float, default=0.2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    print("Loading data …", flush=True)
    df = load_data(args.base_dir, args.annot_tag)
    print(f"  {len(df):,} transcripts, {df['species'].nunique()} species", flush=True)

    X_base, base_names = build_base_matrix(df)
    X_lr, lr_names     = add_interactions(X_base, base_names)

    # labels
    y_all = df["gffcompare_class"].map(
        lambda c: 1 if c in POSITIVE_CLASSES else (0 if c in NEGATIVE_CLASSES else np.nan)
    )
    labelled    = y_all.notna()
    X_lab_base  = X_base.values[labelled]
    X_lab_lr    = X_lr[labelled]
    y_lab       = y_all[labelled].values.astype(int)
    print(f"  Labelled: {labelled.sum():,}  pos={y_lab.sum():,}  neg={(y_lab==0).sum():,}",
          flush=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=42)
    train_idx, test_idx = next(sss.split(X_lab_base, y_lab))

    X_tr_base, X_te_base = X_lab_base[train_idx], X_lab_base[test_idx]
    X_tr_lr,   X_te_lr   = X_lab_lr[train_idx],   X_lab_lr[test_idx]
    y_train, y_test       = y_lab[train_idx],       y_lab[test_idx]

    pdf_path = args.out_dir / "advanced_classifiers.pdf"
    print(f"Writing PDF to {pdf_path}", flush=True)

    with PdfPages(pdf_path) as pdf:
        lr_scores  = run_lr(X_tr_lr, y_train, X_te_lr, y_test,
                            X_lr, lr_names, df, args.out_dir, pdf)
        lgb_scores = run_lgb(X_tr_base, y_train, X_te_base, y_test,
                             X_base.values, base_names, df, args.out_dir, pdf)

        # stash test probabilities for comparison page
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # re-predict on test set for comparison page using saved scores
        lr_proba_test  = lr_scores.loc[labelled, "correct_prob_lr"].values[test_idx]
        lgb_proba_test = lgb_scores.loc[labelled, "correct_prob_lgb"].values[test_idx]

        print("Comparison page …", flush=True)
        page_comparison(lr_scores, lgb_scores,
                        y_test, lr_proba_test, lgb_proba_test, pdf)

    print(f"Done. Outputs in {args.out_dir}", flush=True)

    # print summary table to stdout
    merged = lr_scores[["gffcompare_class", "correct_prob_lr"]].merge(
        lgb_scores[["transcript_id", "correct_prob_lgb"]], left_index=True, right_index=True
    )
    print("\nRescue summary (fraction scoring ≥ 0.5):")
    print(f"{'class':<6} {'N':>6}  {'LR':>6}  {'LGB':>6}  {'both':>6}")
    for cls in CLASS_ORDER:
        sub = lr_scores[lr_scores["gffcompare_class"] == cls]
        sub_lgb = lgb_scores[lgb_scores["gffcompare_class"] == cls]
        n = len(sub)
        if n == 0:
            continue
        lr_r  = (sub["correct_prob_lr"].values  >= 0.5).mean()
        lgb_r = (sub_lgb["correct_prob_lgb"].values >= 0.5).mean()
        both  = ((sub["correct_prob_lr"].values >= 0.5) &
                 (sub_lgb["correct_prob_lgb"].values >= 0.5)).mean()
        print(f"{cls:<6} {n:>6}  {lr_r:>6.3f}  {lgb_r:>6.3f}  {both:>6.3f}")


if __name__ == "__main__":
    main()
