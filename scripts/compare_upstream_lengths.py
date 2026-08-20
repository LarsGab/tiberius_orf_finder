"""Compare ORFfinder scores across different upstream flanking lengths.

For each upstream length (500, 1000, 2000 bp) loads the pre-computed scores
TSVs, assigns TP/FP labels via gffcompare (run once per species, shared), then
plots side-by-side comparisons to answer: does more upstream context improve
TP vs FP discrimination?

Outputs:
  - <out-pdf>           — figure with Sn-Prec curves, violin distributions,
                          F1/AUC bar chart, and per-metric score deltas
  - <out-pdf>.tsv       — summary table (metric × upstream_bp → max-F1, AUC)

Usage
-----
python scripts/compare_upstream_lengths.py \\
  --score-dir  /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --score-tags score_tiberius_epoch_74_up500 score_tiberius_epoch_74_up1000 score_tiberius_epoch_74_up2000 \\
  --tib-tmpl   '/home/gabriell/tiberius_benchmarking/paper/Vertebrata/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf' \\
  --ref-tmpl   '/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/assembly/annot_cds.gff' \\
  --out-pdf    /projects/AI-GUSTUS/tiberius_orf_finder/results/figures/upstream_length_comparison.pdf
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
from sklearn.metrics import roc_auc_score

_SPECIES = [
    "Gallus_gallus",
    "Pristiophorus_japonicus",
    "Bos_taurus",
    "Delphinapterus_leucas",
    "Homo_sapiens",
]
_SP_SHORT = {
    "Gallus_gallus":           "G.gal",
    "Pristiophorus_japonicus": "P.jap",
    "Bos_taurus":              "B.tau",
    "Delphinapterus_leucas":   "D.leu",
    "Homo_sapiens":            "H.sap",
}
_FOCUS_METRICS = [
    ("mean_coding_prob", "Mean P(coding)"),
    ("start_prob",       "P(START) at ATG"),
    ("combined",         "min(coding, START)"),
]
_TAG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# gffcompare
# ---------------------------------------------------------------------------

def _run_gffcompare(tib_gtf: Path, ref_gff: Path, tmpdir: Path) -> set[str]:
    prefix = tmpdir / "gc"
    cmd = [
        "gffcompare", "--strict-match", "-e", "3",
        "-r", str(ref_gff), "-o", str(prefix), str(tib_gtf),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  gffcompare failed:\n{result.stderr[:400]}")
        return set()
    tracking = Path(str(prefix) + ".tracking")
    if not tracking.exists():
        print(f"  WARNING: no .tracking file in {tmpdir}")
        return set()
    tp_ids: set[str] = set()
    for line in tracking.read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5 or parts[3] != "=":
            continue
        try:
            tp_ids.add(parts[4].split(":", 1)[1].split("|")[1])
        except (IndexError, ValueError):
            continue
    return tp_ids


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def _load_tp_labels(tib_tmpl: str, ref_tmpl: str) -> dict[str, set[str]]:
    """Run gffcompare once per species; return {species: tp_transcript_id_set}."""
    tp_by_species: dict[str, set[str]] = {}
    for sp in _SPECIES:
        tib_gtf = Path(tib_tmpl.format(sp=sp))
        ref_gff = Path(ref_tmpl.format(sp=sp))
        if not tib_gtf.exists() or not ref_gff.exists():
            print(f"[cmp] {sp}: missing GTF or reference — no TP labels")
            tp_by_species[sp] = set()
            continue
        print(f"[cmp] {sp}: running gffcompare …", flush=True)
        with tempfile.TemporaryDirectory() as td:
            tp_ids = _run_gffcompare(tib_gtf, ref_gff, Path(td))
        tp_by_species[sp] = tp_ids
        print(f"[cmp] {sp}: {len(tp_ids):,} TPs", flush=True)
    return tp_by_species


def _load_scores(score_dir: Path, tag: str, tp_by_species: dict) -> pd.DataFrame:
    """Load scores TSV for all available species under a given tag."""
    frames = []
    for sp in _SPECIES:
        tsv = score_dir / sp / tag / "scores.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        df["species"] = sp
        df["tp"] = df["transcript_id"].isin(tp_by_species.get(sp, set()))
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No scores found for tag '{tag}'")
    df_all = pd.concat(frames, ignore_index=True)
    df_all["combined"] = np.minimum(df_all["mean_coding_prob"], df_all["start_prob"])
    return df_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _sn_prec_curve(labels: np.ndarray, scores: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, Sn, Prec) sweeping from keep-all to keep-none."""
    n_tp = labels.sum()
    if n_tp == 0:
        ts = np.linspace(0, 1, 100)
        return ts, np.zeros_like(ts), np.zeros_like(ts)
    order = np.argsort(scores)[::-1]
    sl = labels[order]
    ss = scores[order]
    tp_cs = np.cumsum(sl)
    tot_cs = np.arange(1, len(labels) + 1)
    sn   = tp_cs / n_tp
    prec = tp_cs / tot_cs
    return (np.concatenate([[np.nan], ss]),
            np.concatenate([[0.0], sn]),
            np.concatenate([[1.0], prec]))


def _f1(sn: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = sn + prec
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, 2 * sn * prec / d, 0.0)


def _max_f1(sn: np.ndarray, prec: np.ndarray) -> tuple[float, float, float]:
    f = _f1(sn, prec)
    i = np.argmax(f)
    return float(sn[i]), float(prec[i]), float(f[i])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or (~labels.astype(bool)).sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _compute_tag_metrics(df: pd.DataFrame) -> dict:
    labels = df["tp"].astype(int).values
    out = {}
    for col, _ in _FOCUS_METRICS:
        scores = df[col].fillna(0.0).values
        ts, sn, prec = _sn_prec_curve(labels, scores)
        sn_f1, prec_f1, f1 = _max_f1(sn, prec)
        out[col] = {
            "curve": (ts, sn, prec),
            "max_f1": (sn_f1, prec_f1, f1),
            "auc": _auc(labels.astype(bool), scores),
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _short_tag(tag: str) -> str:
    for part in tag.split("_"):
        if part.startswith("up"):
            return part  # e.g. "up500"
    return tag


def _plot_sn_prec_overlay(axes, tag_data: dict, tags: list[str]) -> None:
    """One Sn-Prec panel per focus metric, all tags overlaid."""
    for ax, (col, mlabel) in zip(axes, _FOCUS_METRICS):
        for ci, tag in enumerate(tags):
            m = tag_data[tag]["metrics"]
            ts, sn, prec = m[col]["curve"]
            short = _short_tag(tag)
            f1_sn, f1_prec, f1 = m[col]["max_f1"]
            ax.plot(sn, prec, color=_TAG_COLORS[ci], lw=1.6,
                    label=f"{short}  F1={f1:.3f}")
            ax.scatter([f1_sn], [f1_prec], color=_TAG_COLORS[ci],
                       s=60, zorder=5, marker="*")
        ax.set_title(mlabel, fontsize=9)
        ax.set_xlabel("Sensitivity", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower left")
        ax.tick_params(labelsize=7)


def _plot_violin_comparison(axes, tag_data: dict, tags: list[str]) -> None:
    """TP vs FP violin for each metric (rows) × each tag (columns)."""
    for col_i, (col, mlabel) in enumerate(_FOCUS_METRICS):
        ax = axes[col_i]
        positions = []
        data = []
        colors = []
        tick_pos = []
        tick_labels = []
        group_spacing = 3.0
        for ti, tag in enumerate(tags):
            df = tag_data[tag]["df"]
            short = _short_tag(tag)
            base = ti * group_spacing
            tp_vals  = df.loc[df["tp"],  col].dropna().values
            fp_vals  = df.loc[~df["tp"], col].dropna().values
            for offset, vals, c in [(0.0, tp_vals, "#4C72B0"),
                                    (1.0, fp_vals, "#C44E52")]:
                if len(vals) > 1:
                    positions.append(base + offset)
                    data.append(vals)
                    colors.append(c)
            tick_pos.append(base + 0.5)
            tick_labels.append(short)

        if data:
            parts = ax.violinplot(data, positions=positions,
                                  widths=0.7, showmedians=True,
                                  showextrema=False)
            for pc, c in zip(parts["bodies"], colors):
                pc.set_facecolor(c)
                pc.set_alpha(0.6)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.2)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel(mlabel, fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        if col_i == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor="#4C72B0", alpha=0.6, label="TP"),
                                Patch(facecolor="#C44E52", alpha=0.6, label="FP")],
                      fontsize=7, loc="lower right")


def _plot_auc_bar(ax, tag_data: dict, tags: list[str]) -> None:
    n_metrics = len(_FOCUS_METRICS)
    x = np.arange(n_metrics)
    width = 0.8 / len(tags)
    for ti, tag in enumerate(tags):
        short = _short_tag(tag)
        m = tag_data[tag]["metrics"]
        aucs = [m[col]["auc"] for col, _ in _FOCUS_METRICS]
        ax.bar(x + ti * width - 0.4 + width / 2, aucs, width,
               color=_TAG_COLORS[ti], alpha=0.8, label=short)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in _FOCUS_METRICS], fontsize=8)
    ax.set_ylabel("ROC AUC", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=7)
    ax.set_title("ROC AUC per metric and upstream length", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle=":")


def _plot_f1_bar(ax, tag_data: dict, tags: list[str]) -> None:
    n_metrics = len(_FOCUS_METRICS)
    x = np.arange(n_metrics)
    width = 0.8 / len(tags)
    for ti, tag in enumerate(tags):
        short = _short_tag(tag)
        m = tag_data[tag]["metrics"]
        f1s = [m[col]["max_f1"][2] for col, _ in _FOCUS_METRICS]
        ax.bar(x + ti * width - 0.4 + width / 2, f1s, width,
               color=_TAG_COLORS[ti], alpha=0.8, label=short)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in _FOCUS_METRICS], fontsize=8)
    ax.set_ylabel("max F1", fontsize=8)
    ax.set_ylim(0.55, 0.80)
    ax.legend(fontsize=7)
    ax.set_title("Max F1 per metric and upstream length", fontsize=9)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _build_summary(tag_data: dict, tags: list[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        short = _short_tag(tag)
        m = tag_data[tag]["metrics"]
        for col, mlabel in _FOCUS_METRICS:
            sn_f1, prec_f1, f1 = m[col]["max_f1"]
            rows.append({
                "upstream": short,
                "metric":   mlabel,
                "max_F1":   round(f1, 4),
                "Sn@maxF1": round(sn_f1, 4),
                "Pr@maxF1": round(prec_f1, 4),
                "AUC":      round(m[col]["auc"], 4),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare ORFfinder scores across upstream flanking lengths."
    )
    ap.add_argument("--score-dir",  type=Path, required=True)
    ap.add_argument("--score-tags", nargs="+",
                    default=["score_tiberius_epoch_74_up500",
                             "score_tiberius_epoch_74_up1000",
                             "score_tiberius_epoch_74_up2000"])
    ap.add_argument("--tib-tmpl",   required=True)
    ap.add_argument("--ref-tmpl",   required=True)
    ap.add_argument("--out-pdf",    type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    print("[cmp] Assigning TP labels via gffcompare …", flush=True)
    tp_by_species = _load_tp_labels(args.tib_tmpl, args.ref_tmpl)

    tag_data: dict = {}
    for tag in args.score_tags:
        print(f"[cmp] Loading scores: {tag}", flush=True)
        df = _load_scores(args.score_dir, tag, tp_by_species)
        print(f"[cmp]   {len(df):,} transcripts  TP={df['tp'].sum():,}  "
              f"FP={(~df['tp']).sum():,}", flush=True)
        tag_data[tag] = {
            "df":      df,
            "metrics": _compute_tag_metrics(df),
        }

    # Summary
    summary = _build_summary(tag_data, args.score_tags)
    out_tsv = args.out_pdf.with_suffix(".tsv")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n[cmp] Summary:\n{summary.to_string(index=False)}\n", flush=True)

    # -------------------------------------------------------------------
    # Figure layout
    #   Row 0 (top): Sn-Prec curves — one panel per focus metric
    #   Row 1 (mid): violin TP/FP distributions — one panel per metric
    #   Row 2 (bot): AUC bar chart | max-F1 bar chart
    # -------------------------------------------------------------------
    n_metrics = len(_FOCUS_METRICS)
    fig = plt.figure(figsize=(5 * n_metrics, 14))

    # Row 0: Sn-Prec overlay
    snprec_axes = [fig.add_subplot(3, n_metrics, ci + 1)
                   for ci in range(n_metrics)]
    _plot_sn_prec_overlay(snprec_axes, tag_data, args.score_tags)
    snprec_axes[0].set_title(
        f"Sn–Prec: {_FOCUS_METRICS[0][1]}", fontsize=9)

    # Row 1: violins
    vln_axes = [fig.add_subplot(3, n_metrics, n_metrics + ci + 1)
                for ci in range(n_metrics)]
    _plot_violin_comparison(vln_axes, tag_data, args.score_tags)

    # Row 2: bar charts
    ax_auc = fig.add_subplot(3, 2, 5)
    _plot_auc_bar(ax_auc, tag_data, args.score_tags)

    ax_f1 = fig.add_subplot(3, 2, 6)
    _plot_f1_bar(ax_f1, tag_data, args.score_tags)

    fig.suptitle(
        "ORFfinder upstream-length comparison — vertebrates test species\n"
        "Blue=TP, Red=FP  |  ★ = max-F1 operating point",
        fontsize=11,
    )
    fig.tight_layout()
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[cmp] Saved figure -> {args.out_pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
