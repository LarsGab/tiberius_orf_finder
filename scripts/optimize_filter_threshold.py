"""Find optimal ORFfinder score thresholds for filtering Tiberius predictions.

For each candidate score metric (and their combination), sweeps thresholds
and computes the resulting gene-level Sensitivity and Precision against a
reference annotation. Produces:

  1. Sn-Prec curves — the full operating-point frontier per metric
  2. F1 vs threshold curves — where the peak is
  3. A summary table — optimal threshold, Sn, Prec, F1 per metric
  4. Operating-point annotations — the current 0.2/0.2 filter and the
     iso-precision / iso-F1 alternatives on the min(coding,start) curve

Sn   = TP_kept / TP_total        (fraction of true genes retained)
Prec = TP_kept / total_kept      (fraction of kept genes that are correct)
F1   = 2·Sn·Prec / (Sn + Prec)

TP is defined by gffcompare class_code '=' with --strict-match.

Usage
-----
python scripts/optimize_filter_threshold.py \\
  --score-dir  /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --score-tag  score_tiberius_epoch_74_up500 \\
  --tib-tmpl   '/home/gabriell/tiberius_benchmarking/paper/Vertebrata/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf' \\
  --ref-tmpl   '/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/assembly/annot_cds.gff' \\
  --out-pdf    results/figures/filter_threshold_opt.pdf \\
  --out-tsv    results/figures/filter_threshold_opt.tsv
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

_SP_SHORT = {
    "Gallus_gallus":           "G.gal",
    "Pristiophorus_japonicus": "P.jap",
    "Bos_taurus":              "B.tau",
    "Delphinapterus_leucas":   "D.leu",
    "Homo_sapiens":            "H.sap",
}

# Metrics to sweep — (column_name, display_label, higher_is_better)
_METRICS = [
    ("mean_coding_prob",   "Mean P(coding)",            True),
    ("start_prob",         "P(START) at ATG",           True),
    ("stop_prob",          "P(STOP) at last base",      True),
    ("frac_argmax_coding", "Frac. argmax=coding",       True),
    ("mean_ir_prob",       "Mean P(IR)  [inverted]",    False),  # lower = better
]

_CURRENT_MIN_CODING = 0.2
_CURRENT_MIN_START  = 0.2


# ---------------------------------------------------------------------------
# gffcompare (copied from plot_tiberius_scores.py)
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


def _load_and_label(
    score_dir: Path,
    score_tag: str,
    tib_tmpl: str,
    ref_tmpl: str,
) -> pd.DataFrame:
    """Load scores TSVs and add gffcompare TP labels. Returns pooled DataFrame."""
    frames: list[pd.DataFrame] = []
    for sp in _SPECIES:
        tsv = score_dir / sp / score_tag / "scores.tsv"
        if not tsv.exists():
            print(f"[opt] {sp}: no scores.tsv, skipping")
            continue
        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        df["species"] = sp

        tib_gtf = Path(tib_tmpl.format(sp=sp))
        ref_gff = Path(ref_tmpl.format(sp=sp))
        if not tib_gtf.exists() or not ref_gff.exists():
            print(f"[opt] {sp}: missing GTF or reference, no TP labels")
            df["tp"] = False
        else:
            with tempfile.TemporaryDirectory() as td:
                tp_ids = _run_gffcompare(tib_gtf, ref_gff, Path(td))
            df["tp"] = df["transcript_id"].isin(tp_ids)
            print(f"[opt] {sp}: TP={df['tp'].sum():,}  FP={(~df['tp']).sum():,}")

        frames.append(df)

    if not frames:
        raise RuntimeError("No data loaded.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def _sn_prec_curve(labels: np.ndarray, scores: np.ndarray, higher_is_better: bool = True
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep thresholds and return (thresholds, Sn, Prec) arrays.

    Sorts from most-permissive (keep all) to most-restrictive (keep none).
    At each threshold: keep transcripts with score >= t (or <= t if lower=better).
    Returns parallel arrays of length = number of unique score values + 2.
    """
    n_tp_total = labels.sum()
    if n_tp_total == 0:
        ts = np.linspace(0, 1, 100)
        return ts, np.zeros_like(ts), np.zeros_like(ts)

    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]          # highest first → most permissive first
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    # Cumulative from the "keep-all" end
    tp_cumsum    = np.cumsum(sorted_labels)
    total_cumsum = np.arange(1, len(labels) + 1)

    sn   = tp_cumsum   / n_tp_total
    prec = tp_cumsum   / total_cumsum

    # Threshold at each step: the score of the current (marginal) transcript
    thresholds = sorted_scores

    # Prepend "keep nothing" point and append "keep all" point
    thresholds = np.concatenate([[np.nan], thresholds])
    sn         = np.concatenate([[0.0],   sn])
    prec       = np.concatenate([[1.0],   prec])   # undefined → 1 by convention

    return thresholds, sn, prec


def _f1(sn: np.ndarray, prec: np.ndarray) -> np.ndarray:
    denom = sn + prec
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, 2 * sn * prec / denom, 0.0)


def _find_iso_prec_point(
    thresholds: np.ndarray,
    sn: np.ndarray,
    prec: np.ndarray,
    target_prec: float,
) -> tuple[float, float, float]:
    """Return (threshold, Sn, Prec) at the highest Sn that achieves target_prec."""
    mask = prec >= target_prec
    if not mask.any():
        return float("nan"), float("nan"), target_prec
    idx = np.where(mask)[0][-1]    # last index (most permissive) still at target prec
    return float(thresholds[idx]), float(sn[idx]), float(prec[idx])


def _find_max_f1_point(
    thresholds: np.ndarray,
    sn: np.ndarray,
    prec: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (threshold, Sn, Prec, F1) at the maximum F1 point."""
    f = _f1(sn, prec)
    idx = np.argmax(f)
    return (float(thresholds[idx]), float(sn[idx]),
            float(prec[idx]), float(f[idx]))


# ---------------------------------------------------------------------------
# Current operating point (0.2/0.2 filter)
# ---------------------------------------------------------------------------

def _current_op_point(df: pd.DataFrame) -> tuple[float, float, float]:
    """Sn, Prec, F1 for the current min_coding=0.2, min_start=0.2 filter."""
    keep = (df["mean_coding_prob"] >= _CURRENT_MIN_CODING) & \
           (df["start_prob"]       >= _CURRENT_MIN_START)
    tp_kept    = (df.loc[keep, "tp"]).sum()
    total_kept = keep.sum()
    n_tp_total = df["tp"].sum()
    sn   = tp_kept / n_tp_total   if n_tp_total   > 0 else 0.0
    prec = tp_kept / total_kept   if total_kept   > 0 else 0.0
    f1   = 2 * sn * prec / (sn + prec) if (sn + prec) > 0 else 0.0
    return float(sn), float(prec), float(f1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sn_prec(ax, results: dict, current_sn: float, current_prec: float) -> None:
    """Sn-Prec curves for all metrics on one axis."""
    colors = plt.cm.tab10.colors
    for ci, (metric, label, hib) in enumerate(_METRICS):
        ts, sn, prec = results[metric]["curve"]
        ax.plot(sn, prec, color=colors[ci], lw=1.4, label=label)
        # mark max-F1 point
        _, f1_sn, f1_prec, _ = results[metric]["max_f1"]
        ax.scatter([f1_sn], [f1_prec], color=colors[ci], s=50, zorder=5,
                   marker="*")

    # min(coding, start) combined
    ts, sn, prec = results["combined"]["curve"]
    ax.plot(sn, prec, color="black", lw=2.0, linestyle="--",
            label="min(coding, START)")
    _, f1_sn, f1_prec, _ = results["combined"]["max_f1"]
    ax.scatter([f1_sn], [f1_prec], color="black", s=70, zorder=6, marker="*")

    # current 0.2/0.2 operating point
    ax.scatter([current_sn], [current_prec], color="red", s=100, zorder=7,
               marker="X", label=f"Current (0.2/0.2)\nSn={current_sn:.3f} Pr={current_prec:.3f}")

    ax.set_xlabel("Sensitivity  (TP kept / TP total)", fontsize=9)
    ax.set_ylabel("Precision  (TP kept / total kept)",  fontsize=9)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Sn – Prec trade-off per score metric", fontsize=10)
    ax.axhline(current_prec, color="red", lw=0.6, linestyle=":", alpha=0.6)
    ax.axvline(current_sn,   color="red", lw=0.6, linestyle=":", alpha=0.6)


def _plot_f1_curves(axes, results: dict) -> None:
    """F1 vs threshold for each metric on individual sub-axes."""
    colors = plt.cm.tab10.colors
    entries = list(_METRICS) + [("combined", "min(coding, START)", True)]
    for ax, (ci, (metric, label, hib)) in zip(axes, enumerate(entries)):
        ts, sn, prec = results[metric]["curve"]
        f1 = _f1(sn, prec)
        # Skip the first NaN threshold point
        mask = ~np.isnan(ts)
        ax.plot(ts[mask], f1[mask], color=colors[ci % 10], lw=1.4)
        t_opt, _, _, f1_opt = results[metric]["max_f1"]
        ax.axvline(t_opt, color=colors[ci % 10], lw=0.8, linestyle="--",
                   label=f"opt={t_opt:.3f}  F1={f1_opt:.3f}")
        ax.axvline(_CURRENT_MIN_CODING, color="red", lw=0.8, linestyle=":",
                   alpha=0.7, label=f"current=0.2")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Threshold", fontsize=7)
        ax.set_ylabel("F1", fontsize=7)
        ax.legend(fontsize=6)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)


def _plot_per_species(axes, df_full: pd.DataFrame, metric: str, label: str,
                      hib: bool) -> None:
    """Per-species Sn-Prec curves for a single metric."""
    colors = plt.cm.tab10.colors
    species_list = sorted(df_full["species"].unique())
    for ax, sp in zip(axes, species_list):
        df = df_full[df_full["species"] == sp].dropna(subset=[metric, "tp"])
        if df.empty:
            continue
        labs   = df["tp"].astype(int).values
        scores = df[metric].values
        ts, sn, prec = _sn_prec_curve(labs, scores, hib)
        ax.plot(sn, prec, lw=1.4, label=_SP_SHORT.get(sp, sp))
        # current op point for this species
        keep = (df["mean_coding_prob"] >= _CURRENT_MIN_CODING) & \
               (df["start_prob"]       >= _CURRENT_MIN_START)
        tp_k  = df.loc[keep, "tp"].sum()
        tot_k = keep.sum()
        ntp   = df["tp"].sum()
        if ntp > 0 and tot_k > 0:
            ax.scatter([tp_k / ntp], [tp_k / tot_k],
                       color="red", s=60, marker="X", zorder=5)
        ax.set_title(_SP_SHORT.get(sp, sp), fontsize=8)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Precision", fontsize=8)
        ax.set_xlabel("Sensitivity", fontsize=7)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _build_summary(df: pd.DataFrame, results: dict, current_sn, current_prec, current_f1
                   ) -> pd.DataFrame:
    rows = []
    for metric, label, hib in _METRICS + [("combined", "min(coding,start)", True)]:
        ts, sn, prec = results[metric]["curve"]
        t_opt, sn_opt, prec_opt, f1_opt = results[metric]["max_f1"]
        t_iso, sn_iso, prec_iso = _find_iso_prec_point(
            ts, sn, prec, target_prec=current_prec,
        )
        rows.append({
            "metric":          label,
            "t_max_f1":        round(t_opt,  4),
            "sn_max_f1":       round(sn_opt, 4),
            "prec_max_f1":     round(prec_opt, 4),
            "f1_max":          round(f1_opt, 4),
            "t_iso_prec":      round(t_iso, 4) if t_iso == t_iso else float("nan"),
            "sn_iso_prec":     round(sn_iso, 4) if sn_iso == sn_iso else float("nan"),
            "delta_sn_vs_0.2": round(sn_iso - current_sn, 4) if sn_iso == sn_iso else float("nan"),
        })
    # Also add current operating point row
    rows.insert(0, {
        "metric":          "Current (min_coding=0.2, min_start=0.2)",
        "t_max_f1":        0.2,
        "sn_max_f1":       round(current_sn, 4),
        "prec_max_f1":     round(current_prec, 4),
        "f1_max":          round(current_f1, 4),
        "t_iso_prec":      0.2,
        "sn_iso_prec":     round(current_sn, 4),
        "delta_sn_vs_0.2": 0.0,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Optimize ORFfinder filter thresholds for Tiberius predictions."
    )
    ap.add_argument("--score-dir", type=Path, required=True)
    ap.add_argument("--score-tag", default="score_tiberius_epoch_74_up500")
    ap.add_argument("--tib-tmpl",  required=True,
                    help="Path template with {sp} for Tiberius GTF.")
    ap.add_argument("--ref-tmpl",  required=True,
                    help="Path template with {sp} for reference GFF.")
    ap.add_argument("--out-pdf",   type=Path, required=True)
    ap.add_argument("--out-tsv",   type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args    = _parse_args(argv)
    out_tsv = args.out_tsv or args.out_pdf.with_suffix(".tsv")

    print("[opt] Loading scores and running gffcompare …", flush=True)
    df = _load_and_label(
        args.score_dir, args.score_tag, args.tib_tmpl, args.ref_tmpl,
    )

    # Add combined metric
    df["combined"] = np.minimum(df["mean_coding_prob"], df["start_prob"])

    labels     = df["tp"].astype(int).values
    current_sn, current_prec, current_f1 = _current_op_point(df)
    print(f"[opt] Current filter (0.2/0.2): Sn={current_sn:.4f}  "
          f"Prec={current_prec:.4f}  F1={current_f1:.4f}", flush=True)

    # Compute curves for every metric
    results: dict = {}
    all_metrics = list(_METRICS) + [("combined", "min(coding,start)", True)]
    for metric, label, hib in all_metrics:
        scores = df[metric].fillna(0 if hib else 1).values
        if not hib:
            scores = -scores      # invert so higher = better throughout
        ts, sn, prec = _sn_prec_curve(labels, scores, higher_is_better=True)
        # Un-invert thresholds for display
        display_ts = -ts if not hib else ts
        results[metric] = {
            "curve":   (display_ts, sn, prec),
            "max_f1":  _find_max_f1_point(display_ts, sn, prec),
        }
        t_opt, sn_opt, prec_opt, f1_opt = results[metric]["max_f1"]
        print(f"[opt]   {label:<30s}  max-F1 @ t={t_opt:.3f}: "
              f"Sn={sn_opt:.3f}  Prec={prec_opt:.3f}  F1={f1_opt:.3f}",
              flush=True)

    # Build and save summary table
    summary = _build_summary(df, results, current_sn, current_prec, current_f1)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n[opt] Summary table:\n{summary.to_string(index=False)}\n", flush=True)
    print(f"[opt] Saved table -> {out_tsv}", flush=True)

    # -------------------------------------------------------------------
    # Figure layout:
    #   Row 0    : pooled Sn-Prec curves for all metrics (full width)
    #   Row 1    : F1 vs threshold for each metric (6 subplots)
    #   Row 2    : per-species Sn-Prec curves for min(coding,start) combined
    # -------------------------------------------------------------------
    n_sp      = df["species"].nunique()
    n_metrics = len(all_metrics)
    fig = plt.figure(figsize=(18, 14))

    # Row 0: pooled Sn-Prec
    ax_snprec = fig.add_subplot(3, 1, 1)
    _plot_sn_prec(ax_snprec, results, current_sn, current_prec)

    # Row 1: F1 vs threshold per metric
    f1_axes = [fig.add_subplot(3, n_metrics, n_metrics + ci + 1)
               for ci in range(n_metrics)]
    _plot_f1_curves(f1_axes, results)

    # Row 2: per-species curves for combined metric
    sp_axes = [fig.add_subplot(3, n_sp, 2 * n_sp + si + 1)
               for si in range(n_sp)]
    _plot_per_species(sp_axes, df, "combined", "min(coding,start)", True)
    # Add a subtitle
    fig.text(0.5, 0.355, "Per-species Sn–Prec: min(coding, START)  "
             "[★ = max-F1,  ✕ = current 0.2/0.2]",
             ha="center", fontsize=9)

    fig.suptitle(
        "ORFfinder filter threshold optimisation — Tiberius vertebrates test",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[opt] Saved figure -> {args.out_pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
