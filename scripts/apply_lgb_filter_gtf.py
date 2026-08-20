"""Filter per-species ORF GTF files using pre-computed LightGBM probabilities.

Reads the pooled orf_scores_lgb.tsv produced by train_orf_advanced_classifiers.py,
then for each species streams the input GTF keeping only transcripts with
correct_prob_lgb >= threshold. Also writes a per-species stats line to stdout.

Usage
-----
python scripts/apply_lgb_filter_gtf.py \\
  --scores    /projects/AI-GUSTUS/tiberius_orf_finder/results/filter_analysis/orf_scores_lgb.tsv \\
  --base-dir  /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \\
  --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \\
  --in-gtf    orfs.filtered.gtf \\
  --out-gtf   orfs_lgb_filtered.gtf \\
  [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TID_RE = re.compile(r'transcript_id\s+"([^"]+)"')


def filter_gtf(in_gtf: Path, passing_tids: set[str], out_gtf: Path) -> tuple[int, int]:
    """Stream in_gtf, write lines whose transcript_id is in passing_tids.
    Returns (n_kept_transcripts, n_total_transcripts)."""
    kept_tids: set[str] = set()
    all_tids:  set[str] = set()

    with in_gtf.open() as fin, out_gtf.open("w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            m = TID_RE.search(line)
            if not m:
                continue
            tid = m.group(1)
            all_tids.add(tid)
            if tid in passing_tids:
                kept_tids.add(tid)
                fout.write(line)

    return len(kept_tids), len(all_tids)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores",    required=True, type=Path,
                   help="orf_scores_lgb.tsv from train_orf_advanced_classifiers.py")
    p.add_argument("--base-dir",  required=True, type=Path,
                   help="vertebrates_test root (one subdir per species)")
    p.add_argument("--annot-tag", default="annotate_epoch_74_filt_tpm1cov3len300_lorf")
    p.add_argument("--in-gtf",    default="orfs.filtered.gtf",
                   help="GTF filename inside each species/annot-tag dir")
    p.add_argument("--out-gtf",   default="orfs_lgb_filtered.gtf",
                   help="Output GTF filename (written to same dir as in-gtf)")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    print(f"Loading scores from {args.scores} …", flush=True)
    scores = pd.read_csv(args.scores, sep="\t", usecols=["transcript_id", "species", "correct_prob_lgb"])
    print(f"  {len(scores):,} rows, threshold = {args.threshold}", flush=True)

    print(f"\n{'species':<35} {'kept':>6} {'total':>6}  {'pct':>6}  out")
    print("-" * 80)

    for species, grp in scores.groupby("species"):
        sp_dir = args.base_dir / species / args.annot_tag
        in_gtf  = sp_dir / args.in_gtf
        out_gtf = sp_dir / args.out_gtf

        if not in_gtf.exists():
            print(f"  {species:<33}  SKIP — {in_gtf} not found")
            continue

        passing = set(grp.loc[grp["correct_prob_lgb"] >= args.threshold, "transcript_id"])
        n_kept, n_total = filter_gtf(in_gtf, passing, out_gtf)
        pct = 100 * n_kept / n_total if n_total else 0
        print(f"  {species:<33} {n_kept:>6} {n_total:>6}  {pct:>5.1f}%  {out_gtf}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
