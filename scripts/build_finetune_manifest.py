"""Build a TSV manifest for fine-tuning the ORFfinder classification head.

Scans the vertebrates_training results directory and emits one row per species
that has all three required files:
  - scores.tsv          (from score_tiberius.py)
  - tiberius_seqlen.gtf (Tiberius ab initio prediction)
  - annotation.gff      (reference annotation)

Usage
-----
python scripts/build_finetune_manifest.py \\
  --results-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_training \\
  --score-tag   score_tiberius_epoch_74_up500 \\
  --out-tsv     /projects/AI-GUSTUS/tiberius_orf_finder/results/finetune/manifest_vertebrates_train.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build fine-tuning manifest from vertebrates_training results."
    )
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Root results dir; expects <sp>/assembly/annotation.gff, "
                         "<sp>/tiberius/tiberius_seqlen.gtf, <sp>/<score-tag>/scores.tsv")
    ap.add_argument("--score-tag",   default="score_tiberius_epoch_74_up500")
    ap.add_argument("--out-tsv",     type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rows = []
    missing: list[str] = []

    for sp_dir in sorted(args.results_dir.iterdir()):
        if not sp_dir.is_dir():
            continue
        species    = sp_dir.name
        scores_tsv = sp_dir / args.score_tag / "scores.tsv"
        tib_gtf    = sp_dir / "tiberius" / "tiberius_seqlen.gtf"
        ref_gff    = sp_dir / "assembly" / "annotation.gff"

        have = {
            "scores": scores_tsv.exists() and scores_tsv.stat().st_size > 0,
            "gtf":    tib_gtf.exists()    and tib_gtf.stat().st_size > 0,
            "ref":    ref_gff.exists()    and ref_gff.stat().st_size > 0,
        }
        if all(have.values()):
            rows.append({
                "species":    species,
                "scores_tsv": str(scores_tsv),
                "tib_gtf":    str(tib_gtf),
                "ref_gff":    str(ref_gff),
            })
        else:
            missing_files = [k for k, v in have.items() if not v]
            missing.append(f"  {species}: missing {missing_files}")

    df = pd.DataFrame(rows)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)

    print(f"[manifest] {len(rows)} species with all files -> {args.out_tsv}")
    if missing:
        print(f"[manifest] {len(missing)} species incomplete:")
        for m in missing:
            print(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
