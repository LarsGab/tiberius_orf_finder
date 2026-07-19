"""Diagnose predicted-locus FPs per species using gffcompare tracking codes.

Reads ``merged.tracking`` and ``merged.loci`` produced by evaluate_accuracy
(one gffcompare run per species) and categorises every predicted locus
(XLOC) as TP or FP, then buckets FP loci by the class codes of the
transcripts inside them.

Class-code semantics (per gffcompare):
    TP-like  : = c k m n j e   (structural agreement w/ a ref transcript)
    same-str : o               (unspecified same-strand exonic overlap)
    antisense: s x             (opposite-strand overlap)
    intronic : i               (fully within a ref intron, same strand)
    contains : y               (contains a ref inside its own intron)
    runon    : p               (close, no overlap)
    intergen : u               (no ref overlap at all)

A locus is TP iff it contains at least one TP-like class code; otherwise
FP with the bucket = "dominant" non-TP code among its transcripts
(precedence order below).

Usage::

    python scripts/fp_locus_diagnostic.py \\
        --eval-dir results/vertebrates_test/eval_accuracy_epoch_74_filt_pfilt \\
        --species Gallus_gallus Bos_taurus ... \\
        --out-tsv results/vertebrates_test/fp_locus_diagnostic_pfilt.tsv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# TP-like class codes: structural agreement with a reference transcript.
# ``~`` (gffcompare v0.12+): predicted intron chain overlaps a ref intron
# chain with extra/shifted introns — still a match to a ref gene.
# Only exact intron-chain match is TP for locus precision. This matches
# gffcompare's own "Locus level" precision (verified against Bos merged.stats:
# {"="}-only count matches #matching_loci to within super-loci / query-loci
# rounding). Broader TP sets (adding c/j/~/k) undercount FP by ~50%.
TP_CODES = {"="}
# Precedence for classifying an FP locus by its "worst" transcript. Codes
# that indicate same-strand overlap with a ref (structural mismatch of a
# real gene) come BEFORE opposite-strand / no-overlap codes.
_FP_CODE_ORDER = ["c", "k", "j", "~", "m", "n", "e", "o",
                  "i", "y", "s", "x", "p", "u"]
_FP_BUCKET = {
    "c": "contained_in_ref",             # shorter valid isoform of a ref tx
    "k": "contains_ref",                 # ref tx contained in prediction
    "j": "junction_partial",             # some ref junctions matched, not all
    "~": "loose_intron_match",           # v0.12: partial intron chain overlap
    "m": "retained_intron_full",
    "n": "retained_intron_partial",
    "e": "single_exon_partial",
    "o": "samestrand_overlap_no_match",  # same-strand overlap w/o structure
    "i": "intronic_samestrand",
    "y": "contains_ref_in_intron",
    "s": "antisense",
    "x": "antisense",
    "p": "polymerase_runon",
    "u": "intergenic",
}
_ALL_BUCKETS = ["contained_in_ref", "contains_ref", "junction_partial",
                "loose_intron_match", "retained_intron_full",
                "retained_intron_partial", "single_exon_partial",
                "samestrand_overlap_no_match", "intronic_samestrand",
                "contains_ref_in_intron", "antisense",
                "polymerase_runon", "intergenic", "other"]


def _classify_locus(codes: list[str]) -> tuple[bool, str | None]:
    """Return (is_TP, fp_bucket). fp_bucket is None if is_TP."""
    if any(c in TP_CODES for c in codes):
        return True, None
    for c in _FP_CODE_ORDER:
        if c in codes:
            return False, _FP_BUCKET[c]
    return False, "other"


def _species_stats(eval_dir: Path, species: str) -> dict:
    """Parse gffcompare's merged.tracking for one species."""
    tracking = eval_dir / "gffcompare_runs" / species / "merged.tracking"
    codes_per_locus: dict[str, list[str]] = defaultdict(list)
    tx_per_locus: dict[str, int] = defaultdict(int)
    with tracking.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            xloc = f[1]           # e.g. XLOC_000001
            code = f[3]           # class code
            codes_per_locus[xloc].append(code)
            tx_per_locus[xloc] += 1

    tp = 0
    fp_buckets = defaultdict(int)
    fp_tx_counts = defaultdict(list)   # for size distribution
    for xloc, codes in codes_per_locus.items():
        is_tp, bucket = _classify_locus(codes)
        if is_tp:
            tp += 1
        else:
            fp_buckets[bucket] += 1
            fp_tx_counts[bucket].append(tx_per_locus[xloc])

    total_loci = len(codes_per_locus)
    fp = total_loci - tp
    return {
        "species": species,
        "total_loci": total_loci,
        "tp_loci": tp,
        "fp_loci": fp,
        "locus_precision_pct": 100.0 * tp / max(1, total_loci),
        **{f"fp_{b}": fp_buckets.get(b, 0) for b in _ALL_BUCKETS},
        **{f"fp_{b}_pct_of_fp":
             (100.0 * fp_buckets.get(b, 0) / max(1, fp))
           for b in _ALL_BUCKETS},
    }


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Directory containing gffcompare_runs/<species>/")
    ap.add_argument("--species", nargs="+", required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    rows = []
    for sp in args.species:
        try:
            rows.append(_species_stats(args.eval_dir, sp))
        except FileNotFoundError as e:
            print(f"[skip] {sp}: {e}", flush=True)

    if not rows:
        print("No species processed.", flush=True)
        return 1

    keys = list(rows[0].keys())
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[fp-diag] wrote {args.out_tsv}", flush=True)

    # Concise stdout summary. Show the top same-strand-ref-overlap
    # buckets (the "wrong isoform / partial match" cases the user cares
    # about) alongside intergenic and antisense for context.
    print()
    print(f"{'species':<26} {'#loci':>6} {'#TP':>6} {'#FP':>6} "
          f"{'j':>6} {'~':>6} {'c':>6} {'k':>6} "
          f"{'o':>5} {'i':>5} {'antis':>5} {'intg':>6}")
    for r in rows:
        print(f"{r['species']:<26} {r['total_loci']:>6d} "
              f"{r['tp_loci']:>6d} {r['fp_loci']:>6d} "
              f"{r['fp_junction_partial']:>6d} "
              f"{r['fp_loose_intron_match']:>6d} "
              f"{r['fp_contained_in_ref']:>6d} "
              f"{r['fp_contains_ref']:>6d} "
              f"{r['fp_samestrand_overlap_no_match']:>5d} "
              f"{r['fp_intronic_samestrand']:>5d} "
              f"{r['fp_antisense']:>5d} "
              f"{r['fp_intergenic']:>6d}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
