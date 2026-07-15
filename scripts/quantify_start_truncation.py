"""Quantify start-codon mistakes when the ref ATG lies near the 5' end of
the assembled StringTie transcript.

For each StringTie transcript ``t`` and each same-strand complete ref CDS
``r`` that projects fully onto ``t``, compute the transcript-coordinate
position of the ref ATG (first base of the start codon in reading order).
If that position is <= ``--max-distance`` (default 3), the case is
considered "eligible". For each eligible case, compare the prediction's
ATG transcript-position against the ref ATG and categorise:

* ``correct``        : |pred_atg - ref_atg| <= ``--correct-tolerance``
* ``late_start``     : pred_atg - ref_atg >  tolerance  (truncated 5')
* ``early_start``    : pred_atg - ref_atg < -tolerance  (rare)
* ``no_prediction``  : no predicted ORF on this transcript

Inputs
------
* StringTie GTF (exon features)
* Reference GFF (CDS + stop_codon features)
* Prediction GTF (CDS features; transcript_id matches StringTie tx ids)

Outputs
-------
* Stdout summary (overall + breakdown by ref-ATG transcript-position)
* TSV: ref_tx, st_tx, strand, ref_atg_pos, pred_atg_pos, delta_bp, category
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tiberius_orf.data.label_transcripts import (  # noqa: E402
    parse_stringtie_gtf,
    parse_reference_cds,
    _genomic_to_transcript,
    _ranges_overlap,
)


def ref_atg_genomic_pos(r) -> int:
    """0-based genomic coordinate of the ref ATG's first base in reading
    order. + strand: leftmost base of the leftmost CDS exon. - strand:
    rightmost base of the rightmost CDS exon (i.e. ``end - 1``).
    """
    if r.strand == "+":
        return r.cds_parts[0][0]
    return r.cds_parts[-1][1] - 1


def _parse_attr(attr_col: str, key: str) -> str | None:
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(None, 1)
        if len(parts) == 2 and parts[0] == key:
            return parts[1].strip().strip('"')
    return None


def parse_predicted_cds_per_tx(path: Path) -> dict[str, list[tuple[int, int]]]:
    by_tid: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            tid = _parse_attr(f[8], "transcript_id")
            if tid is None:
                continue
            s = int(f[3]) - 1
            e = int(f[4])
            by_tid[tid].append((s, e))
    return {tid: sorted(intervals) for tid, intervals in by_tid.items()}


def predicted_atg_tx_pos(pred_exons: list[tuple[int, int]], st_tx) -> int | None:
    """Transcript position of the predicted ATG (first CDS base in reading
    order), or None if the prediction is empty or doesn't map.
    """
    if not pred_exons:
        return None
    if st_tx.strand == "+":
        g = pred_exons[0][0]
    else:
        g = pred_exons[-1][1] - 1
    return _genomic_to_transcript(st_tx, g)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stringtie-gtf", type=Path, required=True)
    ap.add_argument("--reference-gff", type=Path, required=True)
    ap.add_argument("--orfs-gtf", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--max-distance", type=int, default=3,
                    help="ref ATG transcript-position <= this counts as "
                         "'near the start'. Default 3.")
    ap.add_argument("--correct-tolerance", type=int, default=3,
                    help="|pred_atg - ref_atg| <= this counts as correct. "
                         "Default 3.")
    args = ap.parse_args(argv)

    print("[quantify_start_truncation] parsing inputs ...", file=sys.stderr)
    st_txs = parse_stringtie_gtf(args.stringtie_gtf)
    refs = parse_reference_cds(args.reference_gff)
    pred_by_tx = parse_predicted_cds_per_tx(args.orfs_gtf)
    print(f"[quantify_start_truncation] st_tx={len(st_txs)} "
          f"refs={len(refs)} pred_tx={len(pred_by_tx)}", file=sys.stderr)

    refs_by_contig: dict[str, list] = defaultdict(list)
    for r in refs.values():
        refs_by_contig[r.contig].append(r)

    rows = []
    counters: Counter = Counter()
    by_ref_pos: dict[int, Counter] = defaultdict(Counter)

    for tid, t in st_txs.items():
        candidates = [
            r for r in refs_by_contig.get(t.contig, [])
            if r.strand == t.strand
            and r.complete
            and _ranges_overlap(r.g_span, t.g_span)
        ]
        for r in candidates:
            # Fast path: only project the single ATG base, not the whole
            # CDS. Eligibility is "ref ATG maps to this tx within
            # max-distance bp of its 5' end". If the ATG falls in an
            # intron of the tx (or off the tx entirely) we skip; the
            # rest of the CDS need not fit (it's a separate failure
            # mode, not what this question is about).
            ref_atg_g = ref_atg_genomic_pos(r)
            ref_atg_t = _genomic_to_transcript(t, ref_atg_g)
            if ref_atg_t is None or ref_atg_t > args.max_distance:
                continue

            pred_exons = pred_by_tx.get(tid)
            pred_atg = (predicted_atg_tx_pos(pred_exons, t)
                        if pred_exons else None)
            if pred_atg is None:
                category = "no_prediction"
                delta = None
            else:
                delta = pred_atg - ref_atg_t
                if abs(delta) <= args.correct_tolerance:
                    category = "correct"
                elif delta > 0:
                    category = "late_start"
                else:
                    category = "early_start"

            counters[category] += 1
            by_ref_pos[ref_atg_t][category] += 1
            rows.append((
                r.tid, tid, t.strand, ref_atg_t,
                pred_atg if pred_atg is not None else "",
                delta if delta is not None else "",
                category,
            ))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8") as fh:
        fh.write("ref_tx\tst_tx\tstrand\tref_atg_pos\tpred_atg_pos\t"
                 "delta_bp\tcategory\n")
        for row in rows:
            fh.write("\t".join(str(c) for c in row) + "\n")

    n_total = sum(counters.values())
    print(f"\n[quantify_start_truncation] eligible cases "
          f"(ref ATG within {args.max_distance} bp of tx start): {n_total}")
    for cat in ("correct", "late_start", "early_start", "no_prediction"):
        c = counters[cat]
        pct = 100.0 * c / max(1, n_total)
        print(f"  {cat:>14s} : {c:6d}  ({pct:5.1f}%)")

    print("\n[quantify_start_truncation] breakdown by ref ATG tx position:")
    print(f"  {'ref_pos':>7s}  {'total':>6s}  {'correct':>7s}  "
          f"{'late':>5s}  {'early':>5s}  {'no_pred':>7s}")
    for pos in sorted(by_ref_pos):
        d = by_ref_pos[pos]
        tot = sum(d.values())
        print(f"  {pos:7d}  {tot:6d}  {d['correct']:7d}  "
              f"{d['late_start']:5d}  {d['early_start']:5d}  "
              f"{d['no_prediction']:7d}")

    print(f"\n[quantify_start_truncation] detail tsv -> {args.out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
