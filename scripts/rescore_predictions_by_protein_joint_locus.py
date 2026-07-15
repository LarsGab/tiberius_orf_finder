"""Pre-merge protein-evidence rescoring across Tiberius and ORF-finder
predictions.

Rationale: at the merge step, Tiberius and ORF-finder isoforms can end
up at the same locus. This script uses Diamond bitscores as a
tie-breaker: within each joint (contig, strand, span-overlap) locus,
keep only transcripts whose best hit bitscore is within ``tol × max``
of the locus's top bitscore. If no transcript in the locus has any
hit above the e-value threshold, keep every isoform (never remove a
locus for lack of evidence — this is a tie-breaker, not a hard filter).

Inputs:
  ``--tib-gtf``          protein-filtered Tiberius GTF
  ``--orf-gtf``          ORF-finder GTF (post-subseq collapse)
  ``--tib-diamond-tsv``  Diamond TSV for Tiberius transcripts
  ``--orf-diamond-tsv``  Diamond TSV for ORF-finder transcripts

Outputs:
  ``--out-tib-gtf``      filtered Tiberius GTF (survivors only)
  ``--out-orf-gtf``      filtered ORF-finder GTF (survivors only)
  ``--report-tsv``       dropped_tid \\t source \\t locus_id \\t
                         self_bitscore \\t locus_best_bitscore \\t reason

Both output GTFs are then merged as usual by
``merge_annotations.py --mode full``.

Usage::

    python scripts/rescore_predictions_by_protein_joint_locus.py \\
        --tib-gtf          .../diamond_filter/<sp>/tiberius/filtered.gtf \\
        --orf-gtf          .../annotate_.../orfs.filtered.gtf \\
        --tib-diamond-tsv  .../diamond_filter/<sp>/tiberius/diamond.tsv \\
        --orf-diamond-tsv  .../annotate_.../soft_protein/diamond.tsv \\
        --out-tib-gtf      .../rescored/<sp>/tib.rescored.gtf \\
        --out-orf-gtf      .../rescored/<sp>/orf.rescored.gtf \\
        --report-tsv       .../rescored/<sp>/rescored.dropped.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


_TRANSCRIPT_ID_RE = re.compile(r'transcript_id "([^"]+)"')


def parse_transcripts_cds(gtf: Path) -> dict[str, dict]:
    """Return ``{tid: {contig, strand, cds_spans: [(s,e),...], span: (s,e)}}``
    for every transcript with at least one CDS line."""
    by_tid: dict[str, dict] = {}
    with gtf.open() as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            m = _TRANSCRIPT_ID_RE.search(f[8])
            if not m:
                continue
            tid = m.group(1)
            rec = by_tid.setdefault(tid, {
                "contig": f[0], "strand": f[6] or ".",
                "cds_spans": [],
            })
            rec["cds_spans"].append((int(f[3]) - 1, int(f[4])))
    for rec in by_tid.values():
        rec["cds_spans"].sort()
        if rec["cds_spans"]:
            rec["span"] = (rec["cds_spans"][0][0], rec["cds_spans"][-1][1])
    return by_tid


def load_best_bitscore(diamond_tsv: Path, evalue_max: float) -> dict[str, float]:
    """Return {qseqid: max_bitscore} across hits passing evalue."""
    best: dict[str, float] = {}
    with diamond_tsv.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            try:
                q = c[0]
                ev = float(c[10])
                bs = float(c[11])
            except (ValueError, IndexError):
                continue
            if ev > evalue_max:
                continue
            if bs > best.get(q, -1.0):
                best[q] = bs
    return best


def cluster_by_span(spans_per_cs: dict[tuple[str, str], list[tuple[int, int, str]]]
                    ) -> dict[str, int]:
    """Sweep-based clustering by span overlap on each (contig, strand).
    Returns ``{tid: locus_id}``."""
    locus_id: dict[str, int] = {}
    next_id = 0
    for cs, spans in spans_per_cs.items():
        spans_sorted = sorted(spans)  # (start, end, tid)
        cur_end = -1
        cur_id = -1
        for s, e, tid in spans_sorted:
            if s > cur_end:
                cur_id = next_id
                next_id += 1
                cur_end = e
            else:
                cur_end = max(cur_end, e)
            locus_id[tid] = cur_id
    return locus_id


def rescore(
    tib_recs: dict[str, dict],
    orf_recs: dict[str, dict],
    tib_bs: dict[str, float],
    orf_bs: dict[str, float],
    tol: float,
) -> tuple[set[str], set[str], list[dict]]:
    """Return (kept_tib_tids, kept_orf_tids, report_rows).

    Rule: in each joint (contig, strand, span-overlap) locus, if
    ``max_bs > 0`` keep transcripts with ``bs >= tol * max_bs``,
    else keep every transcript.
    """
    spans_per_cs: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    src_of: dict[str, str] = {}
    for tid, rec in tib_recs.items():
        if "span" not in rec:
            continue
        spans_per_cs[(rec["contig"], rec["strand"])].append(
            (rec["span"][0], rec["span"][1], tid)
        )
        src_of[tid] = "tiberius"
    for tid, rec in orf_recs.items():
        if "span" not in rec:
            continue
        spans_per_cs[(rec["contig"], rec["strand"])].append(
            (rec["span"][0], rec["span"][1], tid)
        )
        src_of[tid] = "orf"

    locus_of = cluster_by_span(spans_per_cs)
    per_locus: dict[int, list[str]] = defaultdict(list)
    for tid, lid in locus_of.items():
        per_locus[lid].append(tid)

    def score(tid: str) -> float:
        return (tib_bs.get(tid, 0.0) if src_of[tid] == "tiberius"
                else orf_bs.get(tid, 0.0))

    kept_tib: set[str] = set()
    kept_orf: set[str] = set()
    report: list[dict] = []
    for lid, tids in per_locus.items():
        scores = {t: score(t) for t in tids}
        max_bs = max(scores.values()) if scores else 0.0
        threshold = tol * max_bs if max_bs > 0 else 0.0
        for t in tids:
            keep = (max_bs == 0.0) or (scores[t] >= threshold)
            if keep:
                (kept_tib if src_of[t] == "tiberius" else kept_orf).add(t)
            else:
                report.append({
                    "tid": t, "source": src_of[t], "locus_id": lid,
                    "self_bitscore": scores[t],
                    "locus_best_bitscore": max_bs,
                    "reason": f"below_{tol:g}x_locus_top",
                })
    return kept_tib, kept_orf, report


def write_filtered_gtf(src: Path, dst: Path, kept: set[str]) -> tuple[int, int]:
    n_kept = n_total = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            n_total += 1
            m = _TRANSCRIPT_ID_RE.search(line)
            if m and m.group(1) in kept:
                fout.write(line)
                n_kept += 1
            elif not m:
                # Gene lines: write only if any of its transcripts survives.
                # Simpler: always write; downstream merge tolerates orphan
                # gene rows and rebuilds gene structure.
                fout.write(line)
                n_kept += 1
    return n_kept, n_total


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--tib-gtf",         type=Path, required=True)
    ap.add_argument("--orf-gtf",         type=Path, required=True)
    ap.add_argument("--tib-diamond-tsv", type=Path, required=True)
    ap.add_argument("--orf-diamond-tsv", type=Path, required=True)
    ap.add_argument("--out-tib-gtf",     type=Path, required=True)
    ap.add_argument("--out-orf-gtf",     type=Path, required=True)
    ap.add_argument("--report-tsv",      type=Path, required=True)
    ap.add_argument("--tol", type=float, default=0.7,
                    help="Keep transcripts with bitscore >= tol * "
                         "locus_best_bitscore (default 0.7).")
    ap.add_argument("--evalue-max", type=float, default=1e-5)
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    tib_recs = parse_transcripts_cds(args.tib_gtf)
    orf_recs = parse_transcripts_cds(args.orf_gtf)
    tib_bs   = load_best_bitscore(args.tib_diamond_tsv, args.evalue_max)
    orf_bs   = load_best_bitscore(args.orf_diamond_tsv, args.evalue_max)

    kept_tib, kept_orf, report = rescore(
        tib_recs, orf_recs, tib_bs, orf_bs, tol=args.tol,
    )

    n_tib_kept, n_tib_total = write_filtered_gtf(
        args.tib_gtf, args.out_tib_gtf, kept_tib,
    )
    n_orf_kept, n_orf_total = write_filtered_gtf(
        args.orf_gtf, args.out_orf_gtf, kept_orf,
    )

    args.report_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_tsv.open("w") as fh:
        fh.write("dropped_tid\tsource\tlocus_id\tself_bitscore\t"
                 "locus_best_bitscore\treason\n")
        for r in sorted(report, key=lambda x: (x["source"], x["tid"])):
            fh.write(f"{r['tid']}\t{r['source']}\t{r['locus_id']}\t"
                     f"{r['self_bitscore']:.2f}\t"
                     f"{r['locus_best_bitscore']:.2f}\t{r['reason']}\n")

    n_tib_drop = sum(1 for r in report if r["source"] == "tiberius")
    n_orf_drop = sum(1 for r in report if r["source"] == "orf")
    print(f"[rescore] tol={args.tol}  evalue_max={args.evalue_max}", flush=True)
    print(f"[rescore] tib tx in  : {len(tib_recs)}", flush=True)
    print(f"[rescore] orf tx in  : {len(orf_recs)}", flush=True)
    print(f"[rescore] tib w/ hit : {sum(1 for t in tib_recs if t in tib_bs)}", flush=True)
    print(f"[rescore] orf w/ hit : {sum(1 for t in orf_recs if t in orf_bs)}", flush=True)
    print(f"[rescore] tib dropped: {n_tib_drop} "
          f"({100.0*n_tib_drop/max(1,len(tib_recs)):.1f}%)", flush=True)
    print(f"[rescore] orf dropped: {n_orf_drop} "
          f"({100.0*n_orf_drop/max(1,len(orf_recs)):.1f}%)", flush=True)
    print(f"[rescore] tib gtf out: {args.out_tib_gtf} "
          f"({n_tib_kept}/{n_tib_total} lines)", flush=True)
    print(f"[rescore] orf gtf out: {args.out_orf_gtf} "
          f"({n_orf_kept}/{n_orf_total} lines)", flush=True)
    print(f"[rescore] report tsv : {args.report_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
