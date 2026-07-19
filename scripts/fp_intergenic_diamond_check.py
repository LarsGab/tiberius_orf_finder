"""For each intergenic-FP locus (gffcompare code ``u``), check whether any
underlying ORF-finder or Tiberius transcript at the same span has a
Diamond hit against OrthoDB.

A hit against ODB (already run for the soft-protein filter and for the
Tiberius protein-filter) is a strong signal that the "intergenic FP" is
actually a real conserved gene that the reference annotation is missing.

Bucketing per locus by best Diamond bitscore (over both sources):
    no_hit  weak_<50  mod_50-99  strong_100-199  very_strong_>=200

Usage::

    python scripts/fp_intergenic_diamond_check.py \\
        --eval-dir results/vertebrates_test/eval_accuracy_epoch_74_filt_pfilt \\
        --orf-tmpl 'results/vertebrates_test/{sp}/annotate_epoch_74_filt_tpm1cov3len300/orfs.filtered.gtf' \\
        --orf-diam-tmpl 'results/vertebrates_test/{sp}/annotate_epoch_74_filt_tpm1cov3len300/soft_protein/diamond.tsv' \\
        --tib-tmpl '/home/gabriell/tiberius_proteins_analysis/diamond_filter/{sp}/tiberius/filtered.gtf' \\
        --tib-diam-tmpl '/home/gabriell/tiberius_proteins_analysis/diamond_filter/{sp}/tiberius/diamond.tsv' \\
        --species Gallus_gallus Bos_taurus ... \\
        --out-tsv results/vertebrates_test/fp_intergenic_diamond_pfilt.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


_TID_RE = re.compile(r'transcript_id "([^"]+)"')
# Matches parts[1] of a merged.loci row, e.g. "NC_037328.1[+]479368-607442".
_LOCI_LINE_RE = re.compile(r"^(\S+)\[([+-.])\](\d+)-(\d+)$")

# Same set used by fp_locus_diagnostic.py (~ = v0.12 partial intron match).
TP_CODES = {"=", "c", "k", "m", "n", "j", "e", "~"}


def _parse_tracking_intergenic(tracking: Path) -> set[str]:
    """Return XLOC IDs classified as intergenic FP: no TP class code among
    the locus's transcripts AND at least one transcript coded as 'u'.
    Matches fp_locus_diagnostic.py precedence bucketing (u wins if no TP)."""
    codes_per_xloc: dict[str, set[str]] = defaultdict(set)
    with tracking.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            codes_per_xloc[f[1]].add(f[3])
    return {x for x, cs in codes_per_xloc.items()
            if "u" in cs and not (cs & TP_CODES)}


def _parse_loci_spans(loci: Path, xloc_ids: set[str]) -> list[tuple[str, str, str, int, int]]:
    """Return list of (xloc, contig, strand, start, end) for the given XLOCs."""
    out = []
    with loci.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            xloc = parts[0]
            if xloc not in xloc_ids:
                continue
            m = _LOCI_LINE_RE.match(parts[1])
            if not m:
                continue
            contig, strand, s, e = m.groups()
            out.append((xloc, contig, strand, int(s), int(e)))
    return out


def _parse_tx_spans(gtf: Path) -> dict[tuple[str, str], list[tuple[int, int, str]]]:
    """Return {(contig, strand): sorted [(span_start, span_end, tid)]}."""
    per_tx: dict[str, dict] = {}
    with gtf.open() as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in ("exon", "CDS", "transcript"):
                continue
            m = _TID_RE.search(f[8])
            if not m:
                continue
            tid = m.group(1)
            s, e = int(f[3]), int(f[4])
            rec = per_tx.setdefault(tid, {"contig": f[0], "strand": f[6] or ".",
                                           "s": s, "e": e})
            rec["s"] = min(rec["s"], s)
            rec["e"] = max(rec["e"], e)
    buckets: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for tid, rec in per_tx.items():
        buckets[(rec["contig"], rec["strand"])].append((rec["s"], rec["e"], tid))
    for lst in buckets.values():
        lst.sort()
    return buckets


def _best_bitscore_per_tid(diamond_tsv: Path, evalue_max: float = 1e-5
                            ) -> dict[str, float]:
    best: dict[str, float] = {}
    with diamond_tsv.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            try:
                ev = float(c[10])
                bs = float(c[11])
            except ValueError:
                continue
            if ev > evalue_max:
                continue
            q = c[0]
            if bs > best.get(q, -1.0):
                best[q] = bs
    return best


def _tids_overlapping(spans: list[tuple[int, int, str]],
                      lo: int, hi: int) -> list[str]:
    hits = []
    for s, e, tid in spans:
        if s > hi:
            break
        if e < lo:
            continue
        hits.append(tid)
    return hits


def _bucket_bs(bs: float) -> str:
    if bs <= 0.0:
        return "no_hit"
    if bs < 50.0:
        return "weak_lt50"
    if bs < 100.0:
        return "mod_50_99"
    if bs < 200.0:
        return "strong_100_199"
    return "very_strong_ge200"


_BUCKETS = ["no_hit", "weak_lt50", "mod_50_99",
            "strong_100_199", "very_strong_ge200"]


def _species_analyse(sp: str, eval_dir: Path, orf_gtf: Path,
                     tib_gtf: Path, orf_diam: Path, tib_diam: Path) -> dict:
    tracking = eval_dir / "gffcompare_runs" / sp / "merged.tracking"
    loci     = eval_dir / "gffcompare_runs" / sp / "merged.loci"

    intergenic_xlocs = _parse_tracking_intergenic(tracking)
    spans = _parse_loci_spans(loci, intergenic_xlocs)
    if not spans:
        return {"species": sp, "n_intergenic_fp": 0,
                **{b: 0 for b in _BUCKETS}}

    orf_spans_by_cs = _parse_tx_spans(orf_gtf) if orf_gtf.exists() else {}
    tib_spans_by_cs = _parse_tx_spans(tib_gtf) if tib_gtf.exists() else {}
    orf_bs = _best_bitscore_per_tid(orf_diam) if orf_diam.exists() else {}
    tib_bs = _best_bitscore_per_tid(tib_diam) if tib_diam.exists() else {}

    bucket_counts = defaultdict(int)
    for _, contig, strand, lo, hi in spans:
        best = 0.0
        # StringTie's default strand may be "." when unstranded; try both.
        for st in (strand, "+", "-", "."):
            for tid in _tids_overlapping(
                orf_spans_by_cs.get((contig, st), []), lo, hi
            ):
                if tid in orf_bs and orf_bs[tid] > best:
                    best = orf_bs[tid]
            for tid in _tids_overlapping(
                tib_spans_by_cs.get((contig, st), []), lo, hi
            ):
                if tid in tib_bs and tib_bs[tid] > best:
                    best = tib_bs[tid]
            if st == strand:
                continue  # already looked in the exact-strand bucket
        bucket_counts[_bucket_bs(best)] += 1

    return {"species": sp, "n_intergenic_fp": len(spans),
            **{b: bucket_counts.get(b, 0) for b in _BUCKETS}}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--eval-dir",      type=Path, required=True)
    ap.add_argument("--species",       nargs="+", required=True)
    ap.add_argument("--orf-tmpl",      type=str, required=True)
    ap.add_argument("--orf-diam-tmpl", type=str, required=True)
    ap.add_argument("--tib-tmpl",      type=str, required=True)
    ap.add_argument("--tib-diam-tmpl", type=str, required=True)
    ap.add_argument("--out-tsv",       type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rows = []
    for sp in args.species:
        rows.append(_species_analyse(
            sp, args.eval_dir,
            orf_gtf=Path(args.orf_tmpl.format(sp=sp)),
            tib_gtf=Path(args.tib_tmpl.format(sp=sp)),
            orf_diam=Path(args.orf_diam_tmpl.format(sp=sp)),
            tib_diam=Path(args.tib_diam_tmpl.format(sp=sp)),
        ))
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[intergenic-diam] wrote {args.out_tsv}", flush=True)
    print()
    print(f"{'species':<26} {'#intergen_FP':>13}  "
          + "  ".join(f"{b:>16}" for b in _BUCKETS))
    for r in rows:
        n = r["n_intergenic_fp"]
        cells = "  ".join(
            f"{r[b]:>6d} ({100.0*r[b]/max(1,n):>4.1f}%)" for b in _BUCKETS
        )
        print(f"{r['species']:<26} {n:>13d}  {cells}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
