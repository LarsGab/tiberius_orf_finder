"""Compute the upper-bound Sensitivity achievable by any ORF-finder
whose input is a given StringTie assembly, at the four gffcompare
levels used by the ORF-tool benchmark (base/exon/transcript/locus).

Rationale: every ORF-finder we benchmark predicts CDS *within* the
StringTie transcripts. So no ORF tool can recover a reference CDS
feature that isn't representable by the StringTie exon chain. This
script produces per-species upper bounds so they can be drawn as
horizontal reference lines on the accuracy plots.

Definitions of "recoverable" per level:
  base       reference CDS nucleotide is covered by any StringTie exon
             on the same strand.
  exon       reference CDS exon's (start,end) appears exactly in some
             StringTie exon on the same contig+strand.
  transcript reference CDS chain is a strict sub-sequence of some
             StringTie transcript's exon chain on the same contig+strand
             (reuses ``is_subsequence`` from
             ``filter_subsequence_predictions.py``).
  locus      the reference locus's span is overlapped by any StringTie
             transcript on the same strand.

Denominators come from the reference (annot_cds.gff): unique CDS
nucleotides, unique CDS exons, unique reference RNAs (transcripts),
unique reference genes (loci).

Output: one TSV row per species with columns
  species base_S exon_S transcript_S locus_S
No precision numbers — this is a sensitivity ceiling, not a full run.

Usage::

    python scripts/compute_stringtie_upper_bound.py \\
        --species Bos_taurus Gallus_gallus ... \\
        --ref-tmpl        '.../vertebrates_test/{sp}/assembly/annot_cds.gff' \\
        --stringtie-tmpl  '.../vertebrates_test/{sp}/stringtie/stringtie.filt_tpm1cov3len300.gtf' \\
        --out-tsv         '.../benchmark_orf_tools_filt_tpm1cov3len300/stringtie_upper_bound.tsv'
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_SUBSEQ_PATH = _HERE / "filter_subsequence_predictions.py"


def _load_subseq():
    spec = importlib.util.spec_from_file_location(
        "_filter_subseq_ub", _SUBSEQ_PATH,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_filter_subseq_ub"] = mod
    spec.loader.exec_module(mod)
    return mod


is_subsequence = _load_subseq().is_subsequence


_GTF_KEY_RE = re.compile(r'(\w+)\s+"([^"]+)"')
_GFF3_ATTR_RE = re.compile(r'(\w+)=([^;]+)')


def _parse_attrs(col: str) -> dict[str, str]:
    """Handle both GTF (key "val";) and GFF3 (key=val;) attribute cols."""
    attrs: dict[str, str] = {}
    for m in _GTF_KEY_RE.finditer(col):
        attrs.setdefault(m.group(1), m.group(2))
    for m in _GFF3_ATTR_RE.finditer(col):
        attrs.setdefault(m.group(1), m.group(2))
    return attrs


def parse_features(
    path: Path,
    feature: str,
    tid_attr_priority: tuple[str, ...] = ("transcript_id", "Parent", "ID"),
    gid_attr_priority: tuple[str, ...] = ("gene_id", "gene", "GeneID"),
) -> tuple[dict[str, dict], set[tuple[str, str]]]:
    """Return ``(by_tid, all_contig_strand)`` for lines whose col-3 feature
    equals ``feature``. Each entry:
      ``by_tid[tid] = {contig, strand, gid, exons: [(s,e), ...]}``
    ``exons`` sorted ascending, 0-based half-open.
    """
    by_tid: dict[str, dict] = {}
    seen_cs: set[tuple[str, str]] = set()
    with path.open() as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != feature:
                continue
            attrs = _parse_attrs(f[8])
            tid = None
            for k in tid_attr_priority:
                if k in attrs:
                    tid = attrs[k]
                    break
            if tid is None:
                continue
            gid = None
            for k in gid_attr_priority:
                if k in attrs:
                    gid = attrs[k]
                    break
            gid = gid or tid.rsplit(".", 1)[0]
            contig = f[0]
            strand = f[6] or "."
            start = int(f[3]) - 1
            end = int(f[4])
            rec = by_tid.setdefault(tid, {
                "contig": contig, "strand": strand,
                "gid": gid, "exons": [],
            })
            rec["exons"].append((start, end))
            seen_cs.add((contig, strand))
    for rec in by_tid.values():
        rec["exons"].sort()
    return by_tid, seen_cs


def coverage_intervals(intervals: list[tuple[int, int]]) -> int:
    """Union length of half-open intervals (unsorted OK)."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def per_strand_bp_cov_index(
    stringtie_by_tid: dict[str, dict],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Merged per-(contig,strand) exon intervals for fast bp lookup."""
    buckets: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for rec in stringtie_by_tid.values():
        cs = (rec["contig"], rec["strand"])
        buckets[cs].extend(rec["exons"])
    # Merge each bucket into a sorted disjoint interval list.
    out: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for cs, ivs in buckets.items():
        if not ivs:
            out[cs] = []
            continue
        ivs.sort()
        merged: list[tuple[int, int]] = [ivs[0]]
        for s, e in ivs[1:]:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        out[cs] = merged
    return out


def bp_covered(query_intervals: list[tuple[int, int]],
               merged_bucket: list[tuple[int, int]]) -> int:
    """Total nt of ``query_intervals`` covered by ``merged_bucket``.
    ``merged_bucket`` must be sorted and disjoint."""
    from bisect import bisect_right
    if not merged_bucket:
        return 0
    starts = [iv[0] for iv in merged_bucket]
    total = 0
    for qs, qe in query_intervals:
        i = bisect_right(starts, qe) - 1
        # walk backwards from i while overlap possible
        while i >= 0:
            bs, be = merged_bucket[i]
            if be <= qs:
                break
            ov = min(qe, be) - max(qs, bs)
            if ov > 0:
                total += ov
            i -= 1
    return total


def per_strand_exon_set(
    stringtie_by_tid: dict[str, dict],
) -> dict[tuple[str, str], set[tuple[int, int]]]:
    out: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
    for rec in stringtie_by_tid.values():
        cs = (rec["contig"], rec["strand"])
        for iv in rec["exons"]:
            out[cs].add(iv)
    return out


def per_strand_span_index(
    stringtie_by_tid: dict[str, dict],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Merged transcript-span intervals per (contig, strand). Used for
    the locus-overlap check."""
    buckets: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for rec in stringtie_by_tid.values():
        if not rec["exons"]:
            continue
        span = (rec["exons"][0][0], rec["exons"][-1][1])
        buckets[(rec["contig"], rec["strand"])].append(span)
    return {cs: _merge(ivs) for cs, ivs in buckets.items()}


def _merge(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ivs:
        return []
    ivs = sorted(ivs)
    out: list[tuple[int, int]] = [ivs[0]]
    for s, e in ivs[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def per_strand_tid_index(
    stringtie_by_tid: dict[str, dict],
) -> dict[tuple[str, str], list[tuple[int, int, list[tuple[int, int]]]]]:
    """Per-(contig,strand) list of (span_start, span_end, exons)
    sorted by span_start — used for the transcript-level subseq check."""
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for rec in stringtie_by_tid.values():
        exons = rec["exons"]
        if not exons:
            continue
        buckets[(rec["contig"], rec["strand"])].append(
            (exons[0][0], exons[-1][1], exons)
        )
    for lst in buckets.values():
        lst.sort()
    return buckets


def compute_species_upper_bound(
    ref_path: Path,
    stringtie_path: Path,
) -> dict[str, float]:
    ref_by_tid, _ = parse_features(ref_path, feature="CDS")
    st_by_tid, _  = parse_features(stringtie_path, feature="exon")

    if not ref_by_tid:
        return {"base_S": float("nan"), "exon_S": float("nan"),
                "transcript_S": float("nan"), "locus_S": float("nan")}

    # ── base ────────────────────────────────────────────────────────────────
    bp_index = per_strand_bp_cov_index(st_by_tid)
    total_ref_bp = 0
    covered_ref_bp = 0
    for rec in ref_by_tid.values():
        cs = (rec["contig"], rec["strand"])
        merged_ref_exons = _merge(rec["exons"])
        total_ref_bp += sum(e - s for s, e in merged_ref_exons)
        merged_bucket = bp_index.get(cs, [])
        covered_ref_bp += bp_covered(merged_ref_exons, merged_bucket)
    base_S = 100.0 * covered_ref_bp / max(1, total_ref_bp)

    # ── exon ────────────────────────────────────────────────────────────────
    st_exon_set = per_strand_exon_set(st_by_tid)
    total_ref_exons = 0
    matched_ref_exons = 0
    for rec in ref_by_tid.values():
        cs = (rec["contig"], rec["strand"])
        exset = st_exon_set.get(cs, set())
        for iv in rec["exons"]:
            total_ref_exons += 1
            if iv in exset:
                matched_ref_exons += 1
    exon_S = 100.0 * matched_ref_exons / max(1, total_ref_exons)

    # ── transcript ──────────────────────────────────────────────────────────
    # A ref transcript is "recoverable" iff its CDS chain is a strict
    # sub-sequence of some StringTie transcript's exon chain (same contig+
    # strand). Compute once and reuse for locus_S below.
    tid_index = per_strand_tid_index(st_by_tid)
    recoverable_tids: set[str] = set()
    for tid, rec in ref_by_tid.items():
        exons = rec["exons"]
        if not exons:
            continue
        cs = (rec["contig"], rec["strand"])
        candidates = tid_index.get(cs, [])
        a_s, a_e = exons[0][0], exons[-1][1]
        for b_span_s, b_span_e, b_exons in candidates:
            if b_span_s > a_e:
                break
            if b_span_e < a_s:
                continue
            if is_subsequence(exons, b_exons):
                recoverable_tids.add(tid)
                break
    n_tx = len(ref_by_tid)
    transcript_S = 100.0 * len(recoverable_tids) / max(1, n_tx)

    # ── locus ───────────────────────────────────────────────────────────────
    # Group ref transcripts into loci by (contig, strand, gid). A ref locus
    # is counted as recoverable iff at least ONE of its transcripts has a
    # full-CDS-chain subseq match against some StringTie transcript (i.e.,
    # is in ``recoverable_tids``). Span-only overlap is intentionally not
    # enough: an ORF finder can only reproduce a CDS chain that the
    # StringTie transcript's exon chain actually contains.
    locus_map: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for tid, rec in ref_by_tid.items():
        if not rec["exons"]:
            continue
        key = (rec["contig"], rec["strand"], rec["gid"])
        locus_map[key].append(tid)
    n_loci = len(locus_map)
    hit_loci = sum(1 for tids in locus_map.values()
                   if any(t in recoverable_tids for t in tids))
    locus_S = 100.0 * hit_loci / max(1, n_loci)

    return {"base_S": base_S, "exon_S": exon_S,
            "transcript_S": transcript_S, "locus_S": locus_S}


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--species", nargs="+", required=True)
    ap.add_argument("--ref-tmpl", type=str, required=True,
                    help="Path template with '{sp}' for annot_cds.gff.")
    ap.add_argument("--stringtie-tmpl", type=str, required=True,
                    help="Path template with '{sp}' for stringtie GTF.")
    ap.add_argument("--out-tsv", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with args.out_tsv.open("w") as fh:
        fh.write("species\tbase_S\texon_S\ttranscript_S\tlocus_S\n")
        for sp in args.species:
            ref = Path(args.ref_tmpl.format(sp=sp))
            st  = Path(args.stringtie_tmpl.format(sp=sp))
            if not ref.exists():
                print(f"[skip] {sp}: missing ref {ref}", flush=True)
                fh.write(f"{sp}\tNA\tNA\tNA\tNA\n")
                continue
            if not st.exists():
                print(f"[skip] {sp}: missing stringtie {st}", flush=True)
                fh.write(f"{sp}\tNA\tNA\tNA\tNA\n")
                continue
            print(f"[compute] {sp}", flush=True)
            m = compute_species_upper_bound(ref, st)
            fh.write(f"{sp}\t{m['base_S']:.2f}\t{m['exon_S']:.2f}\t"
                     f"{m['transcript_S']:.2f}\t{m['locus_S']:.2f}\n")
            print(f"  base_S={m['base_S']:.2f}  exon_S={m['exon_S']:.2f}  "
                  f"transcript_S={m['transcript_S']:.2f}  locus_S={m['locus_S']:.2f}",
                  flush=True)

    print(f"\nwrote {args.out_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
