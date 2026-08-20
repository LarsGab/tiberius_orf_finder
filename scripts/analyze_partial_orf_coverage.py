#!/usr/bin/env python3
"""Classify partial ORFs by what protein evidence (miniprot) is available to
recover their missing stop codon.

Categories per partial ORF
--------------------------
NO_COVERAGE   : no miniprot alignment overlaps the ORF span on the same strand
SHORT         : overlapping alignment(s) exist but none extends past the ORF's
                3'-truncation point (last CDS end on + strand, first CDS start
                on - strand)
DIRECT        : best alignment extends past the truncation point AND the
                last (or first on - strand) CDS block of that alignment
                spans the truncation → the extension stays within the same
                exon (no intron between ORF end and protein-supported stop)
INTRON        : best alignment extends past the truncation point BUT the
                last CDS block starts (+ strand) or ends (- strand) AFTER
                the truncation → at least one intron separates the ORF end
                from the miniprot-supported stop codon

Usage
-----
python scripts/analyze_partial_orf_coverage.py \\
    --partial  results/<sp>/annotate_<tag>/orfs.partial.gtf \\
    --miniprot results/<sp>/fix_stop/miniprot.gff \\
    --species  Gallus_gallus

Or for a quick multi-species table via the helper at the bottom.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_attr(col: str, key: str) -> str | None:
    m = re.search(rf'{key}\s+"([^"]+)"', col)
    if m:
        return m.group(1)
    m = re.search(rf'{key}=([^;"\s]+)', col)
    if m:
        return m.group(1)
    return None


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_partial_gtf(path: Path) -> list[dict]:
    """Return list of partial ORF records: {tid, contig, strand, trunc_pos}.

    trunc_pos is the genomic position of the 3'-truncation:
      + strand → max CDS end   (0-based, half-open)
      - strand → min CDS start (0-based)
    """
    segs: dict[str, dict] = {}
    for row in csv.reader(open(path), delimiter="\t"):
        if not row or row[0].startswith("#") or len(row) < 9:
            continue
        if row[2] != "CDS":
            continue
        tid = _parse_attr(row[8], "transcript_id")
        if tid is None:
            continue
        s, e = int(row[3]) - 1, int(row[4])  # 0-based half-open
        strand = row[6]
        contig = row[0]
        if tid not in segs:
            segs[tid] = {"tid": tid, "contig": contig, "strand": strand,
                         "segs": []}
        segs[tid]["segs"].append((s, e))

    orfs = []
    for tid, d in segs.items():
        all_s = d["segs"]
        all_s.sort()
        d.pop("segs")
        if d["strand"] == "+":
            d["trunc_pos"] = max(e for _, e in all_s)
            d["span_start"] = min(s for s, _ in all_s)
        else:
            d["trunc_pos"] = min(s for s, _ in all_s)
            d["span_end"] = max(e for _, e in all_s)
        orfs.append(d)
    return orfs


def parse_miniprot_gff(path: Path) -> list[dict]:
    """Return list of miniprot alignments: {mid, contig, strand, cds_segs}.

    cds_segs is sorted ASC list of (start, end) 0-based half-open.
    """
    pending: dict[str, dict] = {}
    alns: list[dict] = []
    for row in csv.reader(open(path), delimiter="\t"):
        if not row or row[0].startswith("#") or len(row) < 9:
            continue
        feat = row[2]
        if feat == "mRNA":
            mid = _parse_attr(row[8], "ID")
            if mid is None:
                continue
            pending[mid] = {"mid": mid, "contig": row[0], "strand": row[6],
                            "cds_segs": []}
        elif feat == "CDS":
            parent = _parse_attr(row[8], "Parent")
            if parent not in pending:
                continue
            s, e = int(row[3]) - 1, int(row[4])
            pending[parent]["cds_segs"].append((s, e))
        elif feat == "stop_codon":
            # not used in classification, but available
            pass

    for aln in pending.values():
        aln["cds_segs"].sort()
        alns.append(aln)
    return alns


# ── classification ────────────────────────────────────────────────────────────

def classify(orf: dict, aln_index: dict[str, list[dict]]) -> str:
    """Classify one partial ORF.

    aln_index: contig → list of miniprot alignments on that contig.
    """
    contig = orf["contig"]
    strand = orf["strand"]
    trunc  = orf["trunc_pos"]

    candidates = aln_index.get(contig, [])

    # filter to same strand and overlapping the ORF span
    if strand == "+":
        orf_start = orf["span_start"]
        orf_end   = trunc  # exclusive
        overlapping = [
            a for a in candidates
            if a["strand"] == "+" and a["cds_segs"]
            and a["cds_segs"][-1][1] > orf_start
            and a["cds_segs"][0][0]  < orf_end
        ]
    else:
        orf_end  = orf["span_end"]
        orf_start = trunc  # inclusive start (0-based)
        overlapping = [
            a for a in candidates
            if a["strand"] == "-" and a["cds_segs"]
            and a["cds_segs"][0][0]  < orf_end
            and a["cds_segs"][-1][1] > orf_start
        ]

    if not overlapping:
        return "NO_COVERAGE"

    # find alignment that reaches furthest past the truncation
    if strand == "+":
        best = max(overlapping, key=lambda a: a["cds_segs"][-1][1])
        mp_last_end = best["cds_segs"][-1][1]
        if mp_last_end <= trunc:
            return "SHORT"
        # Does the last CDS block span the truncation (i.e., start before trunc)?
        mp_last_start = best["cds_segs"][-1][0]
        if mp_last_start < trunc:
            return "DIRECT"
        else:
            return "INTRON"
    else:
        best = min(overlapping, key=lambda a: a["cds_segs"][0][0])
        mp_first_start = best["cds_segs"][0][0]
        if mp_first_start >= trunc:
            return "SHORT"
        mp_first_end = best["cds_segs"][0][1]
        if mp_first_end > trunc:
            return "DIRECT"
        else:
            return "INTRON"


# ── main ──────────────────────────────────────────────────────────────────────

def analyze_species(partial_gtf: Path, miniprot_gff: Path,
                    species: str) -> dict[str, int]:
    orfs = parse_partial_gtf(partial_gtf)
    alns = parse_miniprot_gff(miniprot_gff)

    # build contig index
    aln_index: dict[str, list[dict]] = defaultdict(list)
    for a in alns:
        aln_index[a["contig"]].append(a)

    counts: dict[str, int] = defaultdict(int)
    for orf in orfs:
        cat = classify(orf, aln_index)
        counts[cat] += 1

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"Species: {species}   total partial ORFs: {total}")
    print(f"{'='*60}")
    cats = ["NO_COVERAGE", "SHORT", "DIRECT", "INTRON"]
    for c in cats:
        n = counts[c]
        pct = 100 * n / total if total else 0
        print(f"  {c:<14} {n:5d}  ({pct:5.1f}%)")
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial",  required=True, type=Path)
    ap.add_argument("--miniprot", required=True, type=Path)
    ap.add_argument("--species",  default="unknown")
    args = ap.parse_args()
    analyze_species(args.partial, args.miniprot, args.species)


if __name__ == "__main__":
    main()
