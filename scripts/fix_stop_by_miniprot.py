#!/usr/bin/env python3
"""Fix or recover stop codons in predicted ORFs using miniprot protein-to-genome alignments.

Two cases are handled:

  1. Partial ORFs (no stop codon predicted — transcript truncated at 3' end):
     Extend the CDS to a miniprot-supported stop codon on the genome.

  2. Complete ORFs with an early spurious stop codon:
     If a miniprot alignment extends past the predicted stop AND has a valid
     downstream stop codon verified in the genome FASTA, replace the spurious
     stop with the protein-supported position.

In both cases the stop codon nucleotide triplet is verified in the genome
FASTA before any corrected output is written (analogous to spaln_to_gff.py
in GeneMark-ETP's ProtHint pipeline).

Usage
-----
python scripts/fix_stop_by_miniprot.py \\
    --orfs     results/orfs.gtf \\
    --partial  results/orfs.partial.gtf \\
    --miniprot results/miniprot.gff \\
    --genome   data/genome.fa \\
    --out      results/orfs.fixed.gtf

Output
------
One GTF file containing:
  - Complete ORFs corrected to a protein-supported stop (source unchanged).
  - Partial ORFs recovered with a protein-supported stop (source unchanged).
  - Complete ORFs for which no correction was needed, passed through as-is.
  - Partial ORFs with no protein support are silently omitted.

A summary is printed to stderr.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from pyfaidx import Fasta

STOP_CODONS: frozenset[str] = frozenset({"TAA", "TAG", "TGA"})
_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _rev_comp(seq: str) -> str:
    return seq.translate(_RC)[::-1]


def _parse_attr(col: str, key: str) -> str | None:
    """Extract key from a GTF/GFF3 attribute column."""
    m = re.search(rf'{key}\s+"([^"]+)"', col)
    if m:
        return m.group(1)
    m = re.search(rf'{key}=([^;"\s]+)', col)
    if m:
        return m.group(1)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

class ORF:
    """Predicted CDS parsed from a GTF file."""
    __slots__ = ("tid", "contig", "strand", "segments", "is_partial", "source")

    def __init__(self, tid: str, contig: str, strand: str,
                 is_partial: bool, source: str) -> None:
        self.tid = tid
        self.contig = contig
        self.strand = strand
        self.segments: list[tuple[int, int]] = []  # 0-based half-open, sorted ASC
        self.is_partial = is_partial
        self.source = source

    @property
    def span_start(self) -> int:
        return self.segments[0][0]

    @property
    def span_end(self) -> int:
        return self.segments[-1][1]

    @property
    def coding_len(self) -> int:
        return sum(e - s for s, e in self.segments)


class MpAlignment:
    """Miniprot protein-to-genome CDS alignment with verified stop codon."""
    __slots__ = ("mid", "contig", "strand", "cds_segments", "identity")

    def __init__(self, mid: str, contig: str, strand: str, identity: float) -> None:
        self.mid = mid
        self.contig = contig
        self.strand = strand
        self.cds_segments: list[tuple[int, int]] = []  # 0-based half-open, sorted ASC
        self.identity = identity

    @property
    def span_start(self) -> int:
        return self.cds_segments[0][0]

    @property
    def span_end(self) -> int:
        return self.cds_segments[-1][1]

    def stop_pos(self) -> int:
        """0-based genomic start of the stop codon (lowest coordinate).

        For + strand: 3 nt immediately after the last CDS block.
        For - strand: 3 nt immediately before the first CDS block.
        """
        if self.strand == "+":
            return self.cds_segments[-1][1]
        else:
            return self.cds_segments[0][0] - 3


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_orf_gtf(path: Path, is_partial: bool) -> dict[str, ORF]:
    """Parse a CDS-only GTF produced by annotate.py."""
    orfs: dict[str, ORF] = {}
    for row in csv.reader(open(path), delimiter="\t"):
        if not row or row[0].startswith("#"):
            continue
        if len(row) < 9 or row[2] != "CDS":
            continue
        tid = _parse_attr(row[8], "transcript_id")
        if tid is None:
            continue
        contig, strand, source = row[0], row[6], row[1]
        start = int(row[3]) - 1   # GTF 1-based inclusive → 0-based
        end = int(row[4])          # GTF inclusive → exclusive
        if tid not in orfs:
            orfs[tid] = ORF(tid, contig, strand, is_partial, source)
        orfs[tid].segments.append((start, end))
    for orf in orfs.values():
        orf.segments.sort()
    return orfs


def parse_miniprot_gff(path: Path) -> list[MpAlignment]:
    """Parse a miniprot GFF3, keeping only alignments where StopCodon=1."""
    pending: dict[str, MpAlignment] = {}
    for row in csv.reader(open(path), delimiter="\t"):
        if not row or row[0].startswith("#"):
            continue
        if len(row) < 9:
            continue
        feat = row[2]
        if feat == "mRNA":
            if _parse_attr(row[8], "StopCodon") != "1":
                continue
            mid = _parse_attr(row[8], "ID")
            if mid is None:
                continue
            ident_s = _parse_attr(row[8], "Identity")
            identity = float(ident_s) if ident_s else 0.0
            pending[mid] = MpAlignment(mid, row[0], row[6], identity)
        elif feat == "CDS":
            parent = _parse_attr(row[8], "Parent")
            if parent not in pending:
                continue
            start = int(row[3]) - 1
            end = int(row[4])
            pending[parent].cds_segments.append((start, end))

    result = []
    for aln in pending.values():
        if aln.cds_segments:
            aln.cds_segments.sort()
            result.append(aln)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Overlap index and lookup
# ──────────────────────────────────────────────────────────────────────────────

MpIndex = dict[tuple[str, str], list[MpAlignment]]


def build_mp_index(alns: list[MpAlignment]) -> MpIndex:
    idx: MpIndex = defaultdict(list)
    for a in alns:
        idx[(a.contig, a.strand)].append(a)
    return dict(idx)


def find_overlapping(
    orf: ORF,
    mp_index: MpIndex,
    min_overlap_frac: float,
) -> list[MpAlignment]:
    """Return miniprot alignments that overlap orf by ≥ min_overlap_frac of its CDS."""
    candidates = mp_index.get((orf.contig, orf.strand), [])
    if not candidates:
        return []
    orf_cds_len = orf.coding_len
    result = []
    for mp in candidates:
        # Fast span check
        if mp.span_end <= orf.span_start or mp.span_start >= orf.span_end:
            continue
        # CDS-level overlap
        overlap = 0
        for os, oe in orf.segments:
            for ms, me in mp.cds_segments:
                lo = max(os, ms)
                hi = min(oe, me)
                if hi > lo:
                    overlap += hi - lo
        if overlap >= min_overlap_frac * orf_cds_len:
            result.append(mp)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Stop codon verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_stop(
    contig: str,
    pos: int,       # 0-based genomic start of the 3-nt stop codon window
    strand: str,
    genome: Fasta,
    contig_lens: dict[str, int],
) -> bool:
    """Return True if genome[contig][pos:pos+3] is a stop codon on the given strand."""
    clen = contig_lens.get(contig, 0)
    if pos < 0 or pos + 3 > clen:
        return False
    seq = str(genome[contig][pos:pos + 3]).upper()
    if strand == "-":
        seq = _rev_comp(seq)
    return seq in STOP_CODONS


# ──────────────────────────────────────────────────────────────────────────────
# Extension logic
# ──────────────────────────────────────────────────────────────────────────────

def _merge_adjacent(segs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _build_extended(
    orf_segs: list[tuple[int, int]],
    mp_segs: list[tuple[int, int]],
    is_complete: bool,
    strand: str,
    max_extension: int,
) -> list[tuple[int, int]] | None:
    """Return merged CDS segments for the extended ORF, or None on failure.

    For + strand:
      ext_boundary = last CDS end (partial) or last CDS end − 3 (complete).
      ORF base: orf_segs clipped to ext_boundary.
      Extension:  mp_segs starting at or past ext_boundary.
      Stop codon: 3 nt appended after the last mp CDS segment.

    For − strand (mirrored):
      ext_boundary = first CDS start (partial) or first CDS start + 3 (complete).
      ORF base: orf_segs at or above ext_boundary.
      Extension:  mp_segs ending at or before ext_boundary.
      Stop codon: 3 nt prepended before the first mp CDS segment.

    Returns None if the extension exceeds max_extension or if the resulting
    total coding length is not divisible by 3 (reading-frame sanity check).
    """
    if strand == "+":
        orf_last = max(e for _, e in orf_segs)
        ext_boundary = orf_last - 3 if is_complete else orf_last
        mp_last_end = max(e for _, e in mp_segs)
        stop_end = mp_last_end + 3

        if stop_end - orf_last > max_extension:
            return None

        base = [(s, min(e, ext_boundary)) for s, e in orf_segs if s < ext_boundary]
        base = [(s, e) for s, e in base if s < e]
        ext = [(max(s, ext_boundary), e) for s, e in mp_segs if e > ext_boundary]
        ext = [(s, e) for s, e in ext if s < e]
        stop_seg = (mp_last_end, stop_end)

    else:  # "-"
        orf_first = min(s for s, _ in orf_segs)
        ext_boundary = orf_first + 3 if is_complete else orf_first
        mp_first_start = min(s for s, _ in mp_segs)
        stop_start = mp_first_start - 3

        if orf_first - stop_start > max_extension:
            return None

        base = [(max(s, ext_boundary), e) for s, e in orf_segs if e > ext_boundary]
        base = [(s, e) for s, e in base if s < e]
        ext = [(s, min(e, ext_boundary)) for s, e in mp_segs if s < ext_boundary]
        ext = [(s, e) for s, e in ext if s < e]
        stop_seg = (stop_start, mp_first_start)

    new_segs = _merge_adjacent(base + ext + [stop_seg])
    if not new_segs:
        return None
    total = sum(e - s for s, e in new_segs)
    if total % 3 != 0:
        return None
    return new_segs


# ──────────────────────────────────────────────────────────────────────────────
# Per-ORF fix attempt
# ──────────────────────────────────────────────────────────────────────────────

def try_fix(
    orf: ORF,
    mp_alns: list[MpAlignment],
    genome: Fasta,
    contig_lens: dict[str, int],
    max_extension: int,
) -> list[tuple[int, int]] | None:
    """Try each overlapping miniprot alignment and return new CDS segments on success.

    Alignments are tried in order of most-downstream stop first so that the
    largest protein-supported extension is preferred.
    """
    orf_end = max(e for _, e in orf.segments)
    orf_first = min(s for s, _ in orf.segments)

    if orf.strand == "+":
        # Most-downstream (highest stop_pos) first
        ordered = sorted(mp_alns, key=lambda m: m.cds_segments[-1][1], reverse=True)
    else:
        # Most-downstream (lowest stop_pos) first
        ordered = sorted(mp_alns, key=lambda m: m.cds_segments[0][0])

    for mp in ordered:
        sp = mp.stop_pos()

        if orf.strand == "+":
            # Miniprot stop must end past current ORF end
            if sp + 3 <= orf_end:
                continue
        else:
            # Miniprot stop must start below current ORF first position
            if sp >= orf_first:
                continue

        if not verify_stop(orf.contig, sp, orf.strand, genome, contig_lens):
            continue

        new_segs = _build_extended(
            orf.segments, mp.cds_segments,
            not orf.is_partial, orf.strand, max_extension,
        )
        if new_segs is not None:
            return new_segs

    return None


# ──────────────────────────────────────────────────────────────────────────────
# GTF writing
# ──────────────────────────────────────────────────────────────────────────────

def _gtf_lines(
    tid: str,
    segments: list[tuple[int, int]],
    contig: str,
    strand: str,
    source: str,
) -> list[str]:
    """Emit one CDS GTF line per segment with correct GTF phase."""
    # Compute phases in 5'→3' reading order
    reading_order = sorted(segments) if strand == "+" else sorted(segments, reverse=True)
    phase_map: dict[tuple[int, int], int] = {}
    coding_so_far = 0
    for seg in reading_order:
        phase_map[seg] = (3 - coding_so_far % 3) % 3
        coding_so_far += seg[1] - seg[0]

    lines = []
    for s, e in sorted(segments):
        ph = phase_map[(s, e)]
        lines.append(
            f"{contig}\t{source}\tCDS\t{s + 1}\t{e}\t.\t{strand}\t{ph}"
            f'\ttranscript_id "{tid}"; gene_id "{tid}";'
        )
    return lines


def _passthrough_lines(orf: ORF) -> list[str]:
    """Rebuild GTF lines for an ORF that needs no correction."""
    return _gtf_lines(orf.tid, orf.segments, orf.contig, orf.strand, orf.source)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--orfs", type=Path, required=True,
                    help="Complete predicted ORF GTF (from annotate.py).")
    ap.add_argument("--partial", type=Path, default=None,
                    help="Partial ORF GTF (from annotate.py --partial-out). Optional.")
    ap.add_argument("--miniprot", type=Path, required=True,
                    help="Miniprot GFF3 output (protein-to-genome alignments).")
    ap.add_argument("--genome", type=Path, required=True,
                    help="Genome FASTA (pyfaidx-readable).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output corrected GTF.")
    ap.add_argument("--max-extension", type=int, default=5000,
                    help="Max bp a stop codon may be extended downstream of "
                         "the current ORF end (default 5000).")
    ap.add_argument("--min-overlap-frac", type=float, default=0.3,
                    help="Min fraction of ORF CDS that must overlap the "
                         "miniprot alignment CDS to consider it a match "
                         "(default 0.3).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"Loading complete ORFs:   {args.orfs}", file=sys.stderr)
    complete_orfs = parse_orf_gtf(args.orfs, is_partial=False)
    print(f"  {len(complete_orfs)} transcripts", file=sys.stderr)

    partial_orfs: dict[str, ORF] = {}
    if args.partial is not None:
        print(f"Loading partial ORFs:    {args.partial}", file=sys.stderr)
        partial_orfs = parse_orf_gtf(args.partial, is_partial=True)
        print(f"  {len(partial_orfs)} transcripts", file=sys.stderr)

    print(f"Loading miniprot GFF3:   {args.miniprot}", file=sys.stderr)
    mp_alns = parse_miniprot_gff(args.miniprot)
    print(f"  {len(mp_alns)} alignments with StopCodon=1", file=sys.stderr)
    mp_index = build_mp_index(mp_alns)

    print(f"Loading genome FASTA:    {args.genome}", file=sys.stderr)
    genome = Fasta(str(args.genome), as_raw=True, sequence_always_upper=True)
    contig_lens = {k: len(genome[k]) for k in genome.keys()}

    # Merge complete + partial for a single processing pass
    all_orfs: list[ORF] = list(complete_orfs.values()) + list(partial_orfs.values())

    n_complete_fixed = 0
    n_complete_unchanged = 0
    n_partial_recovered = 0
    n_partial_dropped = 0

    with open(args.out, "w") as fh:
        for orf in sorted(all_orfs, key=lambda o: o.tid):
            overlapping = find_overlapping(orf, mp_index, args.min_overlap_frac)
            new_segs = try_fix(
                orf, overlapping, genome, contig_lens, args.max_extension,
            )

            if new_segs is not None:
                lines = _gtf_lines(
                    orf.tid, new_segs, orf.contig, orf.strand, orf.source,
                )
                if orf.is_partial:
                    n_partial_recovered += 1
                else:
                    n_complete_fixed += 1
            elif not orf.is_partial:
                lines = _passthrough_lines(orf)
                n_complete_unchanged += 1
            else:
                n_partial_dropped += 1
                continue

            for line in lines:
                fh.write(line + "\n")

    print(
        f"Complete ORFs : {n_complete_fixed} fixed, "
        f"{n_complete_unchanged} unchanged.\n"
        f"Partial ORFs  : {n_partial_recovered} recovered, "
        f"{n_partial_dropped} dropped (no protein support).",
        file=sys.stderr,
    )
    print(f"Output: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
