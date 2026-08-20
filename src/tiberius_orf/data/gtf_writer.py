"""Convert per-position ORF labels to GTF.

Given a per-transcript label array (0=IR, 1=START, 2=E1, 3=E2, 4=E0, 5=STOP)
and a StringTie transcript with its genomic exon structure, this module:

1. Extracts complete ORFs (START ... E* ... E0 STOP) as transcript-coord intervals.
2. Projects each ORF onto the genome by walking the transcript's exons.
3. Emits one CDS line per genomic sub-interval (ORFs that cross exon junctions
   produce multiple CDS lines that share the same transcript_id).

GTF lines use 1-based inclusive coordinates per the GTF spec.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .label_transcripts import StringTieTranscript


# label codes (re-imported to keep this module standalone)
IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5

# LORF classification constants (GeneMark-ETP-style 5'-end categories)
LORF_UPSTOP = "LORF_UPSTOP"    # upstream in-frame stop confirms this ATG is the LORF
sORF_UPSTOP = "sORF_UPSTOP"    # upstream stop exists but a longer ORF starts before us
upLORF = "upLORF"               # no upstream stop; upstream ATG exists → longer ORF present
LORF_NOUPSTOP = "LORF_NOUPSTOP"  # no upstream stop, no upstream ATG → likely 5'-partial

_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


def classify_lorf(tx_seq: str, orf_tx_start: int) -> str:
    """Classify the 5' context of an ORF by scanning upstream in-frame for stop codons.

    Walks upstream from *orf_tx_start* in steps of 3 (same reading frame as the
    predicted ATG).  Returns one of the four LORF class constants defined in
    this module.

    LORF_UPSTOP   – in-frame stop upstream; no ATG between it and our ATG.
    sORF_UPSTOP   – in-frame stop upstream; but a farther ATG exists between
                    it and our ATG (our ORF is not the longest after that stop).
    upLORF        – no upstream stop; an in-frame ATG exists upstream
                    (a longer ORF is present in the same frame).
    LORF_NOUPSTOP – no upstream stop and no upstream ATG; transcript is likely
                    5'-incomplete.
    """
    p = orf_tx_start
    upstream_stop: int | None = None
    for pos in range(p - 3, -1, -3):
        if tx_seq[pos : pos + 3].upper() in _STOP_CODONS:
            upstream_stop = pos
            break
    if upstream_stop is not None:
        for pos in range(upstream_stop + 3, p, 3):
            if tx_seq[pos : pos + 3].upper() == "ATG":
                return sORF_UPSTOP
        return LORF_UPSTOP
    else:
        for pos in range(p % 3, p, 3):
            if tx_seq[pos : pos + 3].upper() == "ATG":
                return upLORF
        return LORF_NOUPSTOP


def extract_orfs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return [(tx_start, tx_end), ...] half-open 0-based intervals for each
    complete ORF in the label sequence.

    A complete ORF is a maximal run of the form
        START (E1|E2|E0)+ STOP
    Anything else (lone STARTs, broken cycles, missing STOP) is skipped.
    """
    orfs: list[tuple[int, int]] = []
    L = len(labels)
    i = 0
    while i < L:
        if labels[i] != START:
            i += 1
            continue
        # walk through coding states until we hit STOP, IR, or another START
        j = i + 1
        while j < L and labels[j] in (E1, E2, E0):
            j += 1
        if j < L and labels[j] == STOP:
            orfs.append((i, j + 1))   # half-open, includes STOP base
            i = j + 1
        else:
            i = j   # incomplete ORF, advance past it
    return orfs


def extract_partial_orfs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return [(tx_start, tx_end), ...] for 3'-truncated ORFs.

    A partial ORF is a run  START (E1|E2|E0)+  that reaches the end of
    ``labels`` without hitting STOP.  Unlike complete ORFs, ``tx_end``
    equals the array length (the run simply ends there) and no STOP label
    is included.  Runs terminated by a second START or an IR state are not
    returned — those are mid-sequence interruptions, not 3'-end truncations.
    """
    orfs: list[tuple[int, int]] = []
    L = len(labels)
    i = 0
    while i < L:
        if labels[i] != START:
            i += 1
            continue
        j = i + 1
        while j < L and labels[j] in (E1, E2, E0):
            j += 1
        if j >= L and j > i + 1:
            # Ran off the end with at least one coding state after START
            orfs.append((i, j))
        i = j
    return orfs


def extract_5prime_partial_orfs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return [(0, tx_end)] for a 5'-truncated ORF at the transcript start.

    A 5'-partial ORF begins at position 0 with coding states (E1|E2|E0) —
    no preceding START — and terminates with a STOP.  This represents a CDS
    whose ATG is upstream of the assembled transcript.

    Returns at most one entry; an empty list if position 0 is not a coding
    state or if no STOP is found before the run ends.
    """
    CODING = frozenset({E1, E2, E0})
    L = len(labels)
    if L == 0 or labels[0] not in CODING:
        return []
    j = 0
    while j < L and labels[j] in CODING:
        j += 1
    if j < L and labels[j] == STOP:
        return [(0, j + 1)]
    return []


def tx_interval_to_genomic_segments(
    tx_start: int,
    tx_end: int,
    tx: StringTieTranscript,
) -> list[tuple[int, int]]:
    """Project a transcript half-open interval [tx_start, tx_end) onto the
    genome via tx.exons, returning a list of half-open genomic intervals
    (one per spanned exon, in genomic ascending order).

    Handles both strands: tx.exons is always sorted ASC by g_start, but on the
    minus strand the FIRST exon (in transcript reading order) is the LAST exon
    in genomic order, so we walk exons reversed for `-` strand.
    """
    if tx_start >= tx_end:
        return []

    exons_in_tx_order = list(tx.exons) if tx.strand == "+" else list(reversed(tx.exons))

    segments: list[tuple[int, int]] = []
    cumulative = 0   # tx-coord position at the start of the current exon
    for g_start, g_end in exons_in_tx_order:
        exon_len = g_end - g_start
        # Intersect [tx_start, tx_end) with [cumulative, cumulative+exon_len)
        lo = max(tx_start, cumulative)
        hi = min(tx_end, cumulative + exon_len)
        if lo < hi:
            off_lo = lo - cumulative
            off_hi = hi - cumulative
            if tx.strand == "+":
                segments.append((g_start + off_lo, g_start + off_hi))
            else:
                # offsets count from the high end of this exon on the genome
                segments.append((g_end - off_hi, g_end - off_lo))
        cumulative += exon_len
        if cumulative >= tx_end:
            break

    # Sort genomic ASC for downstream stability
    segments.sort()
    return segments


def _gtf_attr(tx_id: str) -> str:
    return f'transcript_id "{tx_id}"; gene_id "{tx_id}";'


def labels_to_gtf_lines(
    tx_id: str,
    labels: np.ndarray,
    tx: StringTieTranscript,
    source: str,
) -> list[str]:
    """Return one GTF line per genomic CDS segment for all complete ORFs in the
    transcript.

    Per GTF/GFF3 spec, the phase (column 8) is the number of bases that must
    be removed from the start of this CDS feature to reach the first base of
    the next codon: 0 means this segment begins on a codon boundary, 1 means
    skip 1 base, 2 means skip 2 bases. With `cds_offset` = number of coding
    bases preceding this segment within the same ORF, this is
    `(3 - cds_offset % 3) % 3`, NOT `cds_offset % 3` (which swaps 1 and 2).
    """
    out: list[str] = []
    for orf_tx_start, orf_tx_end in extract_orfs(labels):
        # Walk segments in genomic ASC order; remember tx-coord ordering for phase.
        # For phase computation we need the TX-coord position of each segment's
        # first base, then phase = (3 - (tx_pos - orf_tx_start) % 3) % 3
        # (GTF phase = bases to skip to reach the next codon start).
        exons_in_tx_order = (
            list(tx.exons) if tx.strand == "+" else list(reversed(tx.exons))
        )
        cumulative = 0
        per_segment: list[tuple[int, int, int]] = []  # (g_start, g_end, tx_pos_of_first_base)
        for g_start, g_end in exons_in_tx_order:
            exon_len = g_end - g_start
            lo = max(orf_tx_start, cumulative)
            hi = min(orf_tx_end, cumulative + exon_len)
            if lo < hi:
                off_lo = lo - cumulative
                off_hi = hi - cumulative
                if tx.strand == "+":
                    g_lo, g_hi = g_start + off_lo, g_start + off_hi
                else:
                    g_lo, g_hi = g_end - off_hi, g_end - off_lo
                per_segment.append((g_lo, g_hi, lo))
            cumulative += exon_len
            if cumulative >= orf_tx_end:
                break

        per_segment.sort()  # genomic ASC
        for g_lo, g_hi, tx_pos in per_segment:
            phase = (3 - (tx_pos - orf_tx_start) % 3) % 3
            # GTF: 1-based inclusive
            out.append(
                "\t".join([
                    tx.contig,
                    source,
                    "CDS",
                    str(g_lo + 1),
                    str(g_hi),
                    ".",
                    tx.strand,
                    str(phase),
                    _gtf_attr(tx_id),
                ])
            )
    return out


def write_gtf(
    out_path: Path | str,
    labels_by_tx: dict[str, np.ndarray],
    transcripts: dict[str, StringTieTranscript],
    source: str = "tiberius_orf",
) -> int:
    """Write a GTF file from per-transcript labels. Returns the number of CDS lines."""
    n = 0
    with open(out_path, "w") as fh:
        for tx_id in sorted(labels_by_tx):
            if tx_id not in transcripts:
                continue
            lines = labels_to_gtf_lines(
                tx_id, labels_by_tx[tx_id], transcripts[tx_id], source
            )
            for line in lines:
                fh.write(line + "\n")
                n += 1
    return n
