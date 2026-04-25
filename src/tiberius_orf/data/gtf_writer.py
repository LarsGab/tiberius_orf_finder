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

    Frame is computed as `(cds_offset_from_orf_start) % 3`, where cds_offset
    is the number of coding bases preceding this segment within the same ORF.
    Per GTF spec frame is the number of bases until the next codon starts:
    0 means this segment begins on a codon boundary.
    """
    out: list[str] = []
    for orf_tx_start, orf_tx_end in extract_orfs(labels):
        # Walk segments in genomic ASC order; remember tx-coord ordering for frame.
        # For frame computation we need the TX-coord position of each segment's
        # first base, then frame = ( (tx_pos - orf_tx_start) % 3 ).
        # We re-derive tx-coord offsets per segment by walking the tx_interval
        # function's internal logic below.
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
            frame = (tx_pos - orf_tx_start) % 3
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
                    str(frame),
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
