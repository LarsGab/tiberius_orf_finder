"""Project reference CDS onto StringTie-assembled transcripts.

Inputs
------
* StringTie GTF (exons for assembled transcripts)
* Reference GFF/GTF (CDS + stop_codon features for a curated annotation)

For each StringTie transcript the module decides one of:
* ``ir_only``                 - no same-strand reference CDS overlaps  -> all-IR negative
* ``kept_single`` / ``kept_multi`` - >=1 reference CDS fully contained -> keep the longest
* ``dropped_*``               - partial overlap, antisense, or partial reference CDS

Labels per transcript position (int8):
    0 IR   1 START   2 E1   3 E2   4 E0   5 STOP
where ``START`` marks the A of ATG and ``STOP`` marks the last base of the stop
codon. Between them the cycle ``E1, E2, E0`` is repeated, matching the project
convention ``START - E1 - E2 - E0 - ... - E1 - STOP``.

This implementation uses only stdlib + numpy; bricks2marble's ``Anno``/``Transcript``
does not cover StringTie-style exons (its Transcript model stores only CDS parts),
so we parse the two GTFs directly here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# Label classes.
IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5
_FRAME_CYCLE = (E1, E2, E0)  # indexed by (offset_from_start - 1) % 3


@dataclass
class StringTieTranscript:
    tid: str
    contig: str
    strand: str                           # '+' or '-'
    exons: list[tuple[int, int]]          # 0-based half-open; ASC by g_start

    @property
    def length(self) -> int:
        return sum(e - s for s, e in self.exons)

    @property
    def g_span(self) -> tuple[int, int]:
        return self.exons[0][0], self.exons[-1][1]


@dataclass
class RefTranscript:
    tid: str
    contig: str
    strand: str
    cds_parts: list[tuple[int, int]]      # 0-based half-open; ASC by g_start
    has_stop_codon: bool                  # whether a stop_codon feature was seen

    @property
    def total_cds_len(self) -> int:
        return sum(e - s for s, e in self.cds_parts)

    @property
    def g_span(self) -> tuple[int, int]:
        return self.cds_parts[0][0], self.cds_parts[-1][1]

    @property
    def complete(self) -> bool:
        # Complete = multiple of 3 and has a stop codon feature (or the CDS
        # already includes the stop based on a %3 length).  RefSeq typically
        # embeds the stop in CDS; BRAKER splits it out.  Require one of:
        #   * stop_codon feature present (length already accounts for it)
        #   * no stop_codon feature AND CDS length %3 == 0 (CDS includes stop)
        if self.total_cds_len < 6:
            return False
        if self.total_cds_len % 3 != 0:
            return False
        return True


@dataclass
class ProjectionResult:
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    chosen_ref: dict[str, str] = field(default_factory=dict)  # tid -> ref_tid


# ---------- parsing ----------

def _parse_attr(attr_col: str, key: str) -> str | None:
    # Supports both GTF (key "value";) and GFF3 (key=value;).
    for chunk in attr_col.strip().strip(";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk and " " not in chunk.split("=", 1)[0]:
            k, v = chunk.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"')
        else:
            parts = chunk.split(None, 1)
            if len(parts) == 2 and parts[0] == key:
                return parts[1].strip().strip('"')
    return None


def parse_stringtie_gtf(path: Path) -> dict[str, StringTieTranscript]:
    """Parse a StringTie GTF, collecting exon features per transcript."""
    exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}     # tid -> (contig, strand)
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9 or f[2] != "exon":
            continue
        tid = _parse_attr(f[8], "transcript_id")
        if tid is None:
            continue
        start = int(f[3]) - 1       # 1-based incl -> 0-based
        end = int(f[4])             # 1-based incl end -> 0-based exclusive
        exons[tid].append((start, end))
        meta[tid] = (f[0], f[6])

    out: dict[str, StringTieTranscript] = {}
    for tid, es in exons.items():
        contig, strand = meta[tid]
        out[tid] = StringTieTranscript(
            tid=tid, contig=contig, strand=strand,
            exons=sorted(es),
        )
    return out


def parse_reference_cds(path: Path) -> dict[str, RefTranscript]:
    """Parse a reference GFF/GTF, keeping CDS and stop_codon features per transcript."""
    cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    stops: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str]] = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        feat = f[2]
        if feat not in ("CDS", "stop_codon"):
            continue
        tid = (_parse_attr(f[8], "transcript_id")
               or _parse_attr(f[8], "Parent"))
        if tid is None:
            continue
        start = int(f[3]) - 1
        end = int(f[4])
        (cds if feat == "CDS" else stops)[tid].append((start, end))
        meta[tid] = (f[0], f[6])

    out: dict[str, RefTranscript] = {}
    for tid, parts in cds.items():
        # Merge stop_codon intervals into the CDS set if not already covered.
        merged = _merge_intervals(parts + stops.get(tid, []))
        contig, strand = meta[tid]
        out[tid] = RefTranscript(
            tid=tid, contig=contig, strand=strand,
            cds_parts=merged,
            has_stop_codon=bool(stops.get(tid)),
        )
    return out


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---------- coordinate mapping ----------

def _genomic_to_transcript(t: StringTieTranscript, g: int) -> int | None:
    """Return the transcript coordinate of genomic position ``g`` on ``t``,
    or None if ``g`` does not fall inside any exon of ``t``.
    """
    if t.strand == "+":
        offset = 0
        for s, e in t.exons:                 # ascending
            if s <= g < e:
                return offset + (g - s)
            offset += (e - s)
    else:
        offset = 0
        for s, e in reversed(t.exons):       # 5' -> 3' on the transcript
            if s <= g < e:
                return offset + (e - 1 - g)
            offset += (e - s)
    return None


def _project_ref_onto_transcript(
    t: StringTieTranscript, r: RefTranscript,
) -> tuple[int, int] | None:
    """Map every nucleotide of ``r``'s CDS parts to transcript coordinates in
    ``t``; return ``(orf_start, orf_end)`` inclusive if the CDS is fully
    contained AND contiguous on the transcript, else None.
    """
    t_coords: list[int] = []
    for cs, ce in r.cds_parts:
        for g in range(cs, ce):
            tc = _genomic_to_transcript(t, g)
            if tc is None:
                return None
            t_coords.append(tc)
    t_coords.sort()
    expected = list(range(t_coords[0], t_coords[-1] + 1))
    if t_coords != expected:
        return None
    return t_coords[0], t_coords[-1]


# ---------- labeling ----------

def build_labels(length: int,
                 orf: tuple[int, int] | None = None) -> np.ndarray:
    labels = np.zeros(length, dtype=np.int8)   # IR everywhere
    if orf is None:
        return labels
    start, end = orf
    labels[start] = START
    labels[end] = STOP
    for i in range(1, end - start):
        labels[start + i] = _FRAME_CYCLE[(i - 1) % 3]
    return labels


# ---------- main projection logic ----------

def project_labels(
    stringtie_gtf: Path | str,
    reference_gff: Path | str,
) -> ProjectionResult:
    """Run the full labeling pipeline for one species.

    Returns per-transcript labels and a category-count dict.
    """
    st = parse_stringtie_gtf(Path(stringtie_gtf))
    refs = parse_reference_cds(Path(reference_gff))
    # Index reference transcripts by contig for quick lookup.
    refs_by_contig: dict[str, list[RefTranscript]] = defaultdict(list)
    for r in refs.values():
        refs_by_contig[r.contig].append(r)

    result = ProjectionResult()

    for tid, t in st.items():
        # Candidate refs on same contig with genomic overlap.
        t_s, t_e = t.g_span
        overlapping = [
            r for r in refs_by_contig.get(t.contig, [])
            if _ranges_overlap(r.g_span, (t_s, t_e))
        ]
        if not overlapping:
            result.labels[tid] = build_labels(t.length)
            result.stats["ir_only"] += 1
            continue

        same_strand = [r for r in overlapping if r.strand == t.strand]
        if not same_strand:
            result.stats["dropped_antisense_only"] += 1
            continue

        contained = []
        has_partial_ref = False
        for r in same_strand:
            if not r.complete:
                has_partial_ref = True
                continue
            orf = _project_ref_onto_transcript(t, r)
            if orf is not None:
                contained.append((r, orf))

        if not contained:
            if has_partial_ref:
                result.stats["dropped_ref_partial"] += 1
            else:
                result.stats["dropped_not_contained"] += 1
            continue

        # Pick the longest CDS among contained candidates; tie-break by TID.
        contained.sort(key=lambda ro: (-ro[0].total_cds_len, ro[0].tid))
        chosen_r, chosen_orf = contained[0]
        result.labels[tid] = build_labels(t.length, chosen_orf)
        result.chosen_ref[tid] = chosen_r.tid
        result.stats["kept_multi" if len(contained) > 1 else "kept_single"] += 1

    return result


def _ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def write_outputs(
    result: ProjectionResult,
    out_dir: Path | str,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "labels.npz"
    np.savez_compressed(npz_path, **result.labels)

    stats_path = out_dir / "stats.tsv"
    with stats_path.open("w", encoding="utf-8") as fh:
        fh.write("category\tcount\n")
        for k in sorted(result.stats):
            fh.write(f"{k}\t{result.stats[k]}\n")

    return {"labels": npz_path, "stats": stats_path}


def subset_fasta(fasta_in: Path | str, fasta_out: Path | str,
                 keep_ids: set[str]) -> int:
    """Stream ``fasta_in`` and write only records whose id is in ``keep_ids``.

    Returns the number of records written.
    """
    fasta_in = Path(fasta_in)
    fasta_out = Path(fasta_out)
    n = 0
    keep = False
    with fasta_in.open("r", encoding="utf-8") as fh_in, \
            fasta_out.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            if line.startswith(">"):
                header_id = line[1:].strip().split()[0]
                keep = header_id in keep_ids
                if keep:
                    fh_out.write(line)
                    n += 1
            elif keep:
                fh_out.write(line)
    return n


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Project reference CDS onto "
                                 "StringTie-assembled transcripts.")
    ap.add_argument("--stringtie-gtf", type=Path, required=True)
    ap.add_argument("--reference-gff", type=Path, required=True)
    ap.add_argument("--transcripts-fa", type=Path, required=True,
                    help="Transcripts FASTA from `gffread -w`.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    result = project_labels(args.stringtie_gtf, args.reference_gff)
    paths = write_outputs(result, args.out_dir)

    fasta_out = args.out_dir / "transcripts_labelled.fa"
    n_kept = subset_fasta(args.transcripts_fa, fasta_out,
                          keep_ids=set(result.labels))
    print(f"wrote {n_kept} labelled transcripts -> {fasta_out}")
    print(f"labels -> {paths['labels']}")
    print(f"stats  -> {paths['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
