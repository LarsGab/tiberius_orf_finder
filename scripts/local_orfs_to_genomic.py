"""Convert transcript-coordinate ORF calls to a genomic-coordinate CDS GTF.

The three ORF callers we benchmark against Tiberius —
``TransDecoder`` (v1, 5.7.1), ``TD2`` (TransDecoder2), and
``GeneMarkS-T`` (``gmst.pl``) — all take the assembled transcript FASTA
as input and emit ORF coordinates *in transcript coordinates*. To
compare them against the genomic reference via ``gffcompare`` we splice
each ORF through the StringTie exon structure exactly the way
``annotate.py`` does for Tiberius (see
``tiberius_orf.data.tx_to_genome.project_tx_intervals_to_genomic``).

Input formats supported (auto-detected from the CDS row attributes):

* **TransDecoder v1** — ``<tx>.fasta.transdecoder.gff3``. GFF3 with
  per-tx multi-ORFs. Rows: ``<tx_id>\\ttransdecoder\\tCDS\\t<s>\\t<e>\\t.\\t<sig>\\t<phase>\\tID=cds.<tx_id>.p<N>;Parent=<tx_id>.p<N>``.
  Strand column ``<sig>`` is the ORF's orientation *relative to the
  input transcript*: '+' = sense, '-' = antisense.

* **TD2** — ``<tx>.fasta.TD2.gff3``. Identical structure to TD1 but
  ``source`` column is ``TD2``.

* **GeneMarkS-T** — ``gmst.pl --format GFF``. Older GFF2-ish:
  ``<tx_id>\\tGeneMark.hmm\\tCDS\\t<s>\\t<e>\\t<score>\\t<sig>\\t<phase>\\tgene_id "<n>"``.
  One CDS row per predicted ORF; strand is again relative to the
  transcript.

The converter groups CDS rows into ORFs (via the ``Parent=`` / ``ID=``
attribute if present, else by ``gene_id`` in the attribute column, else
by tx_id + a per-tx counter), projects each ORF's transcript-coord
intervals through the matching StringTie transcript's exons, and emits
one genomic CDS GTF line per resulting sub-interval.

Antisense (``strand_rel='-'``) ORFs are dropped by default: Tiberius
only decodes the sense strand of each assembled transcript, so
including antisense hits would be an apples-to-oranges comparison. Add
``--keep-antisense`` to keep them (they are re-projected onto the
opposite genomic strand, using the same exon structure).

Also emits a summary ``.stats.txt`` next to the GTF: how many ORFs were
read, kept, dropped-antisense, missing-tx (ORF for a tx not in
stringtie.gtf), and how many genomic CDS lines were written.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tiberius_orf.data.label_transcripts import parse_stringtie_gtf  # noqa: E402
from tiberius_orf.data.tx_to_genome import project_tx_intervals_to_genomic  # noqa: E402


_PARENT_RE = re.compile(r"Parent=([^;\s]+)")
_ID_RE     = re.compile(r"ID=([^;\s]+)")
_GENE_ID_RE = re.compile(r'gene_id\s+"([^"]+)"')


def _extract_orf_id(
    attrs: str,
    tx_id: str,
    strand_rel: str,
    tx_counter: dict[tuple[str, str], int],
) -> str:
    """Assign a unique ORF identifier to one CDS row.

    Preference order: ``Parent=`` (TD1/TD2 GFF3), ``ID=`` (any GFF3),
    ``gene_id "..."`` (GMST GFF), else a synthetic ``<tx>.<strand>.<n>``
    where ``n`` counts previously-seen ORFs on the same tx+strand. The
    synthetic branch treats every CDS row as its own ORF; that matches
    GMST's one-CDS-row-per-ORF convention.
    """
    m = _PARENT_RE.search(attrs)
    if m:
        return m.group(1)
    m = _ID_RE.search(attrs)
    if m:
        return m.group(1)
    m = _GENE_ID_RE.search(attrs)
    if m:
        return f"{tx_id}.{strand_rel}.{m.group(1)}"
    key = (tx_id, strand_rel)
    tx_counter[key] = tx_counter.get(key, 0) + 1
    return f"{tx_id}.{strand_rel}.orf{tx_counter[key]}"


def parse_local_orf_gff(path: Path) -> dict[str, dict]:
    """Parse a TD1/TD2/GMST local-ORF GFF into per-ORF records.

    Returns ``{orf_id: {"tx_id", "strand_rel", "intervals"}}`` with
    ``intervals`` a sorted list of 0-based half-open ``(start, end)``
    tuples in transcript coordinates.
    """
    orfs: dict[str, dict] = {}
    tx_counter: dict[tuple[str, str], int] = {}
    for raw in Path(path).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        f = raw.rstrip("\r").split("\t")
        if len(f) < 9 or f[2] != "CDS":
            continue
        tx_id = f[0]
        strand_rel = f[6]
        if strand_rel not in ("+", "-"):
            continue
        try:
            s1 = int(f[3])
            e1 = int(f[4])
        except ValueError:
            continue
        s0, e0 = s1 - 1, e1
        orf_id = _extract_orf_id(f[8], tx_id, strand_rel, tx_counter)
        d = orfs.setdefault(orf_id, {"tx_id": tx_id,
                                     "strand_rel": strand_rel,
                                     "intervals": []})
        d["intervals"].append((s0, e0))
    for d in orfs.values():
        d["intervals"].sort()
    return orfs


def _invert_intervals_in_tx(
    intervals: list[tuple[int, int]], tx_len: int,
) -> list[tuple[int, int]]:
    """Flip transcript-coord intervals to the reverse-complement strand.

    An antisense ORF at transcript-coord ``(s0, e0)`` on strand '-' on
    an ``tx_len``-nt transcript sits at ``(tx_len - e0, tx_len - s0)``
    on the RC. The projection helper always walks exons in transcript
    reading order for the requested strand — so if we then hand it a
    negated StringTie strand, the antisense hit lands correctly on the
    other genomic strand.
    """
    out = [(tx_len - e0, tx_len - s0) for s0, e0 in intervals]
    out.sort()
    return out


def convert(
    local_gff: Path,
    stringtie_gtf: Path,
    out_gtf: Path,
    source: str,
    keep_antisense: bool = False,
) -> dict[str, int]:
    """Emit ``out_gtf`` and return per-category counters.

    Counters: ``orfs_in``, ``orfs_kept``, ``orfs_dropped_antisense``,
    ``orfs_dropped_missing_tx``, ``orfs_dropped_zero_projected``,
    ``cds_lines_written``.
    """
    orfs = parse_local_orf_gff(local_gff)
    stringtie = parse_stringtie_gtf(stringtie_gtf)

    counters = {
        "orfs_in": len(orfs),
        "orfs_kept": 0,
        "orfs_dropped_antisense": 0,
        "orfs_dropped_missing_tx": 0,
        "orfs_dropped_zero_projected": 0,
        "cds_lines_written": 0,
    }

    out_gtf.parent.mkdir(parents=True, exist_ok=True)
    with open(out_gtf, "w") as fh:
        for orf_id, d in orfs.items():
            tx_id = d["tx_id"]
            strand_rel = d["strand_rel"]
            intervals = d["intervals"]
            if tx_id not in stringtie:
                counters["orfs_dropped_missing_tx"] += 1
                continue
            tx = stringtie[tx_id]

            if strand_rel == "-":
                if not keep_antisense:
                    counters["orfs_dropped_antisense"] += 1
                    continue
                # antisense: flip intervals in tx-coord space AND flip
                # the projection strand so they land on the opposite
                # genomic strand. Build a stand-in with the swapped
                # strand; exons are the same list.
                intervals = _invert_intervals_in_tx(intervals, tx.length)
                flipped_strand = "-" if tx.strand == "+" else "+"

                class _TxFlipped:
                    contig = tx.contig
                    strand = flipped_strand
                    exons = tx.exons

                lines = project_tx_intervals_to_genomic(
                    orf_id, intervals, _TxFlipped, source, gene_id=tx_id,
                )
            else:
                lines = project_tx_intervals_to_genomic(
                    orf_id, intervals, tx, source, gene_id=tx_id,
                )

            if not lines:
                counters["orfs_dropped_zero_projected"] += 1
                continue
            counters["orfs_kept"] += 1
            counters["cds_lines_written"] += len(lines)
            for ln in lines:
                fh.write(ln + "\n")

    stats_path = out_gtf.with_suffix(out_gtf.suffix + ".stats.txt")
    with open(stats_path, "w") as fh:
        for k, v in counters.items():
            fh.write(f"{k}\t{v}\n")
    return counters


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Project transcript-coord ORF calls to a genomic GTF."
    )
    ap.add_argument("--local-gff", type=Path, required=True,
                    help="TD1/TD2/GMST GFF with CDS rows in transcript coords.")
    ap.add_argument("--stringtie-gtf", type=Path, required=True,
                    help="StringTie GTF that produced the transcript FASTA.")
    ap.add_argument("--out-gtf", type=Path, required=True,
                    help="Output genomic CDS GTF.")
    ap.add_argument("--source", type=str, default="local_orf",
                    help='GTF column-2 tag (e.g. "transdecoder", "td2", "gmst").')
    ap.add_argument("--keep-antisense", action="store_true",
                    help="Keep ORFs called antisense to the transcript "
                         "(default: drop; Tiberius only decodes sense).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    counters = convert(
        args.local_gff, args.stringtie_gtf, args.out_gtf,
        source=args.source, keep_antisense=args.keep_antisense,
    )
    print(f"Input:  {args.local_gff}", flush=True)
    print(f"Output: {args.out_gtf}", flush=True)
    for k, v in counters.items():
        print(f"  {k:32s} {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
