"""Quantify how often StringTie splits one ref gene across multiple loci.

For each protein-coding gene in the reference GFF, count how many distinct
StringTie loci (gene_id) have at least one same-strand transcript whose
genomic span overlaps the ref gene's span. A count > 1 means the ref gene
was split.

Caveats
-------
* "Gene span" is the simple min-start / max-end across all features of
  the gene; for the StringTie locus it's the min/max over all of its
  exons. Overlapping but neighbouring genes can occasionally inflate
  counts (a StringTie locus that legitimately covers one gene can also
  brush against an adjacent gene). For typical loci this is rare.
* Genes with zero overlapping same-strand StringTie loci are reported
  separately (e.g. unexpressed genes); they are not "split", just unseen.

Inputs
------
* StringTie GTF (exon + transcript features; gene_id attribute used)
* Reference GFF: lines with column 3 = ``gene`` and attributes containing
  ``gene_biotype=protein_coding`` (NCBI RefSeq / Ensembl convention).

Outputs
-------
* Stdout summary: total ref genes, count distribution, split rate.
* TSV: ref_gene, contig, strand, span_start, span_end, n_st_loci, st_loci
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tiberius_orf.data.label_transcripts import _ranges_overlap  # noqa: E402


def _parse_attr(attr_col: str, key: str) -> str | None:
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


def parse_ref_protein_coding_genes(
    path: Path,
) -> list[tuple[str, str, str, int, int]]:
    """Return [(gene_id, contig, strand, start_0based, end_0excl)] for
    ``gene`` features with ``gene_biotype=protein_coding``.
    """
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            biotype = _parse_attr(f[8], "gene_biotype")
            if biotype != "protein_coding":
                continue
            gid = (_parse_attr(f[8], "ID")
                   or _parse_attr(f[8], "gene_id")
                   or _parse_attr(f[8], "Name"))
            if gid is None:
                continue
            s = int(f[3]) - 1
            e = int(f[4])
            out.append((gid, f[0], f[6], s, e))
    return out


def parse_stringtie_loci(path: Path) -> dict[str, tuple[str, str, int, int]]:
    """{st_gene_id: (contig, strand, min_start, max_end)} over all exons
    in the locus."""
    by_gid: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] not in ("exon", "transcript"):
                continue
            gid = _parse_attr(f[8], "gene_id")
            if gid is None:
                continue
            s = int(f[3]) - 1
            e = int(f[4])
            rec = by_gid.setdefault(gid, {
                "contig": f[0], "strand": f[6],
                "min_s": s, "max_e": e,
            })
            if s < rec["min_s"]:
                rec["min_s"] = s
            if e > rec["max_e"]:
                rec["max_e"] = e
    return {
        gid: (rec["contig"], rec["strand"], rec["min_s"], rec["max_e"])
        for gid, rec in by_gid.items()
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stringtie-gtf", type=Path, required=True)
    ap.add_argument("--reference-gff", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    args = ap.parse_args(argv)

    print("[quantify_stringtie_gene_split] parsing inputs ...", file=sys.stderr)
    ref_genes = parse_ref_protein_coding_genes(args.reference_gff)
    st_loci = parse_stringtie_loci(args.stringtie_gtf)
    print(f"[quantify_stringtie_gene_split] ref protein-coding genes : "
          f"{len(ref_genes)}", file=sys.stderr)
    print(f"[quantify_stringtie_gene_split] stringtie loci           : "
          f"{len(st_loci)}", file=sys.stderr)

    st_by_cs: dict[tuple[str, str], list[tuple[str, int, int]]] = defaultdict(list)
    for gid, (contig, strand, s, e) in st_loci.items():
        st_by_cs[(contig, strand)].append((gid, s, e))

    rows = []
    counts_dist: Counter = Counter()
    for gid, contig, strand, s, e in ref_genes:
        overlapping = [
            sgid for sgid, ss, se in st_by_cs.get((contig, strand), [])
            if _ranges_overlap((s, e), (ss, se))
        ]
        n = len(overlapping)
        counts_dist[n] += 1
        rows.append((
            gid, contig, strand, s + 1, e, n,
            ";".join(overlapping) if overlapping else "",
        ))

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8") as fh:
        fh.write("ref_gene\tcontig\tstrand\tspan_start\tspan_end\t"
                 "n_st_loci\tst_loci\n")
        for row in rows:
            fh.write("\t".join(str(c) for c in row) + "\n")

    n_total = len(rows)
    n_zero = counts_dist[0]
    n_one = counts_dist[1]
    n_split = sum(v for k, v in counts_dist.items() if k >= 2)
    print(f"\n[quantify_stringtie_gene_split] ref protein-coding genes: {n_total}")
    print(f"  {'n_loci':>6s}  {'n_genes':>8s}   pct")
    for k in sorted(counts_dist):
        v = counts_dist[k]
        pct = 100.0 * v / max(1, n_total)
        print(f"  {k:>6d}  {v:>8d}  {pct:5.1f}%")
    print(f"\n  no st locus  : {n_zero:>6d}  ({100*n_zero/max(1,n_total):5.1f}%)")
    print(f"  one st locus : {n_one:>6d}  ({100*n_one/max(1,n_total):5.1f}%)")
    print(f"  split (>=2)  : {n_split:>6d}  ({100*n_split/max(1,n_total):5.1f}%)")
    print(f"\n[quantify_stringtie_gene_split] detail tsv -> {args.out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
