"""Attribute merged-locus FPs to StringTie assembly errors.

Cross-references two gffcompare runs against the SAME reference annotation:

* ``merged.tracking``   : predicted (post-merge) transcripts vs reference
* ``stvsref.tracking``  : raw StringTie transcripts vs reference

For each merged FP locus (dominant class code in {j, c, k, ~, m, n, e, o}
i.e. same-strand ref-overlap with wrong structure), look up how StringTie
itself handled the SAME reference gene, and attribute the FP to one of:

    stringtie_missed       : no StringTie transcript at that ref gene
    stringtie_split        : ref gene covered by >=2 distinct StringTie loci
    stringtie_missplice    : StringTie's best code is j/~ (junctions off)
    stringtie_short        : StringTie's best code is c (contained in ref;
                             shorter than ref)
    stringtie_correct_but_model_failed
                           : StringTie has an intron-chain match (= or the
                             ref intron chain is a subset of a StringTie
                             transcript's introns; treated permissively as
                             'StringTie had the structure')

Intergenic (u) FPs are counted separately as ``no_ref_match``.

Usage::

    python scripts/fp_attribute_to_stringtie.py \\
        --merged-tracking .../eval_.../gffcompare_runs/<sp>/merged.tracking \\
        --stvsref-tracking .../<sp>/stringtie_vs_ref_full/stvsref.tracking \\
        --species Bos_taurus \\
        --out-tsv ...
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


# Same TP definition used by fp_locus_diagnostic.py (matches gffcompare's
# own Locus-level precision numerator to within super-loci rounding).
TP_CODES = {"="}


def _ref_gene(ref_field: str) -> str | None:
    """Return the normalised reference gene id from a tracking row's
    column 3. Format is 'GENE|TRANSCRIPT' or '-'.

    Strips the ``gene-`` prefix that NCBI ``annotation.gff`` uses but that
    ``annot_cds.gff`` (which we compare against in the merged eval) does
    not, so the same ref gene compares equal across the two tracking
    files."""
    if ref_field in ("", "-"):
        return None
    gene = ref_field.split("|", 1)[0]
    if gene.startswith("gene-"):
        gene = gene[len("gene-"):]
    return gene


def _parse_tracking(tracking: Path) -> dict[str, list[tuple[str, str]]]:
    """Return {xloc: [(class_code, ref_gene_or_None), ...]}."""
    per_xloc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with tracking.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            per_xloc[f[1]].append((f[3], _ref_gene(f[2])))
    return per_xloc


def _stringtie_view_per_ref_gene(stvsref_tracking: Path
                                  ) -> dict[str, dict]:
    """Return {ref_gene: {"xlocs": set, "codes": Counter-like set}}."""
    view: dict[str, dict] = defaultdict(
        lambda: {"xlocs": set(), "codes": []}
    )
    with stvsref_tracking.open() as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            gene = _ref_gene(f[2])
            if not gene:
                continue
            view[gene]["xlocs"].add(f[1])
            view[gene]["codes"].append(f[3])
    return view


def _stringtie_category(entry: dict) -> str:
    """Categorise StringTie's assembly quality at a ref gene."""
    if not entry:
        return "stringtie_missed"
    if len(entry["xlocs"]) >= 2:
        return "stringtie_split"
    codes = set(entry["codes"])
    if codes & {"=", "c"}:
        # gffcompare "c" for stringtie-vs-ref means stringtie tx is
        # CONTAINED in a ref tx — StringTie has ref intron chain as a
        # subset. Treat as "structure present".
        # Also include "k" (stringtie contains ref) as structural match.
        return "stringtie_correct_but_model_failed"
    if "k" in codes:
        return "stringtie_correct_but_model_failed"
    if codes & {"j", "~"}:
        # Some ref junctions matched but not all.
        return "stringtie_missplice"
    if codes & {"m", "n"}:
        return "stringtie_retained_intron"
    if codes & {"i", "y", "o", "p"}:
        return "stringtie_overlap_only"
    return "stringtie_other"


_ATTRIB_BUCKETS = [
    "stringtie_missed",
    "stringtie_split",
    "stringtie_correct_but_model_failed",
    "stringtie_missplice",
    "stringtie_retained_intron",
    "stringtie_short",  # reserved for a stricter future rule
    "stringtie_overlap_only",
    "stringtie_other",
    "no_ref_match",       # intergenic FP (code 'u' only)
]


def _analyse(merged_tracking: Path, stvsref_tracking: Path) -> dict:
    per_xloc  = _parse_tracking(merged_tracking)
    st_view   = _stringtie_view_per_ref_gene(stvsref_tracking)

    total_fp = 0
    attrib_counts: dict[str, int] = defaultdict(int)
    for xloc, rows in per_xloc.items():
        codes = {c for c, _ in rows}
        if codes & TP_CODES:
            continue                 # locus TP
        total_fp += 1
        # Pick the ref gene most transcripts point to (or first non-None).
        ref_genes = [g for _, g in rows if g is not None]
        if not ref_genes:
            attrib_counts["no_ref_match"] += 1
            continue
        # Use the most frequent ref gene as the anchor.
        counts = defaultdict(int)
        for g in ref_genes:
            counts[g] += 1
        ref_gene = max(counts, key=counts.get)
        cat = _stringtie_category(st_view.get(ref_gene))
        attrib_counts[cat] += 1

    return {
        "total_fp_loci": total_fp,
        **{b: attrib_counts.get(b, 0) for b in _ATTRIB_BUCKETS},
    }


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--merged-tracking",  type=Path, required=True)
    ap.add_argument("--stvsref-tracking", type=Path, required=True)
    ap.add_argument("--species",          type=str,  required=True)
    ap.add_argument("--out-tsv",          type=Path, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    r = _analyse(args.merged_tracking, args.stvsref_tracking)
    r = {"species": args.species, **r}
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(r.keys()), delimiter="\t")
        w.writeheader()
        w.writerow(r)
    print(f"[fp-attribute] wrote {args.out_tsv}", flush=True)
    n = r["total_fp_loci"]
    print(f"\n{args.species}  FP={n}")
    for b in _ATTRIB_BUCKETS:
        v = r[b]
        print(f"  {b:<44s} {v:>6d}  ({100.0*v/max(1,n):>5.1f}%)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
