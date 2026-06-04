#!/usr/bin/env python3
"""
Parse the Tiberius paper `*_data_info.tex` tables and emit per-clade species
CSVs (`species,accession,annotation`) under nextflow/conf/<clade>/.

Maps annotation-source labels found in the .tex tables to the three labels
understood by `nextflow/modules/fetch.nf`:
  - RefSeq    -> fetch via NCBI `datasets download genome accession`
                 (used for RefSeq / Genbank / DDBJ / EMBL / ensembl / NCBI
                  rows that carry a GCA_/GCF_ accession)
  - BRAKER    -> stage from --braker_data_dir/<Genus_species>/
  - Phytozome -> stage from --phytozome_data_dir/<Genus_species>/
                 (used for Embryophyta rows from the Figshare bundle;
                  see scripts/stage_mesangiospermae_figshare.py)

Rows that cannot be auto-fetched (e.g. *Thalassiosira pseudonana*, Filloramo
et al. 2021 Borealis archive) are written to a `_manual.csv` sidecar for
manual handling instead of being silently dropped.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLES_DIR = Path(
    "C:/Users/lgabr/work/paper/tiberius_training/"
    "clade_parameters_oxford_bioinformatics/tables"
)
DEFAULT_OUTDIR = REPO_ROOT / "nextflow" / "conf"


# ----- helpers ----------------------------------------------------------

ITALIC_RE = re.compile(r"\\textit\{([^}]+)\}")
ACCESSION_RE = re.compile(r"\bGC[AF]_\d+\.\d+")
TRAILING_SUPERSCRIPT = re.compile(r"\\textsuperscript\{[^}]*\}")


def _strip_latex(s: str) -> str:
    s = TRAILING_SUPERSCRIPT.sub("", s)
    s = s.replace("\\_", "_")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    return s.strip()


def _sanitise_accession(acc: str) -> str:
    """Strip characters that confuse downstream CSV / shell parsing.
    The accession column is informational for non-NCBI rows (BRAKER /
    Phytozome staging branches key off the species name, not the accession),
    so squashing commas and quotes is safe."""
    s = acc.strip()
    s = s.replace('"', "")
    s = s.replace(",", " ")
    s = re.sub(r"\s+", "_", s).strip("_")
    return s


# Known tex-table typos. Entrez doesn't recognise the typo'd binomial, so
# VARUS_RUNLIST_SR returns no SRA runs; and the staged dirs on brain use
# the corrected spellings. Map typo -> correct binomial at parse time.
SPECIES_TYPO_FIXES = {
    "Cylcotella atomus":           "Cyclotella atomus",
    "Pseudo-nitschia multistriata": "Pseudo-nitzschia multistriata",
}


def _normalise_species(name: str) -> str:
    """`Cylcotella atomus` -> `Cyclotella atomus` (typo fixed, whitespace
    collapsed, strain suffix dropped)."""
    name = _strip_latex(name)
    name = re.sub(r"\s+", " ", name).strip()
    # Drop subspecies / strain suffixes the .tex sometimes carries past the
    # binomial (e.g. "Ostreococcus lucimarinus CCE9901"). Keep only the first
    # two whitespace-separated tokens.
    parts = name.split()
    if len(parts) >= 2:
        binomial = f"{parts[0]} {parts[1]}"
    else:
        binomial = name
    return SPECIES_TYPO_FIXES.get(binomial, binomial)


def _iter_table_body(tex: str) -> Iterable[str]:
    """Yield each logical row inside the tabular body, joined on `\\` line
    breaks. Section header rows (e.g. `\\multicolumn{...}{l}{\\textbf{Training}}`)
    are also yielded so the caller can switch splits."""
    flat = re.sub(r"\n\s*", " ", tex)
    for raw in flat.split(r"\\"):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if "&" not in line and "textbf" not in line:
            continue
        # Drop rows that straddle two tables – they contain `\end{tabular}`
        # or `\begin{tabular}` joined onto a header row.
        if r"\end{tabular}" in line or r"\begin{tabular}" in line:
            continue
        yield line


def _split_cells(row: str) -> list[str]:
    # `&` is the column separator in LaTeX tabulars; rare escaped `\&`
    # doesn't appear in these tables, so we don't bother handling it.
    return [c.strip() for c in row.split("&")]


def _emit_section(rows: list[str]) -> str | None:
    """Detect a section header row like `\\textbf{Training}` and return its
    lower-cased label (`training` / `validation` / `testing` / `test`),
    else None."""
    for r in rows:
        m = re.search(r"\\textbf\{([^}]+)\}", r)
        if not m:
            continue
        label = m.group(1).strip().lower()
        if label.startswith("training"):
            return "training"
        if label.startswith("validation"):
            return "validation"
        if label.startswith("test"):
            return "test"
    return None


# ----- per-clade parsers ------------------------------------------------

def _parse_pair_table(
    tex_path: Path,
    annotation_label_map: dict[str, str],
    manual_predicate=None,
):
    """Parser for tables whose row has TWO species pairs (10 cells)
    or just plain rows. Each row:

        species & acc & gen_mb & annot & ngenes & species2 & acc2 & gen_mb2 & annot2 & ngenes2

    The diatoms / chlorophyta / insecta tables match this shape.
    """
    tex = tex_path.read_text(encoding="utf-8")
    split: str | None = None
    out: dict[str, list[dict]] = {"training": [], "validation": [], "test": []}
    manual: list[dict] = []

    for row in _iter_table_body(tex):
        cells = _split_cells(row)
        # Section header rows look like `\multicolumn{...}{l}{\textbf{Training}}`.
        header = _emit_section([row])
        if header:
            split = header
            continue
        if split is None:
            continue
        # Strip the \hline residue from the last cell of any row.
        cells = [re.sub(r"\\hline\b", "", c).strip() for c in cells]
        # A "pair" row has either 5 or 10 cells (one or two species per row).
        pairs = []
        if len(cells) >= 5:
            pairs.append(cells[:5])
        if len(cells) >= 10:
            pairs.append(cells[5:10])

        for cells5 in pairs:
            sp_cell, acc_cell, _gen, annot_cell, _ng = cells5
            sp = _normalise_species(sp_cell)
            if not sp or " " not in sp:
                continue
            acc = _strip_latex(acc_cell)
            acc_match = ACCESSION_RE.search(acc)
            annot_raw = _strip_latex(annot_cell)

            row_dict = {
                "species": sp,
                "accession_raw": acc,
                "annotation_raw": annot_raw,
                "split": split,
            }

            if manual_predicate and manual_predicate(row_dict):
                manual.append(row_dict)
                continue

            pipeline_label = annotation_label_map.get(annot_raw)
            if pipeline_label is None:
                manual.append(row_dict
                              | {"reason": f"unmapped annotation '{annot_raw}'"})
                continue

            if pipeline_label in ("RefSeq",):
                if not acc_match:
                    manual.append(row_dict | {"reason": "no GCA_/GCF_ accession"})
                    continue
                accession = acc_match.group(0)
            else:
                # BRAKER / Phytozome staged – accession column is informational.
                accession = acc_match.group(0) if acc_match else _sanitise_accession(acc)

            out[split].append({
                "species": sp,
                "accession": accession,
                "annotation": pipeline_label,
            })

    return out, manual


def parse_chlorophyta(tex_path: Path):
    return _parse_pair_table(
        tex_path,
        annotation_label_map={
            "RefSeq":  "RefSeq",
            "Genbank": "RefSeq",
            "ensembl": "RefSeq",
            "DDBJ":    "RefSeq",
        },
    )


def parse_diatoms(tex_path: Path):
    # All diatoms species (train/val/test, including *T. pseudonana* and
    # *P. tricornutum*) are pre-staged on brain under
    # /home/nas-hs/projs/Tiberius_diatoms/tiberius_dataprep/data/{annotations,genomes}/
    # so every row uses the BRAKER staging branch in fetch.nf. The
    # source-column distinction in the tex table (BRAKER vs EMBL vs DDBJ vs
    # Genbank vs AUGUSTUS/Geneious) is preserved in scripts/data/diatoms_file_map.tsv
    # but doesn't affect the pipeline.
    return _parse_pair_table(
        tex_path,
        annotation_label_map={
            "BRAKER":            "BRAKER",
            "EMBL":              "BRAKER",
            "DDBJ":              "BRAKER",
            "Genbank":           "BRAKER",
            "AUGUSTUS/Geneious": "BRAKER",
            "ENSEMBL":           "BRAKER",
        },
    )


def parse_fungi(tex_path: Path):
    """Fungi table omits the annotation column – all GCF accessions, RefSeq."""
    tex = tex_path.read_text(encoding="utf-8")
    split: str | None = None
    out: dict[str, list[dict]] = {"training": [], "validation": [], "test": []}
    manual: list[dict] = []

    for row in _iter_table_body(tex):
        header = _emit_section([row])
        if header:
            split = header
            continue
        if split is None:
            continue
        cells = _split_cells(row)
        cells = [re.sub(r"\\hline\b", "", c).strip() for c in cells]

        pairs = []
        if len(cells) >= 4:
            pairs.append(cells[:4])
        if len(cells) >= 8:
            pairs.append(cells[4:8])

        for cells4 in pairs:
            sp_cell, acc_cell, _gen, _ng = cells4
            sp = _normalise_species(sp_cell)
            if not sp or " " not in sp:
                continue
            acc = _strip_latex(acc_cell)
            acc_match = ACCESSION_RE.search(acc)
            if not acc_match:
                manual.append({
                    "species": sp, "accession_raw": acc,
                    "annotation_raw": "RefSeq",
                    "split": split,
                    "reason": "no GCA_/GCF_ accession",
                })
                continue
            out[split].append({
                "species": sp,
                "accession": acc_match.group(0),
                "annotation": "RefSeq",
            })
    return out, manual


def parse_embryophyta(tex_path: Path):
    """Embryophyta (Mesangiospermae) table is single-column-per-row with
    columns: Species & Accession & GenomeMb & #Genes & AnnotationSource & Ref & DOI.
    Rows whose annotation source isn't NCBI are staged from the Figshare bundle
    (annotation label `Phytozome`)."""
    tex = tex_path.read_text(encoding="utf-8")
    split: str | None = None
    out: dict[str, list[dict]] = {"training": [], "validation": [], "test": []}
    manual: list[dict] = []

    section_map = {
        "training":   "training",
        "validation": "validation",
        "testing":    "test",
        "test":       "test",
    }

    for row in _iter_table_body(tex):
        cells = _split_cells(row)
        cells = [re.sub(r"\\hline\b", "", c).strip() for c in cells]
        # Section headers in this table use `\textbf{Training}` without
        # \multicolumn{} – pick them up too.
        header = re.search(r"\\textbf\{(Training|Validation|Testing|Test)\}",
                           " & ".join(cells), re.IGNORECASE)
        if header:
            split = section_map[header.group(1).lower()]
            continue
        if split is None:
            continue
        if len(cells) < 5:
            continue

        sp_cell, acc_cell, _gen, _ng, annot_cell = cells[:5]
        sp = _normalise_species(sp_cell)
        if not sp or " " not in sp:
            continue
        acc_raw = _strip_latex(acc_cell)
        annot_raw = _strip_latex(annot_cell)

        # All Embryophyta rows are staged from the Figshare bundle on brain
        # (Phytozome / OSU / GDR / NCBI alike) – the bundle is the single
        # reproducible source for the published Mesangiospermae model.
        # Keep the original accession token (Phytozome ID, GCA_/GCF_, or
        # "Figshare" placeholder) for documentation only.
        acc_match = ACCESSION_RE.search(acc_raw)
        accession = acc_match.group(0) if acc_match else _sanitise_accession(
            acc_raw if acc_raw and acc_raw != "-" else "Figshare"
        )
        out[split].append({
            "species": sp,
            "accession": accession,
            "annotation": "Phytozome",
        })
    return out, manual


# ----- writer -----------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["species", "accession", "annotation"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_manual(manual: list[dict], path: Path) -> None:
    if not manual:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in manual for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in manual:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = p.parse_args()

    jobs = [
        ("chlorophyta", args.tables_dir / "chlorophyta_data_info.tex", parse_chlorophyta),
        ("diatoms",     args.tables_dir / "diatoms_data_info.tex",     parse_diatoms),
        ("fungi",       args.tables_dir / "fungi_data_info.tex",       parse_fungi),
        ("embryophyta", args.tables_dir / "embryophyta_data_info.tex", parse_embryophyta),
    ]

    for clade, tex_path, parse_fn in jobs:
        if not tex_path.exists():
            print(f"[skip] {tex_path} not found")
            continue
        out, manual = parse_fn(tex_path)
        clade_dir = args.outdir / clade
        for split, rows in out.items():
            _write_csv(rows, clade_dir / f"species_{split}.csv")
            print(f"[{clade}] {split}: {len(rows):3d} rows -> {clade_dir / f'species_{split}.csv'}")
        if manual:
            _write_manual(manual, clade_dir / "manual_review.csv")
            print(f"[{clade}] manual_review.csv: {len(manual)} rows need attention")


if __name__ == "__main__":
    main()
