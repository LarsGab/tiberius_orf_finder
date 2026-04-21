r"""Parse the insecta_data_info.tex table into per-split species lists.

Table sections are introduced by
    \multicolumn{10}{l}{\textbf{Training}}  (or Test / Validation)
Each subsequent data row contains two species, each encoded as
    \textit{Genus species} & GCA\_... & size & annot & genes
up to the next `\hline` that introduces the next section.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


SPLIT_HEADER = re.compile(
    r"\\multicolumn\{10\}\{l\}\{\\textbf\{(Training|Test|Validation)\}\}"
)
SPECIES_CELL = re.compile(
    r"\\textit\{([^}]+)\}\s*&\s*([A-Z]{3}\\_[0-9]+\.[0-9]+)\s*&\s*[^&]+&\s*([A-Za-z]+)"
)


@dataclass(frozen=True)
class Species:
    name: str           # "Aedes aegypti"
    accession: str      # "GCF_002204515.2"
    annotation: str     # "RefSeq" or "BRAKER"
    split: str          # "training" | "val" | "test"

    @property
    def underscored(self) -> str:
        return self.name.replace(" ", "_")


_SPLIT_MAP = {"Training": "training", "Test": "test", "Validation": "val"}


def parse_species_table(tex_path: Path) -> list[Species]:
    text = Path(tex_path).read_text(encoding="utf-8")
    species: list[Species] = []
    current_split: str | None = None

    for raw_line in text.splitlines():
        hdr = SPLIT_HEADER.search(raw_line)
        if hdr:
            current_split = _SPLIT_MAP[hdr.group(1)]
            continue
        if current_split is None:
            continue
        for m in SPECIES_CELL.finditer(raw_line):
            name = m.group(1).strip()
            accession = m.group(2).replace(r"\_", "_")
            annotation = m.group(3).strip()
            species.append(
                Species(name=name, accession=accession,
                        annotation=annotation, split=current_split)
            )
    return _dedup_val_against_other_splits(species)


def _dedup_val_against_other_splits(species: list[Species]) -> list[Species]:
    """Drop any val species whose name also appears in test or training."""
    non_val_names = {s.name for s in species if s.split != "val"}
    return [s for s in species if not (s.split == "val" and s.name in non_val_names)]


def write_csvs(species: list[Species], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("training", "val", "test"):
        rows = [s for s in species if s.split == split]
        p = out_dir / f"species_{split}.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["species", "accession", "annotation"])
            for s in rows:
                w.writerow([s.name, s.accession, s.annotation])
        paths[split] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tex", type=Path, help="insecta_data_info.tex")
    ap.add_argument("out_dir", type=Path, help="directory for species_*.csv")
    args = ap.parse_args(argv)

    species = parse_species_table(args.tex)
    paths = write_csvs(species, args.out_dir)
    for split, p in paths.items():
        n = sum(1 for s in species if s.split == split)
        print(f"{split}: {n} species -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
