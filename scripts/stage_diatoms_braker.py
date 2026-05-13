#!/usr/bin/env python3
"""
One-time staging of the diatoms (Bacillariophyta) BRAKER data dir on brain
into a `<Genus_species>/{genome.fa, annotation.gff}` symlink layout that the
`BRAKER` branch of `nextflow/modules/fetch.nf` can consume after the
"`genome.fa` then legacy `_renamed.fna`" fallback.

Source layout on brain:

  /home/nas-hs/projs/Tiberius_diatoms/tiberius_dataprep/data/genomes/
      <File_stem>.genome.fa
  /home/nas-hs/projs/Tiberius_diatoms/tiberius_dataprep/data/annotations/
      <File_stem>.gtf

Target layout (created here):

  <root>/by_species/<Genus_species>/genome.fa      -> <Genomes>/<File_stem>.genome.fa
  <root>/by_species/<Genus_species>/annotation.gff -> <Annotations>/<File_stem>.gtf

The `--braker_data_dir` Nextflow param then points at `<root>/by_species/`.
The file-stems often differ from the species name by typos in the source dir
(e.g. *Phaeodactylum* -> `Phaeodactylcum`); see
`scripts/data/diatoms_file_map.tsv` for the explicit mapping.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "scripts" / "data" / "diatoms_file_map.tsv"


def _symlink(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        marker = "exists" if dst.is_symlink() or dst.exists() else "link "
        print(f"  [{marker}] {dst} -> {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--genomes-dir", type=Path,
        default=Path("/home/nas-hs/projs/Tiberius_diatoms/tiberius_dataprep/data/genomes"),
    )
    p.add_argument(
        "--annotations-dir", type=Path,
        default=Path("/home/nas-hs/projs/Tiberius_diatoms/tiberius_dataprep/data/annotations"),
    )
    p.add_argument("--out", type=Path, required=True,
                   help="Output root for the by_species/ layout (e.g. "
                        "/home/gabriell/tiberius_diatoms_staged/by_species).")
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.genomes_dir.is_dir():
        sys.exit(f"genomes dir not found: {args.genomes_dir}")
    if not args.annotations_dir.is_dir():
        sys.exit(f"annotations dir not found: {args.annotations_dir}")
    if not args.map.is_file():
        sys.exit(f"map not found: {args.map}")

    missing: list[str] = []
    staged = 0

    with args.map.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            split = row["split"].strip()
            species = row["species"].strip()
            stem = row["file_stem"].strip()

            genome_src = args.genomes_dir / f"{stem}.genome.fa"
            annot_src  = args.annotations_dir / f"{stem}.gtf"
            for src in (genome_src, annot_src):
                if not src.exists():
                    missing.append(f"{species}\t{split}\t{src}")

            dst_dir = args.out / species.replace(" ", "_")
            print(f"[{split:10s}] {species}")
            _symlink(genome_src, dst_dir / "genome.fa",      args.dry_run)
            _symlink(annot_src,  dst_dir / "annotation.gff", args.dry_run)
            staged += 1

    print()
    print(f"staged {staged} species into {args.out}")
    if missing:
        print(f"WARNING: {len(missing)} source files missing:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
