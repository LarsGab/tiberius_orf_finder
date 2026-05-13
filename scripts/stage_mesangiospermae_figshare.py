#!/usr/bin/env python3
"""
One-time staging of the Tiberius Mesangiospermae Figshare bundle into a
`<Genus_species>/{genome.fa, annotation.gff}` layout that
`nextflow/modules/fetch.nf` (`Phytozome` branch) can consume.

Run **on brain** after extracting the Figshare bundle
(doi:10.6084/m9.figshare.32086728) under e.g.
`/home/gabriell/tiberius_mesangiospermae_training_data`:

    {root}/{train,validation,test}/{annotations,genomes}/*.gff3,*.genome.fa

This script reads `scripts/data/embryophyta_file_map.tsv` (the explicit
species → file-stem mapping derived from the bundle's per-split listings)
and creates symlinks at:

    {root}/by_species/<Genus_species>/genome.fa
                                  /annotation.gff

The `Phytozome` branch in fetch.nf points `--phytozome_data_dir` at
`{root}/by_species/`. Symlinks rather than copies keep this idempotent
and cheap.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "scripts" / "data" / "embryophyta_file_map.tsv"

SPLIT_DIR_NAMES = {
    "training":   "train",
    "validation": "validation",
    "test":       "test",
}


def _resolve_sources(root: Path, split: str, file_stem: str) -> tuple[Path, Path]:
    split_dir = root / SPLIT_DIR_NAMES[split]
    genome = split_dir / "genomes"     / f"{file_stem}.genome.fa"
    annot  = split_dir / "annotations" / f"{file_stem}.gff3"
    return genome, annot


def _symlink(src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dry_run:
            print(f"  [exists] {dst}")
            return
        dst.unlink()
    if dry_run:
        print(f"  [link ] {dst} -> {src}")
        return
    os.symlink(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True,
                   help="Figshare extract root (contains train/, validation/, test/).")
    p.add_argument("--out-subdir", default="by_species",
                   help="Per-species symlink dir created under --root.")
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.root.is_dir():
        sys.exit(f"root not found: {args.root}")
    if not args.map.is_file():
        sys.exit(f"map not found: {args.map}")

    out_root = args.root / args.out_subdir
    missing: list[str] = []
    staged = 0

    with args.map.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            split = row["split"].strip()
            species = row["species"].strip()
            stem = row["file_stem"].strip()
            if split not in SPLIT_DIR_NAMES:
                sys.exit(f"unknown split '{split}' for {species}")

            genome_src, annot_src = _resolve_sources(args.root, split, stem)
            for src in (genome_src, annot_src):
                if not src.exists():
                    missing.append(f"{species}\t{split}\t{src}")

            dst_dir = out_root / species.replace(" ", "_")
            print(f"[{split:10s}] {species}")
            _symlink(genome_src, dst_dir / "genome.fa",     args.dry_run)
            _symlink(annot_src,  dst_dir / "annotation.gff", args.dry_run)
            staged += 1

    print()
    print(f"staged {staged} species into {out_root}")
    if missing:
        print(f"WARNING: {len(missing)} source files missing:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
