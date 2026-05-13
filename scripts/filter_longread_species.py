"""Filter a species CSV to those that actually have long-read RNA-seq in SRA.

For each row of the input CSV, query NCBI SRA via ``varus.runlist.fetch_runlist``
with ``longreads=True``. Keep the row only if at least one PacBio SMRT or Oxford
Nanopore RNA-seq run is found. Write the filtered rows to ``--out-csv`` (same
header as the input).

Network-only; runs in a few minutes for ~100 species on a login node.

Usage:
    python scripts/filter_longread_species.py \\
        --in-csv  nextflow/conf/species_training.csv \\
        --out-csv nextflow/conf/species_training_longread.csv \\
        [--tmpdir /tmp/varus_runlist_filter] \\
        [--email you@host] [--api-key XXX]

Requires the new VARUS v2 package importable in the active env (e.g.
`pip install -e /home/gabriell/VARUS[align]` inside the `orffinder` env).
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from varus.runlist import fetch_runlist


def _count_runs(tsv_path: Path) -> int:
    """Return number of run rows (excluding header) in a Runlist.tsv."""
    if not tsv_path.is_file():
        return 0
    with tsv_path.open() as fh:
        return max(0, sum(1 for _ in fh) - 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-csv", required=True, type=Path,
                    help="Input species CSV (columns: species,accession,annotation).")
    ap.add_argument("--out-csv", required=True, type=Path,
                    help="Output CSV with only species that have long-read RNA-seq.")
    ap.add_argument("--tmpdir", type=Path, default=None,
                    help="Where to stage per-species Runlist.tsv files (default: a fresh temp dir).")
    ap.add_argument("--email", default=None,
                    help="NCBI Entrez contact email (or $NCBI_EMAIL).")
    ap.add_argument("--api-key", default=None,
                    help="NCBI API key (or $NCBI_API_KEY); raises rate limit to 10 req/s.")
    ap.add_argument("--keep-tmpdir", action="store_true",
                    help="Don't delete the temp dir on exit.")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.in_csv.is_file():
        sys.exit(f"input CSV not found: {args.in_csv}")

    tmp_root = args.tmpdir or Path(tempfile.mkdtemp(prefix="varus_runlist_filter_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    cleanup_tmp = (args.tmpdir is None and not args.keep_tmpdir)

    try:
        with args.in_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        if "species" not in fieldnames:
            sys.exit(f"input CSV missing required 'species' column: {args.in_csv}")

        keep: list[dict] = []
        for i, row in enumerate(rows, 1):
            species = (row.get("species") or "").strip()
            if not species:
                print(f"[{i}/{len(rows)}] SKIP  (empty species cell)", flush=True)
                continue
            sp_dir = tmp_root / species.replace(" ", "_")
            sp_dir.mkdir(parents=True, exist_ok=True)
            try:
                fetch_runlist(
                    species=species,
                    outdir=sp_dir,
                    longreads=True,
                    email=args.email,
                    api_key=args.api_key,
                )
                n = _count_runs(sp_dir / "Runlist.tsv")
            except Exception as e:
                n = 0
                err = f" ({type(e).__name__}: {e})"
            else:
                err = ""
            if n > 0:
                print(f"[{i}/{len(rows)}] KEEP  {species}  ({n} runs)", flush=True)
                keep.append(row)
            else:
                print(f"[{i}/{len(rows)}] DROP  {species}{err}", flush=True)

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(keep)

        print(f"\n[summary] kept {len(keep)}/{len(rows)} species -> {args.out_csv}", flush=True)
    finally:
        if cleanup_tmp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
