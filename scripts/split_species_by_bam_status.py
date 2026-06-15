"""Split a species CSV by VARUS.bam integrity status.

Consumes:
  * the TSV report from ``check_varus_assembly_integrity.py``
    (columns include ``species`` and ``bam_status``), and
  * the original species CSV (``species,accession,annotation``) that
    drove the pre-crash run.

Emits two CSVs into ``--out-dir``, both with the same schema as the
input CSV:

  * ``reuse_bam.csv``    -- species whose BAM has ``bam_status == ok``.
                           These will go through a reuse-BAM Nextflow
                           entry point (assembly is re-fetched fresh,
                           BAM is reused from /home_old).
  * ``from_scratch.csv`` -- everyone else (BAM missing / empty /
                           truncated / unreadable / no_reads, *or*
                           species absent from the integrity report).
                           These go through the full short-read v2
                           pipeline (assembly fetch + new VARUS run).

Selection is by ``bam_status``, not by the ``overall`` column: the
assemblies are easy to re-download, so we deliberately ignore the
genome/gff status when choosing what to reuse.

Example::

    python scripts/split_species_by_bam_status.py \\
        --integrity-tsv results/integrity_checks/vertebrates_20260615.tsv \\
        --species-csv   nextflow/conf/species_vertebrates_training.csv \\
        --out-dir       results/integrity_checks/split_vertebrates
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--integrity-tsv", type=Path, required=True,
                    help="TSV from check_varus_assembly_integrity.py "
                         "(must have 'species' and 'bam_status' cols).")
    ap.add_argument("--species-csv", type=Path, required=True,
                    help="Original species CSV (species,accession,annotation).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write reuse_bam.csv + from_scratch.csv.")
    ap.add_argument("--reuse-status", default="ok",
                    help="bam_status value(s) considered reusable. Comma-"
                         "separated, default 'ok'.")
    return ap.parse_args(argv)


def _read_integrity(path: Path) -> dict[str, str]:
    """species -> bam_status."""
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            sys.exit(f"{path}: no header")
        missing = {"species", "bam_status"} - set(reader.fieldnames)
        if missing:
            sys.exit(f"{path}: missing column(s): {sorted(missing)}")
        for row in reader:
            sp = (row.get("species") or "").strip()
            if sp:
                out[sp] = (row.get("bam_status") or "").strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reuse_states = {s.strip() for s in args.reuse_status.split(",") if s.strip()}

    bam_status = _read_integrity(args.integrity_tsv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reuse_path = args.out_dir / "reuse_bam.csv"
    scratch_path = args.out_dir / "from_scratch.csv"

    with args.species_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            sys.exit(f"{args.species_csv}: no header")
        if "species" not in reader.fieldnames:
            sys.exit(f"{args.species_csv}: missing 'species' column")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    n_reuse = n_scratch = n_unreported = 0
    with reuse_path.open("w", newline="") as fr, scratch_path.open("w", newline="") as fs:
        wr = csv.DictWriter(fr, fieldnames=fieldnames)
        ws = csv.DictWriter(fs, fieldnames=fieldnames)
        wr.writeheader()
        ws.writeheader()
        for row in rows:
            sp = row["species"].strip()
            status = bam_status.get(sp)
            if status is None:
                n_unreported += 1
                ws.writerow(row)
                continue
            if status in reuse_states:
                wr.writerow(row)
                n_reuse += 1
            else:
                ws.writerow(row)
                n_scratch += 1

    n_total = n_reuse + n_scratch + n_unreported
    print(
        f"SUMMARY  reuse_bam={n_reuse}  from_scratch={n_scratch} "
        f"(of which not_in_integrity_report={n_unreported})  total={n_total}",
        file=sys.stderr,
    )
    print(f"wrote {reuse_path}", file=sys.stderr)
    print(f"wrote {scratch_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
