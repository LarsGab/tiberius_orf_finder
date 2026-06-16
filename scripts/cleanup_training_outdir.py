"""Reclaim disk space from an existing Tiberius ORF finder training outdir.

For every ``<outdir>/<Genus_species>/`` subdirectory this script:

  * removes files that the (older) Nextflow pipeline published unnecessarily:
        assembly/genome.fa.fai
        varus/genome_index/
        varus/genome.fa
        varus/annotation.gff
        varus/introns.gff
        stringtie/genome.fa
        stringtie/annotation.gff

  * gzips remaining large FASTAs in-place (deletes the uncompressed
    file only after the .gz is written successfully):
        assembly/genome.fa     -> assembly/genome.fa.gz
        stringtie/transcripts.fa -> stringtie/transcripts.fa.gz

The script is idempotent: re-running on a cleaned dir is a no-op.

Usage::

    python scripts/cleanup_training_outdir.py \\
        --outdir /home/gabriell/tiberius_orf_finder/results/training_vertebrates

    python scripts/cleanup_training_outdir.py --outdir <dir> --dry-run

Only stdlib is used (``gzip``, ``shutil``, ``pathlib``) so no extra deps.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path


# Per-species paths to delete outright.
REMOVE: list[Path] = [
    Path("assembly/genome.fa.fai"),
    Path("varus/genome_index"),
    Path("varus/genome.fa"),
    Path("varus/annotation.gff"),
    Path("varus/introns.gff"),
    Path("stringtie/genome.fa"),
    Path("stringtie/annotation.gff"),
]

# Per-species FASTAs to gzip (delete uncompressed original on success).
GZIP_TARGETS: list[Path] = [
    Path("assembly/genome.fa"),
    Path("stringtie/transcripts.fa"),
]


def _human(n: int) -> str:
    f = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024:
            return f"{f:.1f} {u}"
        f /= 1024
    return f"{f:.1f} PiB"


def _du(p: Path) -> int:
    """Bytes used by ``p`` (file or recursive dir). 0 if absent."""
    if p.is_symlink() or p.is_file():
        try:
            return p.lstat().st_size
        except OSError:
            return 0
    if p.is_dir():
        total = 0
        for q in p.rglob("*"):
            try:
                total += q.lstat().st_size
            except OSError:
                pass
        return total
    return 0


def _remove(p: Path, *, dry_run: bool) -> tuple[int, str | None]:
    """Delete ``p`` (file or dir). Returns (bytes_freed, log_msg_or_None)."""
    if not p.exists() and not p.is_symlink():
        return 0, None
    size = _du(p)
    if dry_run:
        return size, f"would remove: {p}  ({_human(size)})"
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()
    return size, f"removed: {p}  ({_human(size)})"


def _gzip_in_place(src: Path, *, dry_run: bool, level: int) -> tuple[int, str | None]:
    """Compress ``src`` -> ``src.gz``; delete src on success.

    Returns (bytes_freed, log_msg_or_None). Already-gzipped (.gz present,
    uncompressed gone) is a no-op.
    """
    gz = src.with_name(src.name + ".gz")

    if not src.exists():
        # Already cleaned up, or never produced.
        return 0, None

    src_size = src.stat().st_size

    if dry_run:
        return src_size, f"would gzip:  {src} -> {gz}  ({_human(src_size)})"

    # Write to a sidecar .part file first; atomic-rename on success so a
    # crashed run never leaves a half-written .gz that looks valid.
    tmp = gz.with_name(gz.name + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with src.open("rb") as fin, gzip.open(tmp, "wb", compresslevel=level) as fout:
            shutil.copyfileobj(fin, fout, length=1 << 20)
        tmp.replace(gz)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    gz_size = gz.stat().st_size
    saved = src_size - gz_size
    src.unlink()
    return saved, (
        f"gzipped:     {src} -> {gz}  "
        f"({_human(src_size)} -> {_human(gz_size)}, saved {_human(saved)})"
    )


def _species_dirs(outdir: Path) -> list[Path]:
    return sorted(p for p in outdir.iterdir() if p.is_dir())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--outdir", type=Path, required=True,
        help="Per-species training results dir (the same path that was "
             "passed as --outdir to the Nextflow pipeline).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without touching files.",
    )
    ap.add_argument(
        "--compresslevel", type=int, default=6,
        help="gzip compresslevel 1-9 (default: 6).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.outdir.is_dir():
        sys.exit(f"--outdir not a directory: {args.outdir}")
    if not 1 <= args.compresslevel <= 9:
        sys.exit("--compresslevel must be in 1..9")

    species = _species_dirs(args.outdir)
    if not species:
        sys.exit(f"no species subdirectories under {args.outdir}")

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}cleaning {len(species)} species dirs under {args.outdir}",
          flush=True)

    total_freed = 0
    n_removed = 0
    n_gzipped = 0
    for sd in species:
        per_species_msgs: list[str] = []
        for rel in REMOVE:
            saved, msg = _remove(sd / rel, dry_run=args.dry_run)
            if msg is not None:
                per_species_msgs.append(msg)
                n_removed += 1
            total_freed += saved
        for rel in GZIP_TARGETS:
            saved, msg = _gzip_in_place(
                sd / rel, dry_run=args.dry_run, level=args.compresslevel,
            )
            if msg is not None:
                per_species_msgs.append(msg)
                n_gzipped += 1
            total_freed += saved

        if per_species_msgs:
            print(f"\n# {sd.name}", flush=True)
            for m in per_species_msgs:
                print(f"  {m}", flush=True)

    verb = "would free" if args.dry_run else "freed"
    print(
        f"\n{prefix}{verb}: {_human(total_freed)}  "
        f"(removed={n_removed}, gzipped={n_gzipped}, species={len(species)})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
