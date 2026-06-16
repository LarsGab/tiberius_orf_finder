"""Post-crash integrity check for per-species VARUS.bam and assembly files.

After the brain cluster crash we need to decide, for each species under
``--outdir``, whether the previously produced VARUS.bam plus the staged
assembly (``genome.fa`` + ``annotation.gff``) are still usable for
retraining/re-running TFRecords, or whether the species must be re-fetched
and re-VARUS'd from scratch.

Layout assumed (matches nextflow/main.nf publishDir conventions)::

    <outdir>/<Genus_species>/assembly/genome.fa[.gz]
    <outdir>/<Genus_species>/assembly/annotation.gff
    <outdir>/<Genus_species>/varus/VARUS.bam

What is checked, per species:

  genome.fa[.gz]
    - exists, size > 0  (either genome.fa or genome.fa.gz is accepted;
      gzipped form is read transparently via Python's gzip module)
    - first non-blank line starts with '>'
    - number of contigs and total residue length

  annotation.gff
    - exists, size > 0
    - every non-comment line has >= 9 tab-separated columns
    - count of 'gene' / 'mRNA' / 'CDS' / 'exon' features

  VARUS.bam
    - exists, size > 0
    - ``samtools quickcheck`` passes (header + EOF block intact -> not
      truncated by the crash)
    - ``samtools view -c`` returns a read count (header + index sane
      enough to scan)
    - ``samtools flagstat`` mapping rate (sanity floor configurable via
      ``--min-map-rate``)

Species selection:
  * default: every immediate subdirectory of --outdir,
  * or restrict to the species listed in --species-csv (the same CSV the
    Nextflow pipeline consumes).

Output:
  * a TSV (``--out-tsv``) with one row per species and one column per
    check, plus an OVERALL status column ('ok' | 'reuse_ok_no_bam'
    | 'bad'),
  * a stderr summary,
  * exit code 1 if any species ends up 'bad' (so this can gate a
    re-run wrapper).

Run on brain inside the ``orffinder`` micromamba env (samtools must be
on PATH).

Example::

    micromamba run -n orffinder python scripts/check_varus_assembly_integrity.py \\
        --outdir /home/gabriell/tiberius_orf_finder/results/training_vertebrates \\
        --species-csv nextflow/conf/vertebrates/species_training.csv \\
        --out-tsv     results/training_vertebrates/integrity_check.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _open_text(path: Path) -> io.TextIOBase:
    """Open ``path`` for text reading, transparently decompressing .gz."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _resolve_genome(assembly_dir: Path) -> Path | None:
    """Return whichever of genome.fa / genome.fa.gz exists (plain preferred)."""
    plain = assembly_dir / "genome.fa"
    gz    = assembly_dir / "genome.fa.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    return None


# ---------------------------------------------------------------- helpers


def _which_samtools() -> str:
    exe = shutil.which("samtools")
    if exe is None:
        sys.exit("samtools not found on PATH. Activate the orffinder env first.")
    return exe


def _run(cmd: list[str], timeout: int) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


# ---------------------------------------------------------------- checks


@dataclass
class SpeciesReport:
    species: str
    species_dir: Path
    genome_status: str = "missing"
    genome_bytes: int = 0
    genome_contigs: int = 0
    genome_residues: int = 0
    gff_status: str = "missing"
    gff_bytes: int = 0
    gff_genes: int = 0
    gff_mrnas: int = 0
    gff_cds: int = 0
    bam_status: str = "missing"
    bam_bytes: int = 0
    bam_reads: int = 0
    bam_mapped_pct: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def overall(self) -> str:
        genome_ok = self.genome_status == "ok"
        gff_ok = self.gff_status == "ok"
        bam_ok = self.bam_status == "ok"
        if genome_ok and gff_ok and bam_ok:
            return "ok"
        # Assembly intact but BAM lost/corrupt: still useful — only VARUS
        # needs re-running, not the genome fetch.
        if genome_ok and gff_ok and self.bam_status in {"missing", "truncated", "empty", "unreadable"}:
            return "reuse_assembly_redo_varus"
        return "bad"


def _check_fasta(path: Path | None, rep: SpeciesReport) -> None:
    if path is None or not path.exists():
        rep.genome_status = "missing"
        return
    rep.genome_bytes = path.stat().st_size
    if rep.genome_bytes == 0:
        rep.genome_status = "empty"
        return
    contigs = 0
    residues = 0
    first_nonblank_checked = False
    try:
        with _open_text(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                if not first_nonblank_checked:
                    if not line.startswith(">"):
                        rep.genome_status = "not_fasta"
                        return
                    first_nonblank_checked = True
                if line.startswith(">"):
                    contigs += 1
                else:
                    residues += len(line.strip())
    except (OSError, gzip.BadGzipFile) as e:
        rep.genome_status = "unreadable"
        rep.notes.append(f"genome read error: {e}")
        return
    rep.genome_contigs = contigs
    rep.genome_residues = residues
    if contigs == 0 or residues == 0:
        rep.genome_status = "empty_records"
        return
    rep.genome_status = "ok"


def _check_gff(path: Path, rep: SpeciesReport) -> None:
    if not path.exists():
        rep.gff_status = "missing"
        return
    rep.gff_bytes = path.stat().st_size
    if rep.gff_bytes == 0:
        rep.gff_status = "empty"
        return
    genes = mrnas = cds = 0
    data_rows = 0
    bad_rows = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line or line.startswith("#"):
                    continue
                if not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9:
                    bad_rows += 1
                    continue
                data_rows += 1
                t = f[2]
                if t == "gene":
                    genes += 1
                elif t in ("mRNA", "transcript"):
                    mrnas += 1
                elif t == "CDS":
                    cds += 1
    except OSError as e:
        rep.gff_status = "unreadable"
        rep.notes.append(f"gff read error: {e}")
        return
    rep.gff_genes = genes
    rep.gff_mrnas = mrnas
    rep.gff_cds = cds
    if data_rows == 0:
        rep.gff_status = "no_features"
        return
    if bad_rows:
        rep.notes.append(f"gff has {bad_rows} malformed row(s)")
    # cds is the load-bearing feature for downstream label projection.
    if cds == 0:
        rep.gff_status = "no_cds"
        return
    rep.gff_status = "ok"


def _check_bam(
    path: Path,
    rep: SpeciesReport,
    samtools: str,
    *,
    quickcheck_timeout: int,
    count_timeout: int,
    flagstat_timeout: int,
    min_map_rate: float,
) -> None:
    if not path.exists():
        rep.bam_status = "missing"
        return
    rep.bam_bytes = path.stat().st_size
    if rep.bam_bytes == 0:
        rep.bam_status = "empty"
        return

    # quickcheck: header readable + EOF block present. This is the
    # canonical "is the BAM truncated after a crash" test.
    rc, _, err = _run([samtools, "quickcheck", "-v", str(path)],
                      timeout=quickcheck_timeout)
    if rc != 0:
        rep.bam_status = "truncated"
        if err.strip():
            rep.notes.append(f"quickcheck: {err.strip()}")
        return

    # view -c: scans through the whole BAM. If quickcheck passed but
    # view -c blows up, the body is damaged (rare but possible).
    try:
        rc, out, err = _run([samtools, "view", "-c", str(path)],
                            timeout=count_timeout)
    except subprocess.TimeoutExpired:
        rep.bam_status = "scan_timeout"
        rep.notes.append(f"samtools view -c timed out (>{count_timeout}s)")
        return
    if rc != 0:
        rep.bam_status = "unreadable"
        rep.notes.append(f"view -c rc={rc}: {err.strip()[:200]}")
        return
    try:
        rep.bam_reads = int(out.strip())
    except ValueError:
        rep.bam_status = "unreadable"
        rep.notes.append(f"view -c gave non-int: {out.strip()[:80]}")
        return
    if rep.bam_reads == 0:
        rep.bam_status = "no_reads"
        return

    # flagstat: mapping rate. A BAM with reads but ~0% mapped is
    # typically still usable but worth flagging.
    try:
        rc, out, _ = _run([samtools, "flagstat", str(path)],
                          timeout=flagstat_timeout)
    except subprocess.TimeoutExpired:
        rep.notes.append("flagstat timed out (skipping mapping rate)")
        rep.bam_status = "ok"
        return
    if rc != 0:
        rep.notes.append("flagstat failed (skipping mapping rate)")
        rep.bam_status = "ok"
        return
    for line in out.splitlines():
        # samtools >= 1.13 line: "12345 + 0 mapped (90.12% : N/A)"
        if " mapped (" in line and "primary mapped" not in line:
            try:
                pct = line.split("(", 1)[1].split("%", 1)[0]
                rep.bam_mapped_pct = float(pct)
            except (IndexError, ValueError):
                pass
            break
    if rep.bam_mapped_pct < min_map_rate:
        rep.notes.append(
            f"low mapping rate {rep.bam_mapped_pct:.1f}% < {min_map_rate:.1f}%"
        )
    rep.bam_status = "ok"


# ---------------------------------------------------------------- driver


def _species_dirs(outdir: Path, species_csv: Path | None) -> list[tuple[str, Path]]:
    """Return [(display_species, species_dir)] in stable order."""
    if species_csv is not None:
        out: list[tuple[str, Path]] = []
        with species_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "species" not in reader.fieldnames:
                sys.exit(f"--species-csv {species_csv} missing 'species' column")
            for row in reader:
                sp = row["species"].strip()
                if not sp:
                    continue
                d = outdir / sp.replace(" ", "_")
                out.append((sp, d))
        return out
    # Fallback: every subdir of outdir.
    return sorted(
        ((p.name.replace("_", " "), p) for p in outdir.iterdir() if p.is_dir()),
        key=lambda x: x[0],
    )


_COLS = [
    "species", "overall",
    "genome_status", "genome_bytes", "genome_contigs", "genome_residues",
    "gff_status",    "gff_bytes",    "gff_genes",      "gff_mrnas", "gff_cds",
    "bam_status",    "bam_bytes",    "bam_reads",      "bam_mapped_pct",
    "notes",
]


def _row(rep: SpeciesReport) -> dict[str, object]:
    return {
        "species":         rep.species,
        "overall":         rep.overall,
        "genome_status":   rep.genome_status,
        "genome_bytes":    rep.genome_bytes,
        "genome_contigs":  rep.genome_contigs,
        "genome_residues": rep.genome_residues,
        "gff_status":      rep.gff_status,
        "gff_bytes":       rep.gff_bytes,
        "gff_genes":       rep.gff_genes,
        "gff_mrnas":       rep.gff_mrnas,
        "gff_cds":         rep.gff_cds,
        "bam_status":      rep.bam_status,
        "bam_bytes":       rep.bam_bytes,
        "bam_reads":       rep.bam_reads,
        "bam_mapped_pct":  f"{rep.bam_mapped_pct:.2f}",
        "notes":           "; ".join(rep.notes),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Per-species training results dir (the same path "
                         "passed as --outdir to nextflow/main.nf).")
    ap.add_argument("--species-csv", type=Path, default=None,
                    help="Optional: restrict to species listed in this CSV "
                         "(columns: species,accession,annotation). Missing "
                         "species dirs are still reported as 'bad'.")
    ap.add_argument("--out-tsv", type=Path, default=None,
                    help="Write per-species report here (TSV).")
    ap.add_argument("--min-map-rate", type=float, default=20.0,
                    help="Mapping-rate floor (%%) below which BAM is flagged "
                         "in notes (not failed). Default 20.")
    ap.add_argument("--quickcheck-timeout", type=int, default=60)
    ap.add_argument("--count-timeout", type=int, default=1800,
                    help="samtools view -c timeout, seconds. Default 1800.")
    ap.add_argument("--flagstat-timeout", type=int, default=600)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.outdir.is_dir():
        sys.exit(f"--outdir not a directory: {args.outdir}")

    samtools = _which_samtools()
    species = _species_dirs(args.outdir, args.species_csv)
    if not species:
        sys.exit(f"no species found under {args.outdir}")

    print(f"checking {len(species)} species under {args.outdir}", flush=True)

    reports: list[SpeciesReport] = []
    for sp, d in species:
        rep = SpeciesReport(species=sp, species_dir=d)
        if not d.is_dir():
            rep.notes.append(f"species dir missing: {d}")
            reports.append(rep)
            print(f"  [miss] {sp}: no dir", flush=True)
            continue
        _check_fasta(_resolve_genome(d / "assembly"), rep)
        _check_gff(d / "assembly" / "annotation.gff", rep)
        _check_bam(
            d / "varus" / "VARUS.bam", rep, samtools,
            quickcheck_timeout=args.quickcheck_timeout,
            count_timeout=args.count_timeout,
            flagstat_timeout=args.flagstat_timeout,
            min_map_rate=args.min_map_rate,
        )
        reports.append(rep)
        print(
            f"  {sp:35s}  genome={rep.genome_status:<13s}"
            f"  gff={rep.gff_status:<11s}"
            f"  bam={rep.bam_status:<12s}"
            f"  reads={rep.bam_reads:>10d}"
            f"  map={rep.bam_mapped_pct:5.1f}%"
            f"  -> {rep.overall}",
            flush=True,
        )

    # ---------------------------------------------------------- summary
    n_ok    = sum(1 for r in reports if r.overall == "ok")
    n_reuse = sum(1 for r in reports if r.overall == "reuse_assembly_redo_varus")
    n_bad   = sum(1 for r in reports if r.overall == "bad")
    print(file=sys.stderr)
    print(
        f"SUMMARY  ok={n_ok}  reuse_assembly_redo_varus={n_reuse}  bad={n_bad}  "
        f"total={len(reports)}",
        file=sys.stderr,
    )

    if args.out_tsv is not None:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_tsv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_COLS, delimiter="\t",
                               extrasaction="ignore")
            w.writeheader()
            for r in reports:
                w.writerow(_row(r))
        print(f"wrote {args.out_tsv}", file=sys.stderr)

    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
