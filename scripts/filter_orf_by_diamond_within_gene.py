"""Soft protein filter for ORF-finder predictions: within-gene isoform pruning.

Given a Diamond blastp TSV of ORF-finder peptides (query id =
StringTie transcript_id like ``STRG.G.I``) against a taxon-excluded
OrthoDB partition, drop isoforms of a gene when a sibling isoform of
the same gene has a Diamond hit and this one does not. Genes with no
hits at all keep every isoform (soft filter — we never remove a gene
just because it has no protein evidence).

Rule per gene ``STRG.G`` (over its isoforms ``STRG.G.*``):
  * ``hit_isoforms``    = isoforms with any Diamond hit passing thresholds
  * if ``hit_isoforms`` non-empty  → keep only hit isoforms
  * else                          → keep every isoform

Two input modes:
  (A) ``--proteins-fa`` + ``--genome``  → gffread + diamond makedb + blastp
  (B) ``--diamond-tsv``                 → reuse pre-computed Diamond TSV

Diamond outfmt is the extended 14-column layout emitted by the sister
Tiberius filter (qseqid sseqid pident length mismatch gapopen qstart qend
sstart send evalue bitscore qlen slen).

Usage (mode A)::

    python scripts/filter_orf_by_diamond_within_gene.py \\
        --orfs-gtf    orfs.filtered.gtf \\
        --genome      genome.fa \\
        --proteins-fa Vertebrata_excl_ord.fa \\
        --out-dir     results/soft_protein_filter/<species>/ \\
        --threads 8   --evalue 1e-5

Usage (mode B)::

    python scripts/filter_orf_by_diamond_within_gene.py \\
        --orfs-gtf    orfs.filtered.gtf \\
        --diamond-tsv diamond.tsv \\
        --out-dir     results/soft_protein_filter/<species>/ \\
        --evalue 1e-5
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


_TRANSCRIPT_ID_RE = re.compile(r'transcript_id "([^"]+)"')
_GENE_ID_RE       = re.compile(r'gene_id "([^"]+)"')
_GFF3_ID_RE       = re.compile(r'(?:^|;)ID=([^;]+)')
_GFF3_PARENT_RE   = re.compile(r'(?:^|;)Parent=([^;]+)')

# Diamond outfmt: 14 cols, indices below match _DIAMOND_FMT used by sister script.
_DIAMOND_FMT = (
    "qseqid sseqid pident length mismatch gapopen "
    "qstart qend sstart send evalue bitscore qlen slen"
)


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def extract_peptides(genome: Path, gtf: Path, out_fa: Path) -> Path:
    """Translate CDS in `gtf` to peptides via gffread. Strips '*' stop
    codons and '.' masked residues so Diamond doesn't reject sequences."""
    if shutil.which("gffread") is None:
        raise RuntimeError("gffread not in PATH")
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_fa.with_suffix(".tmp.fa")
    _run(["gffread", "-y", str(tmp), "-g", str(genome), str(gtf)])
    with tmp.open() as fin, out_fa.open("w") as fout:
        for line in fin:
            if line.startswith(">"):
                fout.write(line)
            else:
                fout.write(line.replace("*", "").replace(".", ""))
    tmp.unlink()
    return out_fa


def run_diamond(query_fa: Path, proteins_fa: Path, out_dir: Path,
                threads: int, evalue: float,
                sensitivity: str = "--more-sensitive",
                max_target_seqs: int = 5,
                force: bool = False) -> Path:
    if shutil.which("diamond") is None:
        raise RuntimeError("diamond not in PATH")
    out_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = out_dir / "diamond_db"
    dmnd = db_prefix.with_suffix(".dmnd")
    out_tsv = out_dir / "diamond.tsv"
    if out_tsv.exists() and out_tsv.stat().st_size > 0 and not force:
        print(f"[diamond] reusing existing {out_tsv}", flush=True)
        return out_tsv
    if not dmnd.exists() or force:
        print("[diamond] makedb", flush=True)
        _run(["diamond", "makedb",
              "--in", str(proteins_fa),
              "--db", str(db_prefix),
              "--threads", str(threads)])
    print(f"[diamond] blastp (evalue<={evalue})", flush=True)
    _run(["diamond", "blastp",
          "--query", str(query_fa),
          "--db",    str(dmnd),
          "--out",   str(out_tsv),
          "--outfmt", "6", *_DIAMOND_FMT.split(),
          "--max-target-seqs", str(max_target_seqs),
          "--evalue", str(evalue),
          sensitivity,
          "--threads", str(threads)])
    return out_tsv


def transcripts_with_hit(diamond_tsv: Path, evalue_max: float,
                         pident_min: float = 0.0,
                         bitscore_min: float = 0.0) -> set[str]:
    """Return {qseqid} that has at least one Diamond hit passing thresholds."""
    keep: set[str] = set()
    with diamond_tsv.open() as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 12:
                continue
            try:
                qseqid   = c[0]
                pident   = float(c[2])
                evalue   = float(c[10])
                bitscore = float(c[11])
            except (ValueError, IndexError):
                continue
            if evalue   > evalue_max:   continue
            if pident   < pident_min:   continue
            if bitscore < bitscore_min: continue
            keep.add(qseqid)
    return keep


def _is_gff3(gtf: Path) -> bool:
    """Return True if the file uses GFF3 attribute syntax (ID= / Parent=)."""
    for line in gtf.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if _TRANSCRIPT_ID_RE.search(line):
            return False
        cols = line.split("\t")
        if len(cols) >= 9:
            attrs = cols[8]
            if "ID=" in attrs or "Parent=" in attrs:
                return True
    return False


def _fallback_gene_id(tid: str) -> str:
    """StringTie tids look like STRG.G.I; strip the last '.I' to recover gene."""
    parts = tid.rsplit(".", 1)
    return parts[0] if len(parts) == 2 else tid


def read_gtf_gene_map(gtf: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (tid_to_gid, gid_to_tids).

    Handles both GTF (transcript_id / gene_id quoted attributes) and GFF3
    (ID= / Parent= attributes on mRNA/transcript features).
    For GTF: uses gene_id when distinct from transcript_id, else derives gene
    from transcript_id via _fallback_gene_id.
    """
    if _is_gff3(gtf):
        return _read_gff3_gene_map(gtf)
    tid_to_gid: dict[str, str] = {}
    for line in gtf.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        tm = _TRANSCRIPT_ID_RE.search(line)
        if not tm:
            continue
        tid = tm.group(1)
        gm = _GENE_ID_RE.search(line)
        gid = gm.group(1) if gm else None
        if not gid or gid == tid:
            gid = _fallback_gene_id(tid)
        tid_to_gid.setdefault(tid, gid)
    gid_to_tids: dict[str, list[str]] = defaultdict(list)
    for tid, gid in tid_to_gid.items():
        gid_to_tids[gid].append(tid)
    return tid_to_gid, gid_to_tids


def _read_gff3_gene_map(gtf: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """GFF3 variant: parse mRNA/transcript features for ID= and Parent=."""
    tid_to_gid: dict[str, str] = {}
    for line in gtf.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 9:
            continue
        if cols[2] not in ("mRNA", "transcript"):
            continue
        attrs = cols[8]
        id_m  = _GFF3_ID_RE.search(attrs)
        par_m = _GFF3_PARENT_RE.search(attrs)
        if not id_m:
            continue
        tid = id_m.group(1).strip()
        gid = par_m.group(1).strip() if par_m else _fallback_gene_id(tid)
        tid_to_gid.setdefault(tid, gid)
    gid_to_tids: dict[str, list[str]] = defaultdict(list)
    for tid, gid in tid_to_gid.items():
        gid_to_tids[gid].append(tid)
    return tid_to_gid, gid_to_tids


def prune_within_gene(
    tid_to_gid: dict[str, str],
    gid_to_tids: dict[str, list[str]],
    hit_tids: set[str],
) -> tuple[set[str], dict[str, str]]:
    """Apply the within-gene pruning rule.

    Returns (kept_tids, dropped_map). ``dropped_map[dropped_tid] = keeper``
    where ``keeper`` is any surviving hit-isoform in the same gene (first
    sorted for determinism).
    """
    kept: set[str] = set()
    dropped: dict[str, str] = {}
    for gid, tids in gid_to_tids.items():
        gene_hits = [t for t in tids if t in hit_tids]
        if gene_hits:
            keeper = sorted(gene_hits)[0]
            for t in tids:
                if t in gene_hits:
                    kept.add(t)
                else:
                    dropped[t] = keeper
        else:
            kept.update(tids)
    return kept, dropped


def write_filtered_gtf(src: Path, dst: Path, kept: set[str],
                       gff3: bool = False) -> tuple[int, int]:
    n_written = n_total = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            n_total += 1
            if gff3:
                cols = line.split("\t")
                if len(cols) < 9:
                    continue
                attrs = cols[8]
                feat  = cols[2]
                if feat in ("mRNA", "transcript"):
                    m = _GFF3_ID_RE.search(attrs)
                    if m and m.group(1).strip() in kept:
                        fout.write(line)
                        n_written += 1
                else:
                    m = _GFF3_PARENT_RE.search(attrs)
                    if m and m.group(1).strip() in kept:
                        fout.write(line)
                        n_written += 1
            else:
                m = _TRANSCRIPT_ID_RE.search(line)
                if m and m.group(1) in kept:
                    fout.write(line)
                    n_written += 1
    return n_written, n_total


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--orfs-gtf",    type=Path, required=True)
    ap.add_argument("--out-dir",     type=Path, required=True)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--proteins-fa", type=Path,
                     help="Protein FASTA (mode A: run diamond ourselves).")
    grp.add_argument("--diamond-tsv", type=Path,
                     help="Pre-computed Diamond TSV (mode B).")
    ap.add_argument("--genome", type=Path,
                    help="Genome FASTA (required in mode A).")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--evalue",  type=float, default=1e-5,
                    help="Max e-value for a hit to count (default 1e-5).")
    ap.add_argument("--pident-min",   type=float, default=0.0)
    ap.add_argument("--bitscore-min", type=float, default=0.0)
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.proteins_fa is not None and args.genome is None:
        raise SystemExit("--genome is required when --proteins-fa is set")

    if args.diamond_tsv:
        diamond_tsv = args.diamond_tsv
    else:
        peptides = extract_peptides(
            args.genome, args.orfs_gtf, args.out_dir / "orfs.peptides.fa",
        )
        diamond_tsv = run_diamond(
            peptides, args.proteins_fa, args.out_dir,
            threads=args.threads, evalue=args.evalue,
        )

    hit_tids = transcripts_with_hit(
        diamond_tsv,
        evalue_max=args.evalue,
        pident_min=args.pident_min,
        bitscore_min=args.bitscore_min,
    )
    gff3_mode = _is_gff3(args.orfs_gtf)
    tid_to_gid, gid_to_tids = read_gtf_gene_map(args.orfs_gtf)

    if not tid_to_gid:
        # Fall back to derivation from tids present in GTF (GTF format only).
        tids: set[str] = set()
        for line in args.orfs_gtf.read_text().splitlines():
            m = _TRANSCRIPT_ID_RE.search(line)
            if m:
                tids.add(m.group(1))
        gid_to_tids = defaultdict(list)
        for t in tids:
            g = _fallback_gene_id(t)
            tid_to_gid[t] = g
            gid_to_tids[g].append(t)

    kept, dropped = prune_within_gene(tid_to_gid, gid_to_tids, hit_tids)

    filt_gtf = args.out_dir / "orfs.filtered_soft_protein.gtf"
    report   = args.out_dir / "dropped_soft_protein.tsv"
    n_lines, n_total = write_filtered_gtf(args.orfs_gtf, filt_gtf, kept, gff3=gff3_mode)

    with report.open("w") as fh:
        fh.write("dropped_tid\tkeeper_tid\treason\n")
        for tid, keeper in sorted(dropped.items()):
            fh.write(f"{tid}\t{keeper}\twithin_gene_hit_sibling\n")

    n_in_tx = len(tid_to_gid)
    n_hit   = len(hit_tids & set(tid_to_gid))
    n_drop  = len(dropped)
    n_kept  = len(kept)
    n_genes = len(gid_to_tids)
    n_hit_genes = sum(1 for tids in gid_to_tids.values()
                      if any(t in hit_tids for t in tids))
    print(f"[filter_orf_soft] thresholds : evalue<={args.evalue} "
          f"pident>={args.pident_min} bitscore>={args.bitscore_min}", flush=True)
    print(f"[filter_orf_soft] input tx   : {n_in_tx}", flush=True)
    print(f"[filter_orf_soft] genes      : {n_genes} (with any hit: {n_hit_genes})", flush=True)
    print(f"[filter_orf_soft] tx w/ hit  : {n_hit}", flush=True)
    print(f"[filter_orf_soft] kept tx    : {n_kept}", flush=True)
    print(f"[filter_orf_soft] dropped tx : {n_drop} ({100.0*n_drop/max(1,n_in_tx):.1f}%)", flush=True)
    print(f"[filter_orf_soft] GTF lines  : {n_lines}/{n_total}", flush=True)
    print(f"[filter_orf_soft] out gtf    : {filt_gtf}", flush=True)
    print(f"[filter_orf_soft] report tsv : {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
