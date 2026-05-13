"""Discover Insect species in SRA with long-read RNA-seq AND a reusable annotation.

Queries NCBI SRA for all distinct organisms under Insecta (taxid 50557) that
have at least one PacBio SMRT or Oxford Nanopore RNA-seq submission. Then for
each species:
  - Looks up a RefSeq assembly via the `datasets` CLI (if available).
  - Checks whether a BRAKER directory exists locally (optional).
  - Resolves family/order via NCBI Taxonomy.

Writes a candidate CSV that downstream split logic can consume.

The output CSV has columns::

    species,annotation_source,accession,in_braker,n_runs,family,order

where ``annotation_source`` is one of {RefSeq, BRAKER, ""} (empty = unannotated;
cannot be used as training data without further work).

Usage (on brain, login-node)::

    micromamba activate orffinder
    python scripts/discover_longread_insect_species.py \\
        --out-csv  nextflow/conf/species_longread_candidates.csv \\
        --braker-dir /home/nas-hs/projs/tiberius-insects/data/insects_data_braker \\
        --email "${NCBI_EMAIL:-you@host}" \\
        --api-key "${NCBI_API_KEY:-}"

Requires Bio.Entrez (pulled in by the VARUS v2 package).  The `datasets` CLI is
optional but recommended; if missing, the script falls back to Entrez Assembly
queries (slower, no quality flag).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from Bio import Entrez

log = logging.getLogger("discover_longread_insects")

INSECTA_TAXID = 50557
LONGREAD_TERM = (
    f"txid{INSECTA_TAXID}[orgn] AND biomol_rna[Prop] "
    "AND (PACBIO_SMRT[Platform] OR OXFORD_NANOPORE[Platform])"
)
PAGE_SIZE = 10_000

_ORG_RE = re.compile(r'ScientificName="([^"]+)"')
_RUN_RE = re.compile(r'Run\s+acc="([^"]+)"')


def _configure_entrez(email: str | None, api_key: str | None) -> None:
    Entrez.email = email or os.environ.get("NCBI_EMAIL", "varus@example.org")
    key = api_key or os.environ.get("NCBI_API_KEY")
    if key:
        Entrez.api_key = key


def _esearch_history(term: str) -> tuple[int, str, str]:
    h = Entrez.esearch(db="sra", term=term, usehistory="y", retmax=0)
    try:
        data = h.read()
    finally:
        h.close()
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    root = ET.fromstring(data)
    return (
        int(root.findtext("Count", "0")),
        root.findtext("WebEnv") or "",
        root.findtext("QueryKey") or "1",
    )


def _esummary_page(webenv: str, qkey: str, retstart: int, retries: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            h = Entrez.esummary(
                db="sra",
                WebEnv=webenv,
                query_key=qkey,
                retstart=retstart,
                retmax=PAGE_SIZE,
            )
            try:
                data = h.read()
            finally:
                h.close()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return data
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"esummary failed: {last_err}")


def _iter_organism_runs(xml: str):
    for block in xml.split("<DocSum>")[1:]:
        end = block.find("</DocSum>")
        if end >= 0:
            block = block[:end]
        m = _ORG_RE.search(block)
        organism = m.group(1) if m else None
        for r in _RUN_RE.finditer(block):
            yield organism, r.group(1)


def collect_sra_species(term: str) -> dict[str, set[str]]:
    """Return {species_name: set(run_accessions)} for every SRA hit on ``term``."""
    count, webenv, qkey = _esearch_history(term)
    log.info("SRA esearch returns %d run records", count)
    if count == 0:
        return {}
    out: dict[str, set[str]] = defaultdict(set)
    retstart = 0
    while retstart < count:
        log.info("esummary page retstart=%d", retstart)
        xml = _esummary_page(webenv, qkey, retstart)
        for organism, run_acc in _iter_organism_runs(xml):
            if organism:
                out[organism].add(run_acc)
        retstart += PAGE_SIZE
    return out


def _datasets_available() -> bool:
    return shutil.which("datasets") is not None


def lookup_refseq_via_datasets(species: str, timeout: int = 30) -> str | None:
    """Return the first GCF_* accession for `species`, or None."""
    try:
        proc = subprocess.run(
            ["datasets", "summary", "genome", "taxon", species,
             "--reference", "--as-json-lines"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        acc = obj.get("accession") or ""
        if acc.startswith("GCF_"):
            return acc
    return None


def lookup_refseq_via_entrez(species: str) -> str | None:
    """Fallback when `datasets` is missing. Slower; no quality flag."""
    try:
        h = Entrez.esearch(db="assembly", term=f'"{species}"[orgn] AND latest_refseq[filter]')
        data = h.read()
        h.close()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        root = ET.fromstring(data)
        ids = [e.text for e in root.findall(".//Id") if e.text]
        if not ids:
            return None
        h = Entrez.esummary(db="assembly", id=ids[0])
        data = h.read()
        h.close()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        m = re.search(r"<AssemblyAccession>(GCF_[^<]+)</AssemblyAccession>", data)
        return m.group(1) if m else None
    except Exception:
        return None


def check_braker(species: str, braker_dir: Path | None) -> bool:
    if not braker_dir:
        return False
    candidate = braker_dir / species.replace(" ", "_")
    if not candidate.is_dir():
        return False
    # Mirror fetch.nf's expectation: must have the two files.
    underscored = species.replace(" ", "_")
    fna = candidate / f"{underscored}_renamed.fna"
    gtf = candidate / f"{underscored}_cds_longest.gtf"
    return fna.is_file() and gtf.is_file()


def lookup_taxonomy(species: str) -> tuple[str, str]:
    """Return (family, order) for `species`, or empty strings on failure."""
    try:
        h = Entrez.esearch(db="taxonomy", term=f'"{species}"[Scientific Name]')
        data = h.read()
        h.close()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        root = ET.fromstring(data)
        ids = [e.text for e in root.findall(".//Id") if e.text]
        if not ids:
            return "", ""
        h = Entrez.efetch(db="taxonomy", id=ids[0], rettype="xml")
        data = h.read()
        h.close()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        root = ET.fromstring(data)
        family = order = ""
        for tx in root.findall(".//Taxon/LineageEx/Taxon"):
            rank = (tx.findtext("Rank") or "").lower()
            name = tx.findtext("ScientificName") or ""
            if rank == "family":
                family = name
            elif rank == "order":
                order = name
        return family, order
    except Exception:
        return "", ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--braker-dir", type=Path, default=None,
                    help="Optional path to BRAKER cache (e.g. /home/nas-hs/projs/.../insects_data_braker).")
    ap.add_argument("--email", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--no-taxonomy", action="store_true",
                    help="Skip family/order resolution (faster).")
    ap.add_argument("--no-assembly-lookup", action="store_true",
                    help="Skip RefSeq lookup entirely.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _configure_entrez(args.email, args.api_key)

    species_to_runs = collect_sra_species(LONGREAD_TERM)
    if not species_to_runs:
        sys.exit("No SRA runs found for Insecta long-read RNA-seq — aborting.")
    log.info("Distinct species in SRA: %d", len(species_to_runs))

    use_datasets = _datasets_available() and not args.no_assembly_lookup
    if not args.no_assembly_lookup and not use_datasets:
        log.warning("`datasets` CLI not found; falling back to Entrez Assembly queries")

    rows = []
    for i, (species, runs) in enumerate(sorted(species_to_runs.items()), 1):
        n_runs = len(runs)
        accession = ""
        if not args.no_assembly_lookup:
            accession = (lookup_refseq_via_datasets(species)
                         if use_datasets else lookup_refseq_via_entrez(species)) or ""
        in_braker = check_braker(species, args.braker_dir)
        if not args.no_taxonomy:
            family, order = lookup_taxonomy(species)
        else:
            family, order = "", ""
        ann_src = "RefSeq" if accession else ("BRAKER" if in_braker else "")
        rows.append({
            "species": species,
            "annotation_source": ann_src,
            "accession": accession,
            "in_braker": int(bool(in_braker)),
            "n_runs": n_runs,
            "family": family,
            "order": order,
        })
        if i % 25 == 0 or i == len(species_to_runs):
            log.info("processed %d/%d species", i, len(species_to_runs))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["species", "annotation_source", "accession",
                        "in_braker", "n_runs", "family", "order"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate summary printed to stdout.
    n_total = len(rows)
    n_refseq = sum(1 for r in rows if r["annotation_source"] == "RefSeq")
    n_braker = sum(1 for r in rows if r["in_braker"])
    n_either = sum(1 for r in rows if r["annotation_source"] or r["in_braker"])
    n_5plus  = sum(1 for r in rows if (r["annotation_source"] or r["in_braker"]) and r["n_runs"] >= 5)
    family_hist: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["family"] and (r["annotation_source"] or r["in_braker"]):
            family_hist[r["family"]] += 1

    print(f"\n=== Insecta long-read RNA-seq candidate species ===")
    print(f"total distinct species in SRA:           {n_total}")
    print(f"  with RefSeq assembly:                  {n_refseq}")
    print(f"  with BRAKER cache:                     {n_braker}")
    print(f"  annotated (RefSeq OR BRAKER):          {n_either}")
    print(f"  annotated AND >= 5 SRA runs:           {n_5plus}")
    if family_hist:
        print(f"  families covered (annotated species): {len(family_hist)}")
        top = sorted(family_hist.items(), key=lambda kv: -kv[1])[:10]
        for fam, k in top:
            print(f"    {fam:<30s}  {k} species")
    print(f"\nWrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
