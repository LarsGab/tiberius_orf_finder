"""Look up order-level NCBI taxon IDs for training species.

Reads names.dmp and nodes.dmp from the local NCBI taxonomy dump, finds the
"order" rank ancestor for each training species, and writes a TSV used by the
SLURM protein-preparation pipeline.

Output columns: species_name  taxid  order_name  order_taxid

If --entrez-email is given and Biopython is installed, species not found in
the local dump fall back to NCBI Entrez (3 requests/s).

Usage
-----
python scripts/lookup_order_taxids.py \\
    --names      /home/gabriell/tiberius_proteins_analysis/odb/names.dmp \\
    --nodes      /home/gabriell/tiberius_proteins_analysis/odb/nodes.dmp \\
    --species-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/training_vertebrates_v2 \\
    --out        /projects/AI-GUSTUS/tiberius_orf_finder/results/training_vertebrates_v2/species_order_taxids.tsv \\
    [--entrez-email user@example.com]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_names(names_dmp: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Return (scientific_name → taxid, taxid → scientific_name)."""
    name_to_taxid: dict[str, int] = {}
    taxid_to_name: dict[int, str] = {}
    with names_dmp.open() as fh:
        for line in fh:
            parts = [p.strip() for p in line.rstrip("\n").split("|")]
            if len(parts) < 4:
                continue
            taxid_str, name_txt, _, name_class = parts[0], parts[1], parts[2], parts[3]
            if name_class == "scientific name":
                tid = int(taxid_str)
                name_to_taxid[name_txt] = tid
                taxid_to_name[tid] = name_txt
    return name_to_taxid, taxid_to_name


def parse_nodes(nodes_dmp: Path) -> dict[int, tuple[int, str]]:
    """Return taxid → (parent_taxid, rank)."""
    nodes: dict[int, tuple[int, str]] = {}
    with nodes_dmp.open() as fh:
        for line in fh:
            parts = [p.strip() for p in line.rstrip("\n").split("|")]
            if len(parts) < 3:
                continue
            taxid, parent, rank = int(parts[0]), int(parts[1]), parts[2]
            nodes[taxid] = (parent, rank)
    return nodes


def find_order(taxid: int, nodes: dict[int, tuple[int, str]],
               taxid_to_name: dict[int, str]) -> tuple[int, str] | None:
    """Walk up the tree from taxid to find the order-rank ancestor."""
    visited: set[int] = set()
    current = taxid
    while current not in visited:
        visited.add(current)
        if current not in nodes:
            return None
        parent, rank = nodes[current]
        if rank == "order":
            return current, taxid_to_name.get(current, f"taxid:{current}")
        if current == parent:
            return None
        current = parent
    return None


def entrez_lookup(sci_name: str, email: str) -> int | None:
    """Return NCBI taxid for a scientific name via Entrez. None on failure."""
    try:
        from Bio import Entrez
        Entrez.email = email
        for term in [f'"{sci_name}"[Scientific Name]', sci_name]:
            handle = Entrez.esearch(db="taxonomy", term=term)
            record = Entrez.read(handle)
            handle.close()
            ids = record.get("IdList", [])
            if ids:
                return int(ids[0])
    except Exception as exc:
        print(f"  Entrez lookup failed for '{sci_name}': {exc}", file=sys.stderr)
    return None


def entrez_get_order(taxid: int, email: str) -> tuple[int, str] | None:
    """Return (order_taxid, order_name) from Entrez lineage. None on failure."""
    try:
        from Bio import Entrez
        Entrez.email = email
        handle = Entrez.efetch(db="taxonomy", id=str(taxid), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        if not records:
            return None
        for item in reversed(records[0].get("LineageEx", [])):
            if item.get("Rank") == "order":
                return int(item["TaxId"]), item["ScientificName"]
    except Exception as exc:
        print(f"  Entrez lineage failed for taxid {taxid}: {exc}", file=sys.stderr)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names",        required=True, type=Path, help="NCBI names.dmp")
    p.add_argument("--nodes",        required=True, type=Path, help="NCBI nodes.dmp")
    p.add_argument("--species-dir",  type=Path,
                   help="Directory with one subdir per species (auto-discover)")
    p.add_argument("--species-list", type=Path,
                   help="Text file with one Genus_species per line")
    p.add_argument("--out",          required=True, type=Path, help="Output TSV path")
    p.add_argument("--entrez-email", default="",
                   help="Email for NCBI Entrez fallback (requires Biopython)")
    args = p.parse_args()

    if not args.species_dir and not args.species_list:
        p.error("Provide --species-dir or --species-list")

    if args.species_list:
        species = [s.strip() for s in args.species_list.read_text().splitlines()
                   if s.strip() and not s.startswith("#")]
    else:
        species = sorted(
            d.name for d in args.species_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    print(f"Species to process: {len(species)}", flush=True)
    print(f"Parsing names.dmp …", flush=True)
    name_to_taxid, taxid_to_name = parse_names(args.names)
    print(f"  {len(name_to_taxid):,} scientific names loaded", flush=True)
    print(f"Parsing nodes.dmp …", flush=True)
    nodes = parse_nodes(args.nodes)
    print(f"  {len(nodes):,} nodes loaded\n", flush=True)

    rows: list[tuple[str, int, str, int]] = []
    missing: list[str] = []

    for sp in species:
        sci_name = sp.replace("_", " ")
        taxid = name_to_taxid.get(sci_name)
        if taxid is None:
            missing.append(sp)
            continue
        result = find_order(taxid, nodes, taxid_to_name)
        if result is None:
            print(f"  WARN {sp}: taxid={taxid} but no order ancestor found",
                  file=sys.stderr)
            missing.append(sp)
            continue
        order_taxid, order_name = result
        rows.append((sp, taxid, order_name, order_taxid))
        print(f"  {sp:<50} taxid={taxid:<9} order='{order_name}' ({order_taxid})")

    if missing and args.entrez_email:
        print(f"\nEntrez fallback for {len(missing)} unresolved species …", flush=True)
        for sp in missing:
            sci_name = sp.replace("_", " ")
            print(f"  {sci_name} …", end=" ", flush=True)
            taxid = entrez_lookup(sci_name, args.entrez_email)
            if taxid is None:
                print("NOT FOUND")
                continue
            result = find_order(taxid, nodes, taxid_to_name)
            if result is None:
                # try Entrez lineage if local nodes don't cover this taxid
                result = entrez_get_order(taxid, args.entrez_email)
            if result is None:
                print(f"taxid={taxid} but no order")
                continue
            order_taxid, order_name = result
            print(f"taxid={taxid} → order='{order_name}' ({order_taxid})")
            rows.append((sp, taxid, order_name, order_taxid))
            time.sleep(0.34)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("species_name\ttaxid\torder_name\torder_taxid\n")
        for sp, tid, oname, otid in sorted(rows, key=lambda r: r[0]):
            fh.write(f"{sp}\t{tid}\t{oname}\t{otid}\n")

    print(f"\nWrote {len(rows)} entries → {args.out}")
    not_found = [s for s in missing if not any(r[0] == s for r in rows)]
    if not_found:
        print(f"WARNING: {len(not_found)} species unresolved "
              f"(add --entrez-email for fallback):")
        for s in not_found:
            print(f"  {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
