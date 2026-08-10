"""
Classify fungal taxids by their taxonomic order/class relative to A. fumigatus.
Uses NCBI efetch XML; extracts the record-level TaxId (not lineage sub-taxids).
"""
import urllib.request
import re
import time
import sys
import xml.etree.ElementTree as ET

taxids_file = '/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/all_taxids.txt'
taxids = open(taxids_file).read().split()

base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
email = "lgabriel23@gmx.de"

batch_size = 200

onygenales_ids = []
eurotiomycetes_ids = []
sordariomycetes_ids = []
other_ids = []


def classify_batch(content):
    """Parse NCBI taxonomy XML and classify each taxon."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"XML parse error: {e}", file=sys.stderr)
        return

    for taxon in root.findall('Taxon'):
        tid = taxon.findtext('TaxId', '').strip()
        if not tid:
            continue

        # Collect all taxon names in the lineage
        lineage_names = set()
        lineage_ex = taxon.find('LineageEx')
        if lineage_ex is not None:
            for lin_taxon in lineage_ex.findall('Taxon'):
                name = lin_taxon.findtext('ScientificName', '')
                lineage_names.add(name)

        # Also check the flat Lineage string as fallback
        lineage_str = taxon.findtext('Lineage', '')

        if 'Onygenales' in lineage_names or 'Onygenales' in lineage_str:
            onygenales_ids.append(tid)
        elif 'Eurotiomycetes' in lineage_names or 'Eurotiomycetes' in lineage_str:
            eurotiomycetes_ids.append(tid)
        elif 'Sordariomycetes' in lineage_names or 'Sordariomycetes' in lineage_str:
            sordariomycetes_ids.append(tid)
        else:
            other_ids.append(tid)


for i in range(0, len(taxids), batch_size):
    batch = taxids[i:i+batch_size]
    ids_str = ','.join(batch)
    url = (f"{base}efetch.fcgi?db=taxonomy&id={ids_str}"
           f"&rettype=xml&email={email}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            content = r.read().decode()
        classify_batch(content)
        print(f"Batch {i//batch_size + 1}: processed up to taxid index {i+len(batch)}, "
              f"Onygenales={len(onygenales_ids)}, "
              f"Eurotiomycetes={len(eurotiomycetes_ids)}, "
              f"Sordariomycetes={len(sordariomycetes_ids)}",
              file=sys.stderr)
        time.sleep(0.35)
    except Exception as e:
        print(f"Batch {i}: error {e}", file=sys.stderr)
        time.sleep(1)

print("=== RESULTS ===")
print(f"Onygenales ({len(onygenales_ids)} total): {onygenales_ids}")
print(f"Other Eurotiomycetes ({len(eurotiomycetes_ids)} total): {eurotiomycetes_ids}")
print(f"Sordariomycetes ({len(sordariomycetes_ids)} total): {sordariomycetes_ids[:20]}")
print(f"Other ({len(other_ids)} total, first 20): {other_ids[:20]}")
