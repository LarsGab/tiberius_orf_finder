"""
Classify fungal taxids by lineage relative to Candida parapsilosis.
Candida parapsilosis is Saccharomycetes / Saccharomycetales — excluded.
Closest outgroups:
  1. Taphrinomycotina (sister subphylum to Saccharomycotina)
  2. Pezizomycotina (diverse classes: Eurotiomycetes, Sordariomycetes, etc.)
Uses NCBI efetch XML; proper XML parsing to extract record-level TaxId.
"""
import urllib.request
import re
import time
import sys
import xml.etree.ElementTree as ET

fasta = '/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi_excl_Saccharomycetales.fa'

# Extract unique taxids from filtered FASTA headers (format: >TAXID_1:XXXXX)
print("Extracting taxids from filtered FASTA...")
taxids = set()
with open(fasta) as f:
    for line in f:
        if line.startswith('>'):
            # Header format: >1001832_1:000000  -> taxid is part before '_1:'
            m = re.match(r'>(\d+)_', line)
            if m:
                taxids.add(m.group(1))

taxids = list(taxids)
print(f"Unique taxids: {len(taxids)}")

base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
email = "lgabriel23@gmx.de"
batch_size = 200

taphrinomycotina_ids = []
pezizomycotina_eurotiomycetes = []
pezizomycotina_sordariomycetes = []
pezizomycotina_other = []
other_ids = []


def classify_batch(content):
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"XML parse error: {e}", file=sys.stderr)
        return

    for taxon in root.findall('Taxon'):
        tid = taxon.findtext('TaxId', '').strip()
        if not tid:
            continue

        name_el = taxon.find('ScientificName')
        name = name_el.text.strip() if name_el is not None and name_el.text else 'unknown'

        # Collect lineage names from LineageEx
        lineage_names = set()
        lineage_ex = taxon.find('LineageEx')
        if lineage_ex is not None:
            for lin_taxon in lineage_ex.findall('Taxon'):
                lname = lin_taxon.findtext('ScientificName', '')
                lineage_names.add(lname)

        # Flat lineage string as fallback
        lineage_str = taxon.findtext('Lineage', '')

        def in_lineage(term):
            return term in lineage_names or term in lineage_str

        if in_lineage('Taphrinomycotina'):
            taphrinomycotina_ids.append((tid, name))
        elif in_lineage('Eurotiomycetes'):
            pezizomycotina_eurotiomycetes.append((tid, name))
        elif in_lineage('Sordariomycetes'):
            pezizomycotina_sordariomycetes.append((tid, name))
        elif in_lineage('Pezizomycotina'):
            pezizomycotina_other.append((tid, name))
        else:
            other_ids.append((tid, name))


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
        if i % 1000 == 0 or i + batch_size >= len(taxids):
            print(f"  processed {min(i+batch_size, len(taxids))}/{len(taxids)} | "
                  f"Taphrinomycotina={len(taphrinomycotina_ids)} "
                  f"Eurotiomycetes={len(pezizomycotina_eurotiomycetes)} "
                  f"Sordariomycetes={len(pezizomycotina_sordariomycetes)} "
                  f"Other_Pezizo={len(pezizomycotina_other)}",
                  flush=True)
        time.sleep(0.35)
    except Exception as e:
        print(f"Batch {i}: error {e}", file=sys.stderr)
        time.sleep(1)

print("\n=== RESULTS ===")
print(f"Taphrinomycotina ({len(taphrinomycotina_ids)} total):")
for tid, name in taphrinomycotina_ids:
    print(f"  {tid}  {name}")
print(f"\nEurotiomycetes ({len(pezizomycotina_eurotiomycetes)} total, first 10):")
for tid, name in pezizomycotina_eurotiomycetes[:10]:
    print(f"  {tid}  {name}")
print(f"\nSordariomycetes ({len(pezizomycotina_sordariomycetes)} total, first 10):")
for tid, name in pezizomycotina_sordariomycetes[:10]:
    print(f"  {tid}  {name}")
print(f"\nOther Pezizomycotina ({len(pezizomycotina_other)} total, first 10):")
for tid, name in pezizomycotina_other[:10]:
    print(f"  {tid}  {name}")
print(f"\nOther fungi ({len(other_ids)} total, first 10):")
for tid, name in other_ids[:10]:
    print(f"  {tid}  {name}")
