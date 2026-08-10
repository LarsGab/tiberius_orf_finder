"""
Classify fungal taxids relative to Fusarium pseudograminearum (Sordariomycetes, Hypocreales).
Finds non-Hypocreales Sordariomycetes first, then other Pezizomycotina.
Uses NCBI efetch XML with proper ElementTree parsing.
"""
import urllib.request
import re
import time
import sys
import xml.etree.ElementTree as ET

FASTA = '/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi_excl_Hypocreales.fa'

print("Extracting taxids from filtered FASTA...")
taxids = set()
with open(FASTA) as f:
    for line in f:
        if line.startswith('>'):
            # header format: >TAXID_1:XXXXXX
            tid = line[1:].split('_1:')[0].strip()
            taxids.add(tid)

taxids = list(taxids)
print(f"Unique taxids in filtered FASTA: {len(taxids)}")

base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
email = "lgabriel23@gmx.de"
batch_size = 200

sordariomycetes_ids = []   # non-Hypocreales Sordariomycetes: (tid, name, order)
other_pezizo_ids = []      # other Pezizomycotina: (tid, name, class)
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

        lineage_names = set()
        lineage_ranks = {}   # rank -> name
        lineage_ex = taxon.find('LineageEx')
        if lineage_ex is not None:
            for lin_taxon in lineage_ex.findall('Taxon'):
                name = lin_taxon.findtext('ScientificName', '')
                rank = lin_taxon.findtext('Rank', '')
                lineage_names.add(name)
                if rank:
                    lineage_ranks[rank] = name

        lineage_str = taxon.findtext('Lineage', '')
        name = taxon.findtext('ScientificName', 'unknown')

        if 'Sordariomycetes' in lineage_names or 'Sordariomycetes' in lineage_str:
            order = lineage_ranks.get('order', 'unknown order')
            sordariomycetes_ids.append((tid, name, order))
        elif 'Pezizomycotina' in lineage_names or 'Pezizomycotina' in lineage_str:
            cls = lineage_ranks.get('class', 'unknown class')
            other_pezizo_ids.append((tid, name, cls))
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
        print(f"Batch {i//batch_size + 1}: processed up to index {i+len(batch)}, "
              f"Sordariomycetes={len(sordariomycetes_ids)}, "
              f"other_Pezizo={len(other_pezizo_ids)}",
              file=sys.stderr, flush=True)
        time.sleep(0.35)
    except Exception as e:
        print(f"Batch {i}: error {e}", file=sys.stderr)
        time.sleep(1)

print("\n=== RESULTS ===")
print(f"Non-Hypocreales Sordariomycetes ({len(sordariomycetes_ids)} total):")
for tid, name, order in sorted(sordariomycetes_ids, key=lambda x: x[2]):
    print(f"  {tid}\t{name}\t{order}")

print(f"\nOther Pezizomycotina ({len(other_pezizo_ids)} total, first 20):")
for tid, name, cls in other_pezizo_ids[:20]:
    print(f"  {tid}\t{name}\t{cls}")
