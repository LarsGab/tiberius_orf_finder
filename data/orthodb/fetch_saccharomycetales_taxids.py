import urllib.request, re, time

email = "lgabriel23@gmx.de"
base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Search for all taxids under Saccharomycetales subtree
search_url = (f"{base}esearch.fcgi?db=taxonomy"
              f"&term=Saccharomycetales[subtree]"
              f"&retmax=200000&email={email}")
with urllib.request.urlopen(search_url, timeout=60) as r:
    xml = r.read().decode()

taxids = re.findall(r'<Id>(\d+)</Id>', xml)
print(f"Found {len(taxids)} Saccharomycetales taxids")

with open('/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/saccharomycetales_taxids.txt', 'w') as f:
    f.write('\n'.join(taxids) + '\n')
