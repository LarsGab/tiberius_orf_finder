#!/bin/bash
# Download OrthoDB v11 Fungi partition and remove all sequences belonging to
# order Eurotiales (NCBI taxid 37989 — contains Aspergillus fumigatus).
# Output: ${OUTDIR}/Fungi_excl_Eurotiales.fa.gz
#
# Usage: sbatch scripts/slurm_make_odb_fungi_excl_eurotiales.sh
#
#SBATCH --job-name=odb_excl_eur
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/odb_excl_eur_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/odb_excl_eur_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
OUTDIR=${PROJDIR}/data/orthodb
ODB_URL="https://bioinf.uni-greifswald.de/bioinf/partitioned_odb11/Fungi.fa.gz"
EUROTIALES_TAXID=37989

mkdir -p "${OUTDIR}" "${PROJDIR}/logs"

echo "[$(date -Iseconds)] downloading Fungi.fa.gz"
wget -q --show-progress -O "${OUTDIR}/Fungi.fa.gz" "${ODB_URL}"
echo "[$(date -Iseconds)] download done"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] fetching Eurotiales descendant taxids from NCBI"
python3 - <<'PYEOF'
import urllib.request, json, gzip, re, os, sys

OUTDIR        = os.environ["OUTDIR"]
EUROTIALES    = int(os.environ["EUROTIALES_TAXID"])
IN_GZ         = f"{OUTDIR}/Fungi.fa.gz"
OUT_GZ        = f"{OUTDIR}/Fungi_excl_Eurotiales.fa.gz"
TAXIDS_FILE   = f"{OUTDIR}/eurotiales_taxids.txt"

# --- 1. fetch all descendant taxids -----------------------------------------
url = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    f"?db=taxonomy&term=txid{EUROTIALES}[Subtree]&retmax=100000&retmode=json"
)
with urllib.request.urlopen(url, timeout=120) as r:
    data = json.loads(r.read())
excl = set(int(x) for x in data["esearchresult"]["idlist"])
excl.add(EUROTIALES)
print(f"  Eurotiales subtree: {len(excl)} taxids", flush=True)
with open(TAXIDS_FILE, "w") as fh:
    fh.write("\n".join(str(t) for t in sorted(excl)) + "\n")

# --- 2. filter FASTA ---------------------------------------------------------
# Header format: >taxid_version:proteinnum  (taxid is digits before first '_')
pat = re.compile(r"^>(\d+)_")
kept = 0
skipped = 0
write_record = True

with gzip.open(IN_GZ, "rt") as fin, gzip.open(OUT_GZ, "wt") as fout:
    for line in fin:
        if line.startswith(">"):
            m = pat.match(line)
            if m and int(m.group(1)) in excl:
                write_record = False
                skipped += 1
            else:
                write_record = True
                kept += 1
                fout.write(line)
        else:
            if write_record:
                fout.write(line)

print(f"  kept {kept} sequences, excluded {skipped} Eurotiales sequences")
print(f"  output -> {OUT_GZ}")
PYEOF

echo "[$(date -Iseconds)] done"
