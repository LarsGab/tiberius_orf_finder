#!/bin/bash
# Look up order-level NCBI taxon IDs for all embryophyta training species.
#
# Output:
#   results/training_embryophyta_v2/species_order_taxids.tsv
#
#SBATCH --job-name=taxids_emb_train
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/taxids_emb_train_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/taxids_emb_train_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2
NAMES=/home/gabriell/tiberius_proteins_analysis/odb/names.dmp
NODES=/home/gabriell/tiberius_proteins_analysis/odb/nodes.dmp
OUT=${TRAINDIR}/species_order_taxids.tsv

mkdir -p "${PROJDIR}/logs"

for f in "${NAMES}" "${NODES}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

SPECIES_LIST=$(mktemp /tmp/emb_train_species_XXXXXX.txt)
ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]' > "${SPECIES_LIST}"
echo "[$(date -Iseconds)] Looking up order taxids for $(wc -l < "${SPECIES_LIST}") embryophyta training species …"

python scripts/lookup_order_taxids.py \
    --names         "${NAMES}" \
    --nodes         "${NODES}" \
    --species-list  "${SPECIES_LIST}" \
    --out           "${OUT}" \
    --entrez-email  lgabriel23@gmx.de

rm -f "${SPECIES_LIST}"

echo "[$(date -Iseconds)] done -> ${OUT}"
wc -l "${OUT}"
