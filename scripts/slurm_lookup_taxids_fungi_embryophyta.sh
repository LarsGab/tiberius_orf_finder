#!/bin/bash
# Look up order-level NCBI taxon IDs for Fungi and Embryophyta test species.
# Prerequisite: NCBI taxonomy names.dmp available at ODB path.
#
# Output:
#   results/fungi_test/species_order_taxids.tsv
#   results/training_embryophyta_test_v2/species_order_taxids.tsv
#
#SBATCH --job-name=lookup_taxids
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lookup_taxids_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lookup_taxids_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
NAMES=/home/gabriell/tiberius_proteins_analysis/odb/names.dmp
NODES=/home/gabriell/tiberius_proteins_analysis/odb/nodes.dmp

mkdir -p "${PROJDIR}/logs"

for f in "${NAMES}" "${NODES}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

# ── Fungi ────────────────────────────────────────────────────────────────────
FUNGI_SPECIES_FILE=$(mktemp /tmp/fungi_species_XXXXXX.txt)
cat > "${FUNGI_SPECIES_FILE}" <<'EOF'
Agaricus_bisporus
Aspergillus_fumigatus
Cryphonectria_parasitica
Parastagonospora_nodorum
Puccinia_striiformis
Punctularia_strigosozonata
Tilletiopsis_washingtonensis
EOF

echo "[$(date -Iseconds)] Looking up taxids for Fungi ..."
python scripts/lookup_order_taxids.py \
    --names       "${NAMES}" \
    --nodes       "${NODES}" \
    --species-list "${FUNGI_SPECIES_FILE}" \
    --out         results/fungi_test/species_order_taxids.tsv \
    --entrez-email lgabriel23@gmx.de

echo "[$(date -Iseconds)] Fungi taxids:"
column -t -s$'\t' results/fungi_test/species_order_taxids.tsv

# ── Embryophyta ───────────────────────────────────────────────────────────────
EMB_SPECIES_FILE=$(mktemp /tmp/emb_species_XXXXXX.txt)
cat > "${EMB_SPECIES_FILE}" <<'EOF'
Arabidopsis_thaliana
Eschscholzia_californica
Freycinetia_multiflora
Medicago_truncatula
Mimulus_guttatus
Urochloa_brizantha
EOF

echo "[$(date -Iseconds)] Looking up taxids for Embryophyta ..."
python scripts/lookup_order_taxids.py \
    --names       "${NAMES}" \
    --nodes       "${NODES}" \
    --species-list "${EMB_SPECIES_FILE}" \
    --out         results/training_embryophyta_test_v2/species_order_taxids.tsv \
    --entrez-email lgabriel23@gmx.de

echo "[$(date -Iseconds)] Embryophyta taxids:"
column -t -s$'\t' results/training_embryophyta_test_v2/species_order_taxids.tsv

rm -f "${FUNGI_SPECIES_FILE}" "${EMB_SPECIES_FILE}"
echo "[$(date -Iseconds)] done"
