#!/bin/bash
# Soft protein filter (diamond blastp) for Tiberius ORF predictions on
# Aspergillus_fumigatus, Candida_parapsilosis, and Fusarium_redolens in
# training_fungi_v2.
#
# Per-species ODB databases (order-excluded Fungi OrthoDB):
#   Aspergillus_fumigatus   → Fungi_excl_Eurotiales.fa
#   Candida_parapsilosis    → Fungi_excl_Saccharomycetales.fa
#   Fusarium_redolens       → Fungi_excl_Hypocreales.fa
#
# Output per species:
#   annotate_run006_e<epoch>/soft_protein/
#     orfs.peptides.fa
#     diamond_db.dmnd
#     diamond.tsv
#     orfs.filtered_soft_protein.gtf
#     dropped_soft_protein.tsv
#
# Usage: sbatch ... [epoch]
#   epoch defaults to the highest epoch_N checkpoint in the model dir.
#
#SBATCH --job-name=softprot_tv2
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/softprot_tv2_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/softprot_tv2_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
MODEL_DIR=${PROJDIR}/results/models/cnn_lstm_run006_fungi
ODB_DIR=${PROJDIR}/data/orthodb

EPOCH="${1:-}"
if [[ -z "${EPOCH}" ]]; then
    EPOCH=$(ls "${MODEL_DIR}"/epoch_*.weights.h5 2>/dev/null \
            | sort -V | tail -1 \
            | xargs -I{} basename {} .weights.h5 \
            | sed 's/epoch_//')
    [[ -n "${EPOCH}" ]] || { echo "no epoch checkpoints found in ${MODEL_DIR}" >&2; exit 2; }
fi

declare -a SPECIES=(
    "Aspergillus_fumigatus"
    "Candida_parapsilosis"
    "Fusarium_redolens"
)
declare -a ODB_FILES=(
    "Fungi_excl_Eurotiales.fa"
    "Fungi_excl_Saccharomycetales.fa"
    "Fungi_excl_Hypocreales.fa"
)

species=${SPECIES[$SLURM_ARRAY_TASK_ID]}
odb_file=${ODB_FILES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TRAINDIR}/${species}/assembly/genome.fa
ORFS=${TRAINDIR}/${species}/annotate_run006_e${EPOCH}/orfs.filtered.gtf
ODB=${ODB_DIR}/${odb_file}
OUTDIR=${TRAINDIR}/${species}/annotate_run006_e${EPOCH}/soft_protein

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered_soft_protein.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered_soft_protein.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${ORFS}" "${ODB}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) species=${species} epoch=${EPOCH}"
echo "[$(date -Iseconds)] ODB=${ODB}"

python "${PROJDIR}/scripts/filter_orf_by_diamond_within_gene.py" \
    --orfs-gtf     "${ORFS}" \
    --genome       "${GENOME}" \
    --proteins-fa  "${ODB}" \
    --out-dir      "${OUTDIR}" \
    --threads      "${SLURM_CPUS_PER_TASK}"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered_soft_protein.gtf"
