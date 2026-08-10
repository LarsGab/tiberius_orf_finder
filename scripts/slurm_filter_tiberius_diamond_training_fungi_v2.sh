#!/bin/bash
# Soft protein filter (diamond blastp) for ab initio Tiberius predictions on
# Aspergillus_fumigatus, Candida_parapsilosis, and Fusarium_redolens.
#
# Input:  {SPECIESDIR}/tiberius/tiberius.gtf
# Output: {SPECIESDIR}/tiberius/soft_protein/orfs.filtered_soft_protein.gtf
#
# Aspergillus_fumigatus is read from fungi_test/ (symlinked into training_fungi_v2).
# Candida and Fusarium are read from training_fungi_v2/.
#
# Per-species ODB (order-excluded Fungi OrthoDB):
#   Aspergillus_fumigatus   → Fungi_excl_Eurotiales.fa
#   Candida_parapsilosis    → Fungi_excl_Saccharomycetales.fa
#   Fusarium_redolens       → Fungi_excl_Hypocreales.fa
#
#SBATCH --job-name=tib_prot_tv2
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_prot_tv2_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_prot_tv2_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
ODB_DIR=${PROJDIR}/data/orthodb

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

# Aspergillus lives in fungi_test; expose its tiberius dir under training_fungi_v2.
if [[ "${species}" == "Aspergillus_fumigatus" && \
      ! -e "${TRAINDIR}/Aspergillus_fumigatus/tiberius" ]]; then
    ln -sf "${PROJDIR}/results/fungi_test/Aspergillus_fumigatus/tiberius" \
           "${TRAINDIR}/Aspergillus_fumigatus/tiberius"
fi

GENOME=${TRAINDIR}/${species}/assembly/genome.fa
TIB_GTF=${TRAINDIR}/${species}/tiberius/tiberius.gtf
ODB=${ODB_DIR}/${odb_file}
OUTDIR=${TRAINDIR}/${species}/tiberius/soft_protein

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered_soft_protein.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered_soft_protein.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${TIB_GTF}" "${ODB}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) species=${species}"
echo "[$(date -Iseconds)] TIB_GTF=${TIB_GTF}"
echo "[$(date -Iseconds)] ODB=${ODB}"

python "${PROJDIR}/scripts/filter_orf_by_diamond_within_gene.py" \
    --orfs-gtf     "${TIB_GTF}" \
    --genome       "${GENOME}" \
    --proteins-fa  "${ODB}" \
    --out-dir      "${OUTDIR}" \
    --threads      "${SLURM_CPUS_PER_TASK}"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered_soft_protein.gtf"
