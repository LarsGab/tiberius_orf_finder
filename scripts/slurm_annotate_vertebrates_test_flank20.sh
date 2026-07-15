#!/bin/bash
# Experiment C: re-run annotate.py on 4 vertebrates_test species with
# --flank-bp 20 (real upstream genomic context, RC for '-' strand). The
# control to compare against is annotate_${WEIGHTS_TAG}_pad20 (same K,
# but the K bases are literal N instead of real genome).
#
# Output:  ${RESULTS_DIR}/<sp>/annotate_${WEIGHTS_TAG}_flank20/orfs.gtf
#
#SBATCH --job-name=annot_vt_flank20
#SBATCH --partition=vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vt_flank20_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vt_flank20_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2/epoch_11.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml

WEIGHTS_TAG=$(basename "${WEIGHTS}" .weights.h5)

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Gallus_gallus"
    "Pristiophorus_japonicus"
    "Bos_taurus"
    "Delphinapterus_leucas"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}_flank20
OUTGTF=${OUTDIR}/orfs.gtf

if [[ ! -s "${STRINGTIE}" || ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: stringtie or genome not ready"
    echo "                       stringtie=${STRINGTIE}"
    echo "                       genome=${GENOME}"
    exit 0
fi

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config: ${CONFIG}"   >&2; exit 2; }
mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] out=${OUTGTF}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK}" \
    --flank-bp      20
