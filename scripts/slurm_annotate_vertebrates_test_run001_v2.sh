#!/bin/bash
# Test the current vertebrates training checkpoint with annotate.py on
# the vertebrates test species (nextflow/conf/species_vertebrates_test.csv).
#
# Reads StringTie GTF + genome FASTA from the existing pre-staged test
# data at ${RESULTS_DIR}/<Genus_species>/{stringtie,assembly}/. If you
# regenerate the test split via the recovery pipeline later, switch
# RESULTS_DIR to results/training_vertebrates_test_v2.
#
# Output:  ${RESULTS_DIR}/<sp>/annotate_run001_v2_e${EPOCH}/annotation.gff
#
# Set WEIGHTS to the checkpoint you want to score against. Default is
# epoch_11 (last saved before run001_v2 OOM'd). Switch to best.weights.h5
# or a later epoch from run002 once training resumes.
#
#SBATCH --job-name=annot_vert_test
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vert_test_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vert_test_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

# >>> EDIT THIS to score a different checkpoint <<<
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2/epoch_11.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml

# Tag the output dir with which checkpoint produced it, so you can score
# multiple epochs without overwriting.
WEIGHTS_TAG=$(basename "${WEIGHTS}" .weights.h5)

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Gallus_gallus"
    "Pristiophorus_japonicus"
    "Bos_taurus"
    "Delphinapterus_leucas"
    "Homo_sapiens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}
OUTGTF=${OUTDIR}/annotation.gff

# Skip-not-fail: data may not be staged for this species yet.
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
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] stringtie=${STRINGTIE}"
echo "[$(date -Iseconds)] weights=${WEIGHTS}"
echo "[$(date -Iseconds)] config=${CONFIG}"
echo "[$(date -Iseconds)] out=${OUTGTF}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK}"
