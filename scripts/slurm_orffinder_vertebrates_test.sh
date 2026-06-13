#!/bin/bash
# Run tiberius_orf_finder (annotate.py) on the 9 vertebrate TEST species,
# one species per array task, using the StringTie GTF + genome FASTA
# produced by `submit_vertebrates_test.sh`.
#
# Species whose StringTie output isn't on disk yet (data-prep still in
# progress or failed) are logged as SKIP and exit 0, so the array can be
# launched while `submit_vertebrates_test.sh` is still running and
# re-launched later for the stragglers.
#
# Inputs:  ${RESULTS_DIR}/<Genus_species>/stringtie/stringtie.gtf
#          ${RESULTS_DIR}/<Genus_species>/assembly/genome.fa
# Outputs: ${RESULTS_DIR}/<Genus_species>/orffinder/orffinder.gtf
#
# Usage:  sbatch scripts/slurm_orffinder_vertebrates_test.sh
#
#SBATCH --job-name=orffinder_vert_test
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/orffinder_vert_test_%A_%a.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/orffinder_vert_test_%A_%a.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

# >>> set these two to the checkpoint you want to benchmark <<<
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run001/best.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Archocentrus_centrarchus"
    "Betta_splendens"
    "Takifugu_rubripes"
    "Pristiophorus_japonicus"
    "Gallus_gallus"
    "Zootoca_vivipara"
    "Bos_taurus"
    "Delphinapterus_leucas"
    "Homo_sapiens"
)

species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${RESULTS_DIR}/${species}/orffinder
OUTGTF=${OUTDIR}/orffinder.gtf

# Skip-not-fail: data-prep may still be running for this species.
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

python scripts/annotate.py \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir           "${OUTDIR}" \
    --batch-size    16 \
    --threads       "${SLURM_CPUS_PER_TASK}"


