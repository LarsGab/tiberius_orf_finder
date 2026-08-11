#!/bin/bash
#SBATCH --job-name=annot_partial
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_partial_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_partial_%A_%a.err

# Re-run annotate.py with --partial-out to generate orfs.partial.gtf for the
# vertebrate test species. Required as input for slurm_fix_stop_vertebrates_test.sh.
# Skips species where orfs.partial.gtf already exists.
# Uses the same weights/config/stringtie filter tag as slurm_annotate_vert_test_run009_best.

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

FILT_TAG=filt_tpm1cov3len300
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run009/best.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml
TAG=run009_best_${FILT_TAG}

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Bos_taurus
    Delphinapterus_leucas
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
    Homo_sapiens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
STRINGTIE="${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf"
OUTDIR="${RESULTS_DIR}/${species}/annotate_${TAG}"
PARTIAL_GTF="${OUTDIR}/orfs.partial.gtf"

mkdir -p "${PROJDIR}/logs"

if [[ -s "${PARTIAL_GTF}" ]]; then
    echo "SKIP ${species}: ${PARTIAL_GTF} already exists"
    exit 0
fi

if [[ ! -s "${STRINGTIE}" || ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: stringtie or genome not ready"
    echo "                       stringtie=${STRINGTIE}"
    echo "                       genome=${GENOME}"
    exit 0
fi

for f in "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing: ${f}" >&2; exit 2; }
done

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] stringtie=${STRINGTIE}"
echo "[$(date -Iseconds)] weights=${WEIGHTS}"
echo "[$(date -Iseconds)] out_dir=${OUTDIR}"
echo "[$(date -Iseconds)] partial_out=${PARTIAL_GTF}"

python scripts/annotate.py \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --partial-out   "${PARTIAL_GTF}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK:-4}"

echo "[$(date -Iseconds)] done -> ${PARTIAL_GTF}"
