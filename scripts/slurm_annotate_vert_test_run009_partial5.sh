#!/bin/bash
#SBATCH --job-name=annot_partial5
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_partial5_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_partial5_%A_%a.err

# Re-run annotate.py with --partial5-out to generate orfs.partial5.gtf for the
# 9 vertebrate test species.  Also regenerates orfs.partial.gtf in the same pass.
# Required as input for slurm_fix_stop_vertebrates_test.sh (--partial5 flag).
# Skips species where orfs.partial5.gtf already exists.

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
PARTIAL5_GTF="${OUTDIR}/orfs.partial5.gtf"

mkdir -p "${PROJDIR}/logs"

if [[ -s "${PARTIAL5_GTF}" ]]; then
    echo "SKIP ${species}: ${PARTIAL5_GTF} already exists"
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
echo "[$(date -Iseconds)] partial5_out=${PARTIAL5_GTF}"

python scripts/annotate.py \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --partial-out   "${PARTIAL_GTF}" \
    --partial5-out  "${PARTIAL5_GTF}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK:-4}"

echo "[$(date -Iseconds)] done -> ${PARTIAL5_GTF}"
