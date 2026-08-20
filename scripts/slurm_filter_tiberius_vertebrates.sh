#!/bin/bash
# Filter Tiberius ab initio predictions for the vertebrates_test species using
# ORFfinder model scores (epoch_74, run006, 500 bp upstream context).
#
# Transcripts with mean_coding_prob < 0.2 OR start_prob < 0.2 are dropped.
#
# Output per species:
#   ${RESULTS_DIR}/<sp>/tiberius_filtered_epoch_74/
#     tiberius_filtered.gtf        — passing predictions only
#     tiberius_filtered.scores.tsv — per-transcript scores + pass_filter flag
#
#SBATCH --job-name=filt_tib_vert
#SBATCH --partition=vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/filt_tib_vert_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/filt_tib_vert_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
BENCH=/home/gabriell/tiberius_benchmarking

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml

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
TIB_GTF=${BENCH}/paper/Vertebrata/${species}/results/predictions/tiberius/tiberius_seqlen.gtf
OUTDIR=${RESULTS_DIR}/${species}/tiberius_filtered_epoch_74
OUT_GTF=${OUTDIR}/tiberius_filtered.gtf

if [[ -s "${OUT_GTF}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: tiberius_filtered.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing genome.fa" >&2; exit 0
fi
if [[ ! -s "${TIB_GTF}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing tiberius_seqlen.gtf" >&2; exit 0
fi

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] gtf=${TIB_GTF}"
echo "[$(date -Iseconds)] out=${OUT_GTF}"

python "${PROJDIR}/scripts/filter_by_orf_score.py" \
    --gtf         "${TIB_GTF}" \
    --genome      "${GENOME}" \
    --weights     "${WEIGHTS}" \
    --config      "${CONFIG}" \
    --out-gtf     "${OUT_GTF}" \
    --upstream-bp 500 \
    --min-coding  0.2 \
    --min-start   0.2 \
    --batch-size  200

echo "[$(date -Iseconds)] done -> ${OUT_GTF}"
