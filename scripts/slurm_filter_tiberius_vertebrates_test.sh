#!/bin/bash
# Filter Tiberius ab initio predictions for all 6 vertebrates_test species
# using ORFfinder model scores (epoch_74, run006, 500 bp upstream context).
#
# Source: /home/gabriell/tiberius_benchmarking/paper/Vertebrata/<sp>/
#           results/predictions/tiberius/tiberius_seqlen.gtf
# Output: ${RESULTS_DIR}/<sp>/tiberius_filtered_epoch_74/
#           tiberius_filtered.gtf        — passing predictions
#           tiberius_filtered.scores.tsv — per-transcript scores
#
#SBATCH --job-name=filt_tib_vert_test
#SBATCH --partition=vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/filt_tib_vert_test_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/filt_tib_vert_test_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
BENCH=/home/gabriell/tiberius_benchmarking

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
TIB_GTF=${BENCH}/paper/Vertebrata/${species}/results/predictions/tiberius/tiberius_seqlen.gtf
OUTDIR=${RESULTS_DIR}/${species}/tiberius_filtered_epoch_74
OUT_GTF=${OUTDIR}/tiberius_filtered.gtf

echo "[$(date -Iseconds)] species=${species}"

test -s "${WEIGHTS}"  || { echo "missing weights: ${WEIGHTS}"  >&2; exit 2; }
test -s "${CONFIG}"   || { echo "missing config: ${CONFIG}"    >&2; exit 2; }
test -s "${TIB_GTF}"  || { echo "missing tib gtf: ${TIB_GTF}" >&2; exit 2; }

if [[ -s "${OUT_GTF}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: tiberius_filtered.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi
test -s "${GENOME}" || { echo "missing genome: ${GENOME}" >&2; exit 2; }

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

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
wc -l "${OUT_GTF}"
