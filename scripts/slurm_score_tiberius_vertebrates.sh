#!/bin/bash
# Score Tiberius ab initio predictions (tiberius_seqlen.gtf) for the 5
# vertebrates_test species using the ORFfinder model (epoch_74, run006).
#
# For each species the script:
#   1. Runs score_tiberius.py — forward NN pass on each Tiberius-predicted
#      CDS sequence; outputs a TSV with per-transcript coding-probability
#      scores (mean_coding_prob, start_prob, stop_prob, mean_ir_prob, …).
#
# Output:
#   ${RESULTS_DIR}/<species>/score_tiberius_epoch_74/scores.tsv
#
#SBATCH --job-name=score_tib_vert
#SBATCH --partition=vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/score_tib_vert_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/score_tib_vert_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
BENCH=/home/gabriell/tiberius_benchmarking

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml
WEIGHTS_TAG=epoch_74

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
OUTDIR=${RESULTS_DIR}/${species}/score_tiberius_${WEIGHTS_TAG}
OUT_TSV=${OUTDIR}/scores.tsv

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config: ${CONFIG}"   >&2; exit 2; }

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing genome.fa" >&2
    exit 0
fi
if [[ ! -s "${TIB_GTF}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing tiberius_seqlen.gtf" >&2
    exit 0
fi

if [[ -s "${OUT_TSV}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: scores.tsv already exists"
    exit 0
fi

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species} weights=${WEIGHTS_TAG}"
echo "[$(date -Iseconds)] gtf=${TIB_GTF}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] out=${OUT_TSV}"

python "${PROJDIR}/scripts/score_tiberius.py" \
    --gtf        "${TIB_GTF}" \
    --genome     "${GENOME}" \
    --weights    "${WEIGHTS}" \
    --config     "${CONFIG}" \
    --out-tsv    "${OUT_TSV}" \
    --batch-size 200

echo "[$(date -Iseconds)] done -> ${OUT_TSV}"
