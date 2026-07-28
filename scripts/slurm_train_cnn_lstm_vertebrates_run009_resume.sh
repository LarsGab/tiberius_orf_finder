#!/bin/bash
# Resume vertebrates CNN-LSTM training (run008) from epoch_74
# after job 7610011 was cancelled due to time limit on 2026-07-19.
#
# Picks up:
#   cnn_lstm_vertebrates_run008/epoch_74.weights.h5
# Continues into:
#   cnn_lstm_vertebrates_run009/
# (new outdir so prior checkpoints stay intact)
#
#SBATCH --job-name=train_lstm_vert009_resume
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_vertebrates_run009_resume_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_vertebrates_run009_resume_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

TRAIN_MANIFEST=${PROJDIR}/results/training_vertebrates_v2/tfrecord_manifest_available.tsv
VAL_MANIFEST=${TRAIN_MANIFEST}

PREV_OUTDIR=${PROJDIR}/results/models/cnn_lstm_vertebrates_run008
INIT_WEIGHTS=${PREV_OUTDIR}/epoch_74.weights.h5
INITIAL_EPOCH=74

OUTDIR=${PROJDIR}/results/models/cnn_lstm_vertebrates_run009

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Pre-flight sanity
test -r "${TRAIN_MANIFEST}" || { echo "missing train manifest: ${TRAIN_MANIFEST}" >&2; exit 2; }
test -r "${INIT_WEIGHTS}"   || { echo "missing init weights: ${INIT_WEIGHTS}" >&2; exit 2; }
test -r "${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml" || {
    echo "missing config" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] train_manifest=${TRAIN_MANIFEST} ($(wc -l < ${TRAIN_MANIFEST}) species)"
echo "[$(date -Iseconds)] init_weights=${INIT_WEIGHTS}"
echo "[$(date -Iseconds)] initial_epoch=${INITIAL_EPOCH}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest   "${VAL_MANIFEST}" \
    --config         "${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml" \
    --outdir         "${OUTDIR}" \
    --init-weights   "${INIT_WEIGHTS}" \
    --initial-epoch  "${INITIAL_EPOCH}"
