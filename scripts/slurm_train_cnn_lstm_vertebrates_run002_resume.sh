#!/bin/bash
# Resume the vertebrates CNN-LSTM training (run001_v2) from epoch_11
# after the run001_v2 job 7499475 OOM'd at 64G after 9h49m.
#
# Picks up:
#   ${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2/epoch_11.weights.h5
# Continues into:
#   ${PROJDIR}/results/models/cnn_lstm_vertebrates_run002/
# (new outdir so the original epoch_01..11.weights.h5 stay intact).
#
# Same train/val manifests as run001 (val == train stopgap until val
# tfrecords are regenerated).
#
#SBATCH --job-name=train_lstm_vert002_resume
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_vertebrates_run002_resume_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_vertebrates_run002_resume_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

TRAIN_MANIFEST=${PROJDIR}/results/training_vertebrates_v2/tfrecord_manifest_available.tsv
VAL_MANIFEST=${TRAIN_MANIFEST}

# Resume from the latest saved checkpoint. run002 itself reached epoch 14
# before a silent C-level abort during epoch 15 (job 7504513). Picking
# up from epoch_14 in a new run003 outdir so prior checkpoints stay
# intact for diffing if this resume also dies.
PREV_OUTDIR=${PROJDIR}/results/models/cnn_lstm_vertebrates_run002
INIT_WEIGHTS=${PREV_OUTDIR}/epoch_14.weights.h5
INITIAL_EPOCH=14

OUTDIR=${PROJDIR}/results/models/cnn_lstm_vertebrates_run003

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
