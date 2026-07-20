#!/bin/bash
# Resume embryophyta CNN-LSTM run001 training from epoch 47.
# Increases mem from 64G to 192G to avoid the OOM that killed job 7586365.
#
#SBATCH --job-name=train_lstm_emb001r
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_embryophyta_run001_resume_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_embryophyta_run001_resume_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

TRAIN_MANIFEST=${PROJDIR}/results/training_embryophyta_v2/tfrecord_manifest_available.tsv
VAL_MANIFEST=${TRAIN_MANIFEST}    # stopgap -- see original script header

OUTDIR=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2
INIT_WEIGHTS=${OUTDIR}/epoch_47.weights.h5
INITIAL_EPOCH=47

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

test -r "${TRAIN_MANIFEST}"  || { echo "missing train manifest" >&2; exit 2; }
test -r "${INIT_WEIGHTS}"    || { echo "missing weights: ${INIT_WEIGHTS}" >&2; exit 2; }
test -r "${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml" || { echo "missing config" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] resuming from epoch ${INITIAL_EPOCH} (${INIT_WEIGHTS})"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest   "${VAL_MANIFEST}" \
    --config         "${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml" \
    --outdir         "${OUTDIR}" \
    --init-weights   "${INIT_WEIGHTS}" \
    --initial-epoch  "${INITIAL_EPOCH}"
