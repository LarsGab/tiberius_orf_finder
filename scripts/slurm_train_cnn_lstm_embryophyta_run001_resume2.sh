#!/bin/bash
# Resume embryophyta CNN-LSTM run001_v2 training from the latest epoch checkpoint.
# Detects the highest epoch_N.weights.h5 at job-start time automatically.
#
#SBATCH --job-name=train_emb001_r
#SBATCH --partition=storm,vision,vision-fast
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_emb001_resume_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_emb001_resume_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
OUTDIR=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2
TRAIN_MANIFEST=${PROJDIR}/results/training_embryophyta_v2/tfrecord_manifest_available.tsv

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

INIT_WEIGHTS=$(ls "${OUTDIR}"/epoch_*.weights.h5 2>/dev/null | sort -V | tail -1)
[[ -n "${INIT_WEIGHTS}" ]] || { echo "no epoch checkpoints in ${OUTDIR}" >&2; exit 2; }
INITIAL_EPOCH=$(basename "${INIT_WEIGHTS}" .weights.h5 | sed 's/epoch_//')

echo "[$(date -Iseconds)] resuming from ${INIT_WEIGHTS} (epoch ${INITIAL_EPOCH})"

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest   "${TRAIN_MANIFEST}" \
    --config         "${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml" \
    --outdir         "${OUTDIR}" \
    --init-weights   "${INIT_WEIGHTS}" \
    --initial-epoch  "${INITIAL_EPOCH}"
