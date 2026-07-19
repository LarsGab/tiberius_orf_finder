#!/bin/bash
# Evaluate the embryophyta CNN-LSTM run001 model on the held-out test set.
# Usage: sbatch scripts/slurm_evaluate_cnn_lstm_embryophyta_run001.sh <epoch_int>
#
#SBATCH --job-name=eval_lstm_emb001
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_cnn_lstm_embryophyta_run001_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_cnn_lstm_embryophyta_run001_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
EPOCH="${1:?usage: $0 <epoch_int>}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

mkdir -p "${PROJDIR}/logs" "${PROJDIR}/results/eval"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} epoch=${EPOCH}"

python "${PROJDIR}/scripts/evaluate.py" \
    --test-manifest "${PROJDIR}/results/training_embryophyta_test_v2/tfrecord_manifest_available.tsv" \
    --weights       "${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2/epoch_${EPOCH}.weights.h5" \
    --config        "${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml" \
    --out           "${PROJDIR}/results/eval/cnn_lstm_embryophyta_run001_ep${EPOCH}.tsv"

echo "[$(date -Iseconds)] done -> results/eval/cnn_lstm_embryophyta_run001_ep${EPOCH}.tsv"
