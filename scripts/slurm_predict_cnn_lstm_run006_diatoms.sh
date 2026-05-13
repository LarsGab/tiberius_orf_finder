#!/bin/bash
#SBATCH --job-name=pred_r006_dia
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/predict_run006_diatoms_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/predict_run006_diatoms_%j.err

# Usage:  sbatch scripts/slurm_predict_cnn_lstm_run006_diatoms.sh <epoch_N>
set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder
CLADE=diatoms
EPOCH="${1:?usage: $0 <epoch_int>}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"
mkdir -p "${PROJDIR}/results/predictions/cnn_lstm_run006_${CLADE}_epoch${EPOCH}"

python scripts/predict.py \
    --test-manifest "${PROJDIR}/results/${CLADE}_test/tfrecord_manifest_available.tsv" \
    --weights       "${PROJDIR}/results/models/cnn_lstm_run006_${CLADE}/epoch_${EPOCH}.weights.h5" \
    --config        configs/cnn_lstm_run006.yaml \
    --out-dir       "${PROJDIR}/results/predictions/cnn_lstm_run006_${CLADE}_epoch${EPOCH}" \
    --batch-size 200
