#!/bin/bash
#SBATCH --job-name=train_r006_emb
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/train_run006_embryophyta_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/train_run006_embryophyta_%j.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder
CLADE=embryophyta

mkdir -p "${PROJDIR}/logs"
mkdir -p "${PROJDIR}/results/models/cnn_lstm_run006_${CLADE}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${PROJDIR}/results/${CLADE}_training/tfrecord_manifest_available.tsv" \
    --val-manifest   "${PROJDIR}/results/${CLADE}_validation/tfrecord_manifest_available.tsv" \
    --config         "${PROJDIR}/configs/cnn_lstm_run006.yaml" \
    --outdir         "${PROJDIR}/results/models/cnn_lstm_run006_${CLADE}"
