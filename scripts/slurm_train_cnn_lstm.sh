#!/bin/bash
#SBATCH --job-name=train_cnn_lstm
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/train_cnn_lstm_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/train_cnn_lstm_%j.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder

mkdir -p "${PROJDIR}/logs"
mkdir -p "${PROJDIR}/results/models/cnn_lstm_run001"

eval "$(micromamba shell hook --shell bash)"
micromamba activate storm

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${PROJDIR}/results/training/tfrecord_manifest_available.tsv" \
    --val-manifest   "${PROJDIR}/results/val/tfrecord_manifest_available.tsv" \
    --config         "${PROJDIR}/configs/cnn_lstm.yaml" \
    --outdir         "${PROJDIR}/results/models/cnn_lstm_run001"
