#!/bin/bash
#SBATCH --job-name=train_lstm_lr_r001
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/train_cnn_lstm_longread_run001_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/train_cnn_lstm_longread_run001_%j.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder

mkdir -p "${PROJDIR}/logs"
mkdir -p "${PROJDIR}/results/models/cnn_lstm_longread_run001"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${PROJDIR}/results/training_longread/tfrecord_manifest_available.tsv" \
    --val-manifest   "${PROJDIR}/results/val_longread/tfrecord_manifest_available.tsv" \
    --config         "${PROJDIR}/configs/cnn_lstm_longread_run001.yaml" \
    --outdir         "${PROJDIR}/results/models/cnn_lstm_longread_run001"
