#!/bin/bash
#SBATCH --job-name=predict_lstm_r005
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/predict_cnn_lstm_run005_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/predict_cnn_lstm_run005_%j.err

set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd /home/gabriell/tiberius_orf_finder

mkdir -p /home/gabriell/tiberius_orf_finder/results/predictions/cnn_lstm_run005_epoch197

python scripts/predict.py \
    --test-manifest /home/gabriell/tiberius_orf_finder/results/test/tfrecord_manifest_available.tsv \
    --weights /home/gabriell/tiberius_orf_finder/results/models/cnn_lstm_run005/epoch_197.weights.h5 \
    --config configs/cnn_lstm_run005.yaml \
    --out-dir /home/gabriell/tiberius_orf_finder/results/predictions/cnn_lstm_run005_epoch197 \
    --batch-size 200
