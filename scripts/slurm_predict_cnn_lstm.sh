#!/bin/bash
#SBATCH --job-name=predict_cnn_lstm
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/predict_cnn_lstm_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/predict_cnn_lstm_%j.err

set -euo pipefail

eval "$(/home/gabriell/.local/bin/micromamba shell hook --shell bash)"
micromamba activate storm

cd /home/gabriell/tiberius_orf_finder

python /home/gabriell/tiberius_orf_finder/scripts/predict.py \
    --test-manifest /home/gabriell/tiberius_orf_finder/results/test/tfrecord_manifest_available.tsv \
    --weights       /home/gabriell/tiberius_orf_finder/results/models/cnn_lstm_run001/epoch_15.weights.h5 \
    --config        /home/gabriell/tiberius_orf_finder/configs/cnn_lstm.yaml \
    --out-dir       /home/gabriell/tiberius_orf_finder/results/predictions/cnn_lstm_run001_epoch15 \
    --batch-size    200
