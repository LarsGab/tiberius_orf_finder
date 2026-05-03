#!/bin/bash
#SBATCH --job-name=eval_lstm_r002_best
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/eval_cnn_lstm_run002_best_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/eval_cnn_lstm_run002_best_%j.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder

mkdir -p "${PROJDIR}/logs"
mkdir -p "${PROJDIR}/results/eval"

eval "$(micromamba shell hook --shell bash)"
micromamba activate storm

python "${PROJDIR}/scripts/evaluate.py" \
    --test-manifest "${PROJDIR}/results/test/tfrecord_manifest_available.tsv" \
    --weights       "${PROJDIR}/results/models/cnn_lstm_run002/best.weights.h5" \
    --config        "${PROJDIR}/configs/cnn_lstm.yaml" \
    --out           "${PROJDIR}/results/eval/cnn_lstm_run002_best.tsv"
