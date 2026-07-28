#!/bin/bash
#SBATCH --job-name=train_r006_fun
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_run006_fungi_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_run006_fungi_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
CLADE=fungi

mkdir -p "${PROJDIR}/logs"
mkdir -p "${PROJDIR}/results/models/cnn_lstm_run006_${CLADE}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${PROJDIR}/results/fungi_train_manifest.tsv" \
    --val-manifest   "${PROJDIR}/results/fungi_val_manifest.tsv" \
    --config         "${PROJDIR}/configs/cnn_lstm_run006.yaml" \
    --outdir         "${PROJDIR}/results/models/cnn_lstm_run006_${CLADE}"
