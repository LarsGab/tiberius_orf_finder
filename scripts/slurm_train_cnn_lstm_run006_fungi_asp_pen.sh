#!/bin/bash
# Train CNN-LSTM (run006 architecture) on Aspergillus + Penicillium species only.
# Uses the same config as the full fungi run006 but restricted to 38 Asp/Pen
# training species; validation uses the general fungi val manifest.
#
# Model output: results/models/cnn_lstm_run006_fungi_asp_pen/
#
#SBATCH --job-name=train_r006_asp_pen
#SBATCH --partition=vision,vision-fast
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_r006_asp_pen_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_r006_asp_pen_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

mkdir -p "${PROJDIR}/logs"
mkdir -p "${PROJDIR}/results/models/cnn_lstm_run006_fungi_asp_pen"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${PROJDIR}/results/fungi_train_manifest_asp_pen.tsv" \
    --val-manifest   "${PROJDIR}/results/fungi_val_manifest.tsv" \
    --config         "${PROJDIR}/configs/cnn_lstm_run006.yaml" \
    --outdir         "${PROJDIR}/results/models/cnn_lstm_run006_fungi_asp_pen"
