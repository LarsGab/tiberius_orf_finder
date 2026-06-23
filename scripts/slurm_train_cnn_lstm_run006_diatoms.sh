#!/bin/bash
#SBATCH --job-name=train_r006_dia
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_run006_diatoms_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_run006_diatoms_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
CLADE=diatoms

# Training data (regenerated post-crash, includes partial-ref categories):
TRAIN_MANIFEST=${PROJDIR}/results/training_${CLADE}_v2/tfrecord_manifest_available.tsv

# Validation data: NOT yet regenerated with the updated label_transcripts
# code. As a stopgap we re-use the training manifest so train.py's required
# --val-manifest is satisfied; val_loss will mirror train_loss, so no real
# early-stopping signal until validation tfrecords are regenerated.
VAL_MANIFEST=${TRAIN_MANIFEST}

OUTDIR=${PROJDIR}/results/models/cnn_lstm_run006_${CLADE}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Sanity checks before we waste GPU time.
test -r "${TRAIN_MANIFEST}" || { echo "missing train manifest: ${TRAIN_MANIFEST}" >&2; exit 2; }
test -r "${PROJDIR}/configs/cnn_lstm_run006.yaml" || { echo "missing config" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] train_manifest=${TRAIN_MANIFEST} ($(wc -l < ${TRAIN_MANIFEST}) species)"
echo "[$(date -Iseconds)] val_manifest=${VAL_MANIFEST} (== train; see header comment)"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest   "${VAL_MANIFEST}" \
    --config         "${PROJDIR}/configs/cnn_lstm_run006.yaml" \
    --outdir         "${OUTDIR}"
