#!/bin/bash
# Embryophyta CNN-LSTM training on the post-crash regenerated tfrecords.
# Reads the *currently available* training set (37 of 45 species dirs at
# the time of writing -- excluded species are big-genome / VARUS-no-SRA
# fails documented in results/integrity_checks/split_embryophyta_training/).
#
# val_manifest == train_manifest is a stopgap: embryophyta has no val/test
# split under results/training_embryophyta_{val,test}_v2/ yet. val_loss
# will mirror train_loss, so early-stopping is effectively disabled until
# proper validation tfrecords land.
#
#SBATCH --job-name=train_lstm_emb001
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_embryophyta_run001_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_embryophyta_run001_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

TRAIN_MANIFEST=${PROJDIR}/results/training_embryophyta_v2/tfrecord_manifest_available.tsv
VAL_MANIFEST=${TRAIN_MANIFEST}    # stopgap -- see header

OUTDIR=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Pre-flight sanity checks
test -r "${TRAIN_MANIFEST}" || { echo "missing train manifest: ${TRAIN_MANIFEST}" >&2; exit 2; }
n=$(wc -l < "${TRAIN_MANIFEST}")
test "${n}" -gt 0 || { echo "empty train manifest" >&2; exit 2; }
test -r "${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml" || {
    echo "missing config" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] train_manifest=${TRAIN_MANIFEST} (${n} species)"
echo "[$(date -Iseconds)] val_manifest=${VAL_MANIFEST} (== train; see header)"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/train.py" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest   "${VAL_MANIFEST}" \
    --config         "${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml" \
    --outdir         "${OUTDIR}"
