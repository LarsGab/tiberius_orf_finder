#!/bin/bash
# Vertebrates CNN-LSTM training on the post-crash regenerated tfrecords.
# Reads the *currently available* training set (49 of 68 species at the
# time of writing -- the from-scratch pipeline is still working through
# the remaining species). You can re-launch later (or use a fresh model
# outdir) once more species have tfrecords.
#
# val_manifest == train_manifest is a stopgap: val/test tfrecords haven't
# been regenerated with the new label_transcripts code yet. val_loss will
# mirror train_loss, so early-stopping is effectively disabled until
# proper validation tfrecords land at:
#   results/training_vertebrates_validation_v2/tfrecord_manifest_available.tsv
#
#SBATCH --job-name=train_lstm_vert001
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_vertebrates_run001_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_cnn_lstm_vertebrates_run001_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

TRAIN_MANIFEST=${PROJDIR}/results/training_vertebrates_v2/tfrecord_manifest_available.tsv
VAL_MANIFEST=${TRAIN_MANIFEST}    # stopgap -- see header

OUTDIR=${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Pre-flight sanity checks
test -r "${TRAIN_MANIFEST}" || { echo "missing train manifest: ${TRAIN_MANIFEST}" >&2; exit 2; }
n=$(wc -l < "${TRAIN_MANIFEST}")
test "${n}" -gt 0 || { echo "empty train manifest" >&2; exit 2; }
test -r "${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml" || {
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
    --config         "${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml" \
    --outdir         "${OUTDIR}"
