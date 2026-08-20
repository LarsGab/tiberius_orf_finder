#!/bin/bash
# Train the embryophyta-specific 3-class LGB model.
#
# Prerequisite: slurm_features_training_embryophyta_v2.sh
#
# Output:
#   results/filter_analysis/lgb_embryophyta/lgb_3class_model.pkl
#   results/filter_analysis/lgb_embryophyta/lgb_3class.pdf
#
#SBATCH --job-name=train_lgb_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_lgb_emb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_lgb_emb_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2
ANNOT_TAG=${ANNOT_TAG:-annotate_run001_e300}
OUTDIR=${PROJDIR}/results/filter_analysis/lgb_embryophyta

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] training embryophyta LGB …"
python "${PROJDIR}/scripts/train_orf_lgb_3class.py" \
    --base-dir   "${TRAINDIR}" \
    --annot-tag  "${ANNOT_TAG}" \
    --out-dir    "${OUTDIR}" \
    --n-estimators 500

echo "[$(date -Iseconds)] done -> ${OUTDIR}/lgb_3class_model.pkl"
