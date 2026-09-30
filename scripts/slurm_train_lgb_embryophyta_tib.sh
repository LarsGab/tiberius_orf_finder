#!/bin/bash
# Train a 3-class LGB on Tiberius ab initio predictions for embryophyta training species.
# Excludes Thinopyrum_intermedium (Tiberius OOM at 128G).
#
# Input:
#   results/training_embryophyta_v2/<sp>/tiberius/tib_features.tsv
#
# Output:
#   results/filter_analysis/lgb_embryophyta_tib/lgb_3class_model.pkl
#   results/filter_analysis/lgb_embryophyta_tib/lgb_3class_scores.tsv
#   results/filter_analysis/lgb_embryophyta_tib/lgb_3class.pdf
#
#SBATCH --job-name=train_lgb_emb_tib
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_lgb_emb_tib_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_lgb_emb_tib_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
BASE_DIR=${PROJDIR}/results/training_embryophyta_v2
OUT_DIR=${PROJDIR}/results/filter_analysis/lgb_embryophyta_tib

mkdir -p "${PROJDIR}/logs" "${OUT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

python scripts/train_orf_lgb_3class.py \
    --base-dir          "${BASE_DIR}" \
    --annot-tag         tiberius \
    --features-filename tib_features.tsv \
    --exclude-species   Thinopyrum_intermedium \
    --out-dir           "${OUT_DIR}" \
    --n-estimators      500

echo "[$(date -Iseconds)] done -> ${OUT_DIR}"
