#!/bin/bash
# Train a 3-class LGB on a mix of ORF-finder and Tiberius ab initio predictions
# for embryophyta training species.  Excludes Thinopyrum_intermedium (Tiberius
# OOM at 128G).
#
# Inputs:
#   results/training_embryophyta_v2/<sp>/annotate_run001_e300/orf_features.tsv
#   results/training_embryophyta_v2/<sp>/tiberius/tib_features.tsv
#
# Output:
#   results/filter_analysis/lgb_embryophyta_mix/lgb_3class_model.pkl
#   results/filter_analysis/lgb_embryophyta_mix/lgb_3class_scores.tsv  (has 'source' column)
#   results/filter_analysis/lgb_embryophyta_mix/lgb_3class.pdf
#
#SBATCH --job-name=train_lgb_emb_mix
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_lgb_emb_mix_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_lgb_emb_mix_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
BASE_DIR=${PROJDIR}/results/training_embryophyta_v2
OUT_DIR=${PROJDIR}/results/filter_analysis/lgb_embryophyta_mix

mkdir -p "${PROJDIR}/logs" "${OUT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

python scripts/train_orf_lgb_3class.py \
    --base-dir        "${BASE_DIR}" \
    --sources         annotate_run001_e300/orf_features.tsv \
                      tiberius/tib_features.tsv \
    --exclude-species Thinopyrum_intermedium \
    --out-dir         "${OUT_DIR}" \
    --n-estimators    500

echo "[$(date -Iseconds)] done -> ${OUT_DIR}"
