#!/bin/bash
# Train the fungi 3-class LGB model from features prepared per species.
# Reads orf_features.tsv from each species' annotate_run006_lgb50_prep/ dir
# and pools them via train_orf_lgb_3class.py --base-dir / --annot-tag.
#
#SBATCH --job-name=train_fungi_lgb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_fungi_lgb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_fungi_lgb_%j.err

set -euo pipefail
PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
BASE_DIR=${BASE_DIR:-${PROJDIR}/results/training_fungi_v2}
ANNOT_TAG=${ANNOT_TAG:-annotate_run006_lgb50_prep}
OUT_DIR=${OUT_DIR:-${PROJDIR}/results/filter_analysis/lgb_fungi_v2}

mkdir -p "${OUT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

python "${PROJDIR}/scripts/train_orf_lgb_3class.py" \
    --base-dir  "${BASE_DIR}" \
    --annot-tag "${ANNOT_TAG}" \
    --out-dir   "${OUT_DIR}"

echo "[$(date -Iseconds)] fungi LGB trained -> ${OUT_DIR}/lgb_3class_model.pkl"
