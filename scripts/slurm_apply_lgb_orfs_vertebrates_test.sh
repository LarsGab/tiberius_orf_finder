#!/bin/bash
# Apply the trained 3-class LGB model to ORF predictions for all 6
# vertebrates_test species.
#
# Input (per species):
#   results/vertebrates_test/<sp>/annotate_epoch_74_filt_tpm1cov3len300_lorf/
#     orf_features.tsv
#     orfs.filtered.gtf
#
# Output (per species):
#   orfs_lgb3_filtered.gtf      — transcripts with P(not-wrong) >= 0.5
#   orfs_lgb3_filtered.scores.tsv
#
#SBATCH --job-name=lgb_orfs_test
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_orfs_test.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_orfs_test.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

python scripts/apply_lgb_model_gtf.py \
    --model     results/filter_analysis/lgb_3class_model.pkl \
    --base-dir  results/vertebrates_test \
    --annot-tag annotate_epoch_74_filt_tpm1cov3len300_lorf \
    --in-gtf    orfs.filtered.gtf \
    --out-gtf   orfs_lgb3_filtered.gtf

echo "[$(date -Iseconds)] done"
