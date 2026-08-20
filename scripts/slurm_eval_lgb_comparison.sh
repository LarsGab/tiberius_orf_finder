#!/bin/bash
# Evaluate 12 LGB-filtered gene sets for all 6 vertebrates_test species
# and produce a comparison bar-chart PDF.
#
# Prerequisite jobs:
#   slurm_apply_lgb_orfs_vertebrates_test.sh     (orfs_lgb3_filtered.gtf)
#   slurm_apply_lgb_tiberius_vertebrates_test.sh (tiberius_lgb_filtered.gtf)
#
# Gene sets per species (12 total):
#   ORFs (full/lgb3/lgb3-correct/lgb3-partial)
#   Tiberius (full/lgb3/lgb3-correct/lgb3-partial)
#   Merge: ORFs+Tib full | Tib-c+ORFs-c | Tib-c+ORFs-full
#   BRAKER3
#
# Output:
#   results/vertebrates_test/eval_lgb_comparison/
#     lgb_comparison_table.tsv
#     lgb_comparison.pdf
#     gffcompare_runs/<sp>/...
#
#SBATCH --job-name=eval_lgb_cmp
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lgb_cmp_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lgb_cmp_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
OUTDIR=${PROJDIR}/results/vertebrates_test/eval_lgb_comparison

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] Starting LGB gene-set comparison …"

python scripts/plot_lgb_comparison.py \
    --out-dir "${OUTDIR}"

echo "[$(date -Iseconds)] done → ${OUTDIR}"
echo "Table: ${OUTDIR}/lgb_comparison_table.tsv"
echo "Figure: ${OUTDIR}/lgb_comparison.pdf"

# Print table to log
if [[ -s "${OUTDIR}/lgb_comparison_table.tsv" ]]; then
    column -t -s$'\t' "${OUTDIR}/lgb_comparison_table.tsv"
fi
