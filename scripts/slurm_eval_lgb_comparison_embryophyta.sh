#!/bin/bash
# Evaluate 12 LGB-filtered gene sets for all 6 Embryophyta test species
# and produce a comparison bar-chart PDF.
#
# Prerequisite: slurm_lgb_apply_embryophyta_test.sh
#
#SBATCH --job-name=eval_lgb_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lgb_emb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lgb_emb_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
OUTDIR=${PROJDIR}/results/training_embryophyta_test_v2/eval_lgb_comparison

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] Starting LGB gene-set comparison for Embryophyta …"

python scripts/plot_lgb_comparison.py \
    --kingdom embryophyta \
    --out-dir "${OUTDIR}"

echo "[$(date -Iseconds)] done → ${OUTDIR}"
if [[ -s "${OUTDIR}/lgb_comparison_table.tsv" ]]; then
    column -t -s$'\t' "${OUTDIR}/lgb_comparison_table.tsv"
fi
