#!/bin/bash
# Train the ORFfinder fine-tuning classification head on vertebrate training
# species Tiberius predictions.
#
# Prerequisite: slurm_score_tiberius_train_vertebrates.sh must have finished
# and build_finetune_manifest.py must have been run to produce the manifest.
#
#SBATCH --job-name=train_finetune
#SBATCH --partition=vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_finetune_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/train_finetune_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
MANIFEST=${PROJDIR}/results/finetune/manifest_vertebrates_train.tsv
OUTDIR=${PROJDIR}/results/finetune/head_v1

mkdir -p "${PROJDIR}/logs"

[[ -s "${MANIFEST}" ]] || {
    echo "Manifest not found: ${MANIFEST}" >&2
    echo "Run build_finetune_manifest.py first." >&2
    exit 2
}

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

cd "${PROJDIR}"

echo "[$(date -Iseconds)] manifest=${MANIFEST}"
echo "[$(date -Iseconds)] out=${OUTDIR}"

python "${PROJDIR}/scripts/train_finetune.py" \
    --manifest   "${MANIFEST}" \
    --out-dir    "${OUTDIR}" \
    --val-frac   0.1 \
    --epochs     100 \
    --batch-size 512 \
    --hidden     64 32 \
    --dropout    0.3 \
    --lr         1e-3

echo "[$(date -Iseconds)] done -> ${OUTDIR}"
