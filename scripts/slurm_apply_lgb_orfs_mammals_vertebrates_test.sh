#!/bin/bash
# Apply the 3-class LGB model to ORF predictions for the three mammal
# vertebrates_test species (Bos_taurus, Delphinapterus_leucas, Homo_sapiens).
#
# Prerequisite: slurm_compute_orf_features_mammals_vertebrates_test.sh
#
# Output (per species):
#   results/vertebrates_test/<sp>/annotate_epoch_74_filt_tpm1cov3len300_lorf/
#     orfs_lgb3_filtered.gtf
#     orfs_lgb3_filtered.scores.tsv
#
#SBATCH --job-name=lgb_orfs_mammals
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_orfs_mammals_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_orfs_mammals_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ANNOT_TAG=annotate_epoch_74_filt_tpm1cov3len300_lorf
MODEL=${PROJDIR}/results/filter_analysis/lgb_3class_model.pkl

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

for species in Bos_taurus Delphinapterus_leucas Homo_sapiens; do
    FEAT="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orf_features.tsv"
    IN_GTF="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
    OUT_GTF="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orfs_lgb3_filtered.gtf"

    if [[ ! -s "${FEAT}" ]]; then
        echo "[skip] ${species}: orf_features.tsv not found"
        continue
    fi
    if [[ -s "${OUT_GTF}" && "${FORCE:-0}" != "1" ]]; then
        echo "[skip] ${species}: orfs_lgb3_filtered.gtf already exists"
        continue
    fi

    echo "[$(date -Iseconds)] ${species}"
    python scripts/apply_lgb_model_gtf.py \
        --model    "${MODEL}" \
        --features "${FEAT}" \
        --in-gtf   "${IN_GTF}" \
        --out-gtf  "${OUT_GTF}"

    N=$(awk '$3=="transcript"' "${OUT_GTF}" | wc -l || true)
    echo "[$(date -Iseconds)] ${species}: ${N} transcripts retained"
done

echo "[$(date -Iseconds)] done"
