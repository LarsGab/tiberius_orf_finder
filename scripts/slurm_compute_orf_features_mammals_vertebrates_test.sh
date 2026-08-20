#!/bin/bash
# Compute per-ORF feature table for the three mammal vertebrates_test species.
#
# Prerequisite for each species:
#   annotate_epoch_74_filt_tpm1cov3len300_lorf/orfs.filtered.gtf
#   fix_stop/miniprot_scored.gff   (slurm_miniprothint_mammals_vertebrates_test.sh)
#   fix_stop/miniprothint/hc.gff
#   assembly/genome.fa
#
# Output:
#   results/vertebrates_test/<sp>/annotate_epoch_74_filt_tpm1cov3len300_lorf/orf_features.tsv
#
#SBATCH --job-name=orf_feat_mammals
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/orf_feat_mammals_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/orf_feat_mammals_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ANNOT_TAG=annotate_epoch_74_filt_tpm1cov3len300_lorf

declare -a SPECIES=(Bos_taurus Delphinapterus_leucas Homo_sapiens)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

ORFS="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
MINIPROT="${RESULTS_DIR}/${species}/fix_stop/miniprot_scored.gff"
HINTS="${RESULTS_DIR}/${species}/fix_stop/miniprothint/hc.gff"
GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
OUT="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orf_features.tsv"

mkdir -p "${PROJDIR}/logs"

echo "[$(date -Iseconds)] species=${species}"

test -s "${ORFS}"     || { echo "missing: ${ORFS}"     >&2; exit 2; }
test -s "${MINIPROT}" || { echo "missing: ${MINIPROT}" >&2; exit 2; }
test -s "${HINTS}"    || { echo "missing: ${HINTS}"    >&2; exit 2; }
test -s "${GENOME}"   || { echo "missing: ${GENOME}"   >&2; exit 2; }

if [[ -s "${OUT}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP: already done -> ${OUT}"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

python "${PROJDIR}/scripts/compute_orf_features.py" \
    --orfs-gtf       "${ORFS}" \
    --miniprot-gff   "${MINIPROT}" \
    --hints-gff      "${HINTS}" \
    --genome         "${GENOME}" \
    --out            "${OUT}"

echo "[$(date -Iseconds)] done -> ${OUT}"
wc -l "${OUT}"
