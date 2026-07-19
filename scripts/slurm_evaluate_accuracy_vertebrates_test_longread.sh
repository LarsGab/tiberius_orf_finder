#!/bin/bash
# Evaluate long-read benchmark: ORF predictions from LR StringTie vs
# the reference annotation for the 4 long-read species.
#
# Uses the same evaluate_accuracy.py as the short-read benchmark.
# Tiberius predictions re-use the protein-filtered GTF from the
# existing SR run (same model, same genome).
#
# Output: ${RESULTS_DIR}/eval_accuracy_epoch_74_lr_filt_pfilt/
#
# Run after slurm_annotate_vertebrates_test_longread.sh completes.
#
#SBATCH --job-name=eval_lr_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lr_vt_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lr_vt_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300

EVAL_TAG=${EVAL_TAG:-epoch_74_lr_filt}
OUT_ROOT=${RESULTS_DIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

# Tiberius protein-filtered GTF from the existing SR run — keeps the
# comparison fair (only ORF input changes between SR and LR benchmarks).
DEFAULT_TIB_FILT_TMPL='/home/gabriell/tiberius_proteins_analysis/diamond_filter/{sp}/tiberius/filtered.gtf'
TIB_FILT_TMPL=${TIB_FILT_TMPL:-$DEFAULT_TIB_FILT_TMPL}

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

SPECIES=(Gallus_gallus Bos_taurus Zootoca_vivipara Betta_splendens)

for sp in "${SPECIES[@]}"; do
    src=${RESULTS_DIR}/${sp}/annotate_epoch_74_lr_${FILT_TAG}/orfs.filtered.gtf
    if [[ ! -s "${src}" ]]; then
        echo "[skip preds] ${sp}: missing ${src}"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] running evaluate_accuracy.py (long-read ORF)"
echo "[$(date -Iseconds)] tib_filt_tmpl=${TIB_FILT_TMPL}"

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir  "${PRED_DIR}" \
    --out-dir   "${OUT_ROOT}" \
    --species   "${SPECIES[@]}" \
    --tib-filtered-tmpl "${TIB_FILT_TMPL}"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -60
