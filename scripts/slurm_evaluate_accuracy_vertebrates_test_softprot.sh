#!/bin/bash
# evaluate_accuracy for the vertebrates_test set using the SOFT-protein-
# filtered ORF predictions as prediction.gtf. Mirrors
# slurm_evaluate_accuracy_vertebrates_test.sh but points pred-dir symlinks
# at orfs.filtered_soft_protein.gtf.
#
# Output layout:
#   ${RESULTS_DIR}/eval_accuracy_epoch_74_filt_pfilt_softprot/
#     preds/<sp>/prediction.gtf   (symlinks to orfs.filtered_soft_protein.gtf)
#     accuracy_table.tsv
#     accuracy_figure.pdf
#
# Requires the soft filter to have been run first:
#   sbatch scripts/slurm_filter_orf_by_diamond_vertebrates_test.sh
#
#SBATCH --job-name=eval_acc_sp
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_sp_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_sp_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
TIB_DIR_NAME=annotate_epoch_74_${FILT_TAG}

SOFTPROT_TAG=${SOFTPROT_TAG:-default}
EVAL_TAG=${EVAL_TAG:-epoch_74_filt_pfilt_softprot_${SOFTPROT_TAG}}
OUT_ROOT=${RESULTS_DIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds
# `${VAR:-DEFAULT}` matches the FIRST `}` in DEFAULT as its terminator, so an
# inline default that contains `{sp}` gets mangled. Assign the default to a
# separate var first so the `{sp}` placeholder is preserved verbatim.
DEFAULT_TIB_FILT_TMPL='/home/gabriell/tiberius_proteins_analysis/diamond_filter/{sp}/tiberius/filtered.gtf'
TIB_FILT_TMPL=${TIB_FILT_TMPL:-$DEFAULT_TIB_FILT_TMPL}

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)

for sp in "${SPECIES[@]}"; do
    if [[ "${SOFTPROT_TAG}" == "default" ]]; then
        sp_softdir=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}/soft_protein
    else
        sp_softdir=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}/soft_protein_${SOFTPROT_TAG}
    fi
    src=${sp_softdir}/orfs.filtered_soft_protein.gtf
    if [[ ! -s "${src}" ]]; then
        echo "[skip preds] ${sp}: missing ${src}"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] running evaluate_accuracy.py (soft-protein ORF)"

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir "${PRED_DIR}" \
    --out-dir  "${OUT_ROOT}" \
    --species "${SPECIES[@]}" \
    --tib-filtered-tmpl "${TIB_FILT_TMPL}"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -80
