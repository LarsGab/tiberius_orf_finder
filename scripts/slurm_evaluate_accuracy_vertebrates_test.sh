#!/bin/bash
# Evaluate the current Tiberius ORF-finder predictions (run006 epoch_74,
# tpm1cov3len300-filtered stringtie, subseq-collapsed) against the
# reference alongside ab-initio Tiberius, BRAKER3, and their merge with
# our ORF predictions, using evaluate_accuracy.py.
#
# Output layout:
#   ${RESULTS_DIR}/eval_accuracy_epoch_74_filt/
#     preds/<sp>/prediction.gtf   (symlinks to orfs.filtered.gtf)
#     accuracy_table.tsv
#     accuracy_figure.pdf
#     gffcompare_runs/<sp>/...
#
#SBATCH --job-name=eval_acc_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_vt_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_vt_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
TIB_DIR_NAME=annotate_epoch_74_${FILT_TAG}

EVAL_TAG=${EVAL_TAG:-epoch_74_filt}
OUT_ROOT=${RESULTS_DIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

# Optional path template for the protein-filtered Tiberius GTF; when set,
# it is added as a gene set and used in the merge in place of ab-initio
# Tiberius. Pass via env, e.g.
#   TIB_FILT_TMPL=/home/.../diamond_filter/{sp}/tiberius/filtered.gtf sbatch ...
TIB_FILT_TMPL=${TIB_FILT_TMPL:-}
FILTER_ORF_VS_TIB=${FILTER_ORF_VS_TIB:-0}

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)

# Build prediction.gtf symlink tree.
for sp in "${SPECIES[@]}"; do
    src=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}/orfs.filtered.gtf
    if [[ ! -s "${src}" ]]; then
        echo "[skip preds] ${sp}: missing ${src}"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] running evaluate_accuracy.py"

EXTRA_ARGS=()
if [[ -n "${TIB_FILT_TMPL}" ]]; then
    EXTRA_ARGS+=(--tib-filtered-tmpl "${TIB_FILT_TMPL}")
fi
if [[ "${FILTER_ORF_VS_TIB}" == "1" ]]; then
    EXTRA_ARGS+=(--filter-orf-vs-tib-subseq)
fi

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir "${PRED_DIR}" \
    --out-dir  "${OUT_ROOT}" \
    --species "${SPECIES[@]}" \
    "${EXTRA_ARGS[@]}"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -80
