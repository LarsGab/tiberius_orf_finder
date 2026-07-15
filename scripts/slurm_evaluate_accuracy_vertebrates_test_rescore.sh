#!/bin/bash
# evaluate_accuracy with cross-source protein-evidence rescoring at merge:
# per species, jointly cluster protein-filtered Tiberius + ORF-finder
# predictions and drop transcripts whose Diamond bitscore is <tol× the
# locus best. Feed the survivors as (tib_filtered, prediction) into
# evaluate_accuracy so the standard merge produces a leaner 'merged' row
# and the 'orf_prediction' row also reflects the rescored ORF set.
#
# Uses cached diamond TSVs (Tiberius: existing pipeline output, ORF:
# the soft-protein filter's diamond.tsv).
#
# Output layout:
#   ${RESULTS_DIR}/rescore_by_protein/<sp>/tib.rescored.gtf
#                                        /orf.rescored.gtf
#                                        /rescored.dropped.tsv
#   ${RESULTS_DIR}/eval_accuracy_epoch_74_filt_rescore_${RESCORE_TAG}/
#     preds/<sp>/prediction.gtf   (symlinks to orf.rescored.gtf)
#     accuracy_table.tsv
#     accuracy_figure.pdf
#
# Env vars:
#   RESCORE_TOL    (default 0.7)  keep tx with bs >= TOL * locus_best_bs
#   EVAL_MAX       (default 1e-5) diamond hit e-value cutoff
#   RESCORE_TAG    (default tol${RESCORE_TOL})
#
#SBATCH --job-name=eval_acc_rs
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_rs_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_rs_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
TIB_DIR_NAME=annotate_epoch_74_${FILT_TAG}

RESCORE_TOL=${RESCORE_TOL:-0.7}
EVAL_MAX=${EVAL_MAX:-1e-5}
RESCORE_TAG=${RESCORE_TAG:-tol${RESCORE_TOL}}

RESCORE_ROOT=${RESULTS_DIR}/rescore_by_protein_${RESCORE_TAG}
EVAL_TAG=epoch_74_filt_rescore_${RESCORE_TAG}
OUT_ROOT=${RESULTS_DIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}" "${RESCORE_ROOT}"

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)

TIB_TMPL='/home/gabriell/tiberius_proteins_analysis/diamond_filter/{sp}/tiberius/filtered.gtf'
TIB_DIAM_TMPL='/home/gabriell/tiberius_proteins_analysis/diamond_filter/{sp}/tiberius/diamond.tsv'

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# 1. Per-species rescore.
for sp in "${SPECIES[@]}"; do
    TIB_GTF=${TIB_TMPL/\{sp\}/${sp}}
    TIB_DIAM=${TIB_DIAM_TMPL/\{sp\}/${sp}}
    ORF_GTF=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}/orfs.filtered.gtf
    ORF_DIAM=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}/soft_protein/diamond.tsv
    SP_OUT=${RESCORE_ROOT}/${sp}
    mkdir -p "${SP_OUT}"

    for f in "${TIB_GTF}" "${TIB_DIAM}" "${ORF_GTF}" "${ORF_DIAM}"; do
        [[ -s "${f}" ]] || { echo "[skip] ${sp}: missing ${f}"; continue 2; }
    done

    echo "[$(date -Iseconds)] rescore ${sp}"
    python "${PROJDIR}/scripts/rescore_predictions_by_protein_joint_locus.py" \
        --tib-gtf         "${TIB_GTF}" \
        --orf-gtf         "${ORF_GTF}" \
        --tib-diamond-tsv "${TIB_DIAM}" \
        --orf-diamond-tsv "${ORF_DIAM}" \
        --out-tib-gtf     "${SP_OUT}/tib.rescored.gtf" \
        --out-orf-gtf     "${SP_OUT}/orf.rescored.gtf" \
        --report-tsv      "${SP_OUT}/rescored.dropped.tsv" \
        --tol             "${RESCORE_TOL}" \
        --evalue-max      "${EVAL_MAX}"

    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${SP_OUT}/orf.rescored.gtf" "${PRED_DIR}/${sp}/prediction.gtf"
done

# 2. evaluate_accuracy using the rescored tib_filtered per species.
echo "[$(date -Iseconds)] evaluate_accuracy (rescored tib_filt + rescored ORF)"

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir           "${PRED_DIR}" \
    --out-dir            "${OUT_ROOT}" \
    --species            "${SPECIES[@]}" \
    --tib-filtered-tmpl  "${RESCORE_ROOT}/{sp}/tib.rescored.gtf"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -60
