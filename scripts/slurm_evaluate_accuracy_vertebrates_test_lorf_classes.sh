#!/bin/bash
# Split the LORF-annotated filtered predictions by class and evaluate accuracy
# of LORF_UPSTOP vs LORF_NOUPSTOP subsets independently.
#
# Requires: annotate_epoch_74_filt_tpm1cov3len300_lorf/ outputs to exist for
# all 8 species (job 7798125).
#
# Output:
#   ${RESULTS_DIR}/eval_accuracy_epoch74_lorf_UPSTOP/
#     preds/<sp>/prediction.gtf   (only LORF_UPSTOP CDS lines)
#     accuracy_table.tsv
#   ${RESULTS_DIR}/eval_accuracy_epoch74_lorf_NOUPSTOP/
#     (same for LORF_NOUPSTOP)
#
#SBATCH --job-name=eval_lorf
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lorf_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lorf_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ANNOT_TAG=annotate_epoch_74_filt_tpm1cov3len300_lorf

SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Bos_taurus
    Delphinapterus_leucas
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

for CLASS in LORF_UPSTOP LORF_NOUPSTOP; do
    OUT_ROOT=${RESULTS_DIR}/eval_accuracy_epoch74_lorf_${CLASS}
    PRED_DIR=${OUT_ROOT}/preds
    mkdir -p "${OUT_ROOT}" "${PRED_DIR}"

    echo "[$(date -Iseconds)] === ${CLASS} ==="

    for sp in "${SPECIES[@]}"; do
        SRC="${RESULTS_DIR}/${sp}/${ANNOT_TAG}/orfs.filtered.gtf"
        if [[ ! -s "${SRC}" ]]; then
            echo "[skip] ${sp}: ${SRC} not found"
            continue
        fi
        mkdir -p "${PRED_DIR}/${sp}"
        DEST="${PRED_DIR}/${sp}/prediction.gtf"
        grep "lorf_class \"${CLASS}\"" "${SRC}" > "${DEST}" || true
        N=$(wc -l < "${DEST}")
        echo "[split] ${sp}: ${N} CDS lines -> ${DEST}"
    done

    echo "[$(date -Iseconds)] Running evaluate_accuracy.py for ${CLASS} ..."
    python "${PROJDIR}/scripts/evaluate_accuracy.py" \
        --pred-dir "${PRED_DIR}" \
        --out-dir  "${OUT_ROOT}" \
        --species  "${SPECIES[@]}"

    echo "[$(date -Iseconds)] done -> ${OUT_ROOT}/accuracy_table.tsv"
    column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -40
    echo ""
done
