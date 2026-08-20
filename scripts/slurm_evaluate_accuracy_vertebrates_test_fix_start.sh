#!/bin/bash
# Evaluate start-fixed predictions against reference annotation.
#
#SBATCH --job-name=eval_fix_start
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_fix_start_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_fix_start_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)

OUT_ROOT="${RESULTS_DIR}/eval_accuracy_epoch74_fix_start"
PRED_DIR="${OUT_ROOT}/preds"
mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

EVAL_SPECIES=()
for sp in "${SPECIES[@]}"; do
    src="${RESULTS_DIR}/${sp}/fix_start/orfs.start_fixed.filtered.gtf"
    if [[ ! -s "${src}" ]]; then
        echo "[skip] ${sp}: ${src} not found"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
    EVAL_SPECIES+=("${sp}")
done

echo "[$(date -Iseconds)] Running evaluate_accuracy.py ..."
python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir "${PRED_DIR}" \
    --out-dir  "${OUT_ROOT}" \
    --species  "${EVAL_SPECIES[@]}"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}/accuracy_table.tsv"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -60
