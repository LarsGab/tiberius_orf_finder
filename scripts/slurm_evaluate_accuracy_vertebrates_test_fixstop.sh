#!/bin/bash
# Evaluate vertebrate test species predictions after stop-codon fixing
# (fix_stop_by_miniprot.py output) against the reference annotation.
#
# Pipeline per species before evaluation:
#   fix_stop/orfs.fixed.gtf
#     → filter_subsequence_predictions.py
#     → fix_stop/orfs.fixed.filtered.gtf   (used as prediction.gtf)
#
# Output layout:
#   ${RESULTS_DIR}/eval_accuracy_run009_fixstop/
#     preds/<sp>/prediction.gtf   (symlinks to orfs.fixed.filtered.gtf)
#     accuracy_table.tsv
#     accuracy_figure.pdf
#     gffcompare_runs/<sp>/...
#
# Submit after fix_stop array job completes:
#   sbatch --dependency=afterok:<fix_stop_jobid> this_script.sh
#
#SBATCH --job-name=eval_fixstop
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_fixstop_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_fixstop_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

EVAL_TAG=run009_fixstop
OUT_ROOT=${RESULTS_DIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

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

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: subsequence-filter each species' fixed GTF ───────────────────────
echo "[$(date -Iseconds)] Filtering subsequence predictions ..."
for sp in "${SPECIES[@]}"; do
    fixed="${RESULTS_DIR}/${sp}/fix_stop/orfs.fixed.gtf"
    filt="${RESULTS_DIR}/${sp}/fix_stop/orfs.fixed.filtered.gtf"
    report="${RESULTS_DIR}/${sp}/fix_stop/dropped_subsequences.tsv"

    if [[ ! -s "${fixed}" ]]; then
        echo "[skip subseq] ${sp}: ${fixed} not found"
        continue
    fi

    if [[ ! -s "${filt}" ]]; then
        echo "[subseq] ${sp} ..."
        python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
            --orfs-gtf   "${fixed}" \
            --out-gtf    "${filt}" \
            --report-tsv "${report}"
    else
        echo "[subseq] ${sp}: already done, reusing"
    fi
done

# ── Step 2: build prediction.gtf symlink tree ─────────────────────────────────
echo "[$(date -Iseconds)] Building prediction symlinks ..."
for sp in "${SPECIES[@]}"; do
    src="${RESULTS_DIR}/${sp}/fix_stop/orfs.fixed.filtered.gtf"
    if [[ ! -s "${src}" ]]; then
        echo "[skip preds] ${sp}: missing ${src}"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
done

# ── Step 3: evaluate ──────────────────────────────────────────────────────────
echo "[$(date -Iseconds)] Running evaluate_accuracy.py ..."
python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir "${PRED_DIR}" \
    --out-dir  "${OUT_ROOT}" \
    --species  "${SPECIES[@]}"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -80
