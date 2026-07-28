#!/bin/bash
# Evaluate embryophyta long-read ORF predictions (run001 epoch_39) against
# the reference annotation alongside ab-initio Tiberius and BRAKER3.
#
# Only the 4 species with LR data are included; species without data are
# not available for comparison and are omitted.
#
# Output: ${TESTDIR}/eval_accuracy_run001_e39_lr_filt/
#
#SBATCH --job-name=eval_emb_lr
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_emb_lr_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_emb_lr_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
BENCH=/home/gabriell/tiberius_benchmarking/paper/Embryophyta

EPOCH=${EPOCH:-39}
EVAL_TAG=${EVAL_TAG:-run001_e${EPOCH}_lr_filt}
OUT_ROOT=${TESTDIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

SPECIES=(Arabidopsis_thaliana Eschscholzia_californica Medicago_truncatula Urochloa_brizantha)

for sp in "${SPECIES[@]}"; do
    src=${TESTDIR}/${sp}/annotate_run001_e${EPOCH}_lr_${FILT_TAG:-filt_tpm1cov3len300}/orfs.filtered.gtf
    if [[ ! -s "${src}" ]]; then
        echo "[skip preds] ${sp}: missing ${src}"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] running evaluate_accuracy.py (embryophyta LR)"

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir  "${PRED_DIR}" \
    --out-dir   "${OUT_ROOT}" \
    --species   "${SPECIES[@]}" \
    --ref-tmpl  "${TESTDIR}/{sp}/assembly/annot_cds.gff" \
    --tib-tmpl  "${BENCH}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf" \
    --brk-tmpl  "${BENCH}/{sp}/results/predictions/braker3/braker3.gtf"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -60
