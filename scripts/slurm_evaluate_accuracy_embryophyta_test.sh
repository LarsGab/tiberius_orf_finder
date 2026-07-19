#!/bin/bash
# Evaluate embryophyta ORF predictions (run001 e39) against the reference,
# alongside ab-initio Tiberius, BRAKER3, and their merge, using
# evaluate_accuracy.py.
#
# Species with Tiberius/BRAKER3 in tiberius_benchmarking (6 of 7):
#   excludes Brachypodium_distachyon (only Brachypodium_stacei in benchmarking dir).
#
# Protein-filtered Tiberius is not yet available for embryophyta; add
# --tib-filtered-tmpl once the diamond/miniprot pipeline runs.
#
# Output:
#   results/training_embryophyta_test_v2/eval_accuracy_run001_e39/
#     accuracy_table.tsv
#     accuracy_figure.pdf
#     gffcompare_runs/<sp>/...
#
#SBATCH --job-name=eval_acc_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_emb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_emb_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
BENCH=/home/gabriell/tiberius_benchmarking/paper/Embryophyta

EPOCH=${EPOCH:-39}
EVAL_TAG=${EVAL_TAG:-run001_e${EPOCH}}
OUT_ROOT=${TESTDIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

# 6 species with Tiberius/BRAKER3 benchmarking data.
SPECIES=(Arabidopsis_thaliana Eschscholzia_californica Freycinetia_multiflora Medicago_truncatula Mimulus_guttatus Urochloa_brizantha)

for sp in "${SPECIES[@]}"; do
    src=${TESTDIR}/${sp}/annotate_run001_e${EPOCH}/orfs.filtered.gtf
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

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir  "${PRED_DIR}" \
    --out-dir   "${OUT_ROOT}" \
    --species   "${SPECIES[@]}" \
    --ref-tmpl  "${TESTDIR}/{sp}/assembly/annot_cds.gff" \
    --tib-tmpl  "${BENCH}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf" \
    --brk-tmpl  "${BENCH}/{sp}/results/predictions/braker3/braker3.gtf"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -80
