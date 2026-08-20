#!/bin/bash
# Evaluate fungi ORF predictions against the reference annotation for 7 test
# species alongside ab-initio Tiberius, BRAKER3, and their merge.
# Prediction source: results/fungi_test/<sp>/annotate_run006_e<EPOCH>/orfs.filtered.gtf
#
# Usage: EPOCH=<n> sbatch scripts/slurm_evaluate_accuracy_fungi_test.sh
#   EPOCH defaults to the highest annotate_run006_e* found for Aspergillus_fumigatus.
#
#SBATCH --job-name=eval_acc_fun
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_fun_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_fun_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/fungi_test

if [[ -z "${EPOCH:-}" ]]; then
    EPOCH=$(ls -d "${TESTDIR}"/Aspergillus_fumigatus/annotate_run006_e* 2>/dev/null \
            | sort -t_ -k4 -n | tail -1 | xargs basename 2>/dev/null \
            | sed 's/annotate_run006_e//' || true)
    [[ -n "${EPOCH}" ]] || { echo "cannot determine EPOCH; set it explicitly via EPOCH=<n>" >&2; exit 2; }
fi
EVAL_TAG=${EVAL_TAG:-run006_e${EPOCH}}
OUT_ROOT=${TESTDIR}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

SPECIES=(
    Agaricus_bisporus
    Aspergillus_fumigatus
    Cryphonectria_parasitica
    Parastagonospora_nodorum
    Puccinia_striiformis
    Punctularia_strigosozonata
    Tilletiopsis_washingtonensis
)

for sp in "${SPECIES[@]}"; do
    src=${TESTDIR}/${sp}/annotate_run006_e${EPOCH}/orfs.filtered.gtf
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
    --tib-tmpl  "/home/gabriell/tiberius_benchmarking/paper/Fungi/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf" \
    --brk-tmpl  "/home/gabriell/tiberius_benchmarking/paper/Fungi/{sp}/results/predictions/braker3/braker3.gtf"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -80
