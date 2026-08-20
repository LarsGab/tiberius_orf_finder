#!/bin/bash
# Evaluate LORF class × protein-support subsets for the 6 vertebrates_test
# species that have complete miniprot data in fix_stop/.
#
# For each species: split orfs.filtered.gtf (from the lorf annotation run)
# into LORF_UPSTOP / LORF_NOUPSTOP_prot / LORF_NOUPSTOP_noprot / lorf_filtered
# using miniprot_scored.gff as the protein-support signal, then run
# evaluate_accuracy.py on each subset.
#
# Output:
#   ${RESULTS_DIR}/eval_accuracy_epoch74_lorf_prot_<SUBSET>/
#     preds/<sp>/prediction.gtf
#     accuracy_table.tsv
#
# Subsets evaluated:
#   lorf_filtered       LORF_UPSTOP + LORF_NOUPSTOP_prot  (GeneMark-ETP-style)
#   LORF_NOUPSTOP_prot
#   LORF_NOUPSTOP_noprot
#   (LORF_UPSTOP already evaluated by slurm_evaluate_accuracy_vertebrates_test_lorf_classes.sh)
#
#SBATCH --job-name=eval_lorf_prot
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lorf_prot_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_lorf_prot_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ANNOT_TAG=annotate_epoch_74_filt_tpm1cov3len300_lorf

# Only species with complete miniprot data in fix_stop/
SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: split each species' filtered GTF ─────────────────────────────────
echo "[$(date -Iseconds)] Splitting ORFs by LORF class x protein support ..."
for sp in "${SPECIES[@]}"; do
    ORFS="${RESULTS_DIR}/${sp}/${ANNOT_TAG}/orfs.filtered.gtf"
    MINIPROT="${RESULTS_DIR}/${sp}/fix_stop/miniprot_scored.gff"
    SPLIT_DIR="${RESULTS_DIR}/${sp}/${ANNOT_TAG}/lorf_prot_split"

    if [[ ! -s "${ORFS}" ]]; then
        echo "[skip split] ${sp}: ${ORFS} not found"
        continue
    fi
    if [[ ! -s "${MINIPROT}" ]]; then
        echo "[skip split] ${sp}: ${MINIPROT} not found"
        continue
    fi

    mkdir -p "${SPLIT_DIR}"
    echo "[split] ${sp} ..."
    python "${PROJDIR}/scripts/split_by_protein_support.py" \
        --orfs-gtf     "${ORFS}" \
        --miniprot-gff "${MINIPROT}" \
        --out-dir      "${SPLIT_DIR}"
done

# ── Step 2: evaluate each subset ─────────────────────────────────────────────
for SUBSET in lorf_filtered LORF_NOUPSTOP_prot LORF_NOUPSTOP_noprot; do
    OUT_ROOT="${RESULTS_DIR}/eval_accuracy_epoch74_lorf_prot_${SUBSET}"
    PRED_DIR="${OUT_ROOT}/preds"
    mkdir -p "${OUT_ROOT}" "${PRED_DIR}"

    EVAL_SPECIES=()
    echo "[$(date -Iseconds)] === ${SUBSET} ==="
    for sp in "${SPECIES[@]}"; do
        SRC="${RESULTS_DIR}/${sp}/${ANNOT_TAG}/lorf_prot_split/${SUBSET}.gtf"
        if [[ ! -s "${SRC}" ]]; then
            echo "[skip eval] ${sp}: ${SRC} not found or empty"
            continue
        fi
        mkdir -p "${PRED_DIR}/${sp}"
        ln -sf "${SRC}" "${PRED_DIR}/${sp}/prediction.gtf"
        EVAL_SPECIES+=("${sp}")
        N=$(wc -l < "${SRC}")
        echo "[link] ${sp}: ${N} CDS lines"
    done

    if [[ ${#EVAL_SPECIES[@]} -eq 0 ]]; then
        echo "[skip evaluate_accuracy] no species with data for ${SUBSET}"
        continue
    fi

    echo "[$(date -Iseconds)] Running evaluate_accuracy.py for ${SUBSET} ..."
    python "${PROJDIR}/scripts/evaluate_accuracy.py" \
        --pred-dir "${PRED_DIR}" \
        --out-dir  "${OUT_ROOT}" \
        --species  "${EVAL_SPECIES[@]}"

    echo "[$(date -Iseconds)] done -> ${OUT_ROOT}/accuracy_table.tsv"
    column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -40
    echo ""
done
