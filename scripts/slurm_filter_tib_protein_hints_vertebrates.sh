#!/bin/bash
#SBATCH --job-name=tib_prob03_hint
#SBATCH --partition=batch,snowball,pinky
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_prob03_hint_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_prob03_hint_%j.err
#
# For each vertebrates_test species:
#   1. Run chainedHints.py to produce chained_hints.gff (if missing).
#   2. Run filter_tib_with_protein_hints.py:
#        Tier 1: keep Tib tx with lgb_prob_correct > 0.3
#        Tier 2: keep remaining Tib tx whose introns cover all intron hints
#                of at least one protein chain
#      Output: tiberius_lgb_filtered/tib_prob03_hint.gtf
#   3. Run plot_lgb_comparison.py to regenerate evaluation table (+figure).
#
# Prerequisites:
#   slurm_apply_lgb_tiberius_vertebrates_test.sh   (tiberius_lgb_filtered.gtf)
#   fix_stop pipeline for each species              (hc.gff, miniprot_scored.gff)
#
# Output:
#   results/vertebrates_test/eval_lgb_comparison/lgb_comparison_table.tsv
#   results/vertebrates_test/eval_lgb_comparison/lgb_comparison.pdf

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
SCRIPTS_DIR=${PROJDIR}/scripts
TIBERIUS_REPO=${TIBERIUS_REPO:-/projects/AI-GUSTUS/Tiberius}
OUTDIR=${PROJDIR}/results/vertebrates_test/eval_lgb_comparison

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
    Bos_taurus
    Delphinapterus_leucas
    Homo_sapiens
)

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

for SP in "${SPECIES[@]}"; do
    SP_DIR=${RESULTS_DIR}/${SP}
    HC_GFF=${SP_DIR}/fix_stop/miniprothint/hc.gff
    MINIPROT_SCORED=${SP_DIR}/fix_stop/miniprot_scored.gff
    TIB_LGB_GTF=${SP_DIR}/tiberius_lgb_filtered/tiberius_lgb_filtered.gtf
    HINT_RESCUE_DIR=${SP_DIR}/hint_rescue
    CHAINED_HINTS=${HINT_RESCUE_DIR}/chained_hints.gff
    OUT_GTF=${SP_DIR}/tiberius_lgb_filtered/tib_correct_hint_partial.gtf

    echo ""
    echo "[$(date -Iseconds)] === ${SP} ==="

    # Skip if required inputs are missing
    if [[ ! -s "${TIB_LGB_GTF}" ]]; then
        echo "  [skip] missing ${TIB_LGB_GTF}"
        continue
    fi
    if [[ ! -s "${HC_GFF}" || ! -s "${MINIPROT_SCORED}" ]]; then
        echo "  [skip] missing hc.gff or miniprot_scored.gff"
        continue
    fi

    mkdir -p "${HINT_RESCUE_DIR}"

    # Step 1: chain-tag miniprot hints (fast, CPU-only)
    if [[ ! -s "${CHAINED_HINTS}" ]]; then
        echo "  [chainedHints] running ..."
        python "${TIBERIUS_REPO}/tiberius/scripts/chainedHints.py" \
            "${HC_GFF}" \
            "${MINIPROT_SCORED}" \
            --output "${CHAINED_HINTS}"
        echo "  [chainedHints] $(wc -l < "${CHAINED_HINTS}") lines"
    else
        echo "  [chainedHints] reusing existing ${CHAINED_HINTS}"
    fi
    [[ -s "${CHAINED_HINTS}" ]] || { echo "  [error] chained_hints.gff empty after chainedHints.py" >&2; continue; }

    # Step 2: two-tier Tiberius filter
    echo "  [filter] running filter_tib_with_protein_hints.py ..."
    python "${SCRIPTS_DIR}/filter_tib_with_protein_hints.py" \
        "${TIB_LGB_GTF}" \
        "${CHAINED_HINTS}" \
        "${OUT_GTF}"
    echo "  [filter] $(awk '$3=="transcript"' "${OUT_GTF}" | wc -l) transcripts in ${OUT_GTF}"
done

echo ""
echo "[$(date -Iseconds)] All species processed. Running evaluation ..."

python "${SCRIPTS_DIR}/plot_lgb_comparison.py" \
    --out-dir "${OUTDIR}" \
    --force

echo "[$(date -Iseconds)] done"
echo "Table: ${OUTDIR}/lgb_comparison_table.tsv"
echo "Figure: ${OUTDIR}/lgb_comparison.pdf"

if [[ -s "${OUTDIR}/lgb_comparison_table.tsv" ]]; then
    column -t -s$'\t' "${OUTDIR}/lgb_comparison_table.tsv"
fi
