#!/bin/bash
# Fix upstream start codons for complete LORF_NOUPSTOP ORFs using miniprot.
#
# Input: orfs.filtered.gtf from the lorf-annotated run (has lorf_class attrs)
#        miniprot_scored.gff from fix_stop/ (protein-to-genome alignments)
# Output:
#   ${RESULTS_DIR}/<sp>/fix_start/
#     orfs.start_fixed.gtf          (all complete ORFs; LORF_NOUPSTOP extended
#                                    upstream where miniprot supports it)
#     orfs.start_fixed.filtered.gtf (after subseq-collapse)
# Accuracy:
#   ${RESULTS_DIR}/eval_accuracy_epoch74_fix_start/
#
#SBATCH --job-name=fix_start_vert
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/fix_start_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/fix_start_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ANNOT_TAG=annotate_epoch_74_filt_tpm1cov3len300_lorf

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

ORFS="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
MINIPROT="${RESULTS_DIR}/${species}/fix_stop/miniprot_scored.gff"
HINTS="${RESULTS_DIR}/${species}/fix_stop/miniprothint/hc.gff"
GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
OUT_DIR="${RESULTS_DIR}/${species}/fix_start"

mkdir -p "${PROJDIR}/logs" "${OUT_DIR}"

echo "[$(date -Iseconds)] species=${species}"

test -s "${ORFS}"     || { echo "missing: ${ORFS}"     >&2; exit 2; }
test -s "${MINIPROT}" || { echo "missing: ${MINIPROT}" >&2; exit 2; }
test -s "${HINTS}"    || { echo "missing: ${HINTS}"    >&2; exit 2; }
test -s "${GENOME}"   || { echo "missing: ${GENOME}"   >&2; exit 2; }

if [[ -s "${OUT_DIR}/orfs.start_fixed.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP: already done"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] Running fix_stop_by_miniprot.py --fix-starts ..."
python "${PROJDIR}/scripts/fix_stop_by_miniprot.py" \
    --orfs         "${ORFS}" \
    --miniprot     "${MINIPROT}" \
    --hints        "${HINTS}" \
    --genome       "${GENOME}" \
    --out          "${OUT_DIR}/orfs.start_fixed.gtf" \
    --fix-starts \
    --fix-starts-classes LORF_NOUPSTOP upLORF \
    --max-start-scan 30

echo "[$(date -Iseconds)] subseq-collapse ..."
python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUT_DIR}/orfs.start_fixed.gtf" \
    --out-gtf    "${OUT_DIR}/orfs.start_fixed.filtered.gtf" \
    --report-tsv "${OUT_DIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUT_DIR}/orfs.start_fixed.filtered.gtf"
