#!/bin/bash
# Compute per-transcript feature table and gffcompare class labels for the
# raw Tiberius ab initio predictions (tiberius_seqlen.gtf) across all 6
# vertebrates_test species.
#
# Steps per species:
#   1. Prefix gene_id / transcript_id with species name to ensure global
#      uniqueness  ->  tiberius_seqlen/<sp>/tiberius_seqlen_fixed.gtf
#   2. compute_orf_features.py  ->  tiberius_seqlen/<sp>/tiberius_features.tsv
#   3. gffcompare vs annot_cds.gff  ->  tiberius_seqlen/<sp>/gffcompare/orfs.tracking
#   4. join_tmap_labels.py      ->  add gffcompare_class column in-place
#
#SBATCH --job-name=tib_seqlen_feat
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_seqlen_feat_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_seqlen_feat_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
BENCH=/home/gabriell/tiberius_benchmarking

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

TIB_GTF="${BENCH}/paper/Vertebrata/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
MINIPROT="${RESULTS_DIR}/${species}/fix_stop/miniprot_scored.gff"
HINTS="${RESULTS_DIR}/${species}/fix_stop/miniprothint/hc.gff"
GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
REF="${RESULTS_DIR}/${species}/assembly/annot_cds.gff"

OUTDIR="${RESULTS_DIR}/${species}/tiberius_seqlen"
FIXED_GTF="${OUTDIR}/tiberius_seqlen_fixed.gtf"
FEATURES="${OUTDIR}/tiberius_features.tsv"
GC_DIR="${OUTDIR}/gffcompare"

mkdir -p "${PROJDIR}/logs" "${OUTDIR}" "${GC_DIR}"

echo "[$(date -Iseconds)] species=${species}"

test -s "${TIB_GTF}"  || { echo "missing: ${TIB_GTF}"  >&2; exit 2; }
test -s "${MINIPROT}" || { echo "missing: ${MINIPROT}" >&2; exit 2; }
test -s "${HINTS}"    || { echo "missing: ${HINTS}"    >&2; exit 2; }
test -s "${GENOME}"   || { echo "missing: ${GENOME}"   >&2; exit 2; }
test -s "${REF}"      || { echo "missing: ${REF}"      >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: fix IDs — prefix gene_id and transcript_id with species name ──────
echo "[$(date -Iseconds)] Fixing IDs ..."
sed -E 's/(gene_id|transcript_id) "g([0-9])/\1 "'"${species}"'.g\2/g' \
    "${TIB_GTF}" > "${FIXED_GTF}"
echo "[$(date -Iseconds)] fixed GTF -> ${FIXED_GTF}  ($(wc -l < "${FIXED_GTF}") lines)"

# ── Step 2: compute features ──────────────────────────────────────────────────
echo "[$(date -Iseconds)] Computing features ..."
python "${PROJDIR}/scripts/compute_orf_features.py" \
    --orfs-gtf     "${FIXED_GTF}" \
    --miniprot-gff "${MINIPROT}" \
    --hints-gff    "${HINTS}" \
    --genome       "${GENOME}" \
    --out          "${FEATURES}"
echo "[$(date -Iseconds)] features done -> ${FEATURES}  ($(wc -l < "${FEATURES}") rows)"

# ── Step 3: gffcompare ────────────────────────────────────────────────────────
GC_PREFIX="${GC_DIR}/orfs"
echo "[$(date -Iseconds)] Running gffcompare ..."
gffcompare --strict-match -e 3 -T \
    -r "${REF}" \
    -o "${GC_PREFIX}" \
    "${FIXED_GTF}"
echo "[$(date -Iseconds)] gffcompare done"

TRACKING="${GC_PREFIX}.tracking"
test -s "${TRACKING}" || { echo "ERROR: no .tracking at ${TRACKING}" >&2; ls "${GC_DIR}" >&2; exit 2; }
echo "tracking: ${TRACKING}  ($(wc -l < "${TRACKING}") rows)"

# ── Step 4: join class labels ─────────────────────────────────────────────────
echo "[$(date -Iseconds)] Joining gffcompare labels ..."
python "${PROJDIR}/scripts/join_tmap_labels.py" \
    --tracking "${TRACKING}" \
    --features "${FEATURES}"

echo "[$(date -Iseconds)] done -> ${FEATURES}"
head -1 "${FEATURES}" | tr '\t' '\n' | tail -3 | nl
