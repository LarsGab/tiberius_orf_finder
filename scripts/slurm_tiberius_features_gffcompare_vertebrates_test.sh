#!/bin/bash
# Compute per-transcript feature table and gffcompare class labels for the
# Tiberius ab initio predictions (tiberius_filtered_epoch_74) across all 6
# vertebrates_test species.
#
# Steps per species:
#   1. compute_orf_features.py  -> tiberius_features.tsv
#   2. gffcompare vs annot_cds.gff -> gffcompare/orfs.tracking
#   3. join_tmap_labels.py      -> add gffcompare_class column in-place
#
# Run after slurm_filter_tiberius_vertebrates_test.sh completes.
#
#SBATCH --job-name=tib_feat_gc
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_feat_gc_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_feat_gc_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

OUTDIR="${RESULTS_DIR}/${species}/tiberius_filtered_epoch_74"
GTF="${OUTDIR}/tiberius_filtered.gtf"
MINIPROT="${RESULTS_DIR}/${species}/fix_stop/miniprot_scored.gff"
HINTS="${RESULTS_DIR}/${species}/fix_stop/miniprothint/hc.gff"
GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
REF="${RESULTS_DIR}/${species}/assembly/annot_cds.gff"
FEATURES="${OUTDIR}/tiberius_features.tsv"
GC_DIR="${OUTDIR}/gffcompare"

mkdir -p "${PROJDIR}/logs" "${GC_DIR}"

echo "[$(date -Iseconds)] species=${species}"

test -s "${GTF}"      || { echo "missing: ${GTF}"      >&2; exit 2; }
test -s "${MINIPROT}" || { echo "missing: ${MINIPROT}" >&2; exit 2; }
test -s "${HINTS}"    || { echo "missing: ${HINTS}"    >&2; exit 2; }
test -s "${GENOME}"   || { echo "missing: ${GENOME}"   >&2; exit 2; }
test -s "${REF}"      || { echo "missing: ${REF}"      >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: compute features ──────────────────────────────────────────────────
echo "[$(date -Iseconds)] Computing features ..."
python "${PROJDIR}/scripts/compute_orf_features.py" \
    --orfs-gtf     "${GTF}" \
    --miniprot-gff "${MINIPROT}" \
    --hints-gff    "${HINTS}" \
    --genome       "${GENOME}" \
    --out          "${FEATURES}"
echo "[$(date -Iseconds)] features done -> ${FEATURES}  ($(wc -l < "${FEATURES}") rows)"

# ── Step 2: gffcompare ────────────────────────────────────────────────────────
GC_PREFIX="${GC_DIR}/orfs"
echo "[$(date -Iseconds)] Running gffcompare ..."
gffcompare --strict-match -e 3 -T \
    -r "${REF}" \
    -o "${GC_PREFIX}" \
    "${GTF}"
echo "[$(date -Iseconds)] gffcompare done"

TRACKING="${GC_PREFIX}.tracking"
test -s "${TRACKING}" || { echo "ERROR: no .tracking at ${TRACKING}" >&2; ls "${GC_DIR}" >&2; exit 2; }
echo "tracking: ${TRACKING}  ($(wc -l < "${TRACKING}") rows)"

# ── Step 3: join class labels ─────────────────────────────────────────────────
echo "[$(date -Iseconds)] Joining gffcompare labels ..."
python "${PROJDIR}/scripts/join_tmap_labels.py" \
    --tracking "${TRACKING}" \
    --features "${FEATURES}"

echo "[$(date -Iseconds)] done -> ${FEATURES}"
head -1 "${FEATURES}" | tr '\t' '\n' | tail -3 | nl
