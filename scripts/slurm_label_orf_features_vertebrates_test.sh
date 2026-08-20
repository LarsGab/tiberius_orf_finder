#!/bin/bash
# Add gffcompare class labels to orf_features.tsv for each vertebrates_test species.
#
# Runs gffcompare against annot_cds.gff, then joins the class_code from the
# .tracking file into orf_features.tsv as a new column (gffcompare_class).
#
# gffcompare v0.12.10 produces .tracking (not .tmap) when -T -r are used.
#
# gffcompare class codes (most common):
#   =   exact CDS match (all introns + boundaries)
#   ~   intron-chain match (same splice structure)
#   j   at least one junction match
#   c   contained in a reference exon
#   e   single-exon overlap with reference exon
#   o   generic overlap with reference locus
#   u   intergenic / no reference overlap
#
#SBATCH --job-name=label_orf_feat
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/label_orf_feat_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/label_orf_feat_%A_%a.err

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

ANNOT_DIR="${RESULTS_DIR}/${species}/${ANNOT_TAG}"
ORFS="${ANNOT_DIR}/orfs.filtered.gtf"
REF="${RESULTS_DIR}/${species}/assembly/annot_cds.gff"
FEATURES="${ANNOT_DIR}/orf_features.tsv"
GC_DIR="${ANNOT_DIR}/gffcompare"

mkdir -p "${PROJDIR}/logs" "${GC_DIR}"

echo "[$(date -Iseconds)] species=${species}"

test -s "${ORFS}"     || { echo "missing: ${ORFS}"     >&2; exit 2; }
test -s "${REF}"      || { echo "missing: ${REF}"      >&2; exit 2; }
test -s "${FEATURES}" || { echo "missing: ${FEATURES}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: run gffcompare ────────────────────────────────────────────────────
GC_PREFIX="${GC_DIR}/orfs"

echo "[$(date -Iseconds)] Running gffcompare ..."
gffcompare --strict-match -e 3 -T \
    -r "${REF}" \
    -o "${GC_PREFIX}" \
    "${ORFS}"
echo "[$(date -Iseconds)] gffcompare done"

# gffcompare v0.12.10 with -T produces <prefix>.tracking
TRACKING="${GC_PREFIX}.tracking"
test -s "${TRACKING}" || { echo "ERROR: no .tracking found: ${TRACKING}" >&2; ls "${GC_DIR}" >&2; exit 2; }
echo "tracking: ${TRACKING}  ($(wc -l < "${TRACKING}") rows)"

# ── Step 2: join class codes into orf_features.tsv ───────────────────────────
echo "[$(date -Iseconds)] Joining labels ..."
python "${PROJDIR}/scripts/join_tmap_labels.py" \
    --tracking "${TRACKING}" \
    --features "${FEATURES}"

echo "[$(date -Iseconds)] done -> ${FEATURES}"
head -1 "${FEATURES}" | tr '\t' '\n' | tail -3 | nl
