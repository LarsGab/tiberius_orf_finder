#!/bin/bash
#SBATCH --job-name=tib_hint_rescue
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_hint_rescue_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_hint_rescue_%j.err
#
# Run Tiberius with per-chain protein hints on all partial-gene rescue loci.
# Each sequence in combined_loci.fa carries exactly the hints from one
# protein alignment chain, so Tiberius sees non-contradicting evidence.
#
# Prerequisites:
#   slurm_prepare_hint_rescue_vertebrates_test.sh must have finished, producing:
#     ${OUTDIR}/combined_loci.fa
#     ${OUTDIR}/combined_hints.gff
#     ${OUTDIR}/loci_manifest.tsv
#   Tiberius must be on the hint_integration branch:
#     cd /home/gabriell/Tiberius && git checkout hint_integration
#
# Output:
#   ${OUTDIR}/tiberius_hint_rescue.gtf   — genome-coordinate predictions

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SPECIES=Archocentrus_centrarchus
OUTDIR=${PROJDIR}/results/vertebrates_test/${SPECIES}/hint_rescue
TIBERIUS_REPO=${TIBERIUS_REPO:-/home/gabriell/Tiberius}
HINT_WEIGHT=2.5

mkdir -p "${PROJDIR}/logs"

# ── Input checks ─────────────────────────────────────────────────────────────
for f in "${OUTDIR}/combined_loci.fa" \
          "${OUTDIR}/combined_hints.gff" \
          "${OUTDIR}/loci_manifest.tsv"; do
    [[ -s "${f}" ]] || {
        echo "ERROR: missing input: ${f}" >&2
        echo "       Run slurm_prepare_hint_rescue_vertebrates_test.sh first." >&2
        exit 1
    }
done

# ── Verify hint_integration branch ───────────────────────────────────────────
TIBERIUS_BRANCH=$(git -C "${TIBERIUS_REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "${TIBERIUS_BRANCH}" != "hint_integration" ]]; then
    echo "ERROR: Tiberius is on branch '${TIBERIUS_BRANCH}', expected 'hint_integration'." >&2
    echo "       cd ${TIBERIUS_REPO} && git checkout hint_integration" >&2
    exit 1
fi

N_ENTRIES=$(grep -c '^>' "${OUTDIR}/combined_loci.fa" || echo 0)
echo "[$(date -Iseconds)] ${N_ENTRIES} (locus, chain) entries to predict"

# ── Run Tiberius ──────────────────────────────────────────────────────────────
eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

RAW_GTF=${OUTDIR}/tiberius_hint_rescue_raw.gtf

python "${TIBERIUS_REPO}/tiberius.py" \
    --genome      "${OUTDIR}/combined_loci.fa" \
    --model_cfg   vertebrates \
    --hints       "${OUTDIR}/combined_hints.gff" \
    --hint_weight "${HINT_WEIGHT}" \
    --out         "${RAW_GTF}"

echo "[$(date -Iseconds)] Tiberius done"

# ── Filter (hint-supported, correct strand) + deduplicate + genome coords ─────
# filter_and_merge_rescue_gtf.py works in local coordinates, so it reads
# combined_hints.gff and the manifest to do the coordinate conversion
# internally, keeping only hint-supported, non-duplicate transcripts.
OUT_GTF=${OUTDIR}/tiberius_hint_rescue.gtf
SCRIPTS_DIR=${PROJDIR}/scripts

micromamba activate orffinder

python "${SCRIPTS_DIR}/filter_and_merge_rescue_gtf.py" \
    "${RAW_GTF}" \
    "${OUTDIR}/combined_hints.gff" \
    "${OUTDIR}/loci_manifest.tsv" \
    "${OUT_GTF}"

N_TX=$(awk '$3=="transcript"' "${OUT_GTF}" | wc -l)
echo "[$(date -Iseconds)] done: ${N_TX} transcripts → ${OUT_GTF}"
