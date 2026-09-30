#!/bin/bash
#SBATCH --job-name=prep_hint_rescue
#SBATCH --partition=batch,snowball,pinky
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_hint_rescue_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_hint_rescue_%j.err
#
# Prepare combined_loci.fa + combined_hints.gff for Tiberius hint-guided
# rescue (Archocentrus_centrarchus).
#
# Pipeline:
#   1. chainedHints.py   — tag miniprot_scored.gff hints with chain_id=
#                          using hc.gff as the high-confidence filter.
#   2. prepare_hint_rescue_loci.py — build per-(locus, chain) FASTA entries
#                          and a combined hints GFF with local coordinates.
#
# Prerequisite: slurm_fix_stop_vertebrates_test.sh must have finished so that
#   fix_stop/miniprot_scored.gff  and  fix_stop/miniprothint/hc.gff  exist.
#
# Outputs (all in ${OUTDIR}/):
#   chained_hints.gff    — chain_id-tagged hints (intermediate)
#   combined_loci.fa     — multi-FASTA for Tiberius
#   combined_hints.gff   — per-locus-chain hints (local coords)
#   loci_manifest.tsv    — index → (chr, bed_start, bed_end, chain_id)
#
# After this job finishes, submit:
#   sbatch ${SCRIPTS_DIR}/slurm_tiberius_hint_rescue_vertebrates_test.sh

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SPECIES=Archocentrus_centrarchus
RESULTS_DIR=${PROJDIR}/results/vertebrates_test/${SPECIES}
SCRIPTS_DIR=${PROJDIR}/scripts
TIBERIUS_REPO=${TIBERIUS_REPO:-/home/gabriell/Tiberius}

PARTIAL_GTF=${RESULTS_DIR}/tiberius_seqlen/tiberius_lgb_partial.gtf
CORRECT_GTF=${RESULTS_DIR}/tiberius_seqlen/tiberius_lgb_correct.gtf
HC_GFF=${RESULTS_DIR}/fix_stop/miniprothint/hc.gff
MINIPROT_SCORED=${RESULTS_DIR}/fix_stop/miniprot_scored.gff
GENOME=${RESULTS_DIR}/assembly/genome.fa
ORFS_DIR=${RESULTS_DIR}/annotate_run009_best_filt_tpm1cov3len300
OUTDIR=${RESULTS_DIR}/hint_rescue

mkdir -p "${OUTDIR}" "${PROJDIR}/logs"

# ── Input checks ─────────────────────────────────────────────────────────────
for f in "${PARTIAL_GTF}" "${CORRECT_GTF}" "${HC_GFF}" "${MINIPROT_SCORED}" \
          "${GENOME}" "${GENOME}.fai" "${ORFS_DIR}/orfs.gtf"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing input: ${f}" >&2; exit 1; }
done

# ── Verify Tiberius is on hint_integration branch ────────────────────────────
TIBERIUS_BRANCH=$(git -C "${TIBERIUS_REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "${TIBERIUS_BRANCH}" != "hint_integration" ]]; then
    echo "ERROR: Tiberius is on branch '${TIBERIUS_BRANCH}', expected 'hint_integration'." >&2
    echo "       cd ${TIBERIUS_REPO} && git checkout hint_integration" >&2
    exit 1
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: chain-tag miniprot_scored.gff against hc.gff ─────────────────────
CHAINED_HINTS=${OUTDIR}/chained_hints.gff
if [[ ! -s "${CHAINED_HINTS}" ]]; then
    echo "[$(date -Iseconds)] Running chainedHints.py ..."
    python "${TIBERIUS_REPO}/tiberius/scripts/chainedHints.py" \
        "${HC_GFF}" \
        "${MINIPROT_SCORED}" \
        --output "${CHAINED_HINTS}"
    echo "[$(date -Iseconds)] chained hints: $(wc -l < "${CHAINED_HINTS}") lines"
else
    echo "[$(date -Iseconds)] chained_hints.gff exists, reusing"
fi

# ── Step 2: build multi-FASTA + hints GFF ────────────────────────────────────
echo "[$(date -Iseconds)] Building combined FASTA and hints ..."
ORFS_ARGS="${ORFS_DIR}/orfs.gtf"
[[ -s "${ORFS_DIR}/orfs.partial.gtf" ]] && ORFS_ARGS="${ORFS_ARGS} ${ORFS_DIR}/orfs.partial.gtf"

python "${SCRIPTS_DIR}/prepare_hint_rescue_loci.py" \
    --partial_gtf   "${PARTIAL_GTF}" \
    --correct_gtf   "${CORRECT_GTF}" \
    --chained_hints "${CHAINED_HINTS}" \
    --orfs_gtf      ${ORFS_ARGS} \
    --genome        "${GENOME}" \
    --outdir        "${OUTDIR}" \
    --flank         25000

N_ENTRIES=$(grep -c '^>' "${OUTDIR}/combined_loci.fa" || echo 0)
echo "[$(date -Iseconds)] Done — ${N_ENTRIES} (locus, chain) entries in combined_loci.fa"
echo ""
echo "Next: ensure Tiberius is on hint_integration branch, then submit:"
echo "  sbatch ${SCRIPTS_DIR}/slurm_tiberius_hint_rescue_vertebrates_test.sh"
