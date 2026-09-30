#!/bin/bash
# Prepare per-(locus, chain) hint-rescue inputs for the 6 embryophyta test
# species, using the ORF-trained LGB "partial" set as rescue targets.
#
# Pipeline per species:
#   1. Split tiberius_lgb_orf_filtered.gtf by lgb_class → partial / correct GTFs
#   2. chainedHints.py — tag miniprot hints with chain_id= using hc.gff.
#   3. prepare_hint_rescue_loci.py — per-(locus, chain) FASTA + local-coord hints.
#
# Tiberius repo (hint_integration branch) lives at /projects/AI-GUSTUS/Tiberius.
#
# Outputs per species → hint_rescue/:
#   tiberius_lgb_partial.gtf, tiberius_lgb_correct.gtf
#   chained_hints.gff, combined_loci.fa, combined_hints.gff, loci_manifest.tsv
#
#SBATCH --job-name=prep_hint_rescue_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_hint_rescue_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_hint_rescue_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
SCRIPTS_DIR=${PROJDIR}/scripts
TIBERIUS_REPO=${TIBERIUS_REPO:-/projects/AI-GUSTUS/Tiberius}
ANNOT_TAG=annotate_run001_e300

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    Arabidopsis_thaliana
    Eschscholzia_californica
    Freycinetia_multiflora
    Medicago_truncatula
    Mimulus_guttatus
    Urochloa_brizantha
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=${TESTDIR}/${species}
TIB_LGB=${RESULTS_DIR}/tiberius_lgb_filtered/tiberius_lgb_orf_filtered.gtf
HC_GFF=${RESULTS_DIR}/proteins/miniprothint/hc.gff
MINIPROT_SCORED=${RESULTS_DIR}/proteins/miniprot_scored.gff
GENOME=${RESULTS_DIR}/assembly/genome.fa
ORFS_GTF=${RESULTS_DIR}/${ANNOT_TAG}/orfs.gtf
OUTDIR=${RESULTS_DIR}/hint_rescue

echo "[$(date -Iseconds)] species=${species}"

mkdir -p "${OUTDIR}"

for f in "${TIB_LGB}" "${HC_GFF}" "${MINIPROT_SCORED}" "${GENOME}" \
         "${GENOME}.fai" "${ORFS_GTF}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: split tiberius_lgb_orf_filtered.gtf by lgb_class ────────────────
PARTIAL_GTF=${OUTDIR}/tiberius_lgb_partial.gtf
CORRECT_GTF=${OUTDIR}/tiberius_lgb_correct.gtf
grep 'lgb_class "partial"'  "${TIB_LGB}" > "${PARTIAL_GTF}" || true
grep 'lgb_class "correct"'  "${TIB_LGB}" > "${CORRECT_GTF}" || true

[[ -s "${PARTIAL_GTF}" ]] || { echo "SKIP ${species}: no 'partial' predictions" >&2; exit 0; }
echo "[$(date -Iseconds)] partial CDS lines: $(wc -l < "${PARTIAL_GTF}")   correct: $(wc -l < "${CORRECT_GTF}")"

# ── Step 2: chain-tag miniprot_scored.gff against hc.gff ─────────────────────
CHAINED_HINTS=${OUTDIR}/chained_hints.gff
if [[ ! -s "${CHAINED_HINTS}" ]]; then
    echo "[$(date -Iseconds)] Running chainedHints.py …"
    python "${TIBERIUS_REPO}/tiberius/scripts/chainedHints.py" \
        "${HC_GFF}" \
        "${MINIPROT_SCORED}" \
        --output "${CHAINED_HINTS}"
    echo "  chained hints: $(wc -l < "${CHAINED_HINTS}") lines"
else
    echo "[$(date -Iseconds)] chained_hints.gff exists, reusing"
fi

# ── Step 3: build multi-FASTA + hints GFF ────────────────────────────────────
echo "[$(date -Iseconds)] Building combined_loci.fa and combined_hints.gff …"
python "${SCRIPTS_DIR}/prepare_hint_rescue_loci.py" \
    --partial_gtf   "${PARTIAL_GTF}" \
    --correct_gtf   "${CORRECT_GTF}" \
    --chained_hints "${CHAINED_HINTS}" \
    --orfs_gtf      "${ORFS_GTF}" \
    --genome        "${GENOME}" \
    --outdir        "${OUTDIR}" \
    --flank         25000

N_ENTRIES=$(grep -c '^>' "${OUTDIR}/combined_loci.fa" || echo 0)
echo "[$(date -Iseconds)] done → ${N_ENTRIES} (locus, chain) entries in ${OUTDIR}/combined_loci.fa"
