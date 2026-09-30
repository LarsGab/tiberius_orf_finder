#!/bin/bash
# Approach 1: relaxed HC — build hint_rescue inputs where the "chained_hints"
# come directly from miniprot_scored.gff (every alignment tagged with chain_id
# = <Parent>_<prot>), skipping the hc.gff filter used by chainedHints.py.
#
# This gives best_chain_at_locus a much bigger candidate pool, letting loci
# that had no HC-supported chain (e.g. locus_0000186 in the t193/t194 case)
# now qualify for hint-guided rescue.
#
# Output per species → hint_rescue_relaxed/
#
#SBATCH --job-name=prep_hr_emb_rlx
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_hr_emb_rlx_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_hr_emb_rlx_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
SCRIPTS_DIR=${PROJDIR}/scripts
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
MINIPROT_SCORED=${RESULTS_DIR}/proteins/miniprot_scored.gff
GENOME=${RESULTS_DIR}/assembly/genome.fa
ORFS_GTF=${RESULTS_DIR}/${ANNOT_TAG}/orfs.gtf
OUTDIR=${RESULTS_DIR}/hint_rescue_relaxed

echo "[$(date -Iseconds)] species=${species}"

mkdir -p "${OUTDIR}"

for f in "${TIB_LGB}" "${MINIPROT_SCORED}" "${GENOME}" "${GENOME}.fai" "${ORFS_GTF}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Step 1: split tiberius_lgb_orf_filtered.gtf by lgb_class ────────────────
PARTIAL_GTF=${OUTDIR}/tiberius_lgb_partial.gtf
CORRECT_GTF=${OUTDIR}/tiberius_lgb_correct.gtf
grep 'lgb_class "partial"' "${TIB_LGB}" > "${PARTIAL_GTF}" || true
grep 'lgb_class "correct"' "${TIB_LGB}" > "${CORRECT_GTF}" || true

[[ -s "${PARTIAL_GTF}" ]] || { echo "SKIP ${species}: no 'partial' predictions" >&2; exit 0; }
echo "[$(date -Iseconds)] partial CDS lines: $(wc -l < "${PARTIAL_GTF}")   correct: $(wc -l < "${CORRECT_GTF}")"

# ── Step 2: tag every miniprot alignment with chain_id (NO hc filter) ────────
RELAXED_HINTS=${OUTDIR}/chained_hints.gff
if [[ ! -s "${RELAXED_HINTS}" ]]; then
    echo "[$(date -Iseconds)] Tagging miniprot alignments with chain_id (relaxed) …"
    python "${SCRIPTS_DIR}/tag_chains_from_miniprot.py" \
        "${MINIPROT_SCORED}" \
        "${RELAXED_HINTS}"
    echo "  relaxed chained hints: $(wc -l < "${RELAXED_HINTS}") lines"
else
    echo "[$(date -Iseconds)] chained_hints.gff (relaxed) exists, reusing"
fi

# ── Step 3: build multi-FASTA + hints GFF ────────────────────────────────────
echo "[$(date -Iseconds)] Building combined_loci.fa and combined_hints.gff …"
python "${SCRIPTS_DIR}/prepare_hint_rescue_loci.py" \
    --partial_gtf   "${PARTIAL_GTF}" \
    --correct_gtf   "${CORRECT_GTF}" \
    --chained_hints "${RELAXED_HINTS}" \
    --orfs_gtf      "${ORFS_GTF}" \
    --genome        "${GENOME}" \
    --outdir        "${OUTDIR}" \
    --flank         25000

N_ENTRIES=$(grep -c '^>' "${OUTDIR}/combined_loci.fa" || echo 0)
echo "[$(date -Iseconds)] done → ${N_ENTRIES} (locus, chain) entries"
