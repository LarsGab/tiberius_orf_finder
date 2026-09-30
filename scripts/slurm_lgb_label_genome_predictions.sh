#!/bin/bash
# Apply the ORF-trained embryophyta LGB to the genome-wide Tiberius
# predictions (approach 2) for all 6 embryophyta test species.
# Splits every prediction into correct / partial / wrong (threshold=0).
#
# Output per species → hint_rescue_genome/lgb/:
#   genome_features.tsv                 (from compute_orf_features.py)
#   tiberius_genome_lgb_labeled.gtf     (all 3 classes tagged, with prob attrs)
#   tiberius_genome_lgb_labeled.scores.tsv
#   tiberius_genome_correct.gtf
#   tiberius_genome_partial.gtf
#   tiberius_genome_wrong.gtf
#
#SBATCH --job-name=lgb_gw_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_gw_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_gw_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
MODEL=${PROJDIR}/results/filter_analysis/lgb_embryophyta/lgb_3class_model.pkl

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
G=${RESULTS_DIR}/hint_rescue_genome
OUTDIR=${G}/lgb

GENOME=${RESULTS_DIR}/assembly/genome.fa
MINIPROT=${RESULTS_DIR}/proteins/miniprot_scored.gff
HINTS=${RESULTS_DIR}/proteins/miniprothint/hc.gff
PROT_FA=${RESULTS_DIR}/proteins/protein_top4.fa
IN_GTF=${G}/tiberius_genome_predictions.gtf

FEAT_TSV=${OUTDIR}/genome_features.tsv
LABELED_GTF=${OUTDIR}/tiberius_genome_lgb_labeled.gtf

echo "[$(date -Iseconds)] species=${species}"

for f in "${IN_GTF}" "${GENOME}" "${MINIPROT}" "${HINTS}" "${PROT_FA}" "${MODEL}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

# ── 1. Features ──────────────────────────────────────────────────────────────
if [[ ! -s "${FEAT_TSV}" || "${FORCE:-0}" == "1" ]]; then
    echo "[$(date -Iseconds)] Computing features on ${IN_GTF} …"
    python scripts/compute_orf_features.py \
        --orfs-gtf        "${IN_GTF}" \
        --miniprot-gff    "${MINIPROT}" \
        --hints-gff       "${HINTS}" \
        --genome          "${GENOME}" \
        --proteins-fasta  "${PROT_FA}" \
        --out             "${FEAT_TSV}"
else
    echo "[$(date -Iseconds)] Reusing existing ${FEAT_TSV}"
fi

# ── 2. Apply LGB, keep all classes (threshold=0) ────────────────────────────
echo "[$(date -Iseconds)] Applying LGB (threshold=0, all classes kept) …"
python scripts/apply_lgb_model_gtf.py \
    --model     "${MODEL}" \
    --features  "${FEAT_TSV}" \
    --in-gtf    "${IN_GTF}" \
    --out-gtf   "${LABELED_GTF}" \
    --threshold 0.0

# ── 3. Split by lgb_class ────────────────────────────────────────────────────
for cls in correct partial wrong; do
    out="${OUTDIR}/tiberius_genome_${cls}.gtf"
    grep "lgb_class \"${cls}\"" "${LABELED_GTF}" > "${out}" || true
    n=$(awk -F'\t' '$3=="transcript"' "${out}" | wc -l)
    echo "  ${cls}: ${n} transcripts → ${out}"
done

echo "[$(date -Iseconds)] done"
