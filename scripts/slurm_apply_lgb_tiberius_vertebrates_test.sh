#!/bin/bash
# Apply the trained 3-class LGB model to Tiberius ab initio predictions
# for all 6 vertebrates_test species.
#
# Step 1: compute_orf_features.py on tiberius_seqlen.gtf (protein/structural
#         features using existing fix_stop/miniprot_scored.gff + hc.gff).
#         lorf_class will be absent (not a Tiberius attribute); the LGB model
#         fills those dummy columns with 0 and uses the remaining features.
# Step 2: apply_lgb_model_gtf.py — filter transcripts with P(not-wrong) >= 0.5.
#
# Input:
#   /home/gabriell/tiberius_benchmarking/paper/Vertebrata/<sp>/
#     results/predictions/tiberius/tiberius_seqlen.gtf
#   results/vertebrates_test/<sp>/fix_stop/miniprot_scored.gff
#   results/vertebrates_test/<sp>/fix_stop/miniprothint/hc.gff
#   results/vertebrates_test/<sp>/assembly/genome.fa
#
# Output (per species):
#   results/vertebrates_test/<sp>/tiberius_lgb_filtered/
#     tiberius_features.tsv
#     tiberius_lgb_filtered.gtf
#     tiberius_lgb_filtered.scores.tsv
#
#SBATCH --job-name=lgb_tib_test
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_tib_test_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_tib_test_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
BENCH=/home/gabriell/tiberius_benchmarking

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
TIB_GTF="${BENCH}/paper/Vertebrata/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
MINIPROT="${RESULTS_DIR}/${species}/fix_stop/miniprot_scored.gff"
HINTS="${RESULTS_DIR}/${species}/fix_stop/miniprothint/hc.gff"
OUTDIR="${RESULTS_DIR}/${species}/tiberius_lgb_filtered"
FEAT_TSV="${OUTDIR}/tiberius_features.tsv"
OUT_GTF="${OUTDIR}/tiberius_lgb_filtered.gtf"

echo "[$(date -Iseconds)] species=${species}"

for f in "${GENOME}" "${TIB_GTF}" "${MINIPROT}" "${HINTS}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

if [[ -s "${OUT_GTF}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: tiberius_lgb_filtered.gtf already exists"
    exit 0
fi

mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] Computing ORF features for Tiberius GTF …"
python scripts/compute_orf_features.py \
    --orfs-gtf     "${TIB_GTF}" \
    --miniprot-gff "${MINIPROT}" \
    --hints-gff    "${HINTS}" \
    --genome       "${GENOME}" \
    --out          "${FEAT_TSV}"

echo "[$(date -Iseconds)] Applying LGB model …"
python scripts/apply_lgb_model_gtf.py \
    --model    "${PROJDIR}/results/filter_analysis/lgb_3class_model.pkl" \
    --features "${FEAT_TSV}" \
    --in-gtf   "${TIB_GTF}" \
    --out-gtf  "${OUT_GTF}"

echo "[$(date -Iseconds)] done → ${OUT_GTF}"
N=$(awk -F'\t' '$3=="transcript"' "${OUT_GTF}" | wc -l || true)
echo "[$(date -Iseconds)] ${N} transcripts retained"
