#!/bin/bash
# Apply the 3-class LGB model to ORF and Tiberius predictions for all 6
# Embryophyta test species, using miniprot protein-support features.
#
# Prerequisite: slurm_miniprot_embryophyta_test.sh
#   results/training_embryophyta_test_v2/<sp>/proteins/miniprot_scored.gff
#   results/training_embryophyta_test_v2/<sp>/proteins/miniprothint/hc.gff
#
# Step 1: compute_orf_features.py on orfs.filtered.gtf (with miniprot/hints)
# Step 2: apply_lgb_model_gtf.py → orfs_lgb3_filtered.gtf
# Step 3: compute_orf_features.py on tiberius_seqlen.gtf (with miniprot/hints)
# Step 4: apply_lgb_model_gtf.py → tiberius_lgb_filtered/tiberius_lgb_filtered.gtf
#
#SBATCH --job-name=lgb_emb_test
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_emb_test_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_emb_test_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_embryophyta_test_v2
BENCH=/home/gabriell/tiberius_benchmarking
MODEL=${MODEL:-${PROJDIR}/results/filter_analysis/lgb_embryophyta/lgb_3class_model.pkl}
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

GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
ANNOT_DIR="${RESULTS_DIR}/${species}/${ANNOT_TAG}"
ORFS_GTF="${ANNOT_DIR}/orfs.filtered.gtf"
ORF_FEAT="${ANNOT_DIR}/orf_features.tsv"
ORF_LGB="${ANNOT_DIR}/orfs_lgb3_filtered.gtf"

MINIPROT="${RESULTS_DIR}/${species}/proteins/miniprot_scored.gff"
HINTS="${RESULTS_DIR}/${species}/proteins/miniprothint/hc.gff"

TIB_GTF="${BENCH}/paper/Embryophyta/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
TIB_OUTDIR="${RESULTS_DIR}/${species}/tiberius_lgb_filtered"
TIB_FEAT="${TIB_OUTDIR}/tiberius_features.tsv"
TIB_LGB="${TIB_OUTDIR}/tiberius_lgb_filtered.gtf"

echo "[$(date -Iseconds)] species=${species}"

for f in "${GENOME}" "${ORFS_GTF}" "${MINIPROT}" "${HINTS}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

# ── ORF predictions ─────────────────────────────────────────────────────────
if [[ ! -s "${ORF_LGB}" || "${FORCE:-0}" == "1" ]]; then
    echo "[$(date -Iseconds)] Computing ORF features …"
    python scripts/compute_orf_features.py \
        --orfs-gtf        "${ORFS_GTF}" \
        --miniprot-gff    "${MINIPROT}" \
        --hints-gff       "${HINTS}" \
        --genome          "${GENOME}" \
        --proteins-fasta  "${RESULTS_DIR}/${species}/proteins/protein_top4.fa" \
        --out             "${ORF_FEAT}"

    echo "[$(date -Iseconds)] Applying LGB model to ORF predictions …"
    python scripts/apply_lgb_model_gtf.py \
        --model    "${MODEL}" \
        --features "${ORF_FEAT}" \
        --in-gtf   "${ORFS_GTF}" \
        --out-gtf  "${ORF_LGB}"

    N=$(awk -F'\t' '$3=="transcript"' "${ORF_LGB}" | wc -l || true)
    echo "[$(date -Iseconds)] ORF lgb3: ${N} transcripts retained → ${ORF_LGB}"
else
    echo "[$(date -Iseconds)] SKIP ORF lgb3: already exists"
fi

# ── Tiberius predictions ─────────────────────────────────────────────────────
if [[ ! -s "${TIB_GTF}" ]]; then
    echo "[$(date -Iseconds)] SKIP Tiberius lgb: missing ${TIB_GTF}"
    exit 0
fi

if [[ ! -s "${TIB_LGB}" || "${FORCE:-0}" == "1" ]]; then
    mkdir -p "${TIB_OUTDIR}"
    echo "[$(date -Iseconds)] Computing ORF features for Tiberius GTF …"
    python scripts/compute_orf_features.py \
        --orfs-gtf        "${TIB_GTF}" \
        --miniprot-gff    "${MINIPROT}" \
        --hints-gff       "${HINTS}" \
        --genome          "${GENOME}" \
        --proteins-fasta  "${RESULTS_DIR}/${species}/proteins/protein_top4.fa" \
        --out             "${TIB_FEAT}"

    echo "[$(date -Iseconds)] Applying LGB model to Tiberius predictions …"
    python scripts/apply_lgb_model_gtf.py \
        --model    "${MODEL}" \
        --features "${TIB_FEAT}" \
        --in-gtf   "${TIB_GTF}" \
        --out-gtf  "${TIB_LGB}"

    N=$(awk -F'\t' '$3=="transcript"' "${TIB_LGB}" | wc -l || true)
    echo "[$(date -Iseconds)] Tiberius lgb3: ${N} transcripts retained → ${TIB_LGB}"
else
    echo "[$(date -Iseconds)] SKIP Tiberius lgb: already exists"
fi

echo "[$(date -Iseconds)] done"
