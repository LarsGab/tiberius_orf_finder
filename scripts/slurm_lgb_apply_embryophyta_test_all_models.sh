#!/bin/bash
# Apply all 3 LGB variants (ORF-trained, Tib-trained, Mix-trained) to both
# ORF and Tiberius test predictions for the 6 embryophyta test species.
#
# Reuses the already-computed orf_features.tsv / tiberius_features.tsv
# (features are model-independent).
#
# Output layout per species:
#   annotate_run001_e300/
#     orfs_lgb3_orf_filtered.gtf   (ORF preds filtered by ORF-trained model)
#     orfs_lgb3_tib_filtered.gtf   (ORF preds filtered by Tib-trained model)
#     orfs_lgb3_mix_filtered.gtf   (ORF preds filtered by Mix-trained model)
#   tiberius_lgb_filtered/
#     tiberius_lgb_orf_filtered.gtf
#     tiberius_lgb_tib_filtered.gtf
#     tiberius_lgb_mix_filtered.gtf
#
#SBATCH --job-name=lgb_all3_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_all3_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_all3_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_embryophyta_test_v2
BENCH=/home/gabriell/tiberius_benchmarking
ANNOT_TAG=annotate_run001_e300

MODEL_ORF=${PROJDIR}/results/filter_analysis/lgb_embryophyta/lgb_3class_model.pkl
MODEL_TIB=${PROJDIR}/results/filter_analysis/lgb_embryophyta_tib/lgb_3class_model.pkl
MODEL_MIX=${PROJDIR}/results/filter_analysis/lgb_embryophyta_mix/lgb_3class_model.pkl

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

ANNOT_DIR="${RESULTS_DIR}/${species}/${ANNOT_TAG}"
ORFS_GTF="${ANNOT_DIR}/orfs.filtered.gtf"
ORF_FEAT="${ANNOT_DIR}/orf_features.tsv"

TIB_GTF="${BENCH}/paper/Embryophyta/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
TIB_OUTDIR="${RESULTS_DIR}/${species}/tiberius_lgb_filtered"
TIB_FEAT="${TIB_OUTDIR}/tiberius_features.tsv"

echo "[$(date -Iseconds)] species=${species}"

for f in "${ORFS_GTF}" "${ORF_FEAT}" "${TIB_GTF}" "${TIB_FEAT}" \
         "${MODEL_ORF}" "${MODEL_TIB}" "${MODEL_MIX}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

mkdir -p "${TIB_OUTDIR}"

for entry in "orf:${MODEL_ORF}" "tib:${MODEL_TIB}" "mix:${MODEL_MIX}"; do
    tag=${entry%%:*}
    model=${entry##*:}

    ORF_OUT="${ANNOT_DIR}/orfs_lgb3_${tag}_filtered.gtf"
    TIB_OUT="${TIB_OUTDIR}/tiberius_lgb_${tag}_filtered.gtf"

    echo "[$(date -Iseconds)] === ${tag} model → ORF preds ==="
    python scripts/apply_lgb_model_gtf.py \
        --model    "${model}" \
        --features "${ORF_FEAT}" \
        --in-gtf   "${ORFS_GTF}" \
        --out-gtf  "${ORF_OUT}"
    N=$(awk -F'\t' '$3=="transcript"' "${ORF_OUT}" | wc -l || true)
    echo "  ${N} transcripts retained → ${ORF_OUT}"

    echo "[$(date -Iseconds)] === ${tag} model → Tiberius preds ==="
    python scripts/apply_lgb_model_gtf.py \
        --model    "${model}" \
        --features "${TIB_FEAT}" \
        --in-gtf   "${TIB_GTF}" \
        --out-gtf  "${TIB_OUT}"
    N=$(awk -F'\t' '$3=="transcript"' "${TIB_OUT}" | wc -l || true)
    echo "  ${N} transcripts retained → ${TIB_OUT}"
done

echo "[$(date -Iseconds)] done"
