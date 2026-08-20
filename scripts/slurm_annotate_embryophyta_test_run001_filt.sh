#!/bin/bash
# Run annotate.py on the 7 embryophyta test species using the CNN-LSTM
# run001 embryophyta model, with StringTie pre-filtered at tpm1cov3len300
# (same setting used for vertebrates production runs).
#
# Usage: sbatch --array=0-6 scripts/slurm_annotate_embryophyta_test_run001_filt.sh [epoch]
#   epoch defaults to 300 (latest checkpoint as of 2026-08-17).
#
# Filter rule: length >= 300, cov >= 3, TPM >= 1 (relaxed to TPM >= 0.5 for len >= 3000).
# Filter intermediate: <testdir>/<sp>/stringtie/stringtie.filt_tpm1cov3len300.gtf
#
# Outputs per species:
#   results/training_embryophyta_test_v2/<sp>/annotate_run001_e<epoch>_filt_tpm1cov3len300/
#     orfs.gtf
#     orfs.filtered.gtf
#     dropped_subsequences.tsv
#
#SBATCH --job-name=annot_emb_filt
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_filt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_filt_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2

EPOCH="${1:-300}"
FILT_TAG=filt_tpm1cov3len300
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2/epoch_${EPOCH}.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Brachypodium_distachyon"
    "Eschscholzia_californica"
    "Freycinetia_multiflora"
    "Medicago_truncatula"
    "Mimulus_guttatus"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TESTDIR}/${species}/assembly/genome.fa
RAW_STRINGTIE=${TESTDIR}/${species}/stringtie/stringtie.gtf
FILT_STRINGTIE=${TESTDIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${TESTDIR}/${species}/annotate_run001_e${EPOCH}_${FILT_TAG}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${RAW_STRINGTIE}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

cd "${PROJDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID}"
echo "[$(date -Iseconds)] species=${species}  epoch=${EPOCH}"

# ── Step 1: filter StringTie ────────────────────────────────────────────────
if [[ ! -s "${FILT_STRINGTIE}" ]]; then
    echo "[$(date -Iseconds)] filtering stringtie -> ${FILT_STRINGTIE}"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf  "${RAW_STRINGTIE}" \
        --out-gtf "${FILT_STRINGTIE}"
else
    echo "[$(date -Iseconds)] filtered stringtie exists, reusing"
fi

# ── Step 2: annotate ────────────────────────────────────────────────────────
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] stringtie=${FILT_STRINGTIE}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${FILT_STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK:-4}"

# ── Step 3: subsequence collapse ────────────────────────────────────────────
python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
