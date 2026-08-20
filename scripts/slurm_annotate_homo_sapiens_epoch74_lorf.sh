#!/bin/bash
# Annotate Homo_sapiens with TiberiusORF epoch_74 (filt_tpm1cov3len300, lorf-class).
#
# Prerequisite: slurm_stringtie_homo_sapiens.sh
#
# Output:
#   results/vertebrates_test/Homo_sapiens/annotate_epoch_74_filt_tpm1cov3len300_lorf/
#     orfs.gtf
#     orfs.partial.gtf
#     orfs.filtered.gtf
#
#SBATCH --job-name=annot_homo
#SBATCH --partition=storm,vision-fast,vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_homo_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_homo_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
species=Homo_sapiens

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml

WEIGHTS_TAG=$(basename "${WEIGHTS}" .weights.h5)
FILT_TAG=filt_tpm1cov3len300

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE_RAW=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
STRINGTIE_FILT=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}_${FILT_TAG}_lorf

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] weights=${WEIGHTS_TAG}"
echo "[$(date -Iseconds)] out=${OUTDIR}"

test -s "${WEIGHTS}" || { echo "ERROR: missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "ERROR: missing config: ${CONFIG}"   >&2; exit 2; }
test -s "${GENOME}"  || { echo "ERROR: missing genome: ${GENOME}"   >&2; exit 2; }
test -s "${STRINGTIE_RAW}" || { echo "ERROR: missing stringtie.gtf — run slurm_stringtie_homo_sapiens.sh first" >&2; exit 2; }

if [[ -s "${OUTDIR}/orfs.partial.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP: orfs.partial.gtf already exists"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

if [[ ! -s "${STRINGTIE_FILT}" ]]; then
    echo "[$(date -Iseconds)] filtering stringtie -> ${STRINGTIE_FILT}"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf       "${STRINGTIE_RAW}" \
        --out-gtf      "${STRINGTIE_FILT}" \
        --out-tsv      "${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.decisions.tsv" \
        --min-length   300 \
        --min-cov      3.0 \
        --min-tpm      1.0 \
        --long-length  3000 \
        --min-tpm-long 0.5
fi
test -s "${STRINGTIE_FILT}" || { echo "ERROR: filtered GTF empty: ${STRINGTIE_FILT}" >&2; exit 2; }

cd "${PROJDIR}"

echo "[$(date -Iseconds)] annotate ..."
python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE_FILT}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK}" \
    --lorf-class \
    --partial-out   "${OUTDIR}/orfs.partial.gtf"

echo "[$(date -Iseconds)] subseq collapse ..."
python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

N=$(awk '$3=="transcript"' "${OUTDIR}/orfs.filtered.gtf" | wc -l || true)
echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf (${N} transcripts)"
