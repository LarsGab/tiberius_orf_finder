#!/bin/bash
# Assemble long-read transcripts with StringTie -L and apply the same
# quality filter used for the short-read benchmark (tpm1cov3len300).
#
# Input:  ${RESULTS_DIR}/${sp}/longread/aligned.bam
# Output: ${RESULTS_DIR}/${sp}/longread/stringtie_lr.gtf
#         ${RESULTS_DIR}/${sp}/longread/stringtie_lr.filt_tpm1cov3len300.gtf
#         ${RESULTS_DIR}/${sp}/longread/stringtie_lr.filt_tpm1cov3len300.decisions.tsv
#
# Run after slurm_longread_align.sh completes.
#
#SBATCH --job-name=lr_stringtie
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_stringtie_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_stringtie_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300

declare -a SPECIES=(
    "Gallus_gallus"
    "Bos_taurus"
    "Zootoca_vivipara"
    "Betta_splendens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

OUTDIR=${RESULTS_DIR}/${species}/longread
BAM=${OUTDIR}/aligned.bam

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

[[ -s "${BAM}" ]] || { echo "missing BAM: ${BAM}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] bam=${BAM}"

THREADS=${SLURM_CPUS_PER_TASK:-8}
RAW_GTF=${OUTDIR}/stringtie_lr.gtf
FILT_GTF=${OUTDIR}/stringtie_lr.${FILT_TAG}.gtf
DECISIONS=${OUTDIR}/stringtie_lr.${FILT_TAG}.decisions.tsv

echo "[$(date -Iseconds)] stringtie -L assembly"
stringtie \
    -L \
    -p "${THREADS}" \
    -l STRG_LR \
    -o "${RAW_GTF}" \
    "${BAM}"

[[ -s "${RAW_GTF}" ]] || { echo "stringtie produced empty GTF: ${RAW_GTF}" >&2; exit 2; }

RAW_N=$(grep -c $'\ttranscript\t' "${RAW_GTF}" || echo 0)
echo "[$(date -Iseconds)] raw transcripts: ${RAW_N}"

echo "[$(date -Iseconds)] filter: ${FILT_TAG}"
python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
    --in-gtf       "${RAW_GTF}" \
    --out-gtf      "${FILT_GTF}" \
    --out-tsv      "${DECISIONS}" \
    --min-length   300 \
    --min-cov      3.0 \
    --min-tpm      1.0 \
    --long-length  3000 \
    --min-tpm-long 0.5

[[ -s "${FILT_GTF}" ]] || { echo "filter produced empty GTF: ${FILT_GTF}" >&2; exit 2; }

FILT_N=$(grep -c $'\ttranscript\t' "${FILT_GTF}" || echo 0)
echo "[$(date -Iseconds)] filtered transcripts: ${FILT_N}"
echo "[$(date -Iseconds)] done -> ${FILT_GTF}"
