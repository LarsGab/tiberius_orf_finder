#!/bin/bash
# Assemble long-read transcripts with StringTie -L and filter
# (tpm1cov3len300) for the 4 embryophyta long-read species.
#
# Input:  ${TESTDIR}/${sp}/longread/aligned.bam
# Output: ${TESTDIR}/${sp}/longread/stringtie_lr.gtf
#         ${TESTDIR}/${sp}/longread/stringtie_lr.filt_tpm1cov3len300.gtf
#
#SBATCH --job-name=lr_emb_st
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_emb_st_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_emb_st_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
FILT_TAG=filt_tpm1cov3len300

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Eschscholzia_californica"
    "Medicago_truncatula"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

OUTDIR=${TESTDIR}/${species}/longread
BAM=${OUTDIR}/aligned.bam

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"
[[ -s "${BAM}" ]] || { echo "missing BAM: ${BAM}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] species=${species}"

THREADS=${SLURM_CPUS_PER_TASK:-8}
RAW_GTF=${OUTDIR}/stringtie_lr.gtf
FILT_GTF=${OUTDIR}/stringtie_lr.${FILT_TAG}.gtf
DECISIONS=${OUTDIR}/stringtie_lr.${FILT_TAG}.decisions.tsv

echo "[$(date -Iseconds)] stringtie -L"
stringtie -L -p "${THREADS}" -l STRG_LR -o "${RAW_GTF}" "${BAM}"
[[ -s "${RAW_GTF}" ]] || { echo "stringtie produced empty GTF" >&2; exit 2; }
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

[[ -s "${FILT_GTF}" ]] || { echo "filter produced empty GTF" >&2; exit 2; }
FILT_N=$(grep -c $'\ttranscript\t' "${FILT_GTF}" || echo 0)
echo "[$(date -Iseconds)] filtered transcripts: ${FILT_N}"
echo "[$(date -Iseconds)] done -> ${FILT_GTF}"
