#!/bin/bash
# Experiment: pre-filter the existing vertebrates_test StringTie assemblies
# at the transcript level before running annotate.py with epoch_11.
#
# Filter rule (see scripts/filter_stringtie_gtf.py):
#     length >= 300
#     AND cov >= 3
#     AND TPM >= 1.0  (relaxed to TPM >= 0.5 for length >= 3000)
#
# TPM is per-kb, so long real transcripts drift downward in TPM at the
# same gene-level expression as a short one. Length-aware TPM threshold
# avoids cutting them. cov stays as a hard floor (absolute read support).
#
# Output:  ${RESULTS_DIR}/<sp>/annotate_${WEIGHTS_TAG}_filt_tpm1cov3len300/orfs.gtf
# Filter intermediate: ${RESULTS_DIR}/<sp>/stringtie/stringtie.filt_tpm1cov3len300.gtf
#
#SBATCH --job-name=annot_vt_filt
#SBATCH --partition=vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vt_filt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vt_filt_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2/epoch_11.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml

WEIGHTS_TAG=$(basename "${WEIGHTS}" .weights.h5)
FILT_TAG=filt_tpm1cov3len300

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Gallus_gallus"
    "Pristiophorus_japonicus"
    "Bos_taurus"
    "Delphinapterus_leucas"
    "Homo_sapiens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE_IN=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
STRINGTIE_FILT=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
DECISIONS_TSV=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.decisions.tsv
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}_${FILT_TAG}

if [[ ! -s "${STRINGTIE_IN}" || ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: stringtie or genome not ready"
    echo "                       stringtie=${STRINGTIE_IN}"
    echo "                       genome=${GENOME}"
    exit 0
fi

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config: ${CONFIG}"   >&2; exit 2; }
mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] filter in=${STRINGTIE_IN}"
echo "[$(date -Iseconds)] filter out=${STRINGTIE_FILT}"

python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
    --in-gtf      "${STRINGTIE_IN}" \
    --out-gtf     "${STRINGTIE_FILT}" \
    --out-tsv     "${DECISIONS_TSV}" \
    --min-length  300 \
    --min-cov     3.0 \
    --min-tpm     1.0 \
    --long-length 3000 \
    --min-tpm-long 0.5

test -s "${STRINGTIE_FILT}" || { echo "filter produced no GTF: ${STRINGTIE_FILT}" >&2; exit 2; }

echo "[$(date -Iseconds)] annotate out=${OUTDIR}/orfs.gtf"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE_FILT}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK}"
