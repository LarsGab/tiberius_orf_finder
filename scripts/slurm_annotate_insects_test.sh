#!/bin/bash
# DRUSILLA annotate for insects_test_v2: runs annotate.py with the
# insect-trained cnn_lstm_run006_insects weights + cnn_lstm_run006.yaml config
# on the filtered StringTie assembly. One array task per species.
#
#SBATCH --job-name=annot_ins
#SBATCH --partition=storm,vision-fast,vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_ins_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_ins_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV
CLADE=insects_test_v2

EPOCH=${INSECT_EPOCH:-51}
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_run006_insects/epoch_${EPOCH}.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml
FILT_TAG=filt_tpm1cov3len300
TAG=run006_insects_e${EPOCH}_${FILT_TAG}

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config:  ${CONFIG}"  >&2; exit 2; }

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
STF=${SP_DIR}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${SP_DIR}/annotate_${TAG}

test -s "${GENOME}" || { echo "missing genome ${GENOME}" >&2; exit 3; }
test -s "${STF}"    || { echo "missing stringtie ${STF}" >&2; exit 3; }
mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder
cd "${PROJDIR}"

echo "[$(date -Iseconds)] annotate sp=${SPECIES} tag=${TAG}"

run_timed "${CLADE}" "${SPECIES}" tiberius annotate -- \
    python "${SCRIPTS_DIR}/annotate.py" \
        --stringtie-gtf "${STF}" \
        --genome        "${GENOME}" \
        --weights       "${WEIGHTS}" \
        --config        "${CONFIG}" \
        --out-dir       "${OUTDIR}" \
        --batch-size    200 \
        --threads       "${SLURM_CPUS_PER_TASK}" \
        --lorf-class

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
