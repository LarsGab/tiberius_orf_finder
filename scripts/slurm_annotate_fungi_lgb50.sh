#!/bin/bash
# DRUSILLA annotate for the 50 fungi training species selected for LGB.
# Runs on GPU with fungi Tiberius weights.
#
#SBATCH --job-name=annot_fungi_lgb
#SBATCH --partition=storm,vision-fast,vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_fungi_lgb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_fungi_lgb_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
FUNGI_WEIGHTS=${FUNGI_WEIGHTS:-${PROJDIR}/results/models/cnn_lstm_run006_fungi/epoch_300.weights.h5}
FUNGI_CONFIG=${FUNGI_CONFIG:-${PROJDIR}/configs/cnn_lstm_run006.yaml}

source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV
CLADE=fungi_train_lgb
ANN_TAG=annotate_run006_lgb50_prep

test -s "${FUNGI_WEIGHTS}" || { echo "missing weights ${FUNGI_WEIGHTS}" >&2; exit 2; }

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
STF=${SP_DIR}/stringtie/stringtie.filt_tpm1cov3len300.gtf
OUTDIR=${SP_DIR}/${ANN_TAG}

[[ -s "${GENOME}" && -s "${STF}" ]] || { echo "missing inputs for ${SPECIES}" >&2; exit 3; }
mkdir -p "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP: DRUSILLA output already exists"; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder
cd "${PROJDIR}"

run_timed "${CLADE}" "${SPECIES}" tiberius annotate -- \
    python "${SCRIPTS_DIR}/annotate.py" \
        --stringtie-gtf "${STF}" --genome "${GENOME}" \
        --weights "${FUNGI_WEIGHTS}" --config "${FUNGI_CONFIG}" \
        --out-dir "${OUTDIR}" --batch-size 200 \
        --threads "${SLURM_CPUS_PER_TASK}" --lorf-class

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
