#!/bin/bash
# DRUSILLA annotate on the pyVARUS-derived StringTie assembly.
# Uses the same clade weights/config as the VARUS-based annotate.
# Output goes into <sp>/annotate_pyvarus_<tag>/ so it doesn't collide.
#
#SBATCH --job-name=annot_pyv
#SBATCH --partition=storm,vision-fast,vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_pyv_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_pyv_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}" "${CLADE:?}"
: "${WEIGHTS:?}" "${CONFIG:?}" "${TAG:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
STF=${SP_DIR}/stringtie_pyvarus/stringtie.filt_tpm1cov3len300.gtf
OUTDIR=${SP_DIR}/annotate_pyvarus_${TAG}

[[ -s "${GENOME}" && -s "${STF}" ]] || { echo "SKIP ${SPECIES}" >&2; exit 0; }
mkdir -p "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${SPECIES}: DRUSILLA output exists"; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder
cd "${PROJDIR}"

run_timed "${CLADE}" "${SPECIES}" pyvarus annotate -- \
    python "${SCRIPTS_DIR}/annotate.py" \
        --stringtie-gtf "${STF}" --genome "${GENOME}" \
        --weights "${WEIGHTS}" --config "${CONFIG}" \
        --out-dir "${OUTDIR}" --batch-size 200 \
        --threads "${SLURM_CPUS_PER_TASK}" --lorf-class

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
