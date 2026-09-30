#!/bin/bash
# Run StringTie on the pyVARUS BAM per species; write output into
# <sp>/stringtie_pyvarus/ so it doesn't collide with the legacy VARUS output.
#
#SBATCH --job-name=st_pyvarus
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/st_pyvarus_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/st_pyvarus_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}" "${CLADE:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
SP_DIR=${RESULTS_ROOT}/${SPECIES}
BAM=${SP_DIR}/pyvarus/pyVARUS.bam
OUT_DIR=${SP_DIR}/stringtie_pyvarus
OUT_RAW=${OUT_DIR}/stringtie.gtf
OUT_FILT=${OUT_DIR}/stringtie.filt_tpm1cov3len300.gtf

[[ -s "${BAM}" ]] || { echo "SKIP ${SPECIES}: pyVARUS.bam missing" >&2; exit 0; }
mkdir -p "${OUT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

if [[ ! -s "${OUT_RAW}" ]]; then
    run_timed "${CLADE}" "${SPECIES}" pyvarus stringtie -- \
        stringtie -p "${SLURM_CPUS_PER_TASK}" -o "${OUT_RAW}" "${BAM}"
fi

if [[ ! -s "${OUT_FILT}" ]]; then
    run_timed "${CLADE}" "${SPECIES}" pyvarus stringtie_filter -- \
        python "${SCRIPTS_DIR}/filter_stringtie_gtf.py" \
            --in-gtf "${OUT_RAW}" --out-gtf "${OUT_FILT}" \
            --out-tsv "${OUT_DIR}/stringtie.filt_tpm1cov3len300.decisions.tsv" \
            --min-length 300 --min-cov 3.0 --min-tpm 1.0 --long-length 3000 --min-tpm-long 0.5
fi

echo "[$(date -Iseconds)] done -> ${OUT_FILT}"
