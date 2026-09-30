#!/bin/bash
# Regenerate VARUS-like BAM with pyVARUS (new logic, singularity container)
# for one species per SLURM array task. Output written to
# <sp>/pyvarus/pyVARUS.bam and the auxiliary files (manifest, splicedb log).
#
# Usage:
#   CLADE=vertebrates_test \
#   SPECIES_CSV=/path/to/nextflow/conf/species_vertebrates_test.csv \
#   RESULTS_ROOT=/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \
#   RUNTIME_TSV=<results_root>/runtimes.tsv \
#   NCBI_EMAIL=lgabriel23@gmx.de \
#   sbatch --array=1-9 scripts/slurm_pyvarus_run.sh
#
# SPECIES_CSV columns must include: species,accession[,annotation]
# The species column supplies the sci-name (underscored form → dots→spaces).
#
#SBATCH --job-name=pyvarus
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/pyvarus_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/pyvarus_%A_%a.err

set -euo pipefail

: "${CLADE:?CLADE required}"
: "${SPECIES_CSV:?SPECIES_CSV required}"
: "${RESULTS_ROOT:?RESULTS_ROOT required}"
: "${NCBI_EMAIL:?NCBI_EMAIL required}"

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
SIF=${SIF:-/projects/AI-GUSTUS/pyvarus.sif}
RUNTIME_TSV=${RUNTIME_TSV:-${RESULTS_ROOT}/runtimes.tsv}

mkdir -p "${PROJDIR}/logs"

# Modules
if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load singularity/3.11.3 || true
fi

# Runtime logger
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
# skip header, then Nth row
LINE=$(awk -v n="${TASK_ID}" -F, 'NR>1 && ++i==n{print; exit}' "${SPECIES_CSV}")
[[ -n "${LINE}" ]] || { echo "no row for array index ${TASK_ID}" >&2; exit 2; }
SPECIES=$(awk -F, '{print $1}' <<<"${LINE}")
SPECIES_SCI=${SPECIES//_/ }
[[ -n "${SPECIES}" ]] || { echo "empty species" >&2; exit 2; }

SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
[[ -s "${GENOME}" ]] || { echo "ERROR: missing ${GENOME}" >&2; exit 3; }

OUT_DIR=${SP_DIR}/pyvarus
mkdir -p "${OUT_DIR}"

# Threads
NPROC=${SLURM_CPUS_PER_TASK:-16}

# Bind paths for singularity (must include the SPECIES results dir)
BIND=/projects/AI-GUSTUS,/home/gabriell,${OUT_DIR}
run_vc() { singularity exec --bind "${BIND}" "${SIF}" "$@"; }

echo "[$(date -Iseconds)] pyVARUS — sp=${SPECIES} (${SPECIES_SCI}) clade=${CLADE}"

# 1) runlist
if [[ ! -s "${OUT_DIR}/Runlist.tsv" ]]; then
    run_timed "${CLADE}" "${SPECIES}" pyvarus runlist -- \
        run_vc varus runlist "${SPECIES_SCI}" --outdir "${OUT_DIR}" \
            --email "${NCBI_EMAIL}"
else
    echo "[skip] Runlist.tsv exists"
fi

# 2) index
if [[ ! -d "${OUT_DIR}/genome" ]] || [[ -z "$(ls -A "${OUT_DIR}/genome" 2>/dev/null)" ]]; then
    run_timed "${CLADE}" "${SPECIES}" pyvarus index -- \
        run_vc varus index "${GENOME}" --outdir "${OUT_DIR}/genome" --threads "${NPROC}"
else
    echo "[skip] genome/ index exists"
fi

# 3) run
if [[ ! -s "${OUT_DIR}/VARUS.bam" ]]; then
    run_timed "${CLADE}" "${SPECIES}" pyvarus run -- \
        run_vc varus run "${SPECIES_SCI}" "${GENOME}" \
            --runlist "${OUT_DIR}/Runlist.tsv" \
            --index   "${OUT_DIR}/genome" \
            --outdir  "${OUT_DIR}" \
            --threads "${NPROC}"
else
    echo "[skip] VARUS.bam exists"
fi

# Distinguish from legacy VARUS.bam
if [[ -s "${OUT_DIR}/VARUS.bam" && ! -s "${OUT_DIR}/pyVARUS.bam" ]]; then
    ln -sf "${OUT_DIR}/VARUS.bam" "${OUT_DIR}/pyVARUS.bam"
fi

echo "[$(date -Iseconds)] done -> ${OUT_DIR}/pyVARUS.bam"
