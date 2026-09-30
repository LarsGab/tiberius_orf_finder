#!/bin/bash
# Run Vipsania annotate with --finetune, per species, on the brain GPU nodes.
# One SLURM array task per species. Emits <sp>/vipsania/vip.gtf under the
# clade results tree; runtimes appended to $RUNTIME_TSV.
#
# Usage:
#   CLADE=vertebrates_test \
#   MODEL_NAME=Vertebrata \
#   SPECIES_FILE=/path/to/species_list.txt \
#   RESULTS_ROOT=/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test \
#   RUNTIME_TSV=<results_root>/runtimes.tsv \
#   sbatch --array=1-9 scripts/slurm_vipsania_annotate.sh
#
# SPECIES_FILE: one species name per line.
# MODEL_NAME:   Vipsania model id (Vertebrata, Insecta, Fungi, Embryophyta, ...).
#
#SBATCH --job-name=vipsania_ann
#SBATCH --partition=storm,vision-fast,vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/vipsania_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/vipsania_%A_%a.err

set -euo pipefail

: "${CLADE:?CLADE required}"
: "${MODEL_NAME:?MODEL_NAME required (e.g. Vertebrata / Insecta / Fungi / Embryophyta)}"
: "${SPECIES_FILE:?SPECIES_FILE required}"
: "${RESULTS_ROOT:?RESULTS_ROOT required}"

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
RUNTIME_TSV=${RUNTIME_TSV:-${RESULTS_ROOT}/runtimes.tsv}

mkdir -p "${PROJDIR}/logs"

# Runtime logger
# shellcheck disable=SC1091
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV

# Species selection: task-array index → line in SPECIES_FILE
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")
[[ -n "${SPECIES}" ]] || { echo "no species for array index ${TASK_ID}" >&2; exit 2; }

SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
[[ -s "${GENOME}" ]] || { echo "ERROR: missing ${GENOME}" >&2; exit 3; }

OUT_DIR=${SP_DIR}/vipsania
mkdir -p "${OUT_DIR}"
VIP_GTF=${OUT_DIR}/vip.gtf

echo "[$(date -Iseconds)] Vipsania annotate — sp=${SPECIES} model=${MODEL_NAME} clade=${CLADE}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vipsania

# Vipsania emits .gtf directly when -o has .gtf extension.
# Reduce memory footprint on large genomes via smaller finetune batch.
run_timed "${CLADE}" "${SPECIES}" vipsania annotate -- \
    vipsania annotate "${MODEL_NAME}" "${GENOME}" -o "${VIP_GTF}" \
        --finetune --finetune_B 2 -B 4

echo "[$(date -Iseconds)] done -> ${VIP_GTF}"
