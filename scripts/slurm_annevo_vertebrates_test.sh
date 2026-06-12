#!/bin/bash
# Run ANNEVO inference on the 9 vertebrate TEST species, one species per
# array task, using the per-lineage model checkpoint:
#   Actinopteri : Archocentrus / Betta / Takifugu / Pristiophorus (shark, closest)
#   Aves        : Gallus      / Zootoca (lizard, closest)
#   Mammalia    : Bos / Delphinapterus / Homo
#
# Inputs:  genomes published by `submit_vertebrates_test.sh` at
#          ${RESULTS_DIR}/<Genus_species>/assembly/genome.fa
# Outputs: ${RESULTS_DIR}/<Genus_species>/annevo/annevo.gff
#
# Usage:  sbatch scripts/slurm_annevo_vertebrates_test.sh
#
#SBATCH --job-name=annevo_vert_test
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/annevo_vert_test_%A_%a.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/annevo_vert_test_%A_%a.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ANNEVO_DIR=/home/gabriell/programs/ANNEVO
MODEL_DIR=${ANNEVO_DIR}/saved_model

mkdir -p "${PROJDIR}/logs"

# species_underscored:lineage  (lineage doubles as model basename suffix)
declare -a JOBS=(
    "Archocentrus_centrarchus:Actinopteri"
    "Betta_splendens:Actinopteri"
    "Takifugu_rubripes:Actinopteri"
    "Pristiophorus_japonicus:Actinopteri"
    "Gallus_gallus:Aves"
    "Zootoca_vivipara:Aves"
    "Bos_taurus:Mammalia"
    "Delphinapterus_leucas:Mammalia"
    "Homo_sapiens:Mammalia"
)

entry=${JOBS[$SLURM_ARRAY_TASK_ID]}
species=${entry%:*}
lineage=${entry#*:}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
MODEL=${MODEL_DIR}/ANNEVO_${lineage}.pt
OUTDIR=${RESULTS_DIR}/${species}/annevo
OUTGFF=${OUTDIR}/annevo.gff

test -s "${GENOME}" || { echo "missing genome: ${GENOME}" >&2; exit 2; }
test -s "${MODEL}"  || { echo "missing model: ${MODEL}"   >&2; exit 2; }
mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate ANNEVO

echo "[$(date -Iseconds)] species=${species} lineage=${lineage}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] model=${MODEL}"
echo "[$(date -Iseconds)] outgff=${OUTGFF}"

python "${ANNEVO_DIR}/annotation.py" \
    -g "${GENOME}" \
    -m "${MODEL}" \
    -l "${lineage}" \
    -o "${OUTGFF}" \
    -t 48 \
    --show_log
