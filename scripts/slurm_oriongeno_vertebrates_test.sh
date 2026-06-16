#!/bin/bash
# Run OrionGeno inference on the 9 vertebrate TEST species, one species per
# array task, using the per-clade checkpoint:
#   fish              : Archocentrus / Betta / Takifugu / Pristiophorus (shark, closest)
#   birds             : Gallus
#   other_vertebrates : Zootoca (lizard — Lepidosauria; not bird/fish/mammal)
#   mammals           : Bos / Delphinapterus / Homo
#
# Inputs:  genomes published by `submit_vertebrates_test.sh` at
#          ${RESULTS_DIR}/<Genus_species>/assembly/genome.fa[.gz]
#          (.gz is decompressed into $SLURM_TMPDIR for OrionGeno)
# Outputs: ${RESULTS_DIR}/<Genus_species>/oriongeno/oriongeno.gtf
#
# Usage:  sbatch scripts/slurm_oriongeno_vertebrates_test.sh
#
#SBATCH --job-name=oriongeno_vert_test
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/oriongeno_vert_test_%A_%a.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/oriongeno_vert_test_%A_%a.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
ORIONGENO_DIR=/home/gabriell/programs/OrionGeno
CHECKPOINT_DIR=/home/gabriell/programs/OrionGeno_models/checkpoints

mkdir -p "${PROJDIR}/logs"

# species_underscored:model_basename
declare -a JOBS=(
    "Archocentrus_centrarchus:oriongeno_fish"
    "Betta_splendens:oriongeno_fish"
    "Takifugu_rubripes:oriongeno_fish"
    "Pristiophorus_japonicus:oriongeno_fish"
    "Gallus_gallus:oriongeno_birds"
    "Zootoca_vivipara:oriongeno_other_vertebrates"
    "Bos_taurus:oriongeno_mammals"
    "Delphinapterus_leucas:oriongeno_mammals"
    "Homo_sapiens:oriongeno_mammals"
)

entry=${JOBS[$SLURM_ARRAY_TASK_ID]}
species=${entry%:*}
model=${entry#*:}

source "$(dirname "$0")/_stage_genome.sh"
trap 'stage_genome_cleanup' EXIT
GENOME=$(stage_genome "${RESULTS_DIR}/${species}/assembly")
CHECKPOINT=${CHECKPOINT_DIR}/${model}
OUTDIR=${RESULTS_DIR}/${species}/oriongeno
OUTGTF=${OUTDIR}/oriongeno.gtf

test -e "${CHECKPOINT}"  || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
mkdir -p "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate oriongeno

echo "[$(date -Iseconds)] species=${species} model=${model}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] checkpoint=${CHECKPOINT}"
echo "[$(date -Iseconds)] outgtf=${OUTGTF}"

cd "${ORIONGENO_DIR}"
python main.py \
    --genome "${GENOME}" \
    --output "${OUTGTF}" \
    --checkpoint "${CHECKPOINT}" \
    --length 512000 \
    --flank 64000 \
    --batch-size 8 \
    --output-gene True \
    --output-repeat False \
    --species-name "${species}"
