#!/bin/bash
# Submit short-read (VARUS v2) training-data generation for Diatoms.
# Usage:  sbatch nextflow/submit_diatoms.sh {training|validation|test}
#
# Prereq: run scripts/stage_diatoms_braker.py once on brain to populate
# /home/gabriell/tiberius_diatoms_staged/by_species/<Genus_species>/{genome.fa,annotation.gff}
#
#SBATCH --job-name=tib_dia_dataprep
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/tib_diatoms_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/tib_diatoms_%j.err

set -euo pipefail
SPLIT="${1:?usage: $0 {training|validation|test}}"
case "$SPLIT" in training|validation|test) ;; *) echo "bad split: $SPLIT" >&2; exit 2;; esac

mkdir -p /home/gabriell/tiberius_orf_finder/logs
cd /home/gabriell/tiberius_orf_finder

export PATH="/home/gabriell/programs:$PATH"
export NXF_HOME=/home/gabriell/.nextflow

/home/gabriell/programs/nextflow run nextflow/main_shortread_v2.nf \
    -c nextflow/conf/brain_shortread_v2.config \
    --species_csv nextflow/conf/diatoms/species_${SPLIT}.csv \
    --braker_data_dir /home/gabriell/tiberius_diatoms_staged/by_species \
    --outdir /home/gabriell/tiberius_orf_finder/results/diatoms_${SPLIT} \
    -resume
