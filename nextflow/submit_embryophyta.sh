#!/bin/bash
# Submit short-read (VARUS v2) training-data generation for Embryophyta
# (Mesangiospermae), using the Figshare-staged bundle on brain.
# Usage:  sbatch nextflow/submit_embryophyta.sh {training|validation|test}
#
# Prereq: run scripts/stage_mesangiospermae_figshare.py once on brain to
# populate /home/gabriell/tiberius_mesangiospermae_training_data/by_species/.
#
#SBATCH --job-name=tib_emb_dataprep
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=168:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/tib_embryophyta_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/tib_embryophyta_%j.err

set -euo pipefail
SPLIT="${1:?usage: $0 {training|validation|test}}"
case "$SPLIT" in training|validation|test) ;; *) echo "bad split: $SPLIT" >&2; exit 2;; esac

mkdir -p /home/gabriell/tiberius_orf_finder/logs
cd /home/gabriell/tiberius_orf_finder

export PATH="/home/gabriell/programs:$PATH"
export NXF_HOME=/home/gabriell/.nextflow

# brain_shortread_v2.config defaults --phytozome_data_dir at the Figshare
# by_species/ layout; override here only if you stage elsewhere.
/home/gabriell/programs/nextflow run nextflow/main_shortread_v2.nf \
    -c nextflow/conf/brain_shortread_v2.config \
    --species_csv nextflow/conf/embryophyta/species_${SPLIT}.csv \
    --outdir /home/gabriell/tiberius_orf_finder/results/embryophyta_${SPLIT} \
    -resume
