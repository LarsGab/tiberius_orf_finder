#!/bin/bash
#SBATCH --job-name=tiberius_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/tiberius_test_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/tiberius_test_%j.err

set -euo pipefail

mkdir -p /home/gabriell/tiberius_orf_finder/logs

cd /home/gabriell/tiberius_orf_finder

export PATH="/home/gabriell/programs:$PATH"
export NXF_HOME=/home/gabriell/.nextflow

export NXF_CACHE_DIR=/home/gabriell/tiberius_orf_finder/.nextflow_test

/home/gabriell/programs/nextflow run nextflow/main.nf \
    -c nextflow/conf/brain.config \
    -work-dir /home/gabriell/tiberius_orf_finder/work/test \
    --species_csv nextflow/conf/species_test.csv \
    --varus_dir /home/gabriell/programs/VARUS \
    --varus_impl /home/gabriell/programs/VARUS/Implementation \
    --hisat_dir /home/gabriell/programs/hisat2-2.2.1 \
    --braker_data_dir /home/nas-hs/projs/tiberius-insects/data/insects_data_braker \
    --outdir /home/gabriell/tiberius_orf_finder/results/test \
    -resume
