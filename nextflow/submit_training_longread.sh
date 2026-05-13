#!/bin/bash
#SBATCH --job-name=tiberius_training_lr
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/tiberius_training_lr_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/tiberius_training_lr_%j.err

set -euo pipefail

mkdir -p /home/gabriell/tiberius_orf_finder/logs

cd /home/gabriell/tiberius_orf_finder

export PATH="/home/gabriell/programs:$PATH"
export NXF_HOME=/home/gabriell/.nextflow_training_lr
export NXF_CACHE_DIR=/home/gabriell/tiberius_orf_finder/.nextflow_training_lr

/home/gabriell/programs/nextflow run nextflow/main_longread.nf \
    -c nextflow/conf/brain_longread.config \
    -work-dir /home/gabriell/tiberius_orf_finder/work/training_lr \
    --species_csv nextflow/conf/species_training_longread.csv \
    --reuse_assembly_dir /home/gabriell/tiberius_orf_finder/results/training \
    --outdir /home/gabriell/tiberius_orf_finder/results/training_longread \
    -resume
