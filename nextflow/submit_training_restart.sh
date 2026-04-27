#!/bin/bash
#SBATCH --job-name=nf_training_restart
#SBATCH --time=72:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/training_restart_%j.log

eval "$(micromamba shell hook --shell bash)" && micromamba activate storm

cd /home/gabriell/tiberius_orf_finder

nextflow run nextflow/main.nf \
  -c nextflow/conf/brain.config \
  --species_csv nextflow/conf/species_training_restart.csv \
  --outdir /home/gabriell/tiberius_orf_finder/results/training \
  --braker_data_dir /home/nas-hs/projs/tiberius-insects/data/insects_data_braker \
  --varus_dir /home/gabriell/programs/VARUS \
  --varus_impl /home/gabriell/programs/VARUS/Implementation \
  --hisat_dir /home/gabriell/programs/hisat2-2.2.1
