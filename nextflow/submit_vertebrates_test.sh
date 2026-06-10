#!/bin/bash
# Submit short-read (VARUS v2) data generation for the vertebrate TEST split.
# Usage:  sbatch nextflow/submit_vertebrates_test.sh
#
# Species list: nextflow/conf/species_vertebrates_test.csv
#
#SBATCH --job-name=tib_vert_test_dataprep
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/tib_vertebrates_test_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/tib_vertebrates_test_%j.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder
RUNDIR=${PROJDIR}/runs/vertebrates_test
mkdir -p "${PROJDIR}/logs" "${RUNDIR}"
cd "${RUNDIR}"   # isolated work/ + .nextflow/ per (clade,split) – avoids LOCK contention

export PATH="/home/gabriell/programs:$PATH"
export NXF_HOME=/home/gabriell/.nextflow

/home/gabriell/programs/nextflow run ${PROJDIR}/nextflow/main_shortread_v2.nf \
    -c ${PROJDIR}/nextflow/conf/brain_shortread_v2.config \
    --species_csv ${PROJDIR}/nextflow/conf/species_vertebrates_test.csv \
    --outdir ${PROJDIR}/results/vertebrates_test \
    --varus_profit_condition true \
    -resume
