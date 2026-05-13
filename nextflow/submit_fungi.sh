#!/bin/bash
# Submit short-read (VARUS v2) training-data generation for Fungi.
# Usage:  sbatch nextflow/submit_fungi.sh {training|validation|test}
#
# Fungi has 306 training species – set executor.queueSize / scheduler limits
# on `brain` appropriately if you fan out further than the default queueSize=50.
#
#SBATCH --job-name=tib_fun_dataprep
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/tib_fungi_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/tib_fungi_%j.err

set -euo pipefail
SPLIT="${1:-}"
case "$SPLIT" in
    training|validation|test) ;;
    *) echo "usage: $0 training|validation|test" >&2; exit 2 ;;
esac

PROJDIR=/home/gabriell/tiberius_orf_finder
RUNDIR=${PROJDIR}/runs/fungi_${SPLIT}
mkdir -p "${PROJDIR}/logs" "${RUNDIR}"
cd "${RUNDIR}"   # isolated work/ + .nextflow/ per (clade,split) – avoids LOCK contention

export PATH="/home/gabriell/programs:$PATH"
export NXF_HOME=/home/gabriell/.nextflow

/home/gabriell/programs/nextflow run ${PROJDIR}/nextflow/main_shortread_v2.nf \
    -c ${PROJDIR}/nextflow/conf/brain_shortread_v2.config \
    --species_csv ${PROJDIR}/nextflow/conf/fungi/species_${SPLIT}.csv \
    --outdir ${PROJDIR}/results/fungi_${SPLIT} \
    -resume
