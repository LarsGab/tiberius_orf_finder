#!/bin/bash
# Nextflow mother for the from-scratch half of the vertebrates post-crash
# recovery: re-runs the full short-read v2 pipeline (assembly fetch +
# VARUS v2 + StringTie + label + TFRecord) for species whose pre-crash
# VARUS.bam couldn't be salvaged.
#
# Pair script: scripts/slurm_nextflow_vertebrates_reuse_bam.sh handles
# the species whose BAM survived intact under /home_old.
#
# Pre-req: scripts/split_species_by_bam_status.py has produced
#   results/integrity_checks/split_vertebrates/from_scratch.csv
#
# Usage:  sbatch scripts/slurm_nextflow_vertebrates_from_scratch.sh
#
#SBATCH --job-name=nf_scratch_vert
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/nf_scratch_vert_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/nf_scratch_vert_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

SPECIES_CSV=${PROJDIR}/results/integrity_checks/split_vertebrates/from_scratch.csv

OUTDIR=${PROJDIR}/results/training_vertebrates_v2

NF_CONF=${PROJDIR}/nextflow/conf/brain_shortread_v2.config
NF_MAIN=${PROJDIR}/nextflow/main_shortread_v2.nf

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"
RUNDIR=${OUTDIR}/_nf_run_scratch
mkdir -p "${RUNDIR}"

# Guards ------------------------------------------------------------
test -r "${SPECIES_CSV}"            || { echo "missing from_scratch.csv: ${SPECIES_CSV}" >&2; exit 2; }
test -r "${NF_CONF}" -a -r "${NF_MAIN}" || { echo "missing nextflow config/main" >&2; exit 2; }
case "${OUTDIR}" in
    /home_old/*) echo "refuse: OUTDIR under /home_old is read-only: ${OUTDIR}" >&2; exit 2 ;;
esac

# Env ---------------------------------------------------------------
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder
export NXF_OPTS='-Xms1g -Xmx4g'

cd "${RUNDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[$(date -Iseconds)] species_csv=${SPECIES_CSV}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"
echo "[$(date -Iseconds)] n_species=$(($(wc -l < "${SPECIES_CSV}") - 1))"
nextflow -v

nextflow run "${NF_MAIN}" \
    -c "${NF_CONF}" \
    --species_csv "${SPECIES_CSV}" \
    --outdir      "${OUTDIR}" \
    -work-dir     "${RUNDIR}/work" \
    -with-report  "${RUNDIR}/report.html" \
    -with-trace   "${RUNDIR}/trace.txt" \
    -with-timeline "${RUNDIR}/timeline.html" \
    -resume

echo "[$(date -Iseconds)] done. manifest=${OUTDIR}/tfrecord_manifest.tsv"
