#!/bin/bash
# Nextflow mother for the reuse-BAM half of the vertebrates post-crash
# recovery: re-fetches assembly + reuses /home_old VARUS.bam -> StringTie
# -> labels (with updated partial-ref categories) -> TFRecord.
#
# Pair script: scripts/slurm_nextflow_vertebrates_from_scratch.sh runs
# the full pipeline on the species whose BAM couldn't be salvaged.
#
# Pre-req:
#   1. sbatch scripts/slurm_check_varus_assembly_integrity.sh
#   2. python scripts/split_species_by_bam_status.py \
#        --integrity-tsv <step1 output> \
#        --species-csv   nextflow/conf/species_vertebrates_training.csv \
#        --out-dir       results/integrity_checks/split_vertebrates
#
# Usage:  sbatch scripts/slurm_nextflow_vertebrates_reuse_bam.sh
#
#SBATCH --job-name=nf_reuse_bam_vert
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gabriell/tiberius_orf_finder/logs/nf_reuse_bam_vert_%j.out
#SBATCH --error=/home/gabriell/tiberius_orf_finder/logs/nf_reuse_bam_vert_%j.err

set -euo pipefail

PROJDIR=/home/gabriell/tiberius_orf_finder

# Inputs / outputs --------------------------------------------------
SPECIES_CSV=${PROJDIR}/results/integrity_checks/split_vertebrates/reuse_bam.csv

# >>> CONFIRM: read-only pre-crash results dir under /home_old <<<
OLD_OUTDIR=/home_old/gabriell/tiberius_orf_finder/results_vertebrates/training/

# Fresh writable run dir. Use _v2 (or a date) to keep the salvaged run
# distinct from any future re-run.
OUTDIR=${PROJDIR}/results/training_vertebrates_v2

NF_CONF=${PROJDIR}/nextflow/conf/brain_shortread_v2.config
NF_MAIN=${PROJDIR}/nextflow/main_reuse_bam.nf

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"
RUNDIR=${OUTDIR}/_nf_run
mkdir -p "${RUNDIR}"

# Guards ------------------------------------------------------------
test -r "${SPECIES_CSV}"            || { echo "missing reuse_bam.csv: ${SPECIES_CSV}" >&2; exit 2; }
test -r "${OLD_OUTDIR}"             || { echo "OLD_OUTDIR not readable: ${OLD_OUTDIR}" >&2; exit 2; }
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
echo "[$(date -Iseconds)] old_outdir=${OLD_OUTDIR}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"
echo "[$(date -Iseconds)] n_species=$(($(wc -l < "${SPECIES_CSV}") - 1))"
nextflow -v

nextflow run "${NF_MAIN}" \
    -c "${NF_CONF}" \
    --species_csv "${SPECIES_CSV}" \
    --old_outdir  "${OLD_OUTDIR}" \
    --outdir      "${OUTDIR}" \
    -work-dir     "${RUNDIR}/work" \
    -with-report  "${RUNDIR}/report.html" \
    -with-trace   "${RUNDIR}/trace.txt" \
    -with-timeline "${RUNDIR}/timeline.html" \
    -resume

echo "[$(date -Iseconds)] done. manifest=${OUTDIR}/tfrecord_manifest.tsv"
