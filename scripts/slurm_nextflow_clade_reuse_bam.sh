#!/bin/bash
# Generic reuse-BAM Nextflow mother for any clade. Pairs with
# slurm_check_varus_assembly_integrity_clade.sh +
# split_species_by_bam_status.py.
#
# Re-fetches assembly + reuses /home_old VARUS.bam -> StringTie -> labels
# (with current categories) -> TFRecord.
#
# Usage:
#   CLADE=<clade> [SPLIT=training] OLD_OUTDIR=/home_old/... \
#       sbatch scripts/slurm_nextflow_clade_reuse_bam.sh
#
# Example:
#   CLADE=chlorophyta \
#   OLD_OUTDIR=/home_old/gabriell/tiberius_orf_finder/results_chlorophyta/training \
#       sbatch scripts/slurm_nextflow_clade_reuse_bam.sh
#
# Inputs:
#   ${SPLIT_DIR}/reuse_bam.csv   <- from split_species_by_bam_status.py
#   ${OLD_OUTDIR}                <- read-only /home_old root
# Outputs:
#   ${OUTDIR}/<sp>/{assembly,stringtie,labels,tfrecord}/
#   ${OUTDIR}/tfrecord_manifest.tsv
#
#SBATCH --job-name=nf_reuse_bam
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/nf_reuse_bam_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/nf_reuse_bam_%j.err

set -euo pipefail

: "${CLADE:?must set CLADE}"
SPLIT=${SPLIT:-training}
: "${OLD_OUTDIR:?must set OLD_OUTDIR}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

SPLIT_DIR=${PROJDIR}/results/integrity_checks/split_${CLADE}_${SPLIT}
SPECIES_CSV=${SPLIT_DIR}/reuse_bam.csv

OUTDIR=${PROJDIR}/results/training_${CLADE}_v2
RUNDIR=${OUTDIR}/_nf_run

NF_CONF=${PROJDIR}/nextflow/conf/brain_shortread_v2.config
NF_MAIN=${PROJDIR}/nextflow/main_reuse_bam.nf

mkdir -p "${PROJDIR}/logs" "${OUTDIR}" "${RUNDIR}"

# ---- guards ----
test -r "${SPECIES_CSV}"            || { echo "missing reuse_bam.csv: ${SPECIES_CSV}" >&2; exit 2; }
test -r "${OLD_OUTDIR}"             || { echo "OLD_OUTDIR not readable: ${OLD_OUTDIR}" >&2; exit 2; }
test -r "${NF_CONF}" -a -r "${NF_MAIN}" || { echo "missing nextflow config/main" >&2; exit 2; }
case "${OUTDIR}" in
    /home_old/*) echo "refuse: OUTDIR under /home_old is read-only" >&2; exit 2 ;;
esac

# ---- env ----
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder
export NXF_OPTS='-Xms1g -Xmx4g'

cd "${RUNDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[$(date -Iseconds)] clade=${CLADE} split=${SPLIT}"
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
