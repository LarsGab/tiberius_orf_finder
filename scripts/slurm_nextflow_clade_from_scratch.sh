#!/bin/bash
# Generic from-scratch Nextflow mother for any clade. Pairs with
# slurm_check_varus_assembly_integrity_clade.sh +
# split_species_by_bam_status.py.
#
# Re-runs the full short-read v2 pipeline (assembly fetch + VARUS v2 +
# StringTie + label + TFRecord) for species whose pre-crash VARUS.bam
# couldn't be salvaged.
#
# Usage:
#   CLADE=<clade> [SPLIT=training] \
#       sbatch scripts/slurm_nextflow_clade_from_scratch.sh
#
# Example:
#   CLADE=diatoms \
#       sbatch scripts/slurm_nextflow_clade_from_scratch.sh
#
# Inputs:
#   ${SPLIT_DIR}/from_scratch.csv  <- from split_species_by_bam_status.py
# Outputs:
#   ${OUTDIR}/<sp>/{assembly,varus,stringtie,labels,tfrecord}/
#   ${OUTDIR}/tfrecord_manifest.tsv  (shared with reuse-BAM half)
#
#SBATCH --job-name=nf_scratch
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/nf_scratch_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/nf_scratch_%j.err

set -euo pipefail

: "${CLADE:?must set CLADE}"
SPLIT=${SPLIT:-training}

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

SPLIT_DIR=${PROJDIR}/results/integrity_checks/split_${CLADE}_${SPLIT}
SPECIES_CSV=${SPLIT_DIR}/from_scratch.csv

OUTDIR=${PROJDIR}/results/training_${CLADE}_v2
RUNDIR=${OUTDIR}/_nf_run_scratch

NF_CONF=${PROJDIR}/nextflow/conf/brain_shortread_v2.config
NF_MAIN=${PROJDIR}/nextflow/main_shortread_v2.nf

mkdir -p "${PROJDIR}/logs" "${OUTDIR}" "${RUNDIR}"

# ---- guards ----
test -r "${SPECIES_CSV}"            || { echo "missing from_scratch.csv: ${SPECIES_CSV}" >&2; exit 2; }
test -r "${NF_CONF}" -a -r "${NF_MAIN}" || { echo "missing nextflow config/main" >&2; exit 2; }
case "${OUTDIR}" in
    /home_old/*) echo "refuse: OUTDIR under /home_old is read-only" >&2; exit 2 ;;
esac

# ---- env ----
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder
export NXF_OPTS='-Xms1g -Xmx4g'

# ---- per-clade staging dir overrides ----
# fetch.nf needs --braker_data_dir for diatoms (BRAKER annotation) and
# --phytozome_data_dir for embryophyta (Phytozome annotation). Defaults
# in brain_shortread_v2.config point at paths that no longer exist
# post-migration; override here so each clade gets the right one.
EXTRA_NF_ARGS=()
case "${CLADE}" in
    diatoms)
        EXTRA_NF_ARGS+=(--braker_data_dir /home_old/gabriell/tiberius_diatoms_staged/by_species) ;;
    embryophyta)
        EXTRA_NF_ARGS+=(--phytozome_data_dir /home_old/gabriell/tiberius_mesangiospermae_training_data/by_species) ;;
esac

cd "${RUNDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[$(date -Iseconds)] clade=${CLADE} split=${SPLIT}"
echo "[$(date -Iseconds)] species_csv=${SPECIES_CSV}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"
echo "[$(date -Iseconds)] extra_nf_args=${EXTRA_NF_ARGS[*]:-<none>}"
echo "[$(date -Iseconds)] n_species=$(($(wc -l < "${SPECIES_CSV}") - 1))"
nextflow -v

nextflow run "${NF_MAIN}" \
    -c "${NF_CONF}" \
    --species_csv "${SPECIES_CSV}" \
    --outdir      "${OUTDIR}" \
    "${EXTRA_NF_ARGS[@]}" \
    -work-dir     "${RUNDIR}/work" \
    -with-report  "${RUNDIR}/report.html" \
    -with-trace   "${RUNDIR}/trace.txt" \
    -with-timeline "${RUNDIR}/timeline.html" \
    -resume

echo "[$(date -Iseconds)] done. manifest=${OUTDIR}/tfrecord_manifest.tsv"
