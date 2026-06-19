#!/bin/bash
# Generic post-crash integrity check for any clade's pre-crash training
# data (VARUS.bam + assembly) sitting read-only under /home_old.
#
# Drop-in replacement for slurm_check_varus_assembly_integrity.sh (which
# is hardcoded to vertebrates). Parameterised by CLADE + SPLIT.
#
# Usage:
#   CLADE=<clade> [SPLIT=training] OLD_OUTDIR=/home_old/... \
#       sbatch scripts/slurm_check_varus_assembly_integrity_clade.sh
#
# Examples:
#   CLADE=chlorophyta \
#   OLD_OUTDIR=/home_old/gabriell/tiberius_orf_finder/results_chlorophyta/training \
#       sbatch scripts/slurm_check_varus_assembly_integrity_clade.sh
#
#   CLADE=fungi SPLIT=test \
#   OLD_OUTDIR=/home_old/gabriell/tiberius_orf_finder/results_fungi/test \
#       sbatch scripts/slurm_check_varus_assembly_integrity_clade.sh
#
# Outputs:
#   ${PROJDIR}/results/integrity_checks/<CLADE>_<SPLIT>_<TS>.tsv
#
#SBATCH --job-name=varus_integrity
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/varus_integrity_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/varus_integrity_%j.err

set -euo pipefail

# ---- inputs (env-driven) ----
: "${CLADE:?must set CLADE (chlorophyta | diatoms | fungi | embryophyta | vertebrates)}"
SPLIT=${SPLIT:-training}
: "${OLD_OUTDIR:?must set OLD_OUTDIR (read-only /home_old path with <Genus_species>/{assembly,varus} subdirs)}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

SPECIES_CSV=${PROJDIR}/nextflow/conf/${CLADE}/species_${SPLIT}.csv

REPORT_DIR=${PROJDIR}/results/integrity_checks
TS=$(date +%Y%m%d_%H%M%S)
OUT_TSV=${REPORT_DIR}/${CLADE}_${SPLIT}_${TS}.tsv

mkdir -p "${PROJDIR}/logs" "${REPORT_DIR}"

# ---- guards ----
test -d "${OLD_OUTDIR}" || { echo "OLD_OUTDIR not a dir: ${OLD_OUTDIR}" >&2; exit 2; }
test -r "${OLD_OUTDIR}" || { echo "OLD_OUTDIR not readable: ${OLD_OUTDIR}" >&2; exit 2; }
test -r "${SPECIES_CSV}" || { echo "SPECIES_CSV not readable: ${SPECIES_CSV}" >&2; exit 2; }
case "${OUT_TSV}" in
    /home_old/*) echo "refuse: OUT_TSV under /home_old is read-only" >&2; exit 2 ;;
esac

# ---- env ----
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} partition=${SLURM_JOB_PARTITION:-?}"
echo "[$(date -Iseconds)] clade=${CLADE} split=${SPLIT}"
echo "[$(date -Iseconds)] old_outdir=${OLD_OUTDIR}"
echo "[$(date -Iseconds)] species_csv=${SPECIES_CSV}"
echo "[$(date -Iseconds)] out_tsv=${OUT_TSV}"

python scripts/check_varus_assembly_integrity.py \
    --outdir      "${OLD_OUTDIR}" \
    --species-csv "${SPECIES_CSV}" \
    --out-tsv     "${OUT_TSV}"
rc=$?

echo "[$(date -Iseconds)] check rc=${rc} report=${OUT_TSV}"
exit ${rc}
