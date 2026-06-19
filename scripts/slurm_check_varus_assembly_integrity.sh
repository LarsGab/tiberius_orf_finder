#!/bin/bash
# Post-crash integrity check for the previously generated VARUS.bam +
# assembly files under /home_old/gabriell/...  (read-only after the
# brain crash). Runs scripts/check_varus_assembly_integrity.py.
#
# /home_old is read-only for this user, so all outputs (TSV report and
# SLURM logs) are written under $HOME instead.
#
# Usage:  sbatch scripts/slurm_check_varus_assembly_integrity.sh
#
#SBATCH --job-name=varus_integrity
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/varus_integrity_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/varus_integrity_%j.err

set -euo pipefail

# ---------------------------------------------------------------- paths
# Code + outputs live under /projects/AI-GUSTUS (writable). Data being
# checked is strictly read-only under /home_old.

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

# >>> EDIT THIS to point at the actual old training results dir <<<
# Expected layout under OLD_OUTDIR:
#   <OLD_OUTDIR>/<Genus_species>/assembly/genome.fa
#   <OLD_OUTDIR>/<Genus_species>/assembly/annotation.gff
#   <OLD_OUTDIR>/<Genus_species>/varus/VARUS.bam
OLD_OUTDIR=/home_old/gabriell/tiberius_orf_finder/results_vertebrates/training/

# Restrict the check to the species we plan to retrain on. Leave empty
# to scan every subdirectory of OLD_OUTDIR.
SPECIES_CSV=${PROJDIR}/nextflow/conf/species_vertebrates_training.csv

# Writable output location.
REPORT_DIR=${PROJDIR}/results/integrity_checks
TS=$(date +%Y%m%d_%H%M%S)
OUT_TSV=${REPORT_DIR}/vertebrates_${TS}.tsv

mkdir -p "${PROJDIR}/logs" "${REPORT_DIR}"

# ---------------------------------------------------------------- guards
test -d "${OLD_OUTDIR}" || { echo "OLD_OUTDIR not a dir: ${OLD_OUTDIR}" >&2; exit 2; }
test -r "${OLD_OUTDIR}" || { echo "OLD_OUTDIR not readable: ${OLD_OUTDIR}" >&2; exit 2; }
if [[ -n "${SPECIES_CSV}" ]]; then
    test -r "${SPECIES_CSV}" || { echo "SPECIES_CSV not readable: ${SPECIES_CSV}" >&2; exit 2; }
fi
# Sanity: refuse to write anywhere under /home_old.
case "${OUT_TSV}" in
    /home_old/*) echo "refuse: OUT_TSV under /home_old is read-only: ${OUT_TSV}" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- env
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] old_outdir=${OLD_OUTDIR}"
echo "[$(date -Iseconds)] species_csv=${SPECIES_CSV:-<all subdirs>}"
echo "[$(date -Iseconds)] out_tsv=${OUT_TSV}"
echo "[$(date -Iseconds)] samtools=$(command -v samtools)"
samtools --version | head -n 1

CMD=(python scripts/check_varus_assembly_integrity.py
     --outdir  "${OLD_OUTDIR}"
     --out-tsv "${OUT_TSV}")
if [[ -n "${SPECIES_CSV}" ]]; then
    CMD+=(--species-csv "${SPECIES_CSV}")
fi

"${CMD[@]}"
rc=$?

echo "[$(date -Iseconds)] check rc=${rc}  report=${OUT_TSV}"
exit ${rc}
