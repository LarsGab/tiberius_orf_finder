#!/bin/bash
# Label ORF predictions for fungi training species via gffcompare.
#
# Prerequisite: slurm_annotate_training_fungi_v2_all.sh
#
# For each species:
#   grep -w CDS orfs.filtered.gtf | gffcompare -r annot_cds.gff -T → .tracking
#
# Output per species:
#   results/training_fungi_v2/<sp>/<annot_tag>/gffcompare/gffcmp.tracking
#
#SBATCH --job-name=gffcmp_fung
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-310
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gffcmp_fung_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gffcmp_fung_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
ANNOT_TAG=${ANNOT_TAG:-annotate_run006_e300}

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -v '\.')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

ORFS="${TRAINDIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
REF="${TRAINDIR}/${species}/assembly/annot_cds.gff"
WORK="${TRAINDIR}/${species}/${ANNOT_TAG}/gffcompare"
TRACKING="${WORK}/gffcmp.tracking"

echo "[$(date -Iseconds)] species=${species}"

[[ -s "${ORFS}" ]] || { echo "SKIP ${species}: missing ${ORFS}" >&2; exit 0; }
[[ -s "${REF}"  ]] || { echo "SKIP ${species}: missing ${REF}"  >&2; exit 0; }

if [[ -s "${TRACKING}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: tracking already exists"; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

mkdir -p "${WORK}"

CDS_GFF="${WORK}/orfs_cds.gff"
grep -w CDS "${ORFS}" > "${CDS_GFF}"
[[ -s "${CDS_GFF}" ]] || { echo "ERROR: empty CDS extraction from ${ORFS}" >&2; exit 2; }

gffcompare --strict-match -e 3 -T \
    -r "${REF}" \
    -o "${WORK}/gffcmp" \
    "${CDS_GFF}"

[[ -s "${TRACKING}" ]] || { echo "ERROR: gffcompare produced no tracking file" >&2; exit 2; }
echo "[$(date -Iseconds)] done -> ${TRACKING}"
wc -l "${TRACKING}"
