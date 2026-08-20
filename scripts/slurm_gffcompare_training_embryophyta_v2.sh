#!/bin/bash
# Label ORF predictions for embryophyta training species via gffcompare.
#
# Prerequisite: slurm_annotate_training_embryophyta_v2.sh
#
# Reference: assembly/annotation.gff (full GFF3).
# CDS is extracted from annotation.gff to build annot_cds.gff per species.
#
# Output per species:
#   assembly/annot_cds.gff  (created once if absent)
#   <annot_tag>/gffcompare/gffcmp.tracking
#
#SBATCH --job-name=gffcmp_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-44
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gffcmp_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gffcmp_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2
ANNOT_TAG=${ANNOT_TAG:-annotate_run001_e300}

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

ORFS="${TRAINDIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
ANNOT_GFF="${TRAINDIR}/${species}/assembly/annotation.gff"
ANNOT_CDS="${TRAINDIR}/${species}/assembly/annot_cds.gff"
WORK="${TRAINDIR}/${species}/${ANNOT_TAG}/gffcompare"
TRACKING="${WORK}/gffcmp.tracking"

echo "[$(date -Iseconds)] species=${species}"

[[ -s "${ORFS}" ]]      || { echo "SKIP ${species}: missing ${ORFS}"      >&2; exit 0; }
[[ -s "${ANNOT_GFF}" ]] || { echo "SKIP ${species}: missing ${ANNOT_GFF}" >&2; exit 0; }

if [[ -s "${TRACKING}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: tracking already exists"; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Extract CDS reference (idempotent)
if [[ ! -s "${ANNOT_CDS}" ]]; then
    echo "[$(date -Iseconds)] extracting CDS reference …"
    grep -w CDS "${ANNOT_GFF}" > "${ANNOT_CDS}"
    [[ -s "${ANNOT_CDS}" ]] || { echo "ERROR: no CDS in ${ANNOT_GFF}" >&2; exit 2; }
fi

mkdir -p "${WORK}"

CDS_GFF="${WORK}/orfs_cds.gff"
grep -w CDS "${ORFS}" > "${CDS_GFF}"
[[ -s "${CDS_GFF}" ]] || { echo "ERROR: empty CDS extraction from ${ORFS}" >&2; exit 2; }

gffcompare --strict-match -e 3 -T \
    -r "${ANNOT_CDS}" \
    -o "${WORK}/gffcmp" \
    "${CDS_GFF}"

[[ -s "${TRACKING}" ]] || { echo "ERROR: gffcompare produced no tracking file" >&2; exit 2; }
echo "[$(date -Iseconds)] done -> ${TRACKING}"
wc -l "${TRACKING}"
