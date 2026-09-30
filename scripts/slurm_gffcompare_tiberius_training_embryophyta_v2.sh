#!/bin/bash
# Label Tiberius ab initio predictions for embryophyta training species via gffcompare.
#
# Prerequisite: slurm_run_tiberius_train_embryophyta_v2.sh
#
# Reference: assembly/annot_cds.gff (created by slurm_gffcompare_training_embryophyta_v2.sh)
#
# Output per species:
#   tiberius/gffcompare/gffcmp.tracking
#
#SBATCH --job-name=gffcmp_tib_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-44
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gffcmp_tib_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gffcmp_tib_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

TIB_GTF="${TRAINDIR}/${species}/tiberius/tiberius_seqlen.gtf"
ANNOT_GFF="${TRAINDIR}/${species}/assembly/annotation.gff"
ANNOT_CDS="${TRAINDIR}/${species}/assembly/annot_cds.gff"
WORK="${TRAINDIR}/${species}/tiberius/gffcompare"
TRACKING="${WORK}/gffcmp.tracking"

echo "[$(date -Iseconds)] species=${species}"

[[ -s "${TIB_GTF}" ]]   || { echo "SKIP ${species}: missing ${TIB_GTF}"   >&2; exit 0; }
[[ -s "${ANNOT_GFF}" ]] || { echo "SKIP ${species}: missing ${ANNOT_GFF}" >&2; exit 0; }

if [[ -s "${TRACKING}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: tracking already exists"; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

if [[ ! -s "${ANNOT_CDS}" ]]; then
    echo "[$(date -Iseconds)] extracting CDS reference …"
    grep -w CDS "${ANNOT_GFF}" > "${ANNOT_CDS}"
    [[ -s "${ANNOT_CDS}" ]] || { echo "ERROR: no CDS in ${ANNOT_GFF}" >&2; exit 2; }
fi

mkdir -p "${WORK}"

CDS_GFF="${WORK}/tib_cds.gff"
grep -w CDS "${TIB_GTF}" > "${CDS_GFF}"
[[ -s "${CDS_GFF}" ]] || { echo "ERROR: empty CDS extraction from ${TIB_GTF}" >&2; exit 2; }

gffcompare --strict-match -e 3 -T \
    -r "${ANNOT_CDS}" \
    -o "${WORK}/gffcmp" \
    "${CDS_GFF}"

[[ -s "${TRACKING}" ]] || { echo "ERROR: gffcompare produced no tracking file" >&2; exit 2; }
echo "[$(date -Iseconds)] done -> ${TRACKING}"
wc -l "${TRACKING}"
