#!/bin/bash
# Compute orf_features.tsv and join gffcompare labels for embryophyta training species.
#
# Prerequisite: slurm_gffcompare_training_embryophyta_v2.sh
#
# No miniprot data for training species — protein features will be zero.
#
# Output per species:
#   results/training_embryophyta_v2/<sp>/<annot_tag>/orf_features.tsv
#
#SBATCH --job-name=feat_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-44
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/feat_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/feat_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2
ANNOT_TAG=${ANNOT_TAG:-annotate_run001_e300}

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

ORFS="${TRAINDIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
GENOME="${TRAINDIR}/${species}/assembly/genome.fa"
MINIPROT="${TRAINDIR}/${species}/proteins/miniprot_scored.gff"
HINTS="${TRAINDIR}/${species}/proteins/miniprothint/hc.gff"
TRACKING="${TRAINDIR}/${species}/${ANNOT_TAG}/gffcompare/gffcmp.tracking"
OUT="${TRAINDIR}/${species}/${ANNOT_TAG}/orf_features.tsv"

echo "[$(date -Iseconds)] species=${species}"

for f in "${ORFS}" "${GENOME}" "${MINIPROT}" "${HINTS}" "${TRACKING}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

if [[ -s "${OUT}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: orf_features.tsv already exists"; exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    gunzip -k "${GENOME}.gz"
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] computing features …"
python "${PROJDIR}/scripts/compute_orf_features.py" \
    --orfs-gtf       "${ORFS}" \
    --miniprot-gff   "${MINIPROT}" \
    --hints-gff      "${HINTS}" \
    --genome         "${GENOME}" \
    --out            "${OUT}"

echo "[$(date -Iseconds)] joining gffcompare labels …"
python "${PROJDIR}/scripts/join_tmap_labels.py" \
    --tracking  "${TRACKING}" \
    --features  "${OUT}"

echo "[$(date -Iseconds)] done -> ${OUT}"
wc -l "${OUT}"
