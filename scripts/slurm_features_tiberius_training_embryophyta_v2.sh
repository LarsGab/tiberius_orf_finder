#!/bin/bash
# Compute features + gffcompare labels for Tiberius ab initio predictions on
# embryophyta training species.
#
# Prerequisite: slurm_gffcompare_tiberius_training_embryophyta_v2.sh
#
# Output per species:
#   tiberius/tib_features.tsv
#
#SBATCH --job-name=feat_tib_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-44
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/feat_tib_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/feat_tib_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

TIB_GTF="${TRAINDIR}/${species}/tiberius/tiberius_seqlen.gtf"
GENOME="${TRAINDIR}/${species}/assembly/genome.fa"
MINIPROT="${TRAINDIR}/${species}/proteins/miniprot_scored.gff"
HINTS="${TRAINDIR}/${species}/proteins/miniprothint/hc.gff"
PROT_FA="${TRAINDIR}/${species}/proteins/protein_top4.fa"
TRACKING="${TRAINDIR}/${species}/tiberius/gffcompare/gffcmp.tracking"
OUT="${TRAINDIR}/${species}/tiberius/tib_features.tsv"

echo "[$(date -Iseconds)] species=${species}"

for f in "${TIB_GTF}" "${GENOME}" "${MINIPROT}" "${HINTS}" "${PROT_FA}" "${TRACKING}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

if [[ -s "${OUT}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: tib_features.tsv already exists"; exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    gunzip -k "${GENOME}.gz"
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] computing features …"
python "${PROJDIR}/scripts/compute_orf_features.py" \
    --orfs-gtf        "${TIB_GTF}" \
    --miniprot-gff    "${MINIPROT}" \
    --hints-gff       "${HINTS}" \
    --genome          "${GENOME}" \
    --proteins-fasta  "${PROT_FA}" \
    --out             "${OUT}"

echo "[$(date -Iseconds)] joining gffcompare labels …"
python "${PROJDIR}/scripts/join_tmap_labels.py" \
    --tracking  "${TRACKING}" \
    --features  "${OUT}"

echo "[$(date -Iseconds)] done -> ${OUT}"
wc -l "${OUT}"
