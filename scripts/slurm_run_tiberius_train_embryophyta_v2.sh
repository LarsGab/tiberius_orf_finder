#!/bin/bash
# Run Tiberius ab initio gene prediction for all embryophyta training species.
#
# Output per species:
#   results/training_embryophyta_v2/<sp>/tiberius/tiberius_seqlen.gtf
#
# Downstream:
#   slurm_gffcompare_tiberius_training_embryophyta_v2.sh
#   slurm_features_tiberius_training_embryophyta_v2.sh
#
#SBATCH --job-name=tib_train_emb
#SBATCH --partition=vision,vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=85G
#SBATCH --time=12:00:00
#SBATCH --array=0-44
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_train_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_train_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2
TIBERIUS=/home/gabriell/Tiberius/tiberius.py
TIBERIUS_CFG=angiosperms

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${TRAINDIR}/${species}/assembly/genome.fa"
OUTDIR="${TRAINDIR}/${species}/tiberius"
OUT_GTF="${OUTDIR}/tiberius_seqlen.gtf"

echo "[$(date -Iseconds)] species=${species}"

if [[ -s "${OUT_GTF}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: tiberius_seqlen.gtf already exists"; exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

[[ -s "${GENOME}" ]]  || { echo "SKIP ${species}: missing genome.fa" >&2; exit 0; }
[[ -f "${TIBERIUS}" ]] || { echo "ERROR: ${TIBERIUS} not found" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

mkdir -p "${OUTDIR}"

echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] out=${OUT_GTF}"

python "${TIBERIUS}" \
    --genome    "${GENOME}" \
    --model_cfg "${TIBERIUS_CFG}" \
    --out       "${OUT_GTF}"

[[ -s "${OUT_GTF}" ]] || { echo "ERROR: Tiberius produced no GTF" >&2; exit 2; }
N_TX=$(awk -F'\t' '$3=="transcript"' "${OUT_GTF}" | wc -l || true)
echo "[$(date -Iseconds)] done -> ${OUT_GTF}  (${N_TX} transcripts)"
