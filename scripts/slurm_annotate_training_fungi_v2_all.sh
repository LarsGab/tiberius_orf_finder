#!/bin/bash
# Annotate all fungi training species for LGB training.
# CPU-only: snowball/pinky, 72 CPUs per job.
#
# Usage: sbatch --array=0-310 scripts/slurm_annotate_training_fungi_v2_all.sh [epoch]
#   epoch defaults to the latest checkpoint in MODEL_DIR.
#
# Output per species:
#   results/training_fungi_v2/<sp>/annotate_run006_e<epoch>/
#     orfs.gtf  orfs.filtered.gtf  dropped_subsequences.tsv
#
#SBATCH --job-name=annot_fung_all
#SBATCH --partition=snowball,pinky
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-310
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_fung_all_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_fung_all_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
MODEL_DIR=${PROJDIR}/results/models/cnn_lstm_run006_fungi
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml

export CUDA_VISIBLE_DEVICES=""

EPOCH="${1:-}"
if [[ -z "${EPOCH}" ]]; then
    EPOCH=$(ls "${MODEL_DIR}"/epoch_*.weights.h5 2>/dev/null \
            | sort -V | tail -1 \
            | xargs -I{} basename {} .weights.h5 \
            | sed 's/epoch_//')
    [[ -n "${EPOCH}" ]] || { echo "no epoch checkpoints in ${MODEL_DIR}" >&2; exit 2; }
fi
WEIGHTS=${MODEL_DIR}/epoch_${EPOCH}.weights.h5
ANNOT_TAG=annotate_run006_e${EPOCH}

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -v '\.')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TRAINDIR}/${species}/assembly/genome.fa
STRINGTIE=${TRAINDIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${TRAINDIR}/${species}/${ANNOT_TAG}

echo "[$(date -Iseconds)] species=${species}  epoch=${EPOCH}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: orfs.filtered.gtf already exists"; exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${STRINGTIE}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

mkdir -p "${OUTDIR}"
cd "${PROJDIR}"

echo "[$(date -Iseconds)] annotating …"
python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    32 \
    --threads       "${SLURM_CPUS_PER_TASK:-72}" \
    --lorf-class

echo "[$(date -Iseconds)] subseq collapse …"
python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
