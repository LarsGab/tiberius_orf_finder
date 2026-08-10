#!/bin/bash
# Run annotate.py on Aspergillus_fumigatus, Candida_parapsilosis, and
# Fusarium_redolens in training_fungi_v2 using the CNN-LSTM run006 full-fungi
# model — CPU-only variant (snowball/pinky, 72 CPUs).
#
# Usage: sbatch scripts/slurm_annotate_training_fungi_v2_run006_cpu.sh [epoch]
#   epoch defaults to the highest epoch_N checkpoint in the model dir.
#
#SBATCH --job-name=annot_tv2_cpu
#SBATCH --partition=snowball,pinky
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_tv2_cpu_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_tv2_cpu_%A_%a.err

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
    [[ -n "${EPOCH}" ]] || { echo "no epoch checkpoints found in ${MODEL_DIR}" >&2; exit 2; }
fi
WEIGHTS=${MODEL_DIR}/epoch_${EPOCH}.weights.h5

declare -a SPECIES=(
    "Aspergillus_fumigatus"
    "Candida_parapsilosis"
    "Fusarium_redolens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TRAINDIR}/${species}/assembly/genome.fa
STRINGTIE=${TRAINDIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${TRAINDIR}/${species}/annotate_run006_e${EPOCH}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${STRINGTIE}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"
echo "[$(date -Iseconds)] host=$(hostname) species=${species} epoch=${EPOCH} cpus=${SLURM_CPUS_PER_TASK}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    32 \
    --threads       "${SLURM_CPUS_PER_TASK:-72}"

python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
