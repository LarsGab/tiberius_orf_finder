#!/bin/bash
#SBATCH --job-name=annot_r009
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_r009_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_r009_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run009/best.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml
TAG=run009_best_${FILT_TAG}

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Bos_taurus
    Delphinapterus_leucas
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${TAG}

mkdir -p "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" ]]; then echo "SKIP ${species}"; exit 0; fi

[[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]] && gunzip -k "${GENOME}.gz"

for f in "${GENOME}" "${STRINGTIE}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"
python scripts/annotate.py \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       ${SLURM_CPUS_PER_TASK:-4}

python scripts/filter_subsequence_predictions.py \
    --orfs-gtf    "${OUTDIR}/orfs.gtf" \
    --out-gtf     "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv  "${OUTDIR}/dropped_subsequences.tsv"

echo "done -> ${OUTDIR}/orfs.filtered.gtf"
