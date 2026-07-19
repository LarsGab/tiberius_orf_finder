#!/bin/bash
# Run annotate.py on the 7 embryophyta test species using the CNN-LSTM
# run001 embryophyta model checkpoint.
#
# Usage: sbatch --array=0-6 scripts/slurm_annotate_embryophyta_test_run001.sh [epoch]
#   epoch defaults to 39.
#
# Inputs per species (from the training_embryophyta_test_v2 pipeline output):
#   assembly/genome.fa(.gz)
#   stringtie/stringtie.gtf
#
# Outputs per species:
#   results/training_embryophyta_test_v2/<sp>/annotate_run001_e<epoch>/
#     orfs.gtf
#     orfs.filtered.gtf
#     dropped_subsequences.tsv
#
#SBATCH --job-name=annot_emb_test
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_test_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_test_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2

EPOCH="${1:-39}"
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2/epoch_${EPOCH}.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Brachypodium_distachyon"
    "Eschscholzia_californica"
    "Freycinetia_multiflora"
    "Medicago_truncatula"
    "Mimulus_guttatus"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TESTDIR}/${species}/assembly/genome.fa
STRINGTIE=${TESTDIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${TESTDIR}/${species}/annotate_run001_e${EPOCH}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Resume-skip.
if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

# Decompress genome if only .gz is present.
if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${STRINGTIE}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID}"
echo "[$(date -Iseconds)] species=${species}  epoch=${EPOCH}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] stringtie=${STRINGTIE}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK:-4}"

echo "[$(date -Iseconds)] postprocess: subseq collapse"

python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
