#!/bin/bash
# Run annotate.py on the 7 fungi test species using the CNN-LSTM
# run006 fungi model (genome + StringTie input).
#
# Usage: sbatch scripts/slurm_annotate_fungi_test_run006.sh [epoch]
#   epoch defaults to the value of EPOCH variable (currently newest on disk).
#
# Inputs per species (from fungi_test pipeline output):
#   assembly/genome.fa
#   stringtie/stringtie.gtf
#
# Outputs per species:
#   results/fungi_test/<sp>/annotate_run006_e<epoch>/
#     orfs.gtf
#     orfs.filtered.gtf
#     dropped_subsequences.tsv
#
#SBATCH --job-name=annot_fun_test
#SBATCH --partition=vision,vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_fun_test_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_fun_test_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/fungi_test
MODEL_DIR=${PROJDIR}/results/models/cnn_lstm_run006_fungi
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml

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
    "Agaricus_bisporus"
    "Aspergillus_fumigatus"
    "Cryphonectria_parasitica"
    "Parastagonospora_nodorum"
    "Puccinia_striiformis"
    "Punctularia_strigosozonata"
    "Tilletiopsis_washingtonensis"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TESTDIR}/${species}/assembly/genome.fa
STRINGTIE=${TESTDIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${TESTDIR}/${species}/annotate_run006_e${EPOCH}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

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
