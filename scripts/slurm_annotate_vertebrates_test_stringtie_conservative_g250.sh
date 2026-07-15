#!/bin/bash
# Experiment: rerun StringTie on the vertebrates_test BAMs with the
# built-in --conservative preset combined with -g 250, then annotate
# with epoch_11.
#
# --conservative is StringTie's high-precision preset:
#     -t         (disable end-trimming; more of the read-supported edges kept)
#     -c 1.5     (lower single-tx coverage floor)
#     -f 0.05    (raise minimum isoform fraction; drops minor isoforms
#                 that are typically partial/fragmentary)
# It tightens what StringTie is willing to call a transcript. Combining
# with -g 250 keeps the fragment-merge benefit from the earlier g250
# experiment. Both changes target precision: --conservative from the
# assembly-model side, -g from the fragment-merging side.
#
# Output:
#   ${RESULTS_DIR}/<sp>/stringtie_conservative_g250/stringtie.gtf
#   ${RESULTS_DIR}/<sp>/annotate_${WEIGHTS_TAG}_stringtie_conservative_g250/orfs.gtf
#
#SBATCH --job-name=annot_vt_cons
#SBATCH --partition=vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vt_cons_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_vt_cons_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2/epoch_11.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml

WEIGHTS_TAG=$(basename "${WEIGHTS}" .weights.h5)
ST_TAG=stringtie_conservative_g250

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Gallus_gallus"
    "Pristiophorus_japonicus"
    "Bos_taurus"
    "Delphinapterus_leucas"
    "Homo_sapiens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
BAM=${RESULTS_DIR}/${species}/varus/VARUS.bam
ST_DIR=${RESULTS_DIR}/${species}/${ST_TAG}
ST_GTF=${ST_DIR}/stringtie.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}_${ST_TAG}

if [[ ! -s "${BAM}" || ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: bam or genome not ready"
    echo "                       bam=${BAM}"
    echo "                       genome=${GENOME}"
    exit 0
fi

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config: ${CONFIG}"   >&2; exit 2; }
mkdir -p "${ST_DIR}" "${OUTDIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] bam=${BAM}"
echo "[$(date -Iseconds)] stringtie out=${ST_GTF}"

TMP=$(mktemp -d -p "${TMPDIR:-/tmp}" stringtie.XXXXXX)
trap 'rm -rf "${TMP}"' EXIT

SORTED_BAM=${TMP}/sorted.bam
samtools sort -@ "${SLURM_CPUS_PER_TASK}" -o "${SORTED_BAM}" "${BAM}"

stringtie "${SORTED_BAM}" \
    -o "${ST_GTF}" \
    --conservative \
    -g 250 \
    -p "${SLURM_CPUS_PER_TASK}"

test -s "${ST_GTF}" || { echo "stringtie produced no GTF: ${ST_GTF}" >&2; exit 2; }

echo "[$(date -Iseconds)] annotate out=${OUTDIR}/orfs.gtf"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${ST_GTF}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK}"
