#!/bin/bash
# Run annotate.py (epoch_74, tpm1cov3len300 filter) on the long-read
# StringTie assemblies for the 4 long-read benchmark species.
# Mirrors slurm_annotate_vertebrates_test_filt_tpm1cov3len300_run006_epoch74.sh
# but uses stringtie_lr.filt_tpm1cov3len300.gtf as input and only runs
# on the 4 species with PacBio Iso-Seq data.
#
# Input:  ${RESULTS_DIR}/${sp}/longread/stringtie_lr.filt_tpm1cov3len300.gtf
# Output: ${RESULTS_DIR}/${sp}/annotate_epoch_74_lr_filt_tpm1cov3len300/
#           orfs.gtf              (raw Tiberius output)
#           orfs.filtered.gtf     (after subseq-collapse post-filter)
#           dropped_subsequences.tsv
#
# Run after slurm_longread_stringtie.sh completes.
#
#SBATCH --job-name=annot_lr_vt
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_lr_vt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_lr_vt_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml
WEIGHTS_TAG=epoch_74

declare -a SPECIES=(
    "Gallus_gallus"
    "Bos_taurus"
    "Zootoca_vivipara"
    "Betta_splendens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE_FILT=${RESULTS_DIR}/${species}/longread/stringtie_lr.${FILT_TAG}.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}_lr_${FILT_TAG}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Resume-skip: both annotate.py and the postprocess filter must have completed.
if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

# Decompress genome if only .gz is present.
if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${STRINGTIE_FILT}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] stringtie=${STRINGTIE_FILT}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE_FILT}" \
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
