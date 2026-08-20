#!/bin/bash
# Re-run Tiberius annotation (epoch_74, filt_tpm1cov3len300) with --lorf-class
# and --partial-out to get LORF-class annotations on both complete and partial
# ORFs.  Compares LORF_UPSTOP / sORF_UPSTOP / upLORF / LORF_NOUPSTOP counts
# across species to assess how many partial ORFs have confirmed 5' starts.
#
# Output:
#   ${RESULTS_DIR}/<sp>/annotate_epoch_74_filt_tpm1cov3len300_lorf/
#     orfs.gtf              (complete ORFs with lorf_class attribute)
#     orfs.partial.gtf      (3'-truncated ORFs with lorf_class attribute)
#     orfs.filtered.gtf     (after subseq-collapse)
#
#SBATCH --job-name=annot_lorf
#SBATCH --partition=vision-fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_lorf_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_lorf_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

WEIGHTS=${PROJDIR}/results/models/cnn_lstm_vertebrates_run006/epoch_74.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_run006.yaml

WEIGHTS_TAG=$(basename "${WEIGHTS}" .weights.h5)
FILT_TAG=filt_tpm1cov3len300

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Gallus_gallus"
    "Pristiophorus_japonicus"
    "Bos_taurus"
    "Delphinapterus_leucas"
    "Takifugu_rubripes"
    "Zootoca_vivipara"
    "Archocentrus_centrarchus"
    "Betta_splendens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
STRINGTIE_RAW=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
STRINGTIE_FILT=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${RESULTS_DIR}/${species}/annotate_${WEIGHTS_TAG}_${FILT_TAG}_lorf

test -s "${WEIGHTS}" || { echo "missing weights: ${WEIGHTS}" >&2; exit 2; }
test -s "${CONFIG}"  || { echo "missing config: ${CONFIG}"   >&2; exit 2; }
mkdir -p "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.partial.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.partial.gtf already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" || ! -s "${STRINGTIE_RAW}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing genome.fa or raw stringtie.gtf"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

if [[ ! -s "${STRINGTIE_FILT}" ]]; then
    echo "[$(date -Iseconds)] filtering stringtie -> ${STRINGTIE_FILT}"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf       "${STRINGTIE_RAW}" \
        --out-gtf      "${STRINGTIE_FILT}" \
        --out-tsv      "${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.decisions.tsv" \
        --min-length   300 \
        --min-cov      3.0 \
        --min-tpm      1.0 \
        --long-length  3000 \
        --min-tpm-long 0.5
fi
test -s "${STRINGTIE_FILT}" || { echo "filter produced no GTF: ${STRINGTIE_FILT}" >&2; exit 2; }

cd "${PROJDIR}"

echo "[$(date -Iseconds)] species=${species} weights=${WEIGHTS_TAG}"
echo "[$(date -Iseconds)] annotate out=${OUTDIR}/orfs.gtf"

python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${STRINGTIE_FILT}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK}" \
    --lorf-class \
    --partial-out   "${OUTDIR}/orfs.partial.gtf"

echo "[$(date -Iseconds)] postprocess: subseq collapse"

python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.partial.gtf + orfs.filtered.gtf"
