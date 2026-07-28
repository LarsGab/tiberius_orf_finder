#!/bin/bash
# Run annotate.py (embryophyta run001 epoch_39, tpm1cov3len300 filter) on
# long-read StringTie assemblies for the 4 embryophyta LR species.
# Mirrors slurm_annotate_embryophyta_test_run001.sh but uses the LR
# filtered StringTie GTF.
#
# Input:  ${TESTDIR}/${sp}/longread/stringtie_lr.filt_tpm1cov3len300.gtf
# Output: ${TESTDIR}/${sp}/annotate_run001_e39_lr_filt_tpm1cov3len300/
#           orfs.gtf
#           orfs.filtered.gtf     (after subseq-collapse)
#           dropped_subsequences.tsv
#
#SBATCH --job-name=annot_emb_lr
#SBATCH --partition=vision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_lr_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_lr_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
FILT_TAG=filt_tpm1cov3len300

EPOCH=${EPOCH:-39}
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2/epoch_${EPOCH}.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Eschscholzia_californica"
    "Medicago_truncatula"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TESTDIR}/${species}/assembly/genome.fa
STRINGTIE_FILT=${TESTDIR}/${species}/longread/stringtie_lr.${FILT_TAG}.gtf
OUTDIR=${TESTDIR}/${species}/annotate_run001_e${EPOCH}_lr_${FILT_TAG}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.filtered.gtf already exists"
    exit 0
fi

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
echo "[$(date -Iseconds)] species=${species}  epoch=${EPOCH}"
echo "[$(date -Iseconds)] stringtie=${STRINGTIE_FILT}"
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
