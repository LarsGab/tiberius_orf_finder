#!/bin/bash
# Annotate all embryophyta training species for LGB training.
# GPU: vision partition, batch-size 200.
#
# Usage: sbatch --array=0-44 scripts/slurm_annotate_training_embryophyta_v2.sh [epoch]
#   epoch defaults to 300.
#
# Output per species:
#   results/training_embryophyta_v2/<sp>/annotate_run001_e<epoch>/
#     orfs.gtf  orfs.filtered.gtf  dropped_subsequences.tsv
#
#SBATCH --job-name=annot_emb_train
#SBATCH --partition=vision,vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-44
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_train_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_train_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2
EPOCH="${1:-300}"
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2/epoch_${EPOCH}.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml
ANNOT_TAG=annotate_run001_e${EPOCH}
FILT_TAG=filt_tpm1cov3len300

mkdir -p "${PROJDIR}/logs"

readarray -t SPECIES < <(ls "${TRAINDIR}" | grep -E '^[A-Z][a-z]')
[[ ${SLURM_ARRAY_TASK_ID} -lt ${#SPECIES[@]} ]] || { echo "task id out of range"; exit 0; }
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${TRAINDIR}/${species}/assembly/genome.fa"
RAW_STRINGTIE="${TRAINDIR}/${species}/stringtie/stringtie.gtf"
FILT_STRINGTIE="${TRAINDIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf"
OUTDIR="${TRAINDIR}/${species}/${ANNOT_TAG}"

echo "[$(date -Iseconds)] species=${species}  epoch=${EPOCH}"

if [[ -s "${OUTDIR}/orfs.filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: orfs.filtered.gtf already exists"; exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${RAW_STRINGTIE}" "${WEIGHTS}" "${CONFIG}"; do
    [[ -s "${f}" ]] || { echo "missing: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

mkdir -p "${OUTDIR}"
cd "${PROJDIR}"

if [[ ! -s "${FILT_STRINGTIE}" ]]; then
    echo "[$(date -Iseconds)] filtering stringtie …"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf  "${RAW_STRINGTIE}" \
        --out-gtf "${FILT_STRINGTIE}"
fi

N_TX=$(awk -F'\t' '$3=="transcript"' "${FILT_STRINGTIE}" | wc -l || true)
[[ "${N_TX}" -gt 0 ]] || { echo "SKIP ${species}: 0 transcripts after filtering"; exit 0; }
echo "[$(date -Iseconds)] ${N_TX} transcripts after filtering"

echo "[$(date -Iseconds)] annotating …"
python "${PROJDIR}/scripts/annotate.py" \
    --stringtie-gtf "${FILT_STRINGTIE}" \
    --genome        "${GENOME}" \
    --weights       "${WEIGHTS}" \
    --config        "${CONFIG}" \
    --out-dir       "${OUTDIR}" \
    --batch-size    200 \
    --threads       "${SLURM_CPUS_PER_TASK:-4}" \
    --lorf-class

echo "[$(date -Iseconds)] subseq collapse …"
python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
    --orfs-gtf   "${OUTDIR}/orfs.gtf" \
    --out-gtf    "${OUTDIR}/orfs.filtered.gtf" \
    --report-tsv "${OUTDIR}/dropped_subsequences.tsv"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered.gtf"
