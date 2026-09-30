#!/bin/bash
# Re-annotate 6 embryophyta test species to add lorf_class.
#
# Writes to the same annotate_run001_e300 directory (FORCE=1 by default) so
# that downstream LGB apply and eval pick up the corrected annotations.
# Uses filtered StringTie input (filt_tpm1cov3len300) for consistency with
# the embryophyta training pipeline.
#
# Usage: sbatch --array=0-5 scripts/slurm_annotate_embryophyta_test_lorf.sh [epoch]
#   epoch defaults to 300.
#
#SBATCH --job-name=annot_emb_test_lorf
#SBATCH --partition=vision,vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_test_lorf_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/annot_emb_test_lorf_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
EPOCH="${1:-300}"
FILT_TAG=filt_tpm1cov3len300
WEIGHTS=${PROJDIR}/results/models/cnn_lstm_embryophyta_run001_v2/epoch_${EPOCH}.weights.h5
CONFIG=${PROJDIR}/configs/cnn_lstm_embryophyta_run001.yaml
ANNOT_TAG=annotate_run001_e${EPOCH}

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Eschscholzia_californica"
    "Freycinetia_multiflora"
    "Medicago_truncatula"
    "Mimulus_guttatus"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${TESTDIR}/${species}/assembly/genome.fa"
RAW_STRINGTIE="${TESTDIR}/${species}/stringtie/stringtie.gtf"
FILT_STRINGTIE="${TESTDIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf"
OUTDIR="${TESTDIR}/${species}/${ANNOT_TAG}"

echo "[$(date -Iseconds)] species=${species}  epoch=${EPOCH}"

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
