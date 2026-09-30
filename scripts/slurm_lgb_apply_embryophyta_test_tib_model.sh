#!/bin/bash
# Apply the Tiberius-trained LGB (lgb_embryophyta_tib) to Tiberius test
# predictions for the 6 embryophyta test species.
#
# Reuses the tiberius_features.tsv already computed by
# slurm_lgb_apply_embryophyta_test.sh (with best_protein_coverage).
# Overwrites tiberius_lgb_filtered/tiberius_lgb_filtered.gtf so the eval
# script picks up the new tib_lgb3* gene sets.
#
#SBATCH --job-name=lgb_emb_test_tib
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_emb_test_tib_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_emb_test_tib_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_embryophyta_test_v2
BENCH=/home/gabriell/tiberius_benchmarking
MODEL=${PROJDIR}/results/filter_analysis/lgb_embryophyta_tib/lgb_3class_model.pkl

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    Arabidopsis_thaliana
    Eschscholzia_californica
    Freycinetia_multiflora
    Medicago_truncatula
    Mimulus_guttatus
    Urochloa_brizantha
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

TIB_GTF="${BENCH}/paper/Embryophyta/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
TIB_OUTDIR="${RESULTS_DIR}/${species}/tiberius_lgb_filtered"
TIB_FEAT="${TIB_OUTDIR}/tiberius_features.tsv"
TIB_LGB="${TIB_OUTDIR}/tiberius_lgb_filtered.gtf"

echo "[$(date -Iseconds)] species=${species}"

for f in "${MODEL}" "${TIB_GTF}" "${TIB_FEAT}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

echo "[$(date -Iseconds)] Applying Tiberius-trained LGB to Tiberius predictions …"
python scripts/apply_lgb_model_gtf.py \
    --model    "${MODEL}" \
    --features "${TIB_FEAT}" \
    --in-gtf   "${TIB_GTF}" \
    --out-gtf  "${TIB_LGB}"

N=$(awk -F'\t' '$3=="transcript"' "${TIB_LGB}" | wc -l || true)
echo "[$(date -Iseconds)] Tiberius lgb3: ${N} transcripts retained → ${TIB_LGB}"
