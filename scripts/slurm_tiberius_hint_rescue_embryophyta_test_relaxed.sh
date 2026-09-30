#!/bin/bash
# Approach 1 rescue step: run Tiberius (hint_integration branch) on the
# hint_rescue_relaxed/ per-locus inputs (from
# slurm_prepare_hint_rescue_embryophyta_test_relaxed.sh).
#
# Output per species → hint_rescue_relaxed/tiberius_hint_rescue.gtf
#
#SBATCH --job-name=tib_hr_emb_rlx
#SBATCH --partition=vision,vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_hr_emb_rlx_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_hr_emb_rlx_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
SCRIPTS_DIR=${PROJDIR}/scripts
TIBERIUS_REPO=${TIBERIUS_REPO:-/projects/AI-GUSTUS/Tiberius}
TIBERIUS_CFG=angiosperms
HINT_WEIGHT=2.5

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

OUTDIR=${TESTDIR}/${species}/hint_rescue_relaxed
LOCI_FA=${OUTDIR}/combined_loci.fa
HINTS_GFF=${OUTDIR}/combined_hints.gff
MANIFEST=${OUTDIR}/loci_manifest.tsv
RAW_GTF=${OUTDIR}/tiberius_hint_rescue_raw.gtf
OUT_GTF=${OUTDIR}/tiberius_hint_rescue.gtf

echo "[$(date -Iseconds)] species=${species}"

for f in "${LOCI_FA}" "${HINTS_GFF}" "${MANIFEST}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

N_ENTRIES=$(grep -c '^>' "${LOCI_FA}" || echo 0)
echo "[$(date -Iseconds)] ${N_ENTRIES} (locus, chain) entries to predict"

TIBERIUS_BRANCH=$(git -C "${TIBERIUS_REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "${TIBERIUS_BRANCH}" != "hint_integration" ]]; then
    echo "ERROR: ${TIBERIUS_REPO} is on '${TIBERIUS_BRANCH}', expected 'hint_integration'." >&2
    exit 2
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate tib_test

rm -f "${RAW_GTF}" "${OUT_GTF}"

python "${TIBERIUS_REPO}/tiberius.py" \
    --genome      "${LOCI_FA}" \
    --model_cfg   "${TIBERIUS_CFG}" \
    --hints       "${HINTS_GFF}" \
    --hint_weight "${HINT_WEIGHT}" \
    --out         "${RAW_GTF}"

echo "[$(date -Iseconds)] Tiberius done → ${RAW_GTF}"

micromamba activate orffinder

python "${SCRIPTS_DIR}/filter_and_merge_rescue_gtf.py" \
    "${RAW_GTF}" \
    "${HINTS_GFF}" \
    "${MANIFEST}" \
    "${OUT_GTF}"

N_TX=$(awk '$3=="transcript"' "${OUT_GTF}" | wc -l)
echo "[$(date -Iseconds)] done: ${N_TX} rescued transcripts → ${OUT_GTF}"
