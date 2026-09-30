#!/bin/bash
# Bridge tiberius_proteins pipeline outputs (miniprot + miniprothint) into
# tiberius_orf_finder layout for insects_test_v2, then run chainedHints.py.
#
# Source (tiberius_proteins pipeline):
#   $TP_WORK/miniprot/<sp>_excl_order/{miniprot_parsed.gff, hc.gff, miniprothint.gff}
#
# Target (tiberius_orf_finder):
#   <sp>/proteins/miniprot_scored.gff        ← link to miniprot_parsed.gff
#   <sp>/proteins/miniprothint/hc.gff        ← link to hc.gff
#   <sp>/proteins/miniprothint/miniprothint.gff
#   <sp>/hint_rescue/chained_hints.gff       ← chainedHints.py output
#
#SBATCH --job-name=int_prot_ins
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/int_prot_ins_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/int_prot_ins_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
TP_WORK=${TP_WORK:-/home/gabriell/tiberius_proteins_analysis}
CHAINED_HINTS_PY=${CHAINED_HINTS_PY:-/projects/AI-GUSTUS/Tiberius/tiberius/scripts/chainedHints.py}

source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV
CLADE=insects_test_v2

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")

SP_DIR=${RESULTS_ROOT}/${SPECIES}
SRC=${TP_WORK}/miniprot/${SPECIES}_excl_order

for f in "${SRC}/miniprot_parsed.gff" "${SRC}/hc.gff" "${SRC}/miniprothint.gff" "${CHAINED_HINTS_PY}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 3; }
done

PROT_DIR=${SP_DIR}/proteins
MP_DIR=${PROT_DIR}/miniprothint
HR_DIR=${SP_DIR}/hint_rescue
mkdir -p "${MP_DIR}" "${HR_DIR}"

# Symlinks (source of truth stays in tiberius_proteins_analysis)
ln -sfn "${SRC}/miniprot_parsed.gff" "${PROT_DIR}/miniprot_scored.gff"
ln -sfn "${SRC}/hc.gff"               "${MP_DIR}/hc.gff"
ln -sfn "${SRC}/miniprothint.gff"     "${MP_DIR}/miniprothint.gff"

echo "[$(date -Iseconds)] symlinks staged for ${SPECIES}"
ls -la "${PROT_DIR}/miniprot_scored.gff" "${MP_DIR}/hc.gff" "${MP_DIR}/miniprothint.gff"

# Run chainedHints.py (hc.gff + miniprot_scored.gff → chained_hints.gff)
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

CHAINED=${HR_DIR}/chained_hints.gff
if [[ ! -s "${CHAINED}" || "${FORCE:-0}" == "1" ]]; then
    run_timed "${CLADE}" "${SPECIES}" chained_hints -- \
        python "${CHAINED_HINTS_PY}" \
            "${MP_DIR}/hc.gff" \
            "${PROT_DIR}/miniprot_scored.gff" \
            --output "${CHAINED}"
    echo "[$(date -Iseconds)] chained hints: $(wc -l < "${CHAINED}") lines"
else
    echo "[skip] chained_hints.gff already present"
fi

echo "[$(date -Iseconds)] done -> ${CHAINED}"
