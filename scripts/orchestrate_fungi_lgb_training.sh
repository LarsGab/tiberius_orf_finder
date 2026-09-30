#!/bin/bash
# Fungi LGB training orchestrator: 50 selected training species.
# Chain: (1) prep+miniprot+miniprothint  (2) DRUSILLA annotate  (3) prep-features again to run steps 4-5  (4) train LGB
#
# Usage:
#   ./scripts/orchestrate_fungi_lgb_training.sh
#
# Reads species from nextflow/conf/fungi/species_training_lgb50.txt.

set -euo pipefail

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJDIR}/results/training_fungi_v2}
SPECIES_FILE=${SPECIES_FILE:-${PROJDIR}/nextflow/conf/fungi/species_training_lgb50.txt}
RUNTIME_TSV=${RUNTIME_TSV:-${PROJDIR}/results/filter_analysis/lgb_fungi_v2/runtimes.tsv}
OUT_LGB=${OUT_LGB:-${PROJDIR}/results/filter_analysis/lgb_fungi_v2}

[[ -s "${SPECIES_FILE}" ]] || { echo "missing ${SPECIES_FILE}" >&2; exit 2; }
N=$(wc -l < "${SPECIES_FILE}")
mkdir -p "${OUT_LGB}"

echo "[$(date -Iseconds)] fungi LGB training: N=${N} species → ${OUT_LGB}"

# Stage 1: prep (miniprot + miniprothint on Fungi.fa)
J_PREP=$(sbatch --parsable --array=1-${N} \
    --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
    "${SCRIPTS_DIR}/slurm_fungi_lgb_prep_features.sh")
echo "[stage1] prep (miniprot+miniprothint) = ${J_PREP}"

# Stage 2: DRUSILLA annotate (GPU)
J_ANN=$(sbatch --parsable --dependency=afterok:${J_PREP} --array=1-${N} \
    --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
    "${SCRIPTS_DIR}/slurm_annotate_fungi_lgb50.sh")
echo "[stage2] annotate = ${J_ANN}"

# Stage 3: re-run prep script (this time it does gffcompare + features since annotate output now exists)
J_FEAT=$(sbatch --parsable --dependency=afterok:${J_ANN} --array=1-${N} \
    --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
    "${SCRIPTS_DIR}/slurm_fungi_lgb_prep_features.sh")
echo "[stage3] features = ${J_FEAT}"

# Stage 4: train LGB
J_TRAIN=$(sbatch --parsable --dependency=afterok:${J_FEAT} \
    --export=ALL,BASE_DIR=${RESULTS_ROOT},ANNOT_TAG=annotate_run006_lgb50_prep,OUT_DIR=${OUT_LGB} \
    "${SCRIPTS_DIR}/slurm_train_fungi_lgb.sh")
echo "[stage4] train = ${J_TRAIN}"

echo
echo "Chain: prep=${J_PREP} → annotate=${J_ANN} → features=${J_FEAT} → train=${J_TRAIN}"
echo "Model will land at: ${OUT_LGB}/lgb_3class_model.pkl"
