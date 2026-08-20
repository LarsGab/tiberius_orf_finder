#!/bin/bash
# Master submission script for the training-species protein pipeline.
#
# Submits all 4 steps with correct SBATCH dependencies:
#
#   STEP1 (prepare_proteins) ──► STEP2 (miniprot_miniprothint) ──►
#                                                                    STEP4 (orf_features)
#   STEP3 (annotate_orfs) ───────────────────────────────────────►
#
# Prerequisite: run lookup_order_taxids.py first to generate species_order_taxids.tsv:
#
#   python scripts/lookup_order_taxids.py \
#       --names  /home/gabriell/tiberius_proteins_analysis/odb/names.dmp \
#       --nodes  /home/gabriell/tiberius_proteins_analysis/odb/nodes.dmp \
#       --species-dir /projects/AI-GUSTUS/tiberius_orf_finder/results/training_vertebrates_v2 \
#       --out    /projects/AI-GUSTUS/tiberius_orf_finder/results/training_vertebrates_v2/species_order_taxids.tsv \
#       --entrez-email YOUR_EMAIL
#
# Usage:
#   bash scripts/slurm_submit_training_protein_pipeline.sh [--skip-annotate] [--skip-proteins]
#
# Flags:
#   --skip-proteins   Skip steps 1+2 (proteins already computed)
#   --skip-annotate   Skip step 3  (annotation already done)

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPT_DIR=${PROJDIR}/scripts
TAXIDS_TSV=${PROJDIR}/results/training_vertebrates_v2/species_order_taxids.tsv

SKIP_PROTEINS=0
SKIP_ANNOTATE=0

for arg in "$@"; do
    case "${arg}" in
        --skip-proteins) SKIP_PROTEINS=1 ;;
        --skip-annotate) SKIP_ANNOTATE=1 ;;
        *) echo "Unknown flag: ${arg}" >&2; exit 1 ;;
    esac
done

mkdir -p "${PROJDIR}/logs"

# ── Check prerequisites ───────────────────────────────────────────────────────
[[ -s "${TAXIDS_TSV}" ]] || {
    echo "ERROR: ${TAXIDS_TSV} not found."
    echo "Run lookup_order_taxids.py first (see header of this script)."
    exit 1
}
N_TAXIDS=$(tail -n +2 "${TAXIDS_TSV}" | wc -l)
echo "species_order_taxids.tsv: ${N_TAXIDS} entries"
[[ "${N_TAXIDS}" -ge 60 ]] || echo "WARNING: fewer than 60 species found — check the TSV"

# ── STEP 1: prepare proteins (ODB filter + diamond + protein_top4.fa) ─────────
JID1=""
if [[ "${SKIP_PROTEINS}" == "0" ]]; then
    JID1=$(sbatch --parsable \
        "${SCRIPT_DIR}/slurm_prepare_proteins_training_v2.sh")
    echo "STEP1 submitted: ${JID1}  (prepare_proteins)"
fi

# ── STEP 2: miniprot + miniprothint ──────────────────────────────────────────
JID2=""
if [[ "${SKIP_PROTEINS}" == "0" ]]; then
    DEP2="afterok:${JID1}"
    JID2=$(sbatch --parsable \
        --dependency="${DEP2}" \
        "${SCRIPT_DIR}/slurm_miniprot_miniprothint_training_v2.sh")
    echo "STEP2 submitted: ${JID2}  (miniprot_miniprothint, after ${JID1})"
fi

# ── STEP 3: annotate ORFs (GPU, independent of steps 1-2) ────────────────────
JID3=""
if [[ "${SKIP_ANNOTATE}" == "0" ]]; then
    JID3=$(sbatch --parsable \
        "${SCRIPT_DIR}/slurm_annotate_orfs_training_v2.sh")
    echo "STEP3 submitted: ${JID3}  (annotate_orfs)"
fi

# ── STEP 4: gffcompare + orf_features ────────────────────────────────────────
DEP4_JIDS=()
[[ -n "${JID2}" ]] && DEP4_JIDS+=("${JID2}")
[[ -n "${JID3}" ]] && DEP4_JIDS+=("${JID3}")

if [[ "${#DEP4_JIDS[@]}" -gt 0 ]]; then
    # SLURM AND-dependency: afterok:JID_A:JID_B (both must succeed)
    DEP4="afterok:$(IFS=':'; echo "${DEP4_JIDS[*]}")"
    JID4=$(sbatch --parsable \
        --dependency="${DEP4}" \
        "${SCRIPT_DIR}/slurm_orf_features_gffcmp_training_v2.sh")
    echo "STEP4 submitted: ${JID4}  (orf_features_gffcmp, after ${DEP4})"
else
    JID4=$(sbatch --parsable \
        "${SCRIPT_DIR}/slurm_orf_features_gffcmp_training_v2.sh")
    echo "STEP4 submitted: ${JID4}  (orf_features_gffcmp, no dependency)"
fi

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \${USER} --format='%.10i %.12j %.8T %.10M %R'"
echo ""
echo "Logs: ${PROJDIR}/logs/"
echo ""
echo "Once STEP4 finishes, train the LGB model with:"
echo "  python scripts/train_orf_lgb_3class.py \\"
echo "    --base-dir  '${PROJDIR}/results/training_vertebrates_v2' \\"
echo "    --annot-tag annotate_train_lorf \\"
echo "    --out-dir   '${PROJDIR}/results/filter_analysis'"
