#!/bin/bash
# Run miniprot + miniprothint for all 6 Embryophyta test species.
#
# Prerequisite: slurm_prepare_proteins_embryophyta_test.sh (protein_top4.fa)
#
# Output per species (results/training_embryophyta_test_v2/<sp>/proteins/):
#   miniprot_scored.gff
#   miniprothint/hc.gff
#
#SBATCH --job-name=miniprot_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/miniprot_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/miniprot_emb_%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load singularity/3.11.3

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/training_embryophyta_test_v2
SCORING_MATRIX=/home/gabriell/Tiberius/conf/blosum62.csv
SIF=${PROJDIR}/sif/tiberius_2.0.2.sif
CPUS=${SLURM_CPUS_PER_TASK:-16}

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
GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
PROTEIN_FA=${RESULTS_DIR}/${species}/proteins/protein_top4.fa
OUT_DIR=${RESULTS_DIR}/${species}/proteins
SCORED_GFF=${OUT_DIR}/miniprot_scored.gff
MINIPROTHINT_DIR=${OUT_DIR}/miniprothint

echo "[$(date -Iseconds)] species=${species}"

[[ -s "${SCORING_MATRIX}" ]] || { echo "ERROR: missing ${SCORING_MATRIX}" >&2; exit 2; }
[[ -s "${GENOME}" ]]         || { echo "SKIP ${species}: missing genome.fa" >&2; exit 0; }
[[ -s "${PROTEIN_FA}" ]]     || { echo "SKIP ${species}: missing protein_top4.fa (run prepare step first)" >&2; exit 0; }

if [[ -s "${MINIPROTHINT_DIR}/hc.gff" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: miniprothint/hc.gff already exists"; exit 0
fi

mkdir -p "${OUT_DIR}" "${MINIPROTHINT_DIR}"

run_tool() {
    singularity exec \
        --bind /projects/AI-GUSTUS,/home/gabriell \
        "${SIF}" "$@"
}

# ── Step 1: miniprot --aln | miniprot_boundary_scorer ────────────────────────
if [[ ! -s "${SCORED_GFF}" ]]; then
    echo "[$(date -Iseconds)] Running miniprot --aln | miniprot_boundary_scorer …"
    run_tool miniprot \
        --aln \
        -t "${CPUS}" \
        "${GENOME}" \
        "${PROTEIN_FA}" \
    | run_tool miniprot_boundary_scorer \
        -s "${SCORING_MATRIX}" \
        -o "${SCORED_GFF}"
    echo "[$(date -Iseconds)] miniprot_scored.gff: $(wc -l < "${SCORED_GFF}") lines"
else
    echo "[$(date -Iseconds)] Reusing ${SCORED_GFF}"
fi

# ── Step 2: miniprothint ──────────────────────────────────────────────────────
echo "[$(date -Iseconds)] Running miniprothint.py …"
run_tool miniprothint.py \
    "${SCORED_GFF}" \
    --workdir          "${MINIPROTHINT_DIR}" \
    --ignoreCoverage \
    --topNperSeed      10 \
    --minScoreFraction 0.5

[[ -s "${MINIPROTHINT_DIR}/hc.gff" ]] || {
    echo "ERROR: miniprothint produced no hc.gff" >&2; exit 2
}
echo "[$(date -Iseconds)] done → ${MINIPROTHINT_DIR}/hc.gff"
wc -l "${MINIPROTHINT_DIR}/hc.gff"
