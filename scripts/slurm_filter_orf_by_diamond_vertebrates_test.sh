#!/bin/bash
# Soft protein filter for the ORF-finder predictions on the 8
# vertebrates_test species.
#
# Per species: gffread → diamond blastp against the per-species
# order-excluded Vertebrata OrthoDB → within-gene isoform pruning.
# Never drops a whole gene; only trims unhit isoforms from a gene that
# has at least one hit sibling.
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/annotate_epoch_74_filt_tpm1cov3len300/soft_protein/
#     orfs.peptides.fa
#     diamond_db.dmnd
#     diamond.tsv
#     orfs.filtered_soft_protein.gtf
#     dropped_soft_protein.tsv
#
# Prereqs (rebuild these first if missing):
#   ${WORK_DIR}/odb/filtered/<sp>_excl_order.fa
#
#SBATCH --job-name=orf_softprot
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/orf_softprot_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/orf_softprot_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
TIB_DIR_NAME=annotate_epoch_74_${FILT_TAG}
WORK_DIR=/home/gabriell/tiberius_proteins_analysis

EVAL_MAX=${EVAL_MAX:-1e-5}
PIDENT_MIN=${PIDENT_MIN:-0}
BITSCORE_MIN=${BITSCORE_MIN:-0}
# Distinct subdir per threshold combination so multiple runs coexist.
SOFTPROT_TAG=${SOFTPROT_TAG:-default}

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
ORFS=${RESULTS_DIR}/${species}/${TIB_DIR_NAME}/orfs.filtered.gtf
ODB=${WORK_DIR}/odb/filtered/${species}_excl_order.fa
if [[ "${SOFTPROT_TAG}" == "default" ]]; then
    OUTDIR=${RESULTS_DIR}/${species}/${TIB_DIR_NAME}/soft_protein
else
    OUTDIR=${RESULTS_DIR}/${species}/${TIB_DIR_NAME}/soft_protein_${SOFTPROT_TAG}
fi
# Diamond TSV is thresholds-independent; symlink from the default subdir so
# stricter thresholds don't rerun the ~2h blastp step.
SHARED_DIAM=${RESULTS_DIR}/${species}/${TIB_DIR_NAME}/soft_protein/diamond.tsv
SHARED_DB=${RESULTS_DIR}/${species}/${TIB_DIR_NAME}/soft_protein/diamond_db.dmnd

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

# Reuse the diamond outputs from the default subdir if they exist.
if [[ -s "${SHARED_DIAM}" && ! -s "${OUTDIR}/diamond.tsv" ]]; then
    ln -sf "${SHARED_DIAM}" "${OUTDIR}/diamond.tsv"
fi
if [[ -s "${SHARED_DB}" && ! -s "${OUTDIR}/diamond_db.dmnd" ]]; then
    ln -sf "${SHARED_DB}" "${OUTDIR}/diamond_db.dmnd"
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

for f in "${GENOME}" "${ORFS}" "${ODB}"; do
    if [[ ! -s "${f}" ]]; then
        echo "[$(date -Iseconds)] missing input: ${f}" >&2
        exit 2
    fi
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] species=${species} evalue<=${EVAL_MAX} pident>=${PIDENT_MIN}"

python "${PROJDIR}/scripts/filter_orf_by_diamond_within_gene.py" \
    --orfs-gtf     "${ORFS}" \
    --genome       "${GENOME}" \
    --proteins-fa  "${ODB}" \
    --out-dir      "${OUTDIR}" \
    --threads      "${SLURM_CPUS_PER_TASK}" \
    --evalue       "${EVAL_MAX}" \
    --pident-min   "${PIDENT_MIN}" \
    --bitscore-min "${BITSCORE_MIN}"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.filtered_soft_protein.gtf"
