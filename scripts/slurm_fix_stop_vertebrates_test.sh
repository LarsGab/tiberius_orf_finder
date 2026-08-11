#!/bin/bash
#SBATCH --job-name=fix_stop_vert
#SBATCH --partition=batch,snowball,pinky
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/fix_stop_vert_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/fix_stop_vert_%A_%a.err

# Fix stop codons in ORF predictions for the 9 vertebrate test species.
# Requires slurm_annotate_vert_test_run009_partial.sh to have finished first
# (to generate orfs.partial.gtf alongside orfs.gtf).
#
# Pipeline per species:
#   1. Diamond: query Tiberius peptides against order-excluded ODB → top-5 species
#   2. Filter ODB to those 5 species' proteins
#   3. miniprot --gff: align filtered proteins to genome
#   4. fix_stop_by_miniprot.py: extend partial ORFs and fix early stops in
#      complete ORFs using the miniprot alignments
#
# Inputs:
#   ${RESULTS_DIR}/${species}/assembly/genome.fa
#   ${RESULTS_DIR}/${species}/annotate_run009_best_filt_tpm1cov3len300/orfs.gtf
#   ${RESULTS_DIR}/${species}/annotate_run009_best_filt_tpm1cov3len300/orfs.partial.gtf
#   ${WORK_DIR}/odb/filtered/${species}_excl_order.fa[.gz]
#   ${WORK_DIR}/peptides/${species}.pep.fa  (optional: used for Diamond pre-filter)
#
# Output:
#   ${RESULTS_DIR}/${species}/fix_stop/orfs.fixed.gtf

set -euo pipefail
source /etc/profile.d/modules.sh
module load singularity/3.11.3

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
WORK_DIR=/home/gabriell/tiberius_proteins_analysis
TIBERIUS_SIF=${TIBERIUS_SIF:-docker://larsgabriel23/tiberius:2.0.2}
TIBERIUS_REPO=${TIBERIUS_REPO:-/home/gabriell/Tiberius}

FILT_TAG=filt_tpm1cov3len300
TAG=run009_best_${FILT_TAG}
TOP_N=5
CPUS=${SLURM_CPUS_PER_TASK:-16}

declare -a SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Bos_taurus
    Delphinapterus_leucas
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
    Homo_sapiens
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
ODB_PROTEINS="${WORK_DIR}/odb/filtered/${species}_excl_order.fa"
[[ ! -s "${ODB_PROTEINS}" && -s "${ODB_PROTEINS}.gz" ]] && ODB_PROTEINS="${ODB_PROTEINS}.gz"
TIBERIUS_PEPTIDES="${WORK_DIR}/peptides/${species}.pep.fa"
ORF_DIR="${RESULTS_DIR}/${species}/annotate_${TAG}"
OUT_DIR="${RESULTS_DIR}/${species}/fix_stop"

echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] odb_proteins=${ODB_PROTEINS}"
echo "[$(date -Iseconds)] tiberius_peptides=${TIBERIUS_PEPTIDES}"
echo "[$(date -Iseconds)] orf_dir=${ORF_DIR}"
echo "[$(date -Iseconds)] out_dir=${OUT_DIR}"

if [[ ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: genome not staged: ${GENOME}"
    exit 0
fi
[[ -s "${ODB_PROTEINS}" ]] || { echo "ERROR: ODB proteins not found: ${ODB_PROTEINS}" >&2; exit 1; }
[[ -s "${ORF_DIR}/orfs.gtf" ]] || {
    echo "ERROR: orfs.gtf not found: ${ORF_DIR}/orfs.gtf" >&2; exit 1;
}
[[ -s "${ORF_DIR}/orfs.partial.gtf" ]] || {
    echo "ERROR: orfs.partial.gtf missing or empty (run slurm_annotate_vert_test_run009_partial.sh first)" >&2
    exit 1
}

if [[ -s "${OUT_DIR}/orfs.fixed.gtf" ]]; then
    echo "SKIP ${species}: ${OUT_DIR}/orfs.fixed.gtf already exists"
    exit 0
fi

mkdir -p "${OUT_DIR}"
mkdir -p "${PROJDIR}/logs"

run_tool() {
    if [[ -n "${TIBERIUS_SIF:-}" ]]; then
        singularity exec "${TIBERIUS_SIF}" "$@"
    else
        "$@"
    fi
}

# ── Step 1: Diamond pre-filter to top-N species ───────────────────────────────
PROTEINS_FOR_MINIPROT="${ODB_PROTEINS}"

if [[ ! -s "${TIBERIUS_PEPTIDES}" ]]; then
    echo "[$(date -Iseconds)] WARN: Tiberius peptides not found, using full ODB for miniprot"
else
    PREPROCESSED_FA="${OUT_DIR}/protein_top${TOP_N}.fa"
    DIAMOND_DB="${OUT_DIR}/prot_db"
    DIAMOND_HITS="${OUT_DIR}/diamond_hits.tsv"

    if [[ ! -s "${PREPROCESSED_FA}" ]]; then
        echo "[$(date -Iseconds)] Building Diamond DB ..."
        run_tool diamond makedb \
            --in "${ODB_PROTEINS}" \
            --db "${DIAMOND_DB}" \
            --threads "${CPUS}"

        echo "[$(date -Iseconds)] Running Diamond blastp ..."
        run_tool diamond blastp \
            --query   "${TIBERIUS_PEPTIDES}" \
            --db      "${DIAMOND_DB}.dmnd" \
            --out     "${DIAMOND_HITS}" \
            --outfmt  6 qseqid sseqid pident length evalue bitscore qlen slen \
            --evalue  1e-5 \
            --max-target-seqs 200 \
            --very-sensitive \
            --threads "${CPUS}"

        echo "[$(date -Iseconds)] Ranking species (top ${TOP_N}) ..."
        (cd "${OUT_DIR}" && \
         python3 "${TIBERIUS_REPO}/tiberius/scripts/rank_species_from_diamond.py" \
             "${DIAMOND_HITS}" "${TOP_N}" \
             > "${OUT_DIR}/species_rank.tsv")

        echo "[$(date -Iseconds)] Top ${TOP_N} species:"
        cat "${OUT_DIR}/top_species.txt"

        ODB_STREAM="cat"
        [[ "${ODB_PROTEINS}" == *.gz ]] && ODB_STREAM="zcat"
        ${ODB_STREAM} "${ODB_PROTEINS}" | awk '
        BEGIN {
            while ((getline < "'"${OUT_DIR}/top_species.txt"'") > 0) {
                wanted[$1] = 1
            }
        }
        /^>/ {
            hdr = substr($0, 2)
            split(hdr, a, /[ \t]/)
            id = a[1]
            sp = id; sub(/_.*/, "", sp)
            keep = (sp in wanted)
        }
        keep { print }
        ' > "${PREPROCESSED_FA}"

        N_FILT=$(grep -c '^>' "${PREPROCESSED_FA}" || echo 0)
        echo "[$(date -Iseconds)] Filtered DB: ${N_FILT} sequences"
    else
        echo "[$(date -Iseconds)] Diamond pre-filter already done, reusing ${PREPROCESSED_FA}"
    fi
    PROTEINS_FOR_MINIPROT="${PREPROCESSED_FA}"
fi

# ── Step 2: miniprot alignment ────────────────────────────────────────────────
MINIPROT_GFF="${OUT_DIR}/miniprot.gff"

if [[ ! -s "${MINIPROT_GFF}" ]]; then
    echo "[$(date -Iseconds)] Running miniprot --gff ..."
    run_tool miniprot \
        -t   "${CPUS}" \
        --gff \
        "${GENOME}" \
        "${PROTEINS_FOR_MINIPROT}" \
        > "${MINIPROT_GFF}"
    echo "[$(date -Iseconds)] miniprot done: $(wc -l < "${MINIPROT_GFF}") lines"
else
    echo "[$(date -Iseconds)] miniprot GFF exists, reusing"
fi

# ── Step 3: fix stop codons ───────────────────────────────────────────────────
eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] Running fix_stop_by_miniprot.py ..."
python "${PROJDIR}/scripts/fix_stop_by_miniprot.py" \
    --orfs     "${ORF_DIR}/orfs.gtf" \
    --partial  "${ORF_DIR}/orfs.partial.gtf" \
    --miniprot "${MINIPROT_GFF}" \
    --genome   "${GENOME}" \
    --out      "${OUT_DIR}/orfs.fixed.gtf"

echo "[$(date -Iseconds)] done -> ${OUT_DIR}/orfs.fixed.gtf"
