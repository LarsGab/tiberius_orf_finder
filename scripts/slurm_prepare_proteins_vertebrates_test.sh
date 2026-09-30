#!/bin/bash
# Diamond pre-filter: identify the 4 closest ODB species (outside target order)
# for all 9 vertebrates_test species, then extract protein_top4.fa. Same logic
# as slurm_prepare_proteins_mammals_vertebrates_test.sh but for all 9 species.
#
# Peptide source: Tiberius predictions
#   annotate_epoch_74_filt_tpm1cov3len300_lorf/orfs.filtered.gtf + assembly/genome.fa
# with tiberius_benchmarking fallback if the primary ORF GTF is missing.
#
# Input:
#   results/vertebrates_test/<sp>/assembly/genome.fa
#   results/vertebrates_test/<sp>/annotate_epoch_74_filt_tpm1cov3len300_lorf/orfs.filtered.gtf
#   /home/gabriell/tiberius_proteins_analysis/odb/filtered/<sp>_excl_order.fa[.gz]
#
# Output:
#   results/vertebrates_test/<sp>/fix_stop/protein_top4.fa
#   results/vertebrates_test/<sp>/fix_stop/diamond_hits.tsv
#   results/vertebrates_test/<sp>/fix_stop/top_species.txt
#
#SBATCH --job-name=prep_prot_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_prot_vt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_prot_vt_%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load singularity/3.11.3

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
WORK_DIR=/home/gabriell/tiberius_proteins_analysis
TIBERIUS_REPO=/home/gabriell/Tiberius
BENCH=/home/gabriell/tiberius_benchmarking

SIF=${PROJDIR}/sif/tiberius_2.0.2.sif
if [[ ! -s "${SIF}" ]]; then
    mkdir -p "${PROJDIR}/sif"
    (flock -x 200
     [[ ! -s "${SIF}" ]] && singularity pull "${SIF}" docker://larsgabriel23/tiberius:2.0.2
    ) 200>"${SIF}.lock" || true
    [[ -s "${SIF}" ]] || { echo "ERROR: SIF pull failed" >&2; exit 1; }
fi

run_tool() { singularity exec --bind /projects/AI-GUSTUS,/home/gabriell "${SIF}" "$@"; }

TOP_N=4
CPUS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    "Gallus_gallus"
    "Pristiophorus_japonicus"
    "Bos_taurus"
    "Delphinapterus_leucas"
    "Takifugu_rubripes"
    "Zootoca_vivipara"
    "Archocentrus_centrarchus"
    "Betta_splendens"
    "Homo_sapiens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
ORFS_GTF="${RESULTS_DIR}/${species}/annotate_epoch_74_filt_tpm1cov3len300_lorf/orfs.filtered.gtf"
FIX_DIR="${RESULTS_DIR}/${species}/fix_stop"
PROTEIN_TOP4="${FIX_DIR}/protein_top4.fa"
ODB="${WORK_DIR}/odb/filtered/${species}_excl_order.fa"
[[ ! -s "${ODB}" && -s "${ODB}.gz" ]] && ODB="${ODB}.gz"

echo "[$(date -Iseconds)] species=${species}"

if [[ -s "${PROTEIN_TOP4}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP: protein_top4.fa already exists (set FORCE=1 to rerun)"
    exit 0
fi

[[ -s "${GENOME}" ]] || { echo "ERROR: genome not found: ${GENOME}" >&2; exit 2; }
[[ -s "${ODB}" ]]    || { echo "ERROR: ODB not found: ${ODB}" >&2; exit 2; }

mkdir -p "${FIX_DIR}"

# ── Peptide source ─────────────────────────────────────────────────────────────
PEPTIDES="${FIX_DIR}/query_peptides.fa"

if [[ -s "${PEPTIDES}" ]]; then
    echo "[$(date -Iseconds)] reusing existing query peptides: ${PEPTIDES}"
elif [[ -s "${ORFS_GTF}" ]]; then
    echo "[$(date -Iseconds)] extracting peptides from ORF predictions ..."
    run_tool gffread "${ORFS_GTF}" -g "${GENOME}" -y "${PEPTIDES}" || true
    if [[ -s "${PEPTIDES}" ]]; then
        N=$(grep -c '^>' "${PEPTIDES}" || echo 0)
        echo "[$(date -Iseconds)] extracted ${N} peptides from ORF predictions"
    else
        echo "[$(date -Iseconds)] gffread produced no output from ORF GTF"
        rm -f "${PEPTIDES}"
    fi
fi

# Tiberius benchmarking fallback
if [[ ! -s "${PEPTIDES}" ]]; then
    TIB_GTF="${BENCH}/paper/Vertebrata/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
    TIB_GENOME="${BENCH}/Vertebrata/${species}/genome.fa"
    if [[ -s "${TIB_GTF}" && -s "${TIB_GENOME}" ]]; then
        echo "[$(date -Iseconds)] falling back to Tiberius benchmarking GTF ..."
        run_tool gffread "${TIB_GTF}" -g "${TIB_GENOME}" -y "${PEPTIDES}" || true
        [[ -s "${PEPTIDES}" ]] && echo "[$(date -Iseconds)] extracted $(grep -c '^>' "${PEPTIDES}" || echo 0) peptides from Tiberius GTF"
    fi
fi

[[ -s "${PEPTIDES}" ]] || { echo "ERROR: no peptides available for ${species}" >&2; exit 2; }

# Strip stop-codon dots (.) from peptide sequences — Diamond rejects them
echo "[$(date -Iseconds)] stripping stop-codon dots from peptide sequences ..."
sed -i '/^>/!s/\.//g' "${PEPTIDES}"

# ── Diamond pre-filter → protein_top4.fa ──────────────────────────────────────
DIAMOND_DB="${FIX_DIR}/prot_db"
DIAMOND_HITS="${FIX_DIR}/diamond_hits.tsv"

if [[ ! -s "${DIAMOND_HITS}" ]]; then
    echo "[$(date -Iseconds)] building Diamond DB ..."
    run_tool diamond makedb \
        --in      "${ODB}" \
        --db      "${DIAMOND_DB}" \
        --threads "${CPUS}"

    echo "[$(date -Iseconds)] running Diamond blastp ..."
    run_tool diamond blastp \
        --query            "${PEPTIDES}" \
        --db               "${DIAMOND_DB}.dmnd" \
        --out              "${DIAMOND_HITS}" \
        --outfmt           6 qseqid sseqid pident length evalue bitscore qlen slen \
        --evalue           1e-5 \
        --max-target-seqs  200 \
        --very-sensitive \
        --threads          "${CPUS}"
    echo "[$(date -Iseconds)] Diamond done: $(wc -l < "${DIAMOND_HITS}") hits"
fi

echo "[$(date -Iseconds)] ranking species (top ${TOP_N}) ..."
(cd "${FIX_DIR}" && \
 python3 "${TIBERIUS_REPO}/tiberius/scripts/rank_species_from_diamond.py" \
     "${DIAMOND_HITS}" "${TOP_N}" \
     > "${FIX_DIR}/species_rank.tsv")

[[ -s "${FIX_DIR}/top_species.txt" ]] || { echo "ERROR: rank_species produced no top_species.txt" >&2; exit 2; }
echo "[$(date -Iseconds)] top ${TOP_N} species:"
cat "${FIX_DIR}/top_species.txt"

ODB_STREAM="cat"
[[ "${ODB}" == *.gz ]] && ODB_STREAM="zcat"
${ODB_STREAM} "${ODB}" | awk '
BEGIN {
    while ((getline < "'"${FIX_DIR}/top_species.txt"'") > 0) {
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
' > "${PROTEIN_TOP4}"

N=$(grep -c '^>' "${PROTEIN_TOP4}" || echo 0)
echo "[$(date -Iseconds)] protein_top4.fa: ${N} sequences"
[[ "${N}" -gt 0 ]] || { echo "ERROR: protein_top4.fa is empty" >&2; exit 2; }
echo "[$(date -Iseconds)] done -> ${PROTEIN_TOP4}"
