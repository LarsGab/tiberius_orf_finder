#!/bin/bash
# Soft protein filter for ab-initio Tiberius predictions on the 6 embryophyta
# test species that have Tiberius/BRAKER3 benchmarking data.
#
# Uses the pre-computed diamond_hits.tsv (Tiberius peptides vs order-excluded
# OrthoDB) already present in each species' proteins/ directory — no GPU needed.
#
# Within-gene rule (filter_orf_by_diamond_within_gene.py mode B):
#   if any isoform of a gene has a Diamond hit → keep only hit isoforms
#   if no isoform has any hit              → keep every isoform (soft filter)
#
# Input:
#   proteins/diamond_hits.tsv   (pre-computed, Tiberius peptides as query)
#   tiberius_benchmarking/.../tiberius_seqlen.gtf
#
# Output:
#   <testdir>/<sp>/tiberius_protein_filter/
#     tiberius_filtered.gtf           (passing predictions)
#     orfs.filtered_soft_protein.gtf  (same, script default name)
#
#SBATCH --job-name=filt_tib_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/filt_tib_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/filt_tib_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
BENCH=/home/gabriell/tiberius_benchmarking/paper/Embryophyta

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Eschscholzia_californica"
    "Freycinetia_multiflora"
    "Medicago_truncatula"
    "Mimulus_guttatus"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

TIB_GTF=${BENCH}/${species}/results/predictions/tiberius/tiberius_seqlen.gtf
DIAMOND_TSV=${TESTDIR}/${species}/proteins/diamond_hits.tsv
OUTDIR=${TESTDIR}/${species}/tiberius_protein_filter

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/tiberius_filtered.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: tiberius_filtered.gtf already exists"
    exit 0
fi

for f in "${TIB_GTF}" "${DIAMOND_TSV}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate gpu

echo "[$(date -Iseconds)] host=$(hostname) species=${species}"
echo "[$(date -Iseconds)] tib_gtf=${TIB_GTF}"
echo "[$(date -Iseconds)] diamond_tsv=${DIAMOND_TSV}"

python "${PROJDIR}/scripts/filter_orf_by_diamond_within_gene.py" \
    --orfs-gtf    "${TIB_GTF}" \
    --diamond-tsv "${DIAMOND_TSV}" \
    --out-dir     "${OUTDIR}" \
    --evalue      1e-5

# Expose output under the canonical name used by --tib-filtered-tmpl.
ln -sf "${OUTDIR}/orfs.filtered_soft_protein.gtf" "${OUTDIR}/tiberius_filtered.gtf"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/tiberius_filtered.gtf"
wc -l "${OUTDIR}/tiberius_filtered.gtf"
