#!/bin/bash
# Align PacBio Iso-Seq reads for the 4 embryophyta long-read species
# with minimap2 (splice:hq preset) and sort/index with samtools.
#
# Input:  ${PROJDIR}/data/longread_embryophyta/${sp}/reads.fq.gz
# Output: ${TESTDIR}/${sp}/longread/aligned.bam  (+ .bai, flagstat.txt)
#
#SBATCH --job-name=lr_emb_align
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_emb_align_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_emb_align_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Eschscholzia_californica"
    "Medicago_truncatula"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

READS=${PROJDIR}/data/longread_embryophyta/${species}/reads.fq.gz
GENOME=${TESTDIR}/${species}/assembly/genome.fa
OUTDIR=${TESTDIR}/${species}/longread

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

for f in "${READS}" "${GENOME}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

TMPDIR_SORT=${PROJDIR}/tmp/lr_emb_sort/${species}
mkdir -p "${TMPDIR_SORT}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

if ! command -v minimap2 &>/dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load minimap2 2>/dev/null || true
fi
command -v minimap2 || { echo "minimap2 not found" >&2; exit 2; }

echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] minimap2=$(minimap2 --version 2>&1 || true)"

THREADS=${SLURM_CPUS_PER_TASK:-16}
SORT_THREADS=4

minimap2 \
    -ax splice:hq \
    -t $(( THREADS - SORT_THREADS )) \
    --MD \
    "${GENOME}" \
    "${READS}" \
| samtools sort \
    -@ ${SORT_THREADS} \
    -m 2G \
    -T "${TMPDIR_SORT}/sort" \
    -o "${OUTDIR}/aligned.bam"

samtools index "${OUTDIR}/aligned.bam"
samtools flagstat "${OUTDIR}/aligned.bam" | tee "${OUTDIR}/flagstat.txt"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/aligned.bam"
rm -rf "${TMPDIR_SORT}"
