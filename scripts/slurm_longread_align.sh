#!/bin/bash
# Align PacBio Iso-Seq reads to each species genome with minimap2
# (splice:hq preset) and sort/index the result with samtools.
#
# Input:  ${PROJDIR}/data/longread/${sp}/reads.fq.gz
# Output: ${RESULTS_DIR}/${sp}/longread/aligned.bam  (+ .bai)
#
# Run after slurm_longread_download.sh completes.
#
# Usage: sbatch [--dependency=afterok:<download_jobid>] scripts/slurm_longread_align.sh
#
#SBATCH --job-name=lr_align
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_align_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_align_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

declare -a SPECIES=(
    "Gallus_gallus"
    "Bos_taurus"
    "Zootoca_vivipara"
    "Betta_splendens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

READS=${PROJDIR}/data/longread/${species}/reads.fq.gz
GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
OUTDIR=${RESULTS_DIR}/${species}/longread

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

for f in "${READS}" "${GENOME}"; do
    [[ -s "${f}" ]] || { echo "missing input: ${f}" >&2; exit 2; }
done

# Use /projects scratch for samtools sort temp files (not /tmp which is per-node local).
TMPDIR_SORT=${PROJDIR}/tmp/lr_sort/${species}
mkdir -p "${TMPDIR_SORT}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# minimap2 may not be bundled in the orffinder env; try module system.
if ! command -v minimap2 &>/dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load minimap2 2>/dev/null || true
fi
command -v minimap2 || { echo "minimap2 not found — install with: micromamba install -n orffinder -c bioconda minimap2" >&2; exit 2; }

echo "[$(date -Iseconds)] minimap2=$(minimap2 --version 2>&1 || true)"
echo "[$(date -Iseconds)] samtools=$(samtools --version | head -1)"
echo "[$(date -Iseconds)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[$(date -Iseconds)] species=${species}"
echo "[$(date -Iseconds)] reads=${READS}"
echo "[$(date -Iseconds)] genome=${GENOME}"
echo "[$(date -Iseconds)] outdir=${OUTDIR}"

THREADS=${SLURM_CPUS_PER_TASK:-16}
SORT_THREADS=4  # dedicated to samtools sort; rest go to minimap2

minimap2 \
    -ax splice:hq \
    -t $(( THREADS - SORT_THREADS )) \
    --MD \
    "${GENOME}" \
    "${READS}" \
| samtools sort \
    -@ ${SORT_THREADS} \
    -m 4G \
    -T "${TMPDIR_SORT}/sort" \
    -o "${OUTDIR}/aligned.bam"

samtools index "${OUTDIR}/aligned.bam"

# Alignment summary.
samtools flagstat "${OUTDIR}/aligned.bam" | tee "${OUTDIR}/flagstat.txt"

echo "[$(date -Iseconds)] done -> ${OUTDIR}/aligned.bam"

rm -rf "${TMPDIR_SORT}"
