#!/bin/bash
# Run StringTie on the Homo_sapiens VARUS BAM, then apply the
# tpm1cov3len300 filter used for all other vertebrates_test species.
#
# Also creates assembly/ symlinks from top-level genome.fa / annot_cds.gff
# if those files exist there but not yet under assembly/.
#
# Input:
#   results/vertebrates_test/Homo_sapiens/varus/VARUS.bam
#   results/vertebrates_test/Homo_sapiens/assembly/genome.fa
#
# Output:
#   results/vertebrates_test/Homo_sapiens/stringtie/stringtie.gtf
#   results/vertebrates_test/Homo_sapiens/stringtie/stringtie.filt_tpm1cov3len300.gtf
#
#SBATCH --job-name=st_homo
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/st_homo_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/st_homo_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SP_DIR=${PROJDIR}/results/vertebrates_test/Homo_sapiens

FILT_TAG=filt_tpm1cov3len300
BAM=${SP_DIR}/varus/VARUS.bam
GENOME=${SP_DIR}/assembly/genome.fa
ST_DIR=${SP_DIR}/stringtie
ST_GTF=${ST_DIR}/stringtie.gtf
ST_FILT=${ST_DIR}/stringtie.${FILT_TAG}.gtf

mkdir -p "${PROJDIR}/logs" "${ST_DIR}"

echo "[$(date -Iseconds)] Homo_sapiens StringTie pipeline"

# ── Stage assembly/ symlinks from top-level if needed ────────────────────────
mkdir -p "${SP_DIR}/assembly"
for fname in genome.fa annot_cds.gff; do
    if [[ -s "${SP_DIR}/${fname}" && ! -e "${SP_DIR}/assembly/${fname}" ]]; then
        ln -sf "../${fname}" "${SP_DIR}/assembly/${fname}"
        echo "[$(date -Iseconds)] linked ${fname} -> assembly/${fname}"
    fi
done

[[ -s "${BAM}" ]]    || { echo "ERROR: VARUS.bam not found: ${BAM}" >&2; exit 2; }
[[ -s "${GENOME}" ]] || { echo "ERROR: genome.fa not found: ${GENOME}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── StringTie ─────────────────────────────────────────────────────────────────
if [[ -s "${ST_GTF}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] reusing ${ST_GTF}"
else
    TMP=$(mktemp -d -p "${TMPDIR:-/tmp}" stringtie.XXXXXX)
    trap 'rm -rf "${TMP}"' EXIT

    SORTED_BAM=${TMP}/sorted.bam
    echo "[$(date -Iseconds)] sorting BAM ..."
    samtools sort -@ "${SLURM_CPUS_PER_TASK:-16}" -o "${SORTED_BAM}" "${BAM}"

    echo "[$(date -Iseconds)] running StringTie ..."
    stringtie "${SORTED_BAM}" \
        -o "${ST_GTF}" \
        -p "${SLURM_CPUS_PER_TASK:-16}"

    [[ -s "${ST_GTF}" ]] || { echo "ERROR: stringtie produced no GTF" >&2; exit 2; }
    echo "[$(date -Iseconds)] stringtie done: $(wc -l < "${ST_GTF}") lines"
fi

# ── tpm1cov3len300 filter ─────────────────────────────────────────────────────
if [[ -s "${ST_FILT}" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] reusing ${ST_FILT}"
else
    echo "[$(date -Iseconds)] filtering stringtie -> ${ST_FILT}"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf       "${ST_GTF}" \
        --out-gtf      "${ST_FILT}" \
        --out-tsv      "${ST_DIR}/stringtie.${FILT_TAG}.decisions.tsv" \
        --min-length   300 \
        --min-cov      3.0 \
        --min-tpm      1.0 \
        --long-length  3000 \
        --min-tpm-long 0.5
fi

[[ -s "${ST_FILT}" ]] || { echo "ERROR: filtered GTF is empty" >&2; exit 2; }

N_RAW=$(awk '$3=="transcript"' "${ST_GTF}"  | wc -l || true)
N_FILT=$(awk '$3=="transcript"' "${ST_FILT}" | wc -l || true)
echo "[$(date -Iseconds)] done: ${N_RAW} raw -> ${N_FILT} filtered transcripts"
