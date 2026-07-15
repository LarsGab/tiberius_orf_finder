#!/bin/bash
# TD2 (TransDecoder2) --precise on the tpm1cov3len300-filtered
# StringTie assemblies. Sibling of the default variant + the unfiltered
# precise script. CPU only (cuDNN mismatch on vision GPUs).
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/benchmark_orf_tools/transdecoder2_precise_filt_tpm1cov3len300/
#     transcripts.fa
#     transcripts.fa.TD2.gff3
#     orfs.gtf
#     orfs.gtf.stats.txt
#     td2_workdir/
#
#SBATCH --job-name=td2p_vt_f
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2p_vt_f_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2p_vt_f_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
TOOL=transdecoder2_precise_${FILT_TAG}

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
STRINGTIE_RAW=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
STRINGTIE=${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${RESULTS_DIR}/${species}/benchmark_orf_tools/${TOOL}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.gtf already exists (set FORCE=1 to rerun)"
    exit 0
fi

if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" || ! -s "${STRINGTIE_RAW}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing genome.fa or raw stringtie.gtf"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
# `td2` is the dedicated CPU-only env — see the default variant script
# for the rationale.
micromamba activate td2

if [[ ! -s "${STRINGTIE}" ]]; then
    echo "[$(date -Iseconds)] filtering stringtie -> ${STRINGTIE}"
    python "${PROJDIR}/scripts/filter_stringtie_gtf.py" \
        --in-gtf       "${STRINGTIE_RAW}" \
        --out-gtf      "${STRINGTIE}" \
        --out-tsv      "${RESULTS_DIR}/${species}/stringtie/stringtie.${FILT_TAG}.decisions.tsv" \
        --min-length   300 \
        --min-cov      3.0 \
        --min-tpm      1.0 \
        --long-length  3000 \
        --min-tpm-long 0.5
fi
test -s "${STRINGTIE}" || { echo "filter produced no GTF: ${STRINGTIE}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=""

cd "${OUTDIR}"
echo "[$(date -Iseconds)] species=${species} variant=precise filter=${FILT_TAG}"

SHARED_FA=${RESULTS_DIR}/${species}/benchmark_orf_tools/transcripts.${FILT_TAG}.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

TD2_WORKDIR=${OUTDIR}/td2_workdir
rm -rf "${TD2_WORKDIR}"

TD2.LongOrfs -t transcripts.fa -O "${TD2_WORKDIR}" -@ "${SLURM_CPUS_PER_TASK}"
TD2.Predict  -t transcripts.fa -O "${TD2_WORKDIR}" --precise --verbose

LOCAL_GFF=${OUTDIR}/transcripts.fa.TD2.gff3
if [[ ! -s "${LOCAL_GFF}" && -s "${TD2_WORKDIR}/transcripts.fa.TD2.gff3" ]]; then
    LOCAL_GFF=${TD2_WORKDIR}/transcripts.fa.TD2.gff3
fi
test -s "${LOCAL_GFF}" || {
    echo "TD2 produced no GFF3. Looked in:" >&2
    echo "  ${OUTDIR}/transcripts.fa.TD2.gff3" >&2
    echo "  ${TD2_WORKDIR}/transcripts.fa.TD2.gff3" >&2
    ls -la "${OUTDIR}" "${TD2_WORKDIR}" >&2 || true
    exit 3
}

python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${LOCAL_GFF}" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        td2_precise

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
