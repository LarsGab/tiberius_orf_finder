#!/bin/bash
# TransDecoder v5.7.1 on the tpm1cov3len300-filtered StringTie
# assemblies. Sibling of slurm_benchmark_transdecoder1_vertebrates_test.sh
# but with the filtered StringTie as input.
#
# Filters the raw stringtie.gtf on the fly if the filtered variant
# is not yet present. Resume-skips species whose orfs.gtf already
# exists (safe to rerun with the full 0-5 array).
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/benchmark_orf_tools/transdecoder1_filt_tpm1cov3len300/
#     transcripts.fa                     (link to shared filtered fasta)
#     transcripts.fa.transdecoder.gff3
#     orfs.gtf                            (genomic CDS)
#     orfs.gtf.stats.txt
#
#SBATCH --job-name=td1_vt_f
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td1_vt_f_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td1_vt_f_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
TD1_DIR=/home/gabriell/programs/TransDecoder-TransDecoder-v5.7.1
FILT_TAG=filt_tpm1cov3len300
TOOL=transdecoder1_${FILT_TAG}

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

# gunzip genome.fa.gz -> genome.fa on the fly if only the compressed
# form is present (Archocentrus_centrarchus, Betta_splendens).
if [[ ! -s "${GENOME}" && -s "${GENOME}.gz" ]]; then
    echo "[$(date -Iseconds)] gunzipping ${GENOME}.gz"
    gunzip -k "${GENOME}.gz"
fi

if [[ ! -s "${GENOME}" || ! -s "${STRINGTIE_RAW}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing genome.fa or raw stringtie.gtf"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Filter raw StringTie GTF on the fly if the filtered variant is not
# yet present. Same parameters as slurm_annotate_vertebrates_test_filt_tpm1cov3len300.sh.
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

export PATH="${TD1_DIR}:${PATH}"

cd "${OUTDIR}"

echo "[$(date -Iseconds)] species=${species}"

# Shared filtered transcript FASTA (distinct from the unfiltered one so
# the two benchmarks don't overwrite each other's inputs).
SHARED_FA=${RESULTS_DIR}/${species}/benchmark_orf_tools/transcripts.${FILT_TAG}.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

"${TD1_DIR}/TransDecoder.LongOrfs" -t transcripts.fa
"${TD1_DIR}/TransDecoder.Predict"  -t transcripts.fa

LOCAL_GFF=${OUTDIR}/transcripts.fa.transdecoder.gff3
test -s "${LOCAL_GFF}" || { echo "TD1 produced no GFF3" >&2; exit 3; }

python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${LOCAL_GFF}" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        transdecoder

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
