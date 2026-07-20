#!/bin/bash
# Run TransDecoder v1 (TD1) on the 7 embryophyta test species.
# Mirrors slurm_benchmark_transdecoder1_vertebrates_test.sh.
#
#SBATCH --job-name=td1_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td1_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td1_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
TD1_DIR=/home/gabriell/programs/TransDecoder-TransDecoder-v5.7.1
TOOL=transdecoder1

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Brachypodium_distachyon"
    "Eschscholzia_californica"
    "Freycinetia_multiflora"
    "Medicago_truncatula"
    "Mimulus_guttatus"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TESTDIR}/${species}/assembly/genome.fa
STRINGTIE=${TESTDIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${TESTDIR}/${species}/benchmark_orf_tools/${TOOL}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: orfs.gtf already exists"
    exit 0
fi

if [[ ! -s "${STRINGTIE}" || ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing inputs"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

export PATH="${TD1_DIR}:${PATH}"

cd "${OUTDIR}"
echo "[$(date -Iseconds)] host=$(hostname) species=${species}"

SHARED_FA=${TESTDIR}/${species}/benchmark_orf_tools/transcripts.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
# Filter zero-length sequences — TD1 aborts on empty records.
FILTERED_FA=${OUTDIR}/transcripts_filtered.fa
awk '/^>/{if(seq!="")print prev"\n"seq; prev=$0; seq=""; next}
     {seq=seq$0}
     END{if(seq!="")print prev"\n"seq}' "${SHARED_FA}" > "${FILTERED_FA}"
# Use a distinct symlink name so TD1 creates a fresh transdecoder_dir
# (avoids stale files from any prior failed run on transcripts.fa).
ln -sf "${FILTERED_FA}" transcripts_filt.fa
n_tx=$(grep -c '^>' transcripts_filt.fa || true)
echo "  transcripts_filt.fa: ${n_tx} records (after empty-seq filter)"

"${TD1_DIR}/TransDecoder.LongOrfs" -t transcripts_filt.fa
"${TD1_DIR}/TransDecoder.Predict"  -t transcripts_filt.fa

LOCAL_GFF=${OUTDIR}/transcripts_filt.fa.transdecoder.gff3
test -s "${LOCAL_GFF}" || { echo "TD1 produced no GFF3" >&2; exit 3; }

python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${LOCAL_GFF}" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        transdecoder1

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
