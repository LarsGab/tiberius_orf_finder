#!/bin/bash
# Run TransDecoder v5.7.1 (TransDecoder1) on the 4 vertebrates_test
# species and project ORFs to a genomic GTF for gffcompare.
#
# Input (same as Tiberius): the transcript FASTA gffread produces from
#   ${RESULTS_DIR}/<sp>/stringtie/stringtie.gtf + ${RESULTS_DIR}/<sp>/assembly/genome.fa
# TransDecoder is run without any homology evidence — LongOrfs then Predict
# with defaults. The prior Phaeodactylum benchmark run took the same
# stance (see /home_old/gabriell/evidence_pipeline/tiberius_orf_finder/
# Phaeodactylum_tricornutum/transdecoder/).
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/benchmark_orf_tools/transdecoder1/
#     transcripts.fa                     (linked / rebuilt if missing)
#     transcripts.fa.transdecoder.gff3   (raw TD1 output, tx coords)
#     transcripts.fa.transdecoder.pep    (raw TD1 output)
#     orfs.gtf                            (genomic CDS via local_orfs_to_genomic.py)
#     orfs.gtf.stats.txt                  (per-species conversion counters)
#
#SBATCH --job-name=td1_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td1_vt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td1_vt_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
TD1_DIR=/home/gabriell/programs/TransDecoder-TransDecoder-v5.7.1
TOOL=transdecoder1

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
STRINGTIE=${RESULTS_DIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${RESULTS_DIR}/${species}/benchmark_orf_tools/${TOOL}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}"

if [[ ! -s "${STRINGTIE}" || ! -s "${GENOME}" ]]; then
    echo "[$(date -Iseconds)] SKIP ${species}: missing inputs"
    echo "  stringtie=${STRINGTIE}"
    echo "  genome=${GENOME}"
    exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Add TransDecoder v1 to PATH (dominates the newer TD2 shims in the env
# because we use the fully-qualified TD1 binaries below, but this also
# helps if PerlLib lookup relies on PATH ordering).
export PATH="${TD1_DIR}:${PATH}"

cd "${OUTDIR}"

echo "[$(date -Iseconds)] species=${species}"

# 1. Shared transcript FASTA (built once, reused by TD2 + gmst too).
SHARED_FA=${RESULTS_DIR}/${species}/benchmark_orf_tools/transcripts.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

# 2. TransDecoder — two-step, no homology.
"${TD1_DIR}/TransDecoder.LongOrfs" -t transcripts.fa
"${TD1_DIR}/TransDecoder.Predict"  -t transcripts.fa

# TD1 writes outputs into CWD:
#   transcripts.fa.transdecoder.gff3
#   transcripts.fa.transdecoder.bed
#   transcripts.fa.transdecoder.pep
#   transcripts.fa.transdecoder.cds
LOCAL_GFF=${OUTDIR}/transcripts.fa.transdecoder.gff3
test -s "${LOCAL_GFF}" || { echo "TD1 produced no GFF3" >&2; exit 3; }

# 3. Project transcript-coord CDS to genomic GTF (drops antisense ORFs).
python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${LOCAL_GFF}" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        transdecoder

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
