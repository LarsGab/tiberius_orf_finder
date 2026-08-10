#!/bin/bash
# Run TransDecoder2 (--precise) on Aspergillus_fumigatus, Candida_parapsilosis,
# and Fusarium_redolens in training_fungi_v2.
# Mirrors slurm_benchmark_transdecoder2_precise_fungi_test.sh.
#
#SBATCH --job-name=td2p_tv2
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2p_tv2_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2p_tv2_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
TOOL=transdecoder2_precise

declare -a SPECIES=(
    "Aspergillus_fumigatus"
    "Candida_parapsilosis"
    "Fusarium_redolens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME=${TRAINDIR}/${species}/assembly/genome.fa
STRINGTIE=${TRAINDIR}/${species}/stringtie/stringtie.gtf
OUTDIR=${TRAINDIR}/${species}/benchmark_orf_tools/${TOOL}

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

# Fake psauron to bypass broken libtorch_cuda.so (ncclCommResume missing).
FAKE_BIN_DIR=$(mktemp -d)
cp "${PROJDIR}/scripts/fake_psauron.sh" "${FAKE_BIN_DIR}/psauron"
chmod +x "${FAKE_BIN_DIR}/psauron"
export PATH="${FAKE_BIN_DIR}:${PATH}"
export CUDA_VISIBLE_DEVICES=""

cd "${OUTDIR}"
echo "[$(date -Iseconds)] host=$(hostname) species=${species} variant=precise"

SHARED_FA=${TRAINDIR}/${species}/benchmark_orf_tools/transcripts.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
# Drop empty seqs, uppercase, replace non-ACGTN with N.
# TD2.LongOrfs crashes on IUPAC ambiguity codes in some gffread outputs.
python3 -c "
import re, sys
hdr = None; seq = []
for line in open(sys.argv[1]):
    line = line.rstrip()
    if line.startswith('>'):
        if hdr and seq:
            s = re.sub(r'[^ACGTN]', 'N', ''.join(seq).upper())
            if s: print(hdr); print(s)
        hdr = line; seq = []
    else:
        seq.append(line)
if hdr and seq:
    s = re.sub(r'[^ACGTN]', 'N', ''.join(seq).upper())
    if s: print(hdr); print(s)
" "${SHARED_FA}" > transcripts_clean.fa
n_tx=$(grep -c '^>' transcripts_clean.fa)
echo "  transcripts_clean.fa: ${n_tx} records (sanitized)"

TD2_WORKDIR=${OUTDIR}/td2_workdir
rm -rf "${TD2_WORKDIR}"

TD2.LongOrfs -t transcripts_clean.fa -O "${TD2_WORKDIR}" -@ "${SLURM_CPUS_PER_TASK}"
TD2.Predict  -t transcripts_clean.fa -O "${TD2_WORKDIR}" --precise --verbose

LOCAL_GFF=${OUTDIR}/transcripts_clean.fa.TD2.gff3
if [[ ! -s "${LOCAL_GFF}" && -s "${TD2_WORKDIR}/transcripts_clean.fa.TD2.gff3" ]]; then
    LOCAL_GFF=${TD2_WORKDIR}/transcripts_clean.fa.TD2.gff3
fi
test -s "${LOCAL_GFF}" || { echo "TD2 produced no GFF3" >&2; exit 3; }

python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${LOCAL_GFF}" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        td2_precise

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
