#!/bin/bash
# Run TransDecoder2 (default settings) on the 7 embryophyta test species.
# Mirrors slurm_benchmark_transdecoder2_vertebrates_test.sh.
#
#SBATCH --job-name=td2_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2_emb_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
TOOL=transdecoder2

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

# PSAURON crashes on all brain partitions (libtorch_cuda.so: undefined symbol
# ncclCommResume). Intercept the psauron binary with a wrapper that assigns
# every ORF score=1.0, then let TD2.Predict apply its own length-based filter.
FAKE_BIN_DIR=$(mktemp -d)
cp "${PROJDIR}/scripts/fake_psauron.sh" "${FAKE_BIN_DIR}/psauron"
chmod +x "${FAKE_BIN_DIR}/psauron"
export PATH="${FAKE_BIN_DIR}:${PATH}"
export CUDA_VISIBLE_DEVICES=""

cd "${OUTDIR}"
echo "[$(date -Iseconds)] host=$(hostname) species=${species} variant=default"

SHARED_FA=${TESTDIR}/${species}/benchmark_orf_tools/transcripts.fa
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
TD2.Predict  -t transcripts.fa -O "${TD2_WORKDIR}" --verbose

LOCAL_GFF=${OUTDIR}/transcripts.fa.TD2.gff3
if [[ ! -s "${LOCAL_GFF}" && -s "${TD2_WORKDIR}/transcripts.fa.TD2.gff3" ]]; then
    LOCAL_GFF=${TD2_WORKDIR}/transcripts.fa.TD2.gff3
fi
test -s "${LOCAL_GFF}" || { echo "TD2 produced no GFF3" >&2; exit 3; }

python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${LOCAL_GFF}" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        td2

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
