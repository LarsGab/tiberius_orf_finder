#!/bin/bash
# Run GeneMarkS-T on Aspergillus_fumigatus, Candida_parapsilosis, and
# Fusarium_redolens in training_fungi_v2.
# Mirrors slurm_benchmark_gmst_fungi_test.sh.
#
#SBATCH --job-name=gmst_tv2
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gmst_tv2_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gmst_tv2_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
GMST_DIR=/home_old/gabriell/programs/ETP/bin/gmst
GMST_PL=${GMST_DIR}/gmst.pl
TOOL=gmst

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

test -x "${GMST_PL}" || { echo "missing gmst.pl: ${GMST_PL}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${OUTDIR}"
echo "[$(date -Iseconds)] host=$(hostname) species=${species}"

SHARED_FA=${TRAINDIR}/${species}/benchmark_orf_tools/transcripts.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

ln -sf "${GMST_DIR}/MetaGeneMark_v1.mod" MetaGeneMark_v1.mod
ln -sf "${GMST_DIR}/par_EST.default"     par_EST.default

set +e
"${GMST_PL}" \
    --strand direct \
    --format GFF \
    --output "${OUTDIR}/gmst.gff" \
    "${OUTDIR}/transcripts.fa" \
    > "${OUTDIR}/gmst.log" 2>&1
gmst_rc=$?
set -e

if [[ ${gmst_rc} -ne 0 || ! -s "${OUTDIR}/gmst.gff" ]]; then
    echo "gmst.pl failed (rc=${gmst_rc}) — tail of gmst.log:" >&2
    tail -60 "${OUTDIR}/gmst.log" >&2 || true
    exit 3
fi

python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${OUTDIR}/gmst.gff" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        gmst

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
