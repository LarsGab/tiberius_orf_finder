#!/bin/bash
# Run GeneMarkS-T (gmst.pl) on the vertebrates_test species and project
# ORFs to a genomic GTF for gffcompare.
#
# Uses the native gmst.pl install at /home_old/gabriell/programs/ETP/bin/gmst/
# rather than the copy bundled inside braker3.sif (which lives at an
# uncooperative absolute path inside the container). The native binary
# is the same tool from the same GeneMark-ETP release — verified to run
# successfully via probe job 7537269 (2026-07-06).
#
# gmst.pl reads its HMM model (MetaGeneMark_v1.mod, par_EST.default)
# from CWD by default, so we invoke it from the ETP gmst/ dir with
# absolute input/output paths. gmst.log + intermediate .lst files then
# also land in that dir — we mv the log back to ${OUTDIR} for
# per-species bookkeeping.
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/benchmark_orf_tools/gmst/
#     transcripts.fa       (link into shared per-species fasta)
#     gmst.gff             (raw gmst.pl output, tx coords)
#     gmst.log             (gmst.pl stderr)
#     orfs.gtf             (genomic CDS)
#     orfs.gtf.stats.txt
#
#SBATCH --job-name=gmst_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gmst_vt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gmst_vt_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
GMST_DIR=/home_old/gabriell/programs/ETP/bin/gmst
GMST_PL=${GMST_DIR}/gmst.pl
TOOL=gmst

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
    exit 0
fi

test -x "${GMST_PL}" || { echo "missing gmst.pl: ${GMST_PL}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${OUTDIR}"
echo "[$(date -Iseconds)] species=${species}"

# 1. Shared transcript FASTA.
SHARED_FA=${RESULTS_DIR}/${species}/benchmark_orf_tools/transcripts.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

# 2. gmst.pl. --strand direct = only sense strand of each transcript.
# --format GFF (native GeneMark GFF2). --output pins the emitted GFF
# to an absolute path so we don't have to chdir into GMST_DIR (which
# would pollute the shared tool dir with per-species .lst files).
# We do have to bring the model files (MetaGeneMark_v1.mod,
# par_EST.default) into CWD though — symlink them so gmst.pl finds
# them without needing --parname / --format overrides.
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

# 3. Project transcript-coord CDS rows to genomic GTF.
python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
    --local-gff     "${OUTDIR}/gmst.gff" \
    --stringtie-gtf "${STRINGTIE}" \
    --out-gtf       "${OUTDIR}/orfs.gtf" \
    --source        gmst

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
