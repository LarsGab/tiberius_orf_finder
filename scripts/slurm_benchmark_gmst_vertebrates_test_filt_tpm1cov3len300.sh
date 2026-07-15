#!/bin/bash
# GeneMarkS-T (gmst.pl) on the tpm1cov3len300-filtered StringTie
# assemblies. Native gmst.pl at /home_old/gabriell/programs/ETP/bin/gmst/
# (same as the unfiltered variant — no singularity needed).
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/benchmark_orf_tools/gmst_filt_tpm1cov3len300/
#     transcripts.fa
#     gmst.gff
#     gmst.log
#     orfs.gtf
#     orfs.gtf.stats.txt
#
#SBATCH --job-name=gmst_vt_f
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gmst_vt_f_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/gmst_vt_f_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
GMST_DIR=/home_old/gabriell/programs/ETP/bin/gmst
GMST_PL=${GMST_DIR}/gmst.pl
FILT_TAG=filt_tpm1cov3len300
TOOL=gmst_${FILT_TAG}

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

test -x "${GMST_PL}" || { echo "missing gmst.pl: ${GMST_PL}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

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

cd "${OUTDIR}"
echo "[$(date -Iseconds)] species=${species} filter=${FILT_TAG}"

SHARED_FA=${RESULTS_DIR}/${species}/benchmark_orf_tools/transcripts.${FILT_TAG}.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

# gmst.pl needs model files in CWD. Symlink them.
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
