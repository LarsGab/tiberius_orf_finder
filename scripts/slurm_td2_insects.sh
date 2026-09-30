#!/bin/bash
# TransDecoder2 (default) on filtered StringTie assemblies for insects_test_v2.
# One SLURM array task per species. CPU-only (see vertebrates TD2 script for
# the CUDA-VISIBLE-DEVICES rationale).
#
#SBATCH --job-name=td2_ins
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2_ins_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2_ins_%A_%a.err

set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${RUNTIME_TSV:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
SCRIPTS_DIR=${PROJDIR}/scripts
source "${SCRIPTS_DIR}/lib/log_runtime.sh"
export RUNTIME_TSV
CLADE=insects_test_v2
FILT_TAG=filt_tpm1cov3len300
TOOL=transdecoder2_${FILT_TAG}

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
SPECIES=$(sed -n "${TASK_ID}p" "${SPECIES_FILE}")

SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
STRINGTIE=${SP_DIR}/stringtie/stringtie.${FILT_TAG}.gtf
OUTDIR=${SP_DIR}/benchmark_orf_tools/${TOOL}

[[ -s "${GENOME}" && -s "${STRINGTIE}" ]] || { echo "SKIP ${SPECIES}: missing inputs" >&2; exit 0; }
mkdir -p "${OUTDIR}"

if [[ -s "${OUTDIR}/orfs.gtf" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP: orfs.gtf already exists"; exit 0
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate td2
export CUDA_VISIBLE_DEVICES=""

SHARED_FA=${SP_DIR}/benchmark_orf_tools/transcripts.${FILT_TAG}.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    run_timed "${CLADE}" "${SPECIES}" td2 gffread -- \
        gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi

cd "${OUTDIR}"
ln -sf "${SHARED_FA}" transcripts.fa

TD2_WORKDIR=${OUTDIR}/td2_workdir
rm -rf "${TD2_WORKDIR}"

run_timed "${CLADE}" "${SPECIES}" td2 longorfs -- \
    TD2.LongOrfs -t transcripts.fa -O "${TD2_WORKDIR}" -@ "${SLURM_CPUS_PER_TASK}"
run_timed "${CLADE}" "${SPECIES}" td2 predict -- \
    TD2.Predict  -t transcripts.fa -O "${TD2_WORKDIR}" --verbose

LOCAL_GFF=${OUTDIR}/transcripts.fa.TD2.gff3
[[ -s "${LOCAL_GFF}" ]] || LOCAL_GFF=${TD2_WORKDIR}/transcripts.fa.TD2.gff3
[[ -s "${LOCAL_GFF}" ]] || { echo "TD2 produced no GFF3" >&2; exit 3; }

micromamba activate orffinder
run_timed "${CLADE}" "${SPECIES}" td2 local_to_genomic -- \
    python "${PROJDIR}/scripts/local_orfs_to_genomic.py" \
        --local-gff     "${LOCAL_GFF}" \
        --stringtie-gtf "${STRINGTIE}" \
        --out-gtf       "${OUTDIR}/orfs.gtf" \
        --source        td2

echo "[$(date -Iseconds)] done -> ${OUTDIR}/orfs.gtf"
