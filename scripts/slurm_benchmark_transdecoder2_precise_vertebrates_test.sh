#!/bin/bash
# Run TD2 (TransDecoder2) with --precise on the 4 vertebrates_test
# species and project ORFs to a genomic GTF for gffcompare.
#
# --precise on TD2.Predict pairs with the same flag on LongOrfs (which
# is what the prior Phaeodactylum benchmark used); it sets stricter
# minimum-protein-length thresholds inside the caller.
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/benchmark_orf_tools/transdecoder2_precise/
#     transcripts.fa
#     transcripts.fa.TD2.gff3
#     transcripts.fa.TD2.pep
#     orfs.gtf
#     orfs.gtf.stats.txt
#
#SBATCH --job-name=td2p_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2p_vt_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/td2p_vt_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
TOOL=transdecoder2_precise

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

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# TD2 PSAURON's CNN fails on brain `vision` GPUs with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Force CPU inference.
export CUDA_VISIBLE_DEVICES=""

cd "${OUTDIR}"
echo "[$(date -Iseconds)] species=${species} variant=precise"

SHARED_FA=${RESULTS_DIR}/${species}/benchmark_orf_tools/transcripts.fa
if [[ ! -s "${SHARED_FA}" ]]; then
    mkdir -p "$(dirname "${SHARED_FA}")"
    gffread -w "${SHARED_FA}" -g "${GENOME}" "${STRINGTIE}"
fi
ln -sf "${SHARED_FA}" transcripts.fa
n_tx=$(grep -c '^>' transcripts.fa)
echo "  transcripts.fa: ${n_tx} records"

# TD2.LongOrfs refuses to overwrite an existing output dir; nuke stale.
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
