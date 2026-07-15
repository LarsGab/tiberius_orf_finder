#!/bin/bash
# Re-run local_orfs_to_genomic.py with --keep-antisense on every
# existing TD1 / TD2 / TD2-precise / GMST output (both unfiltered and
# tpm1cov3len300-filtered), saving each as orfs_keep_antisense.gtf
# (side by side with the default sense-only orfs.gtf) so the two can
# be compared without rerunning the tools themselves.
#
# Antisense ORFs are transcript-strand-'-' calls: they'd be encoded on
# the reverse complement of the assembled transcript. Tiberius never
# decodes those, so the default converter policy is to drop them (see
# orfs.gtf.stats.txt: `orfs_dropped_antisense`). This variant keeps
# them and re-projects onto the opposite genomic strand.
#
# Output per (species, tool[, filt]):
#   ${sp}/benchmark_orf_tools/<tool>[_filt_tpm1cov3len300]/
#       orfs_keep_antisense.gtf
#       orfs_keep_antisense.gtf.stats.txt
#
#SBATCH --job-name=reconv_as
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/reconv_as_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/reconv_as_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
CONV=${PROJDIR}/scripts/local_orfs_to_genomic.py

SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Bos_taurus
    Delphinapterus_leucas
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
)

# Each variant is a '|'-delimited quad:
#   <tool_subdir>|<stringtie_suffix>|<local_gff_basename>|<source_tag>
# stringtie_suffix is the piece between "stringtie" and ".gtf" — empty
# for the unfiltered input. gff_basename is passed to `find -name`; the
# first hit within the tool subdir is used, so we also pick up the
# td2_workdir/ fallback for TD2 releases that write there.
declare -a VARIANTS=(
    # unfiltered variants (input: stringtie.gtf)
    "transdecoder1||transcripts.fa.transdecoder.gff3|transdecoder"
    "transdecoder2||transcripts.fa.TD2.gff3|td2"
    "transdecoder2_precise||transcripts.fa.TD2.gff3|td2_precise"
    "gmst||gmst.gff|gmst"
    # tpm1cov3len300-filtered variants (input: stringtie.filt_tpm1cov3len300.gtf)
    "transdecoder1_filt_tpm1cov3len300|.filt_tpm1cov3len300|transcripts.fa.transdecoder.gff3|transdecoder"
    "transdecoder2_filt_tpm1cov3len300|.filt_tpm1cov3len300|transcripts.fa.TD2.gff3|td2"
    "transdecoder2_precise_filt_tpm1cov3len300|.filt_tpm1cov3len300|transcripts.fa.TD2.gff3|td2_precise"
    "gmst_filt_tpm1cov3len300|.filt_tpm1cov3len300|gmst.gff|gmst"
)

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

n_ok=0
n_skip=0
n_fail=0

for sp in "${SPECIES[@]}"; do
    for v in "${VARIANTS[@]}"; do
        IFS='|' read -r subdir st_suffix gff_name src_tag <<< "${v}"

        TOOL_DIR=${RESULTS_DIR}/${sp}/benchmark_orf_tools/${subdir}
        STRINGTIE=${RESULTS_DIR}/${sp}/stringtie/stringtie${st_suffix}.gtf
        OUT_GTF=${TOOL_DIR}/orfs_keep_antisense.gtf

        # First-hit glob so we also find td2_workdir/transcripts.fa.TD2.gff3.
        LOCAL_GFF=$(find "${TOOL_DIR}" -type f -name "${gff_name}" 2>/dev/null | head -n1 || true)

        if [[ -z "${LOCAL_GFF}" || ! -s "${LOCAL_GFF}" || ! -s "${STRINGTIE}" ]]; then
            echo "[skip] ${sp}/${subdir}: missing local gff or stringtie"
            n_skip=$((n_skip + 1))
            continue
        fi

        echo "[run] ${sp}/${subdir}  <- $(basename "${LOCAL_GFF}")"
        if python "${CONV}" \
            --local-gff     "${LOCAL_GFF}" \
            --stringtie-gtf "${STRINGTIE}" \
            --out-gtf       "${OUT_GTF}" \
            --source        "${src_tag}" \
            --keep-antisense; then
            n_ok=$((n_ok + 1))
        else
            echo "[FAIL] ${sp}/${subdir}" >&2
            n_fail=$((n_fail + 1))
        fi
    done
done

echo
echo "[$(date -Iseconds)] summary: ok=${n_ok} skip=${n_skip} fail=${n_fail}"
