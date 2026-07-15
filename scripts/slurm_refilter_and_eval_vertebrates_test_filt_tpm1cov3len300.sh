#!/bin/bash
# Re-filter Tiberius `orfs.gtf` for vertebrates_test with the new
# near-subsequence tolerances in filter_subsequence_predictions.py, then
# re-run gffcompare against the reference for all 5 ORF tools. TD1/TD2/
# GMST inputs are unchanged; only the Tiberius column changes.
#
# Output layout:
#   ${RESULTS_DIR}/<sp>/annotate_epoch_74_filt_tpm1cov3len300/
#     orfs.filtered.${TOL_TAG}.gtf
#     dropped_subsequences.${TOL_TAG}.tsv
#   ${RESULTS_DIR}/benchmark_orf_tools_filt_tpm1cov3len300_${TOL_TAG}/
#     <sp>/gffcompare/<tool>.stats
#     accuracy_table.tsv
#
# Tolerance settings are passed via env vars (defaults in the script):
#   TERM_OH=90 SPLICE_SHIFT=6 ALLOW_SKIP=1 sbatch <this>
#
#SBATCH --job-name=refilt_eval_f
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/refilt_eval_f_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/refilt_eval_f_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
FILT_TAG=filt_tpm1cov3len300
TIB_DIR_NAME=annotate_epoch_74_${FILT_TAG}

TERM_OH=${TERM_OH:-90}
SPLICE_SHIFT=${SPLICE_SHIFT:-6}
ALLOW_SKIP=${ALLOW_SKIP:-1}

TOL_TAG="oh${TERM_OH}_ss${SPLICE_SHIFT}_skip${ALLOW_SKIP}"
OUT_ROOT=${RESULTS_DIR}/benchmark_orf_tools_${FILT_TAG}_${TOL_TAG}

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)
TOOLS=(tiberius transdecoder1 transdecoder2 transdecoder2_precise gmst)

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

SKIP_FLAG=""
if [[ "${ALLOW_SKIP}" == "1" ]]; then
    SKIP_FLAG="--allow-exon-skip"
fi

echo "[$(date -Iseconds)] tolerances: overhang=${TERM_OH} shift=${SPLICE_SHIFT} allow_skip=${ALLOW_SKIP}"
echo "[$(date -Iseconds)] out_root=${OUT_ROOT}"

# 1. Re-filter Tiberius orfs.gtf per species.
for sp in "${SPECIES[@]}"; do
    TIB_DIR=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}
    RAW=${TIB_DIR}/orfs.gtf
    FILTERED=${TIB_DIR}/orfs.filtered.${TOL_TAG}.gtf
    REPORT=${TIB_DIR}/dropped_subsequences.${TOL_TAG}.tsv
    if [[ ! -s "${RAW}" ]]; then
        echo "[skip refilter] ${sp}: no ${RAW}"
        continue
    fi
    echo "[$(date -Iseconds)] refilter ${sp} -> ${FILTERED}"
    python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
        --orfs-gtf            "${RAW}" \
        --out-gtf             "${FILTERED}" \
        --report-tsv          "${REPORT}" \
        --terminal-overhang-nt "${TERM_OH}" \
        --splice-shift-nt      "${SPLICE_SHIFT}" \
        ${SKIP_FLAG}
done

# 2. gffcompare per (species, tool).
TABLE=${OUT_ROOT}/accuracy_table.tsv
printf "species\ttool\tbase_S\tbase_P\texon_S\texon_P\ttranscript_S\ttranscript_P\tlocus_S\tlocus_P\n" > "${TABLE}"

for sp in "${SPECIES[@]}"; do
    SP_OUT=${OUT_ROOT}/${sp}/gffcompare
    mkdir -p "${SP_OUT}"

    REF=${RESULTS_DIR}/${sp}/assembly/annot_cds.gff
    if [[ ! -s "${REF}" ]]; then
        echo "[skip eval] ${sp}: missing reference ${REF}"
        continue
    fi

    declare -A GTF
    GTF[tiberius]=${RESULTS_DIR}/${sp}/${TIB_DIR_NAME}/orfs.filtered.${TOL_TAG}.gtf
    GTF[transdecoder1]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/transdecoder1_${FILT_TAG}/orfs.gtf
    GTF[transdecoder2]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/transdecoder2_${FILT_TAG}/orfs.gtf
    GTF[transdecoder2_precise]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/transdecoder2_precise_${FILT_TAG}/orfs.gtf
    GTF[gmst]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/gmst_${FILT_TAG}/orfs.gtf

    for tool in "${TOOLS[@]}"; do
        gtf=${GTF[$tool]}
        if [[ ! -s "${gtf}" ]]; then
            echo "[skip] ${sp}/${tool}: missing ${gtf}"
            continue
        fi
        PREFIX=${SP_OUT}/${tool}
        awk -F '\t' '$3 == "CDS"' "${gtf}" > "${PREFIX}_cds.gff"
        gffcompare --strict-match -e 3 -T -r "${REF}" -o "${PREFIX}" "${PREFIX}_cds.gff" \
            > "${PREFIX}.gffcompare.log" 2>&1 || true

        STATS=${PREFIX}.stats
        if [[ ! -s "${STATS}" ]]; then
            echo "[warn] ${sp}/${tool}: no ${STATS}"
            continue
        fi

        base=$(awk '/Base level:/       {print $3"\t"$5; exit}' "${STATS}")
        exon=$(awk '/Exon level:/       {print $3"\t"$5; exit}' "${STATS}")
        tran=$(awk '/Transcript level:/ {print $3"\t"$5; exit}' "${STATS}")
        locus=$(awk '/Locus level:/     {print $3"\t"$5; exit}' "${STATS}")
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${sp}" "${tool}" \
            "${base}" "${exon}" "${tran}" "${locus}" >> "${TABLE}"
    done
done

echo
echo "[$(date -Iseconds)] accuracy table -> ${TABLE}"
column -t -s$'\t' "${TABLE}"
