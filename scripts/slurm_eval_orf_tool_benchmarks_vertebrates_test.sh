#!/bin/bash
# gffcompare Tiberius + TransDecoder1 + TransDecoder2 + TransDecoder2(--precise)
# + GeneMarkS-T against the reference for each vertebrates_test species,
# then aggregate into a single accuracy_table.tsv.
#
# Runs after the 4 tool jobs (td1, td2, td2_precise, gmst) have written
# ${RESULTS_DIR}/<sp>/benchmark_orf_tools/<tool>/orfs.gtf. Tiberius's
# baseline GTF is the same one the eval_observations script uses:
# ${RESULTS_DIR}/<sp>/annotate_epoch_11/orfs.gtf. Override with $TIB_TAG
# if you want a different Tiberius variant.
#
# Output:
#   ${RESULTS_DIR}/benchmark_orf_tools/
#     <sp>/gffcompare/<tool>.stats
#     accuracy_table.tsv
#
#SBATCH --job-name=eval_orf_tools_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_orf_tools_vt_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_orf_tools_vt_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
TIB_TAG=${TIB_TAG:-annotate_epoch_11}
OUT_ROOT=${RESULTS_DIR}/benchmark_orf_tools

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)
TOOLS=(tiberius transdecoder1 transdecoder2 transdecoder2_precise gmst)

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

TABLE=${OUT_ROOT}/accuracy_table.tsv
{
    printf "species\ttool\tbase_S\tbase_P\texon_S\texon_P\ttranscript_S\ttranscript_P\tlocus_S\tlocus_P\n"
} > "${TABLE}"

for sp in "${SPECIES[@]}"; do
    SP_OUT=${OUT_ROOT}/${sp}/gffcompare
    mkdir -p "${SP_OUT}"

    REF=${RESULTS_DIR}/${sp}/assembly/annot_cds.gff
    if [[ ! -s "${REF}" ]]; then
        echo "[skip] ${sp}: missing reference ${REF}"
        continue
    fi

    declare -A GTF
    GTF[tiberius]=${RESULTS_DIR}/${sp}/${TIB_TAG}/orfs.gtf
    GTF[transdecoder1]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/transdecoder1/orfs.gtf
    GTF[transdecoder2]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/transdecoder2/orfs.gtf
    GTF[transdecoder2_precise]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/transdecoder2_precise/orfs.gtf
    GTF[gmst]=${RESULTS_DIR}/${sp}/benchmark_orf_tools/gmst/orfs.gtf

    for tool in "${TOOLS[@]}"; do
        gtf=${GTF[$tool]}
        if [[ ! -s "${gtf}" ]]; then
            echo "[skip] ${sp}/${tool}: missing ${gtf}"
            continue
        fi
        PREFIX=${SP_OUT}/${tool}
        # -e 3 permits 3-bp boundary slack (matches evaluate_accuracy.py).
        awk -F '\t' '$3 == "CDS"' "${gtf}" > "${PREFIX}_cds.gff"
        gffcompare --strict-match -e 3 -T -r "${REF}" -o "${PREFIX}" "${PREFIX}_cds.gff" \
            > "${PREFIX}.gffcompare.log" 2>&1 || true

        STATS=${PREFIX}.stats
        if [[ ! -s "${STATS}" ]]; then
            echo "[warn] ${sp}/${tool}: no ${STATS}"
            continue
        fi

        # Parse Sensitivity | Precision for base / exon / transcript / locus.
        # Robust to variable spacing.
        # Format: "        Base level:    96.4     |    46.3    |"
        # tokens: Base level: 96.4 | 46.3 | -> $3 sensitivity, $5 precision
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
