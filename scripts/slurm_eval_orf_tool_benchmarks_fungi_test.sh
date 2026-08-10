#!/bin/bash
# gffcompare Tiberius (run006) + TransDecoder1 + TransDecoder2 +
# TransDecoder2(--precise) + GeneMarkS-T against the reference for each
# fungi test species. Produces accuracy_table.tsv + PDF plots.
#
# Run after td1/td2/td2p/gmst array jobs complete (or submit immediately
# — "[skip]" lines are printed for any missing inputs).
#
# Override TIB_TAG to use a different Tiberius annotation run.
#
#SBATCH --job-name=eval_orf_fun
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_orf_fun_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_orf_fun_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/fungi_test
OUT_ROOT=${TESTDIR}/benchmark_orf_tools

# Resolve TIB_TAG: use env var if set, else find the newest annotate_run006_e* dir
# across any species (they should all use the same epoch).
if [[ -z "${TIB_TAG:-}" ]]; then
    TIB_TAG=$(ls -d "${TESTDIR}"/Aspergillus_fumigatus/annotate_run006_e* 2>/dev/null \
              | sort -t_ -k4 -n | tail -1 | xargs basename 2>/dev/null || true)
    [[ -n "${TIB_TAG}" ]] || { echo "cannot determine TIB_TAG; set it explicitly" >&2; exit 2; }
fi
echo "[$(date -Iseconds)] TIB_TAG=${TIB_TAG}"

SPECIES=(Agaricus_bisporus Aspergillus_fumigatus Cryphonectria_parasitica Parastagonospora_nodorum Puccinia_striiformis Punctularia_strigosozonata Tilletiopsis_washingtonensis)
TOOLS=(tiberius transdecoder1 transdecoder2 transdecoder2_precise gmst)

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

TABLE=${OUT_ROOT}/accuracy_table.tsv
printf "species\ttool\tbase_S\tbase_P\texon_S\texon_P\ttranscript_S\ttranscript_P\tlocus_S\tlocus_P\n" > "${TABLE}"

for sp in "${SPECIES[@]}"; do
    SP_OUT=${OUT_ROOT}/${sp}/gffcompare
    mkdir -p "${SP_OUT}"

    REF=${TESTDIR}/${sp}/assembly/annot_cds.gff
    if [[ ! -s "${REF}" ]]; then
        echo "[skip] ${sp}: missing reference ${REF}"
        continue
    fi

    declare -A GTF
    GTF[tiberius]=${TESTDIR}/${sp}/${TIB_TAG}/orfs.gtf
    GTF[transdecoder1]=${TESTDIR}/${sp}/benchmark_orf_tools/transdecoder1/orfs.gtf
    GTF[transdecoder2]=${TESTDIR}/${sp}/benchmark_orf_tools/transdecoder2/orfs.gtf
    GTF[transdecoder2_precise]=${TESTDIR}/${sp}/benchmark_orf_tools/transdecoder2_precise/orfs.gtf
    GTF[gmst]=${TESTDIR}/${sp}/benchmark_orf_tools/gmst/orfs.gtf

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
            echo "[warn] ${sp}/${tool}: no stats file"
            continue
        fi
        base=$(awk '/Base level:/       {print $3"\t"$5; exit}' "${STATS}")
        exon=$(awk '/Exon level:/       {print $3"\t"$5; exit}' "${STATS}")
        tran=$(awk '/Transcript level:/ {print $3"\t"$5; exit}' "${STATS}")
        locus=$(awk '/Locus level:/     {print $3"\t"$5; exit}' "${STATS}")
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${sp}" "${tool}" \
            "${base}" "${exon}" "${tran}" "${locus}" >> "${TABLE}"
    done
    unset GTF
done

echo "[$(date -Iseconds)] accuracy table -> ${TABLE}"
column -t -s$'\t' "${TABLE}"

# StringTie sensitivity upper bound.
UB_TSV=${OUT_ROOT}/stringtie_upper_bound.tsv
echo ""
echo "[$(date -Iseconds)] computing stringtie upper bound"
python "${PROJDIR}/scripts/compute_stringtie_upper_bound.py" \
    --species        "${SPECIES[@]}" \
    --ref-tmpl       "${TESTDIR}/{sp}/assembly/annot_cds.gff" \
    --stringtie-tmpl "${TESTDIR}/{sp}/stringtie/stringtie.gtf" \
    --out-tsv        "${UB_TSV}"
column -t -s$'\t' "${UB_TSV}"

# Generate plots at both locus and transcript levels.
for lvl in locus transcript; do
    echo "[$(date -Iseconds)] plotting ${lvl} level"
    python "${PROJDIR}/scripts/plot_orf_tool_accuracy.py" \
        --table           "${TABLE}" \
        --level           "${lvl}" \
        --upper-bound-tsv "${UB_TSV}" \
        --out             "${OUT_ROOT}/${lvl}_accuracy.pdf"
done

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}/{locus,transcript}_accuracy.pdf"
