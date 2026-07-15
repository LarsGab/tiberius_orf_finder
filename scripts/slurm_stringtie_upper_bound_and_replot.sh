#!/bin/bash
# Compute the per-species StringTie upper-bound sensitivity for the two
# StringTie inputs the ORF-tool benchmark uses (raw + tpm1cov3len300
# filtered) and regenerate the benchmark plots with the resulting
# horizontal reference line at each of --level {locus,transcript}.
#
# Output layout:
#   ${RESULTS_DIR}/benchmark_orf_tools/stringtie_upper_bound.tsv
#   ${RESULTS_DIR}/benchmark_orf_tools_filt_tpm1cov3len300/stringtie_upper_bound.tsv
#     (shared by the *_oh90_ss6_skip{0,1} tolerance variants too)
#   ${RESULTS_DIR}/benchmark_orf_tools*/[locus|transcript]_accuracy.pdf
#     (overwritten with the upper-bound overlay)
#
#SBATCH --job-name=st_ub_replot
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/st_ub_replot_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/st_ub_replot_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)

REF_TMPL='/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/assembly/annot_cds.gff'
ST_RAW_TMPL='/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/stringtie/stringtie.gtf'
ST_FILT_TMPL='/projects/AI-GUSTUS/tiberius_orf_finder/results/vertebrates_test/{sp}/stringtie/stringtie.filt_tpm1cov3len300.gtf'

UB_RAW=${RESULTS_DIR}/benchmark_orf_tools/stringtie_upper_bound.tsv
UB_FILT=${RESULTS_DIR}/benchmark_orf_tools_filt_tpm1cov3len300/stringtie_upper_bound.tsv

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] compute raw stringtie upper bound"
python "${PROJDIR}/scripts/compute_stringtie_upper_bound.py" \
    --species        "${SPECIES[@]}" \
    --ref-tmpl       "${REF_TMPL}" \
    --stringtie-tmpl "${ST_RAW_TMPL}" \
    --out-tsv        "${UB_RAW}"

echo "[$(date -Iseconds)] compute filtered stringtie upper bound"
python "${PROJDIR}/scripts/compute_stringtie_upper_bound.py" \
    --species        "${SPECIES[@]}" \
    --ref-tmpl       "${REF_TMPL}" \
    --stringtie-tmpl "${ST_FILT_TMPL}" \
    --out-tsv        "${UB_FILT}"

# For each benchmark dir, pick the matching UB tsv, regenerate both levels.
for d in benchmark_orf_tools benchmark_orf_tools_filt_tpm1cov3len300 benchmark_orf_tools_filt_tpm1cov3len300_oh90_ss6_skip0 benchmark_orf_tools_filt_tpm1cov3len300_oh90_ss6_skip1; do
    tab=${RESULTS_DIR}/${d}/accuracy_table.tsv
    [[ -s "${tab}" ]] || { echo "[skip] no accuracy_table.tsv in ${d}"; continue; }
    if [[ "${d}" == "benchmark_orf_tools" ]]; then
        ub="${UB_RAW}"
    else
        ub="${UB_FILT}"
    fi
    for lvl in locus transcript; do
        out=${RESULTS_DIR}/${d}/${lvl}_accuracy.pdf
        echo "[$(date -Iseconds)] plot ${d}  level=${lvl}"
        python "${PROJDIR}/scripts/plot_orf_tool_accuracy.py" \
            --table           "${tab}" \
            --level           "${lvl}" \
            --upper-bound-tsv "${ub}" \
            --out             "${out}"
    done
done

echo "[$(date -Iseconds)] done"
column -t -s$'\t' "${UB_RAW}"
echo
column -t -s$'\t' "${UB_FILT}"
