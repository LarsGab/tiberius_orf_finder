#!/bin/bash
# Re-run the three eval-observation quantification scripts on the
# 4 vertebrates_test species for BOTH inference variants:
#   * ir_start : --restrict-start-to-ir-start
#   * pad20    : --prefix-pad-n 20
#
# Reads orfs.gtf from each variant's annotate output dir and writes the
# per-variant eval outputs under:
#   ${RESULTS_DIR}/<sp>/eval_${VARIANT_TAG}/
#
# Combined per-variant summaries:
#   ${RESULTS_DIR}/eval_observations_summary_ir_start.txt
#   ${RESULTS_DIR}/eval_observations_summary_pad20.txt
#
#SBATCH --job-name=eval_obs_vt_variants
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_obs_vt_variants_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_obs_vt_variants_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

WEIGHTS_TAG=epoch_11

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas)
VARIANTS=(ir_start ir_start_p80 pad20 ir_start_pad20 flank20 flank20_clip)

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

for variant in "${VARIANTS[@]}"; do
    COMBINED=${RESULTS_DIR}/eval_observations_summary_${variant}.txt
    : > "${COMBINED}"

    for sp in "${SPECIES[@]}"; do
        SP=${RESULTS_DIR}/${sp}
        ANNOT_DIR=${SP}/annotate_${WEIGHTS_TAG}_${variant}
        ORFS=${ANNOT_DIR}/orfs.gtf
        OUT=${SP}/eval_${variant}
        mkdir -p "${OUT}"

        REF=${SP}/assembly/annotation.gff
        ST=${SP}/stringtie/stringtie.gtf

        if [[ ! -s "${ORFS}" ]]; then
            echo "[$(date -Iseconds)] SKIP ${sp}/${variant}: missing ${ORFS}" \
                | tee -a "${COMBINED}"
            continue
        fi
        for f in "${REF}" "${ST}"; do
            test -s "${f}" || { echo "missing input ${f}" >&2; exit 2; }
        done

        echo "[$(date -Iseconds)] === ${sp} / ${variant} ==="
        SUMMARY=${OUT}/summary.txt
        {
            echo "================================================================"
            echo "Species: ${sp}    Variant: ${variant}"
            echo "================================================================"

            echo
            echo "--- filter_subsequence_predictions ---"
            python "${PROJDIR}/scripts/filter_subsequence_predictions.py" \
                --orfs-gtf   "${ORFS}" \
                --out-gtf    "${OUT}/orfs.filtered.gtf" \
                --report-tsv "${OUT}/dropped_subsequences.tsv"

            echo
            echo "--- quantify_start_truncation ---"
            python "${PROJDIR}/scripts/quantify_start_truncation.py" \
                --stringtie-gtf "${ST}" \
                --reference-gff "${REF}" \
                --orfs-gtf      "${ORFS}" \
                --out-tsv       "${OUT}/start_truncation.tsv"

            echo
            echo "--- quantify_stringtie_gene_split ---"
            python "${PROJDIR}/scripts/quantify_stringtie_gene_split.py" \
                --stringtie-gtf "${ST}" \
                --reference-gff "${REF}" \
                --out-tsv       "${OUT}/stringtie_gene_split.tsv"
        } 2>&1 | tee "${SUMMARY}"

        cat "${SUMMARY}" >> "${COMBINED}"
        echo >> "${COMBINED}"
    done

    echo "[$(date -Iseconds)] variant=${variant} DONE -> ${COMBINED}"
done
