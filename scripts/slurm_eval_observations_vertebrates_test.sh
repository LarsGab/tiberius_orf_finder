#!/bin/bash
# Run the three eval-observation quantification scripts on the
# vertebrates test species (excluding Homo_sapiens whose data prep failed).
#
# Outputs go to:
#   ${RESULTS_DIR}/<sp>/eval/
#     orfs.filtered.gtf
#     dropped_subsequences.tsv
#     start_truncation.tsv
#     stringtie_gene_split.tsv
#     summary.txt          (concatenated stdout of the three scripts)
#
# A combined per-species summary lands at:
#   ${RESULTS_DIR}/eval_observations_summary.txt
#
# Usage:  sbatch scripts/slurm_eval_observations_vertebrates_test.sh
#
#SBATCH --job-name=eval_obs_vert
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_obs_vert_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_obs_vert_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas)

mkdir -p "${PROJDIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

COMBINED=${RESULTS_DIR}/eval_observations_summary.txt
: > "${COMBINED}"

for sp in "${SPECIES[@]}"; do
    SP=${RESULTS_DIR}/${sp}
    OUT=${SP}/eval
    mkdir -p "${OUT}"

    REF=${SP}/assembly/annotation.gff
    ST=${SP}/stringtie/stringtie.gtf
    ORFS=${SP}/annotate_epoch_11/orfs.gtf

    for f in "${REF}" "${ST}" "${ORFS}"; do
        test -s "${f}" || { echo "missing input ${f}" >&2; exit 2; }
    done

    echo "[$(date -Iseconds)] === ${sp} ==="
    SUMMARY=${OUT}/summary.txt
    {
        echo "================================================================"
        echo "Species: ${sp}"
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

echo
echo "[$(date -Iseconds)] DONE. combined summary -> ${COMBINED}"
