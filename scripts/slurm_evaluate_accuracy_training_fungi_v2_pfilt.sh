#!/bin/bash
# Evaluate ORF-finder (protein-filtered) + ab initio Tiberius (protein-filtered)
# + their merge for Aspergillus_fumigatus, Candida_parapsilosis, Fusarium_redolens
# in training_fungi_v2.  Mirrors eval_accuracy_epoch_74_filt_pfilt for vertebrates.
#
# Prereqs:
#   slurm_annotate_training_fungi_v2_run006_cpu.sh       (ORF finder)
#   slurm_filter_orf_diamond_training_fungi_v2.sh        (ORF protein filter)
#   slurm_filter_tiberius_diamond_training_fungi_v2.sh   (Tiberius protein filter)
#
# Output:
#   results/training_fungi_v2/eval_accuracy_run006_pfilt/
#     accuracy_table.tsv
#     accuracy_figure.pdf
#     gffcompare_runs/
#
# Gene sets in the figure:
#   orf_prediction    — protein-filtered ORF finder (orfs.filtered_soft_protein.gtf)
#   tiberius          — ab initio Tiberius (tiberius.gtf)
#   tiberius_filtered — protein-filtered Tiberius (tiberius/soft_protein/*.gtf)
#   merged            — tiberius_filtered + orf_prediction
#
#SBATCH --job-name=eval_acc_tv2_pfilt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_tv2_pfilt_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_acc_tv2_pfilt_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_fungi_v2
OUT_ROOT=${TRAINDIR}/eval_accuracy_run006_pfilt
PRED_DIR=${OUT_ROOT}/preds

SPECIES=(Aspergillus_fumigatus Candida_parapsilosis Fusarium_redolens)

mkdir -p "${PROJDIR}/logs" "${OUT_ROOT}" "${PRED_DIR}"

# Ensure Aspergillus tiberius dir is visible under training_fungi_v2.
ASP_TIB_DIR=${TRAINDIR}/Aspergillus_fumigatus/tiberius
if [[ ! -e "${ASP_TIB_DIR}" ]]; then
    ln -sf "${PROJDIR}/results/fungi_test/Aspergillus_fumigatus/tiberius" "${ASP_TIB_DIR}"
    echo "[$(date -Iseconds)] symlinked Aspergillus tiberius dir"
fi

# Auto-detect the annotate epoch (newest annotate_run006_e* dir on Aspergillus).
EPOCH=$(ls -d "${TRAINDIR}/Aspergillus_fumigatus/annotate_run006_e"* 2>/dev/null \
        | sort -V | tail -1 | xargs -I{} basename {} | sed 's/annotate_run006_e//')
[[ -n "${EPOCH}" ]] || { echo "cannot detect EPOCH from annotate dirs" >&2; exit 2; }
echo "[$(date -Iseconds)] EPOCH=${EPOCH}"

# Build pred_dir: protein-filtered ORF finder predictions.
for sp in "${SPECIES[@]}"; do
    src=${TRAINDIR}/${sp}/annotate_run006_e${EPOCH}/soft_protein/orfs.filtered_soft_protein.gtf
    if [[ ! -s "${src}" ]]; then
        echo "[skip preds] ${sp}: missing ${src}"
        continue
    fi
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

echo "[$(date -Iseconds)] running evaluate_accuracy.py (protein-filtered)"

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir          "${PRED_DIR}" \
    --out-dir           "${OUT_ROOT}" \
    --species           "${SPECIES[@]}" \
    --ref-tmpl          "${TRAINDIR}/{sp}/assembly/annot_cds.gff" \
    --tib-tmpl          "${TRAINDIR}/{sp}/tiberius/tiberius.gtf" \
    --tib-filtered-tmpl "${TRAINDIR}/{sp}/tiberius/soft_protein/orfs.filtered_soft_protein.gtf"

echo "[$(date -Iseconds)] done -> ${OUT_ROOT}/accuracy_figure.pdf"
column -t -s$'\t' "${OUT_ROOT}/accuracy_table.tsv" | head -80
