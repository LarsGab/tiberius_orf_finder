#!/bin/bash
# pyVARUS Table B for vertebrates_test:
#   1. Run pyVARUS per species → <sp>/pyvarus/pyVARUS.bam
#   2. StringTie on pyVARUS.bam → <sp>/stringtie_pyvarus/stringtie.filt.gtf
#   3. DRUSILLA annotate → <sp>/annotate_pyvarus_epoch74/orfs.filtered.gtf
#   4. LGB tool filter (reusing existing miniprot + Vipsania + TD2 outputs)
#   5. Evaluate + build report
#
# Rest of tool outputs (Vipsania, TD2, Tiberius, BRAKER3, protein hints) are
# unchanged relative to Table A; only the DRUSILLA input changes. So merges
# get computed against the new pyVARUS DRUSILLA.
#
# Usage: EVAL_TAG=run010_pyvarus ./scripts/orchestrate_pyvarus_table_b_vertebrates.sh
#
# ⚠ Requires NCBI_EMAIL for pyVARUS runlist step.

set -euo pipefail

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJDIR}/results/vertebrates_test}
BENCH_ROOT=${BENCH_ROOT:-/home/gabriell/tiberius_benchmarking/paper}
LGB_MODEL=${LGB_MODEL:-${PROJDIR}/results/filter_analysis/lgb_3class_model.pkl}
WEIGHTS=${WEIGHTS:-${PROJDIR}/results/models/cnn_lstm_vertebrates_run001_v2/epoch_74.weights.h5}
CONFIG=${CONFIG:-${PROJDIR}/configs/cnn_lstm_vertebrates_run001.yaml}
NCBI_EMAIL=${NCBI_EMAIL:?NCBI_EMAIL required (e.g. lgabriel23@gmx.de)}

EVAL_TAG=${EVAL_TAG:?EVAL_TAG required (e.g. run010_pyvarus)}
OUT_ROOT=${RESULTS_ROOT}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds
RUNTIME_TSV=${OUT_ROOT}/runtimes.tsv
SPECIES_CSV=${SPECIES_CSV:-${PROJDIR}/nextflow/conf/species_vertebrates_test.csv}

SPECIES=(Gallus_gallus Pristiophorus_japonicus Bos_taurus Delphinapterus_leucas Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens Homo_sapiens)
MAMMALS=(Bos_taurus Delphinapterus_leucas Homo_sapiens)
N=${#SPECIES[@]}

mkdir -p "${OUT_ROOT}" "${PRED_DIR}"
SPECIES_FILE=${OUT_ROOT}/species.txt
printf '%s\n' "${SPECIES[@]}" > "${SPECIES_FILE}"

echo "[$(date -Iseconds)] pyVARUS Table B vertebrates tag=${EVAL_TAG} N=${N}"

# ─── Stage 1: pyVARUS BAM regeneration (heavy) ──────────────────────────────
J_PYV=$(sbatch --parsable --array=1-${N} \
    --export=ALL,CLADE=vertebrates_test,SPECIES_CSV=${SPECIES_CSV},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV},NCBI_EMAIL=${NCBI_EMAIL} \
    "${SCRIPTS_DIR}/slurm_pyvarus_run.sh")
echo "[stage1] pyVARUS = ${J_PYV}"

# ─── Stage 2: StringTie on pyVARUS.bam ──────────────────────────────────────
J_ST=$(sbatch --parsable --dependency=afterok:${J_PYV} --array=1-${N} \
    --export=ALL,CLADE=vertebrates_test,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
    "${SCRIPTS_DIR}/slurm_stringtie_on_pyvarus.sh")
echo "[stage2] stringtie = ${J_ST}"

# ─── Stage 3: DRUSILLA annotate ────────────────────────────────────────────
J_ANN=$(sbatch --parsable --dependency=afterok:${J_ST} --array=1-${N} \
    --export=ALL,CLADE=vertebrates_test,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV},WEIGHTS=${WEIGHTS},CONFIG=${CONFIG},TAG=epoch74 \
    "${SCRIPTS_DIR}/slurm_annotate_on_pyvarus.sh")
echo "[stage3] annotate = ${J_ANN}"

# ─── Stage 4: symlink DRUSILLA into preds/, then LGB filter (Vip+TD2 reuse Table A LGB outputs) ─
# Since Vipsania and TD2 filtered outputs from Table A are unchanged, we only
# rebuild the merges (they include pyVARUS-DRUSILLA as orf_prediction).
FINAL=${OUT_ROOT}/slurm_final_eval.sh
cat > "${FINAL}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval_pyv_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_pyv_vt_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_pyv_vt_%j.err
set -euo pipefail
: "${OUT_ROOT:?}" "${RESULTS_ROOT:?}" "${BENCH_ROOT:?}" "${RUNTIME_TSV:?}" "${EVAL_TAG:?}" "${SPECIES_FILE:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

# Build preds/ from the pyVARUS DRUSILLA output
mapfile -t SPECIES < "${SPECIES_FILE}"
for sp in "${SPECIES[@]}"; do
    src=${RESULTS_ROOT}/${sp}/annotate_pyvarus_epoch74/orfs.filtered.gtf
    if [[ -s "${src}" ]]; then
        mkdir -p "${OUT_ROOT}/preds/${sp}"
        ln -sf "${src}" "${OUT_ROOT}/preds/${sp}/prediction.gtf"
    fi
done

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

TIB_FILT_TMPL=${RESULTS_ROOT}/{sp}/tiberius_lgb_filtered/tib_correct_hint_partial.gtf
VIP_TMPL=${RESULTS_ROOT}/{sp}/vipsania/vip.gtf
VIP_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/vipsania/vipsania_correct_hint_partial.gtf
TD2_TMPL=${RESULTS_ROOT}/{sp}/benchmark_orf_tools/transdecoder2_filt_tpm1cov3len300/orfs.gtf
TD2_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/td2/td2_correct_hint_partial.gtf

MAMMALS=(Bos_taurus Delphinapterus_leucas Homo_sapiens)
NONMAMMALS=(Gallus_gallus Pristiophorus_japonicus Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)

run_eval_batch() {
    local sub_out=$1; local clade=$2; shift 2
    mkdir -p "${sub_out}/preds"
    for sp in "$@"; do
        [[ -s "${OUT_ROOT}/preds/${sp}/prediction.gtf" ]] || continue
        mkdir -p "${sub_out}/preds/${sp}"
        ln -sf "${OUT_ROOT}/preds/${sp}/prediction.gtf" "${sub_out}/preds/${sp}/prediction.gtf"
    done
    python "${PROJDIR}/scripts/evaluate_accuracy.py" \
        --pred-dir "${sub_out}/preds" --out-dir "${sub_out}" --species "$@" \
        --ref-tmpl "${RESULTS_ROOT}/{sp}/assembly/annot_cds.gff" \
        --tib-tmpl "${BENCH_ROOT}/${clade}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf" \
        --brk-tmpl "${BENCH_ROOT}/${clade}/{sp}/results/predictions/braker3/braker3.gtf" \
        --tib-filtered-tmpl "${TIB_FILT_TMPL}" \
        --vip-tmpl "${VIP_TMPL}" --vip-filtered-tmpl "${VIP_FILT_TMPL}" \
        --td2-tmpl "${TD2_TMPL}" --td2-filtered-tmpl "${TD2_FILT_TMPL}"
}

run_eval_batch "${OUT_ROOT}/_mammals"    Mammalia   "${MAMMALS[@]}"
run_eval_batch "${OUT_ROOT}/_nonmammals" Vertebrata "${NONMAMMALS[@]}"

head -1 "${OUT_ROOT}/_nonmammals/accuracy_table.tsv" > "${OUT_ROOT}/accuracy_table.tsv"
tail -n +2 "${OUT_ROOT}/_nonmammals/accuracy_table.tsv" >> "${OUT_ROOT}/accuracy_table.tsv"
tail -n +2 "${OUT_ROOT}/_mammals/accuracy_table.tsv"    >> "${OUT_ROOT}/accuracy_table.tsv"

python "${PROJDIR}/scripts/collect_runtimes.py" --inputs "${RUNTIME_TSV}" --out-dir "${OUT_ROOT}"
if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 150 "${OUT_ROOT}/_nonmammals/accuracy_figure.pdf" "${OUT_ROOT}/accuracy_figure_nonmammals" >/dev/null || true
    pdftoppm -png -r 150 "${OUT_ROOT}/_mammals/accuracy_figure.pdf"    "${OUT_ROOT}/accuracy_figure_mammals"    >/dev/null || true
fi

python "${PROJDIR}/scripts/build_benchmark_report.py" \
    --eval-dir "${OUT_ROOT}" \
    --title "Vertebrates test — pyVARUS.bam campaign (${EVAL_TAG})"

echo "[$(date -Iseconds)] pyVARUS Table B REPORT -> ${OUT_ROOT}/REPORT.md"
EOF
chmod +x "${FINAL}"

J_FIN=$(sbatch --parsable --dependency=afterok:${J_ANN} \
    --export=ALL,OUT_ROOT=${OUT_ROOT},RESULTS_ROOT=${RESULTS_ROOT},BENCH_ROOT=${BENCH_ROOT},RUNTIME_TSV=${RUNTIME_TSV},EVAL_TAG=${EVAL_TAG},SPECIES_FILE=${SPECIES_FILE} \
    "${FINAL}")
echo "[stage4] eval+report = ${J_FIN}"

echo
echo "Chain: pyVARUS=${J_PYV} → stringtie=${J_ST} → annotate=${J_ANN} → eval=${J_FIN}"
echo "Report: ${OUT_ROOT}/REPORT.md"
