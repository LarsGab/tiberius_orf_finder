#!/bin/bash
# Top-level orchestrator: expanded vertebrates_test benchmark (Table A, VARUS.bam).
#
# Chains SLURM submissions with --dependency=afterok:
#   1. Vipsania annotate array   (one task per species; skipped if all vip.gtf exist)
#   2. LGB-tool-filter array     (per species × [vipsania, td2])
#   3. evaluate_accuracy + collect_runtimes + build_benchmark_report (short CPU job)
#
# Run on the brain login node — this only submits jobs, no compute here.
#
# Usage:
#   EVAL_TAG=run010_varus ./scripts/orchestrate_benchmark_vertebrates.sh
#
# Env overrides:
#   PROJDIR, RESULTS_ROOT, SCRIPTS_DIR, LGB_MODEL, TIB_PROJDIR, NCBI_EMAIL

set -euo pipefail

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJDIR}/results/vertebrates_test}
BENCH_ROOT=${BENCH_ROOT:-/home/gabriell/tiberius_benchmarking/paper}
LGB_MODEL=${LGB_MODEL:-${PROJDIR}/results/filter_analysis/lgb_3class_model.pkl}

EVAL_TAG=${EVAL_TAG:?EVAL_TAG required (e.g. run010_varus)}
OUT_ROOT=${RESULTS_ROOT}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds
RUNTIME_TSV=${OUT_ROOT}/runtimes.tsv

# ─── Species set (same as run009 + Homo_sapiens) ────────────────────────────
SPECIES=(
    Gallus_gallus
    Pristiophorus_japonicus
    Bos_taurus
    Delphinapterus_leucas
    Takifugu_rubripes
    Zootoca_vivipara
    Archocentrus_centrarchus
    Betta_splendens
    Homo_sapiens
)
MAMMALS=(Bos_taurus Delphinapterus_leucas Homo_sapiens)
N=${#SPECIES[@]}

mkdir -p "${OUT_ROOT}" "${PRED_DIR}"
SPECIES_FILE=${OUT_ROOT}/species.txt
printf '%s\n' "${SPECIES[@]}" > "${SPECIES_FILE}"

echo "[$(date -Iseconds)] tag=${EVAL_TAG}  N=${N}  out=${OUT_ROOT}"

# ─── Reuse-source per (row, species) ────────────────────────────────────────
# DRUSILLA (orf_prediction) — annotate_run009_best_filt_tpm1cov3len300/orfs.filtered.gtf
# Tiberius (raw)            — paper tree, Vertebrata or Mammalia (below)
# Tiberius (filtered)       — tiberius_lgb_filtered/tib_correct_hint_partial.gtf
# BRAKER3                   — paper tree
# Vipsania (new)            — <sp>/vipsania/vip.gtf   (produced by phase 1)
# TD2 (existing)            — benchmark_orf_tools/transdecoder2_filt_tpm1cov3len300/orfs.gtf

# helper: return "Mammalia" for mammals, "Vertebrata" otherwise
clade_for() {
    local sp=$1
    for m in "${MAMMALS[@]}"; do [[ "${sp}" == "${m}" ]] && { echo Mammalia; return; }; done
    echo Vertebrata
}

# ─── Symlink tree preds/<sp>/prediction.gtf ─────────────────────────────────
DRUSILLA_TAG=annotate_run009_best_filt_tpm1cov3len300
for sp in "${SPECIES[@]}"; do
    src=${RESULTS_ROOT}/${sp}/${DRUSILLA_TAG}/orfs.filtered.gtf
    if [[ -s "${src}" ]]; then
        mkdir -p "${PRED_DIR}/${sp}"
        ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
    else
        echo "[warn] missing DRUSILLA for ${sp}: ${src}" >&2
    fi
done

# ─── Phase 1: Vipsania (skip if all vip.gtf exist) ──────────────────────────
missing_vip=0
for sp in "${SPECIES[@]}"; do
    [[ -s "${RESULTS_ROOT}/${sp}/vipsania/vip.gtf" ]] || missing_vip=$((missing_vip+1))
done

VIP_DEP=""
if [[ ${missing_vip} -gt 0 ]]; then
    echo "[phase1] Vipsania needed for ${missing_vip}/${N} species — submitting array"
    JOB=$(sbatch --parsable --array=1-${N} \
                 --export=ALL,CLADE=vertebrates_test,MODEL_NAME=Vertebrata,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
                 "${SCRIPTS_DIR}/slurm_vipsania_annotate.sh")
    echo "[phase1] Vipsania job = ${JOB}"
    VIP_DEP="--dependency=afterok:${JOB}"
else
    echo "[phase1] all vipsania/vip.gtf present — skipping"
fi

# ─── Phase 2: LGB-filter Vipsania + TD2 per species (SLURM array) ───────────
# We write a small array-driver script inline that dispatches on TASK_ID.
DRIVER=${OUT_ROOT}/slurm_lgb_filter_array.sh
cat > "${DRIVER}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=lgb_filter
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_filter_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_filter_%A_%a.err
set -euo pipefail

: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${LGB_MODEL:?}" "${RUNTIME_TSV:?}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
NLINES=$(wc -l < "${SPECIES_FILE}")
IDX=$(( (TASK_ID - 1) / 2 + 1 ))   # 1..N
KIND=$(( (TASK_ID - 1) % 2 ))      # 0=vipsania, 1=td2

SPECIES=$(sed -n "${IDX}p" "${SPECIES_FILE}")
[[ -n "${SPECIES}" ]] || { echo "bad species index ${IDX}" >&2; exit 2; }

SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
MINIPROT=${SP_DIR}/fix_stop/miniprot_scored.gff
HINTS=${SP_DIR}/fix_stop/miniprothint/hc.gff
CHAINED=${SP_DIR}/hint_rescue/chained_hints.gff

if [[ ${KIND} -eq 0 ]]; then
    TOOL=vipsania
    IN_GTF=${SP_DIR}/vipsania/vip.gtf
else
    TOOL=td2
    IN_GTF=${SP_DIR}/benchmark_orf_tools/transdecoder2_filt_tpm1cov3len300/orfs.gtf
fi
OUT_DIR=${SP_DIR}/lgb_tool_filter/${TOOL}

if [[ ! -s "${IN_GTF}" ]]; then
    echo "[skip] ${TOOL}:${SPECIES} — missing ${IN_GTF}"
    exit 0
fi

bash /projects/AI-GUSTUS/tiberius_orf_finder/scripts/filter_tool_gtf_with_lgb.sh \
    --tool "${TOOL}" --species "${SPECIES}" --clade vertebrates_test \
    --in-gtf "${IN_GTF}" --out-dir "${OUT_DIR}" \
    --model "${LGB_MODEL}" \
    --miniprot "${MINIPROT}" --hints "${HINTS}" --chained "${CHAINED}" \
    --genome "${GENOME}" \
    --runtime-tsv "${RUNTIME_TSV}"
EOF
chmod +x "${DRIVER}"

NTASK=$(( N * 2 ))
echo "[phase2] LGB-tool-filter array size = ${NTASK}"
LGB_JOB=$(sbatch --parsable ${VIP_DEP} --array=1-${NTASK} \
                 --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},LGB_MODEL=${LGB_MODEL},RUNTIME_TSV=${RUNTIME_TSV} \
                 "${DRIVER}")
echo "[phase2] LGB-filter job = ${LGB_JOB}"

# ─── Phase 3: eval + collect_runtimes + report ──────────────────────────────
FINAL=${OUT_ROOT}/slurm_final_eval.sh
cat > "${FINAL}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval_report
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_report_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_report_%j.err
set -euo pipefail

: "${OUT_ROOT:?}" "${RESULTS_ROOT:?}" "${BENCH_ROOT:?}" "${RUNTIME_TSV:?}" "${EVAL_TAG:?}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

TIB_FILT_TMPL=${RESULTS_ROOT}/{sp}/tiberius_lgb_filtered/tib_correct_hint_partial.gtf
VIP_TMPL=${RESULTS_ROOT}/{sp}/vipsania/vip.gtf
VIP_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/vipsania/vipsania_correct_hint_partial.gtf
TD2_TMPL=${RESULTS_ROOT}/{sp}/benchmark_orf_tools/transdecoder2_filt_tpm1cov3len300/orfs.gtf
TD2_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/td2/td2_correct_hint_partial.gtf

# Per-species Tiberius/BRAKER3 template: clade differs for mammals vs. others.
# We call evaluate_accuracy.py in two batches (mammals under Mammalia, rest
# under Vertebrata) and concatenate the accuracy tables.
MAMMALS=(Bos_taurus Delphinapterus_leucas Homo_sapiens)
NONMAMMALS=(Gallus_gallus Pristiophorus_japonicus Takifugu_rubripes Zootoca_vivipara Archocentrus_centrarchus Betta_splendens)

run_eval_batch() {
    local sub_out=$1; shift
    local clade=$1; shift
    mkdir -p "${sub_out}"
    # Rebuild pred subdir with just the species in this batch
    local sub_pred=${sub_out}/preds
    mkdir -p "${sub_pred}"
    for sp in "$@"; do
        [[ -s "${OUT_ROOT}/preds/${sp}/prediction.gtf" ]] || continue
        mkdir -p "${sub_pred}/${sp}"
        ln -sf "${OUT_ROOT}/preds/${sp}/prediction.gtf" "${sub_pred}/${sp}/prediction.gtf"
    done

    python "${PROJDIR}/scripts/evaluate_accuracy.py" \
        --pred-dir "${sub_pred}" \
        --out-dir  "${sub_out}" \
        --species  "$@" \
        --ref-tmpl "${RESULTS_ROOT}/{sp}/assembly/annot_cds.gff" \
        --tib-tmpl "${BENCH_ROOT}/${clade}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf" \
        --brk-tmpl "${BENCH_ROOT}/${clade}/{sp}/results/predictions/braker3/braker3.gtf" \
        --tib-filtered-tmpl "${TIB_FILT_TMPL}" \
        --vip-tmpl          "${VIP_TMPL}" \
        --vip-filtered-tmpl "${VIP_FILT_TMPL}" \
        --td2-tmpl          "${TD2_TMPL}" \
        --td2-filtered-tmpl "${TD2_FILT_TMPL}"
}

run_eval_batch "${OUT_ROOT}/_mammals"    Mammalia   "${MAMMALS[@]}"
run_eval_batch "${OUT_ROOT}/_nonmammals" Vertebrata "${NONMAMMALS[@]}"

# Concat the two accuracy tables into OUT_ROOT
head -1 "${OUT_ROOT}/_nonmammals/accuracy_table.tsv" > "${OUT_ROOT}/accuracy_table.tsv"
tail -n +2 "${OUT_ROOT}/_nonmammals/accuracy_table.tsv" >> "${OUT_ROOT}/accuracy_table.tsv"
tail -n +2 "${OUT_ROOT}/_mammals/accuracy_table.tsv"    >> "${OUT_ROOT}/accuracy_table.tsv"

# Re-render figure from the combined table using the non-mammal figure as base;
# the per-clade PDFs are also kept in _mammals/, _nonmammals/ for inspection.
cp "${OUT_ROOT}/_nonmammals/accuracy_figure.pdf" "${OUT_ROOT}/accuracy_figure_nonmammals.pdf"
cp "${OUT_ROOT}/_mammals/accuracy_figure.pdf"    "${OUT_ROOT}/accuracy_figure_mammals.pdf"

# Runtimes
python "${PROJDIR}/scripts/collect_runtimes.py" \
    --inputs "${RUNTIME_TSV}" --out-dir "${OUT_ROOT}"

# Optional: render figures to PNG for embedding, if pdftoppm is present
if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 150 "${OUT_ROOT}/accuracy_figure_nonmammals.pdf" \
             "${OUT_ROOT}/accuracy_figure_nonmammals" >/dev/null || true
    pdftoppm -png -r 150 "${OUT_ROOT}/accuracy_figure_mammals.pdf" \
             "${OUT_ROOT}/accuracy_figure_mammals" >/dev/null || true
fi

python "${PROJDIR}/scripts/build_benchmark_report.py" \
    --eval-dir "${OUT_ROOT}" \
    --title "Vertebrates test — VARUS.bam campaign (${EVAL_TAG})"

echo "[$(date -Iseconds)] REPORT written to ${OUT_ROOT}/REPORT.md"
EOF
chmod +x "${FINAL}"

FINAL_JOB=$(sbatch --parsable --dependency=afterok:${LGB_JOB} \
                   --export=ALL,OUT_ROOT=${OUT_ROOT},RESULTS_ROOT=${RESULTS_ROOT},BENCH_ROOT=${BENCH_ROOT},RUNTIME_TSV=${RUNTIME_TSV},EVAL_TAG=${EVAL_TAG} \
                   "${FINAL}")
echo "[phase3] eval+report job = ${FINAL_JOB}"

echo
echo "Submitted job chain: vipsania=${JOB:-'(skipped)'} → lgb_filter=${LGB_JOB} → eval_report=${FINAL_JOB}"
echo "Watch:  squeue -u \$USER --job ${LGB_JOB},${FINAL_JOB}"
echo "Report will land at:  ${OUT_ROOT}/REPORT.md"
