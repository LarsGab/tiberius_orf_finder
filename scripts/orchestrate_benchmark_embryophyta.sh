#!/bin/bash
# Top-level orchestrator: expanded embryophyta_test benchmark (Table A, VARUS.bam).
# Same 3-phase chain as vertebrates; skips Brachypodium_distachyon (no paper-tree
# Tiberius or BRAKER3). Uses the embryophyta-specific LGB model.
#
# Usage:
#   EVAL_TAG=run010_varus ./scripts/orchestrate_benchmark_embryophyta.sh

set -euo pipefail

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJDIR}/results/training_embryophyta_test_v2}
BENCH=${BENCH:-/home/gabriell/tiberius_benchmarking/paper/Embryophyta}
LGB_MODEL=${LGB_MODEL:-${PROJDIR}/results/filter_analysis/lgb_embryophyta/lgb_3class_model.pkl}

EVAL_TAG=${EVAL_TAG:?EVAL_TAG required (e.g. run010_varus)}
OUT_ROOT=${RESULTS_ROOT}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds
RUNTIME_TSV=${OUT_ROOT}/runtimes.tsv

# 6 species with paper-tree Tiberius + BRAKER3
SPECIES=(Arabidopsis_thaliana Eschscholzia_californica Freycinetia_multiflora Medicago_truncatula Mimulus_guttatus Urochloa_brizantha)
N=${#SPECIES[@]}

mkdir -p "${OUT_ROOT}" "${PRED_DIR}"
SPECIES_FILE=${OUT_ROOT}/species.txt
printf '%s\n' "${SPECIES[@]}" > "${SPECIES_FILE}"

echo "[$(date -Iseconds)] tag=${EVAL_TAG}  N=${N}  out=${OUT_ROOT}"

DRUSILLA_TAG=annotate_run001_e300_filt_tpm1cov3len300
for sp in "${SPECIES[@]}"; do
    src=${RESULTS_ROOT}/${sp}/${DRUSILLA_TAG}/orfs.filtered.gtf
    if [[ -s "${src}" ]]; then
        mkdir -p "${PRED_DIR}/${sp}"
        ln -sf "${src}" "${PRED_DIR}/${sp}/prediction.gtf"
    else
        echo "[warn] missing DRUSILLA for ${sp}: ${src}" >&2
    fi
done

# ─── Phase 1: Vipsania ──────────────────────────────────────────────────────
missing_vip=0
for sp in "${SPECIES[@]}"; do
    [[ -s "${RESULTS_ROOT}/${sp}/vipsania/vip.gtf" ]] || missing_vip=$((missing_vip+1))
done

VIP_DEP=""
if [[ ${missing_vip} -gt 0 ]]; then
    echo "[phase1] Vipsania needed for ${missing_vip}/${N} species — submitting array"
    JOB=$(sbatch --parsable --array=1-${N} \
                 --export=ALL,CLADE=embryophyta_test,MODEL_NAME=Streptophyta,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
                 "${SCRIPTS_DIR}/slurm_vipsania_annotate.sh")
    echo "[phase1] Vipsania job = ${JOB}"
    VIP_DEP="--dependency=afterok:${JOB}"
else
    echo "[phase1] all vipsania/vip.gtf present — skipping"
fi

# ─── Phase 2: LGB-filter Vipsania + TD2 per species ─────────────────────────
DRIVER=${OUT_ROOT}/slurm_lgb_filter_array.sh
cat > "${DRIVER}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=lgb_filter_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_filter_emb_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_filter_emb_%A_%a.err
set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${LGB_MODEL:?}" "${RUNTIME_TSV:?}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
IDX=$(( (TASK_ID - 1) / 2 + 1 ))
KIND=$(( (TASK_ID - 1) % 2 ))
SPECIES=$(sed -n "${IDX}p" "${SPECIES_FILE}")
[[ -n "${SPECIES}" ]] || { echo "bad species index" >&2; exit 2; }

SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
MINIPROT=${SP_DIR}/proteins/miniprot_scored.gff
HINTS=${SP_DIR}/proteins/miniprothint/hc.gff
CHAINED=${SP_DIR}/hint_rescue/chained_hints.gff

if [[ ${KIND} -eq 0 ]]; then
    TOOL=vipsania
    IN_GTF=${SP_DIR}/vipsania/vip.gtf
else
    TOOL=td2
    IN_GTF=${SP_DIR}/benchmark_orf_tools/transdecoder2/orfs.gtf
fi
OUT_DIR=${SP_DIR}/lgb_tool_filter/${TOOL}

if [[ ! -s "${IN_GTF}" ]]; then
    echo "[skip] ${TOOL}:${SPECIES} — missing ${IN_GTF}"
    exit 0
fi

bash /projects/AI-GUSTUS/tiberius_orf_finder/scripts/filter_tool_gtf_with_lgb.sh \
    --tool "${TOOL}" --species "${SPECIES}" --clade embryophyta_test \
    --in-gtf "${IN_GTF}" --out-dir "${OUT_DIR}" \
    --model "${LGB_MODEL}" \
    --miniprot "${MINIPROT}" --hints "${HINTS}" --chained "${CHAINED}" \
    --genome "${GENOME}" \
    --runtime-tsv "${RUNTIME_TSV}"
EOF
chmod +x "${DRIVER}"

NTASK=$(( N * 2 ))
LGB_JOB=$(sbatch --parsable ${VIP_DEP} --array=1-${NTASK} \
                 --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},LGB_MODEL=${LGB_MODEL},RUNTIME_TSV=${RUNTIME_TSV} \
                 "${DRIVER}")
echo "[phase2] LGB-filter job = ${LGB_JOB}"

# ─── Phase 3: tier1+tier2 for Tiberius, then eval + report ─────────────────
FINAL=${OUT_ROOT}/slurm_final_eval.sh
cat > "${FINAL}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval_report_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_report_emb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_report_emb_%j.err
set -euo pipefail
: "${OUT_ROOT:?}" "${RESULTS_ROOT:?}" "${BENCH:?}" "${RUNTIME_TSV:?}" "${EVAL_TAG:?}" "${SPECIES_FILE:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
source "${PROJDIR}/scripts/lib/log_runtime.sh"
export RUNTIME_TSV

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Tier1+Tier2 filter for Tiberius (uses existing LGB-scored GTF)
mapfile -t SPECIES < "${SPECIES_FILE}"
for sp in "${SPECIES[@]}"; do
    TIB_LGB=${RESULTS_ROOT}/${sp}/tiberius_lgb_filtered/tiberius_lgb_filtered.gtf
    CHAINED=${RESULTS_ROOT}/${sp}/hint_rescue/chained_hints.gff
    OUT=${RESULTS_ROOT}/${sp}/tiberius_lgb_filtered/tib_correct_hint_partial.gtf
    if [[ -s "${TIB_LGB}" && -s "${CHAINED}" ]]; then
        if [[ ! -s "${OUT}" ]]; then
            run_timed embryophyta_test "${sp}" tiberius lgb_tier12 -- \
                python "${PROJDIR}/scripts/filter_tib_with_protein_hints.py" \
                    "${TIB_LGB}" "${CHAINED}" "${OUT}"
        else
            echo "[skip] ${sp}: tib_correct_hint_partial.gtf exists"
        fi
    else
        echo "[warn] ${sp}: missing tib_lgb_filtered.gtf or chained_hints.gff"
    fi
done

# Templates
TIB_FILT_TMPL=${RESULTS_ROOT}/{sp}/tiberius_lgb_filtered/tib_correct_hint_partial.gtf
VIP_TMPL=${RESULTS_ROOT}/{sp}/vipsania/vip.gtf
VIP_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/vipsania/vipsania_correct_hint_partial.gtf
TD2_TMPL=${RESULTS_ROOT}/{sp}/benchmark_orf_tools/transdecoder2/orfs.gtf
TD2_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/td2/td2_correct_hint_partial.gtf

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir "${OUT_ROOT}/preds" \
    --out-dir  "${OUT_ROOT}" \
    --species  "${SPECIES[@]}" \
    --ref-tmpl "${RESULTS_ROOT}/{sp}/assembly/annot_cds.gff" \
    --tib-tmpl "${BENCH}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf" \
    --brk-tmpl "${BENCH}/{sp}/results/predictions/braker3/braker3.gtf" \
    --tib-filtered-tmpl "${TIB_FILT_TMPL}" \
    --vip-tmpl          "${VIP_TMPL}" \
    --vip-filtered-tmpl "${VIP_FILT_TMPL}" \
    --td2-tmpl          "${TD2_TMPL}" \
    --td2-filtered-tmpl "${TD2_FILT_TMPL}"

python "${PROJDIR}/scripts/collect_runtimes.py" --inputs "${RUNTIME_TSV}" --out-dir "${OUT_ROOT}"

if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 150 "${OUT_ROOT}/accuracy_figure.pdf" "${OUT_ROOT}/accuracy_figure" >/dev/null || true
fi

python "${PROJDIR}/scripts/build_benchmark_report.py" \
    --eval-dir "${OUT_ROOT}" \
    --title "Embryophyta test — VARUS.bam campaign (${EVAL_TAG})" \
    ${FIG_PNG:+--figure-png "$FIG_PNG"}

echo "[$(date -Iseconds)] REPORT -> ${OUT_ROOT}/REPORT.md"
EOF
chmod +x "${FINAL}"

FINAL_JOB=$(sbatch --parsable --dependency=afterok:${LGB_JOB} \
                   --export=ALL,OUT_ROOT=${OUT_ROOT},RESULTS_ROOT=${RESULTS_ROOT},BENCH=${BENCH},RUNTIME_TSV=${RUNTIME_TSV},EVAL_TAG=${EVAL_TAG},SPECIES_FILE=${SPECIES_FILE} \
                   "${FINAL}")
echo "[phase3] eval+report job = ${FINAL_JOB}"

echo
echo "Submitted: vipsania=${JOB:-'(skipped)'} → lgb_filter=${LGB_JOB} → eval_report=${FINAL_JOB}"
echo "Report:    ${OUT_ROOT}/REPORT.md"
