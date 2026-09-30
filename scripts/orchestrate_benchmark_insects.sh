#!/bin/bash
# Top-level orchestrator: expanded insects_test_v2 benchmark (Table A, VARUS.bam).
# From-scratch pipeline — no upstream artifacts existed for insects. Chains:
#   0. prep (unzip genome + annot_cds + StringTie filter)
#   1. DRUSILLA annotate (Tiberius insects weights)
#   2. TransDecoder2
#   3. Protein integration (needs tiberius_proteins pipeline finished separately)
#   4. Vipsania (MODEL_NAME=Insecta)
#   5. LGB tool filter (Vipsania + TD2 + Tiberius tier-1/2)
#   6. eval + report
#
# Usage:
#   EVAL_TAG=run010_varus \
#   PROT_DEP_JID=<jid_of_tiberius_proteins_step6_miniprot> \
#   ./scripts/orchestrate_benchmark_insects.sh
#
# If PROT_DEP_JID is unset, protein integration will run immediately (assumes
# tiberius_proteins pipeline already finished).

set -euo pipefail

PROJDIR=${PROJDIR:-/projects/AI-GUSTUS/tiberius_orf_finder}
SCRIPTS_DIR=${SCRIPTS_DIR:-${PROJDIR}/scripts}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJDIR}/results/training_insects_test_v2}
BENCH=${BENCH:-/home/gabriell/tiberius_benchmarking/paper/Insecta}
LGB_MODEL=${LGB_MODEL:-${PROJDIR}/results/filter_analysis/lgb_3class_model.pkl}
INSECT_EPOCH=${INSECT_EPOCH:-51}

EVAL_TAG=${EVAL_TAG:?EVAL_TAG required (e.g. run010_varus)}
OUT_ROOT=${RESULTS_ROOT}/eval_accuracy_${EVAL_TAG}
PRED_DIR=${OUT_ROOT}/preds
RUNTIME_TSV=${OUT_ROOT}/runtimes.tsv

SPECIES=(
    Bombyx_mori
    Cataglyphis_hispanica
    Colias_croceus
    Danaus_plexippus
    Drosophila_melanogaster
    Leptidea_sinapis
    Nymphalis_io
    Osmia_bicornis
    Tribolium_castaneum
    Vanessa_cardui
    Zerene_cesonia
)
N=${#SPECIES[@]}
LAST=$((N-1))

mkdir -p "${OUT_ROOT}" "${PRED_DIR}"
SPECIES_FILE=${OUT_ROOT}/species.txt
printf '%s\n' "${SPECIES[@]}" > "${SPECIES_FILE}"

echo "[$(date -Iseconds)] tag=${EVAL_TAG} N=${N} out=${OUT_ROOT}"

# ─── Phase 0: assembly prep + StringTie filter ──────────────────────────────
J_PREP=$(sbatch --parsable --array=1-${N} \
              --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
              "${SCRIPTS_DIR}/slurm_prepare_insects_assembly.sh")
echo "[phase0] prep = ${J_PREP}"

# ─── Phase 1: DRUSILLA annotate (GPU) ───────────────────────────────────────
J_ANN=$(sbatch --parsable --dependency=afterok:${J_PREP} --array=1-${N} \
              --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV},INSECT_EPOCH=${INSECT_EPOCH} \
              "${SCRIPTS_DIR}/slurm_annotate_insects_test.sh")
echo "[phase1] annotate = ${J_ANN}"

# ─── Phase 2: TD2 (CPU, needs annotate transcripts.fa) ──────────────────────
J_TD2=$(sbatch --parsable --dependency=afterok:${J_ANN} --array=1-${N} \
              --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
              "${SCRIPTS_DIR}/slurm_td2_insects.sh")
echo "[phase2] td2 = ${J_TD2}"

# ─── Phase 3: Vipsania (independent, GPU) ───────────────────────────────────
J_VIP=$(sbatch --parsable --dependency=afterok:${J_PREP} --array=1-${N} \
              --export=ALL,CLADE=insects_test_v2,MODEL_NAME=Insecta,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
              "${SCRIPTS_DIR}/slurm_vipsania_annotate.sh")
echo "[phase3] vipsania = ${J_VIP}"

# ─── Phase 4: protein integration (depends on external tiberius_proteins job) ─
INT_DEP="--dependency=afterok:${J_PREP}"
if [[ -n "${PROT_DEP_JID:-}" ]]; then
    INT_DEP="--dependency=afterok:${J_PREP}:${PROT_DEP_JID}"
    echo "[phase4] integrate proteins waits on tiberius_proteins jid ${PROT_DEP_JID}"
fi
J_INT=$(sbatch --parsable ${INT_DEP} --array=1-${N} \
              --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},RUNTIME_TSV=${RUNTIME_TSV} \
              "${SCRIPTS_DIR}/slurm_integrate_insect_proteins.sh")
echo "[phase4] integrate = ${J_INT}"

# ─── Phase 5: DRUSILLA symlinks + LGB tool filter (2 kinds × N species) ──────
DRUSILLA_TAG=annotate_run006_insects_e${INSECT_EPOCH}_filt_tpm1cov3len300
for sp in "${SPECIES[@]}"; do
    mkdir -p "${PRED_DIR}/${sp}"
    ln -sf "${RESULTS_ROOT}/${sp}/${DRUSILLA_TAG}/orfs.filtered.gtf" "${PRED_DIR}/${sp}/prediction.gtf"
done

DRIVER=${OUT_ROOT}/slurm_lgb_filter_array.sh
cat > "${DRIVER}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=lgb_filter_ins
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_filter_ins_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lgb_filter_ins_%A_%a.err
set -euo pipefail
: "${SPECIES_FILE:?}" "${RESULTS_ROOT:?}" "${LGB_MODEL:?}" "${RUNTIME_TSV:?}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
IDX=$(( (TASK_ID - 1) / 2 + 1 ))
KIND=$(( (TASK_ID - 1) % 2 ))
SPECIES=$(sed -n "${IDX}p" "${SPECIES_FILE}")

SP_DIR=${RESULTS_ROOT}/${SPECIES}
GENOME=${SP_DIR}/assembly/genome.fa
MINIPROT=${SP_DIR}/proteins/miniprot_scored.gff
HINTS=${SP_DIR}/proteins/miniprothint/hc.gff
CHAINED=${SP_DIR}/hint_rescue/chained_hints.gff
FILT_TAG=filt_tpm1cov3len300

if [[ ${KIND} -eq 0 ]]; then
    TOOL=vipsania
    IN_GTF=${SP_DIR}/vipsania/vip.gtf
else
    TOOL=td2
    IN_GTF=${SP_DIR}/benchmark_orf_tools/transdecoder2_${FILT_TAG}/orfs.gtf
fi
OUT_DIR=${SP_DIR}/lgb_tool_filter/${TOOL}

[[ -s "${IN_GTF}" ]] || { echo "[skip] ${TOOL}:${SPECIES} — missing ${IN_GTF}"; exit 0; }

bash /projects/AI-GUSTUS/tiberius_orf_finder/scripts/filter_tool_gtf_with_lgb.sh \
    --tool "${TOOL}" --species "${SPECIES}" --clade insects_test_v2 \
    --in-gtf "${IN_GTF}" --out-dir "${OUT_DIR}" \
    --model "${LGB_MODEL}" \
    --miniprot "${MINIPROT}" --hints "${HINTS}" --chained "${CHAINED}" \
    --genome "${GENOME}" --runtime-tsv "${RUNTIME_TSV}"
EOF
chmod +x "${DRIVER}"

NTASK=$((N*2))
J_LGB=$(sbatch --parsable --dependency=afterok:${J_ANN}:${J_TD2}:${J_VIP}:${J_INT} --array=1-${NTASK} \
              --export=ALL,SPECIES_FILE=${SPECIES_FILE},RESULTS_ROOT=${RESULTS_ROOT},LGB_MODEL=${LGB_MODEL},RUNTIME_TSV=${RUNTIME_TSV} \
              "${DRIVER}")
echo "[phase5] lgb_filter = ${J_LGB}"

# ─── Phase 6: Tiberius LGB (raw Tiberius pred → LGB scored → tier1+2) + eval + report ─
FINAL=${OUT_ROOT}/slurm_final_eval.sh
cat > "${FINAL}" <<'EOF'
#!/bin/bash
#SBATCH --job-name=eval_report_ins
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_report_ins_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/eval_report_ins_%A_%a.err
set -euo pipefail
: "${OUT_ROOT:?}" "${RESULTS_ROOT:?}" "${BENCH:?}" "${RUNTIME_TSV:?}" "${EVAL_TAG:?}" "${SPECIES_FILE:?}" "${LGB_MODEL:?}"

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
source "${PROJDIR}/scripts/lib/log_runtime.sh"
export RUNTIME_TSV

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Apply LGB to raw Tiberius (paper-tree seqlen) per species; run tier1+2
mapfile -t SPECIES < "${SPECIES_FILE}"
for sp in "${SPECIES[@]}"; do
    TIB_RAW=${BENCH}/${sp}/results/predictions/tiberius/tiberius_seqlen.gtf
    SP_DIR=${RESULTS_ROOT}/${sp}
    GENOME=${SP_DIR}/assembly/genome.fa
    MINIPROT=${SP_DIR}/proteins/miniprot_scored.gff
    HINTS=${SP_DIR}/proteins/miniprothint/hc.gff
    CHAINED=${SP_DIR}/hint_rescue/chained_hints.gff
    OUT=${SP_DIR}/lgb_tool_filter/tiberius
    if [[ -s "${TIB_RAW}" && -s "${MINIPROT}" && -s "${CHAINED}" ]]; then
        if [[ ! -s "${OUT}/tiberius_correct_hint_partial.gtf" ]]; then
            bash "${PROJDIR}/scripts/filter_tool_gtf_with_lgb.sh" \
                --tool tiberius --species "${sp}" --clade insects_test_v2 \
                --in-gtf "${TIB_RAW}" --out-dir "${OUT}" \
                --model "${LGB_MODEL}" \
                --miniprot "${MINIPROT}" --hints "${HINTS}" --chained "${CHAINED}" \
                --genome "${GENOME}" --runtime-tsv "${RUNTIME_TSV}"
        fi
    fi
done

TIB_TMPL=${BENCH}/{sp}/results/predictions/tiberius/tiberius_seqlen.gtf
BRK_TMPL=${BENCH}/{sp}/results/predictions/braker3/braker3.gtf
TIB_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/tiberius/tiberius_correct_hint_partial.gtf
VIP_TMPL=${RESULTS_ROOT}/{sp}/vipsania/vip.gtf
VIP_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/vipsania/vipsania_correct_hint_partial.gtf
TD2_TMPL=${RESULTS_ROOT}/{sp}/benchmark_orf_tools/transdecoder2_filt_tpm1cov3len300/orfs.gtf
TD2_FILT_TMPL=${RESULTS_ROOT}/{sp}/lgb_tool_filter/td2/td2_correct_hint_partial.gtf

python "${PROJDIR}/scripts/evaluate_accuracy.py" \
    --pred-dir "${OUT_ROOT}/preds" --out-dir "${OUT_ROOT}" \
    --species  "${SPECIES[@]}" \
    --ref-tmpl "${RESULTS_ROOT}/{sp}/assembly/annot_cds.gff" \
    --tib-tmpl "${TIB_TMPL}" --brk-tmpl "${BRK_TMPL}" \
    --tib-filtered-tmpl "${TIB_FILT_TMPL}" \
    --vip-tmpl "${VIP_TMPL}" --vip-filtered-tmpl "${VIP_FILT_TMPL}" \
    --td2-tmpl "${TD2_TMPL}" --td2-filtered-tmpl "${TD2_FILT_TMPL}"

python "${PROJDIR}/scripts/collect_runtimes.py" --inputs "${RUNTIME_TSV}" --out-dir "${OUT_ROOT}"

if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 150 "${OUT_ROOT}/accuracy_figure.pdf" "${OUT_ROOT}/accuracy_figure" >/dev/null || true
fi

python "${PROJDIR}/scripts/build_benchmark_report.py" \
    --eval-dir "${OUT_ROOT}" \
    --title    "Insects test — VARUS.bam campaign (${EVAL_TAG})"

echo "[$(date -Iseconds)] REPORT -> ${OUT_ROOT}/REPORT.md"
EOF
chmod +x "${FINAL}"

J_FIN=$(sbatch --parsable --dependency=afterok:${J_LGB} \
              --export=ALL,OUT_ROOT=${OUT_ROOT},RESULTS_ROOT=${RESULTS_ROOT},BENCH=${BENCH},RUNTIME_TSV=${RUNTIME_TSV},EVAL_TAG=${EVAL_TAG},SPECIES_FILE=${SPECIES_FILE},LGB_MODEL=${LGB_MODEL} \
              "${FINAL}")
echo "[phase6] eval+report = ${J_FIN}"

echo
echo "Chain: prep=${J_PREP} → annotate=${J_ANN} → td2=${J_TD2}   vipsania=${J_VIP}   integrate=${J_INT}"
echo "       → lgb_filter=${J_LGB} → eval_report=${J_FIN}"
echo "Report: ${OUT_ROOT}/REPORT.md"
