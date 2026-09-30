#!/bin/bash
# Apply the DRUSILLA LGB filter pipeline to an arbitrary tool's GTF
# (Vipsania, TransDecoder2, or Tiberius) for a single species.
#
# Steps (each optionally timed via lib/log_runtime.sh):
#   1. compute_orf_features.py   → <out_dir>/features.tsv
#   2. apply_lgb_model_gtf.py    → <out_dir>/<tool>_lgb3_filtered.gtf
#   3. filter_tib_with_protein_hints.py → <out_dir>/<tool>_correct_hint_partial.gtf
#
# Assumes upstream artifacts (miniprot_scored.gff, hc.gff, chained_hints.gff,
# genome.fa, LGB model) already exist. Emits informative errors if not.
#
# Usage:
#   filter_tool_gtf_with_lgb.sh \
#     --tool vipsania \
#     --species Gallus_gallus \
#     --clade vertebrates_test \
#     --in-gtf   /path/to/vipsania.gtf \
#     --out-dir  /path/to/lgb_out/vipsania/ \
#     --model    /path/to/lgb_3class_model.pkl \
#     --miniprot /path/to/miniprot_scored.gff \
#     --hints    /path/to/hc.gff \
#     --chained  /path/to/chained_hints.gff \
#     --genome   /path/to/genome.fa \
#     [--runtime-tsv /path/to/runtimes.tsv]

set -euo pipefail

TOOL=""; SPECIES=""; CLADE=""
IN_GTF=""; OUT_DIR=""; MODEL=""
MINIPROT=""; HINTS=""; CHAINED=""; GENOME=""
RUNTIME_TSV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tool)         TOOL=$2;         shift 2;;
        --species)      SPECIES=$2;      shift 2;;
        --clade)        CLADE=$2;        shift 2;;
        --in-gtf)       IN_GTF=$2;       shift 2;;
        --out-dir)      OUT_DIR=$2;      shift 2;;
        --model)        MODEL=$2;        shift 2;;
        --miniprot)     MINIPROT=$2;     shift 2;;
        --hints)        HINTS=$2;        shift 2;;
        --chained)      CHAINED=$2;      shift 2;;
        --genome)       GENOME=$2;       shift 2;;
        --runtime-tsv)  RUNTIME_TSV=$2;  shift 2;;
        *) echo "Unknown flag: $1" >&2; exit 2;;
    esac
done

: "${TOOL:?--tool required}"
: "${SPECIES:?--species required}"
: "${CLADE:?--clade required}"
: "${IN_GTF:?--in-gtf required}"
: "${OUT_DIR:?--out-dir required}"
: "${MODEL:?--model required}"
: "${MINIPROT:?--miniprot required}"
: "${HINTS:?--hints required}"
: "${CHAINED:?--chained required}"
: "${GENOME:?--genome required}"

SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd)

# Optional runtime logger
if [[ -n "${RUNTIME_TSV}" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPTS_DIR}/lib/log_runtime.sh"
    export RUNTIME_TSV
    TIMED=(run_timed "${CLADE}" "${SPECIES}" "${TOOL}")
else
    TIMED=(command)
fi

# Preflight
for f in "${IN_GTF}" "${MODEL}" "${MINIPROT}" "${HINTS}" "${CHAINED}" "${GENOME}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing input: ${f}" >&2; exit 3; }
done
mkdir -p "${OUT_DIR}"

FEATURES="${OUT_DIR}/features.tsv"
LGB_GTF="${OUT_DIR}/${TOOL}_lgb3_filtered.gtf"
KEPT_GTF="${OUT_DIR}/${TOOL}_correct_hint_partial.gtf"

echo "[$(date -Iseconds)] tool=${TOOL} sp=${SPECIES} clade=${CLADE}"

# Step 1: features. Vipsania and TD2 GTFs don't carry the DRUSILLA-specific
# lorf_class attribute; --fallback-lorf-class synthesizes it from the upstream
# stop scan. Harmless for Tiberius/DRUSILLA GTFs (attribute already present).
"${TIMED[@]}" features -- \
    python "${SCRIPTS_DIR}/compute_orf_features.py" \
        --orfs-gtf     "${IN_GTF}" \
        --miniprot-gff "${MINIPROT}" \
        --hints-gff    "${HINTS}" \
        --genome       "${GENOME}" \
        --out          "${FEATURES}" \
        --fallback-lorf-class

# Step 2: LGB apply
"${TIMED[@]}" lgb_apply -- \
    python "${SCRIPTS_DIR}/apply_lgb_model_gtf.py" \
        --model    "${MODEL}" \
        --features "${FEATURES}" \
        --in-gtf   "${IN_GTF}" \
        --out-gtf  "${LGB_GTF}"

# Step 3: tier1+tier2 filter
"${TIMED[@]}" lgb_tier12 -- \
    python "${SCRIPTS_DIR}/filter_tib_with_protein_hints.py" \
        "${LGB_GTF}" "${CHAINED}" "${KEPT_GTF}"

echo "[$(date -Iseconds)] done  — kept: ${KEPT_GTF}"
