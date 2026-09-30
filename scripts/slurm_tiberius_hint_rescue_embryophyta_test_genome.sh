#!/bin/bash
# Approach 2: run Tiberius (hint_integration branch) on the FULL genome per
# species, feeding the same top-chain hints per partial locus in genome
# coordinates.  This gives Tiberius its normal sliding-window context (not
# just a ±25 kb window around each partial locus) so it can, e.g., commit to
# a long hinted intron without an artificial locus boundary nearby.
#
# Hints file = concatenation of chained_hints.gff lines whose chain_id is the
# top chain of any partial locus (from loci_manifest.tsv), i.e. exactly the
# same hint set used per-locus but written in genome coords.
#
# Output per species → hint_rescue_genome/tiberius_genome_predictions.gtf
#
#SBATCH --job-name=tib_hr_emb_gw
#SBATCH --partition=vision,vision-fast,storm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_hr_emb_gw_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/tib_hr_emb_gw_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TESTDIR=${PROJDIR}/results/training_embryophyta_test_v2
TIBERIUS_REPO=${TIBERIUS_REPO:-/projects/AI-GUSTUS/Tiberius}
TIBERIUS_CFG=angiosperms
HINT_WEIGHT=2.5

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    Arabidopsis_thaliana
    Eschscholzia_californica
    Freycinetia_multiflora
    Medicago_truncatula
    Mimulus_guttatus
    Urochloa_brizantha
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

RESULTS_DIR=${TESTDIR}/${species}
GENOME=${RESULTS_DIR}/assembly/genome.fa
CHAINED=${RESULTS_DIR}/hint_rescue/chained_hints.gff
MANIFEST=${RESULTS_DIR}/hint_rescue/loci_manifest.tsv
OUTDIR=${RESULTS_DIR}/hint_rescue_genome
HINTS_GFF=${OUTDIR}/top_chain_hints_genome.gff
RAW_GTF=${OUTDIR}/tiberius_genome_predictions_raw.gtf
OUT_GTF=${OUTDIR}/tiberius_genome_predictions.gtf

echo "[$(date -Iseconds)] species=${species}"

mkdir -p "${OUTDIR}"

for f in "${GENOME}" "${GENOME}.fai" "${CHAINED}" "${MANIFEST}"; do
    [[ -s "${f}" ]] || { echo "SKIP ${species}: missing ${f}" >&2; exit 0; }
done

TIBERIUS_BRANCH=$(git -C "${TIBERIUS_REPO}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
if [[ "${TIBERIUS_BRANCH}" != "hint_integration" ]]; then
    echo "ERROR: ${TIBERIUS_REPO} is on '${TIBERIUS_BRANCH}', expected 'hint_integration'." >&2
    exit 2
fi

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# ── Build genome-coord hints file = subset of chained_hints for top chains ──
if [[ ! -s "${HINTS_GFF}" ]]; then
    echo "[$(date -Iseconds)] Extracting top-chain hints for partial loci …"
    python - <<PY
import re, sys
used = set()
with open("${MANIFEST}") as fh:
    next(fh)
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 6 and p[5] and p[5] != "none":
            used.add(p[5])
print(f"top chain_ids: {len(used)}", file=sys.stderr)
kept = 0
with open("${CHAINED}") as fin, open("${HINTS_GFF}", "w") as fout:
    for line in fin:
        if line.startswith("#") or not line.strip():
            continue
        m = re.search(r"chain_id=([^;\s]+)", line)
        if m and m.group(1) in used:
            fout.write(line)
            kept += 1
print(f"wrote {kept} hint lines", file=sys.stderr)
PY
    echo "  hints file: $(wc -l < "${HINTS_GFF}") lines"
else
    echo "[$(date -Iseconds)] hints file exists, reusing"
fi

# ── Run Tiberius on the whole genome with those hints ────────────────────────
micromamba activate tib_test

rm -f "${RAW_GTF}" "${OUT_GTF}"

echo "[$(date -Iseconds)] Running Tiberius genome-wide …"
python "${TIBERIUS_REPO}/tiberius.py" \
    --genome      "${GENOME}" \
    --model_cfg   "${TIBERIUS_CFG}" \
    --hints       "${HINTS_GFF}" \
    --hint_weight "${HINT_WEIGHT}" \
    --out         "${RAW_GTF}"

# ── Post-filter: keep predictions overlapping partial loci ──────────────────
# The raw output already IS the genome-wide prediction; we keep a copy under
# tiberius_genome_predictions.gtf for downstream eval.
cp "${RAW_GTF}" "${OUT_GTF}"

N_TX=$(awk '$3=="transcript"' "${OUT_GTF}" | wc -l)
echo "[$(date -Iseconds)] done: ${N_TX} genome-wide transcripts → ${OUT_GTF}"
