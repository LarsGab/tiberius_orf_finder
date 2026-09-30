#!/bin/bash
# Run miniprot --aln | miniprot_boundary_scorer + miniprothint for the three
# mammal vertebrates_test species (Bos_taurus, Delphinapterus_leucas, Homo_sapiens).
#
# Prerequisite for Bos/Delphin: slurm_fix_stop_vertebrates_test.sh (provides
#   protein_top5.fa). Homo_sapiens falls back to the full ODB if protein_top5.fa
#   is absent; submit --array=2 again after fix_stop for Homo if you prefer
#   the species-filtered set.
#
# Prerequisite: slurm_prepare_proteins_mammals_vertebrates_test.sh (protein_top4.fa)
#
# Output (per species):
#   results/vertebrates_test/<sp>/fix_stop/miniprot_scored.gff
#   results/vertebrates_test/<sp>/fix_stop/miniprothint/hc.gff
#
#SBATCH --job-name=mphint_mammals
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/mphint_mammals_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/mphint_mammals_%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load singularity/3.11.3

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/vertebrates_test
SCORING_MATRIX=/home/gabriell/Tiberius/conf/blosum62.csv

SIF=${PROJDIR}/sif/tiberius_2.0.2.sif
if [[ ! -s "${SIF}" ]]; then
    mkdir -p "${PROJDIR}/sif"
    (flock -x 200
     [[ ! -s "${SIF}" ]] && singularity pull "${SIF}" docker://larsgabriel23/tiberius:2.0.2
    ) 200>"${SIF}.lock" || true
    [[ -s "${SIF}" ]] || { echo "ERROR: SIF pull failed" >&2; exit 1; }
fi

run_tool() { singularity exec --bind /projects/AI-GUSTUS,/home/gabriell "${SIF}" "$@"; }

CPUS=${SLURM_CPUS_PER_TASK:-16}
mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(Bos_taurus Delphinapterus_leucas Homo_sapiens)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

GENOME="${RESULTS_DIR}/${species}/assembly/genome.fa"
FIX_DIR="${RESULTS_DIR}/${species}/fix_stop"
SCORED_GFF="${FIX_DIR}/miniprot_scored.gff"
MINIPROTHINT_DIR="${FIX_DIR}/miniprothint"
HC_GFF="${MINIPROTHINT_DIR}/hc.gff"
PROTEIN_TOP4="${FIX_DIR}/protein_top4.fa"

echo "[$(date -Iseconds)] species=${species}"

if [[ -s "${HC_GFF}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: miniprothint/hc.gff already exists"
    exit 0
fi

if [[ ! -s "${GENOME}" ]]; then
    echo "SKIP ${species}: genome not found: ${GENOME}"
    exit 0
fi

# Require pre-filtered proteins (run slurm_prepare_proteins_mammals_vertebrates_test.sh first).
if [[ ! -s "${PROTEIN_TOP4}" ]]; then
    echo "ERROR: protein_top4.fa not found for ${species}: ${PROTEIN_TOP4}" >&2
    echo "       Run slurm_prepare_proteins_mammals_vertebrates_test.sh first." >&2
    exit 2
fi
PROTEINS="${PROTEIN_TOP4}"
echo "[$(date -Iseconds)] protein source: protein_top4.fa ($(grep -c '^>' "${PROTEINS}") seqs)"

[[ -s "${SCORING_MATRIX}" ]] || { echo "ERROR: missing ${SCORING_MATRIX}" >&2; exit 2; }

mkdir -p "${FIX_DIR}" "${MINIPROTHINT_DIR}"

# ── Step 1: miniprot --aln | miniprot_boundary_scorer ────────────────────────
if [[ ! -s "${SCORED_GFF}" ]]; then
    echo "[$(date -Iseconds)] Running miniprot --aln | miniprot_boundary_scorer ..."
    run_tool miniprot \
        --aln \
        -t "${CPUS}" \
        "${GENOME}" \
        "${PROTEINS}" \
    | run_tool miniprot_boundary_scorer \
        -s "${SCORING_MATRIX}" \
        -o "${SCORED_GFF}"
    echo "[$(date -Iseconds)] miniprot_scored.gff: $(wc -l < "${SCORED_GFF}") lines"
else
    echo "[$(date -Iseconds)] Reusing ${SCORED_GFF}"
fi

# Sanitise scored.gff: miniprot_boundary_scorer occasionally outputs merged/truncated lines.
# Keep only valid 9-column GFF records with a known feature type and an attribute field.
python3 -c "
import sys
FEATURES = {'mRNA','CDS','intron','stop_codon','start_codon'}
gff = sys.argv[1]
tmp = gff + '.clean'
kept = dropped = 0
with open(gff) as f, open(tmp, 'w') as out:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if len(cols) == 9 and cols[0] and cols[2] in FEATURES and '=' in cols[8]:
            out.write(line); kept += 1
        else:
            dropped += 1
import os; os.replace(tmp, gff)
print(f'GFF sanitised: kept {kept}, dropped {dropped}', flush=True)

# Pass 2: remove child records whose mRNA parent was lost during sanitisation
mrna_ids = set()
with open(gff) as f:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if cols[2] == 'mRNA':
            for attr in cols[8].split(';'):
                attr = attr.strip()
                if attr.startswith('ID='):
                    mrna_ids.add(attr[3:].strip()); break
tmp2 = gff + '.orphan_clean'
kept2 = dropped2 = 0
with open(gff) as f, open(tmp2, 'w') as out:
    for line in f:
        cols = line.rstrip('\n').split('\t')
        if cols[2] == 'mRNA':
            out.write(line); kept2 += 1
        else:
            parent = next((a.strip()[7:] for a in cols[8].split(';') if a.strip().startswith('Parent=')), None)
            if parent in mrna_ids:
                out.write(line); kept2 += 1
            else:
                dropped2 += 1
os.replace(tmp2, gff)
print(f'Orphan filter: kept {kept2}, dropped {dropped2}', flush=True)
" "${SCORED_GFF}"

# ── Step 2: miniprothint ──────────────────────────────────────────────────────
echo "[$(date -Iseconds)] Running miniprothint.py ..."
run_tool miniprothint.py \
    "${SCORED_GFF}" \
    --workdir          "${MINIPROTHINT_DIR}" \
    --ignoreCoverage \
    --topNperSeed      10 \
    --minScoreFraction 0.5

[[ -s "${HC_GFF}" ]] || { echo "ERROR: miniprothint produced no hc.gff" >&2; exit 2; }
echo "[$(date -Iseconds)] done -> ${HC_GFF}"
wc -l "${HC_GFF}"
