#!/bin/bash
# Prepare top-4 protein FASTA for each Fungi test species.
#
# For each species:
#   1. Extract peptides from Tiberius GTF (falls back to ORF predictions
#      for Aspergillus_fumigatus where tiberius_seqlen.gtf is missing).
#   2. BFS on nodes.dmp → order-excluded filter script on /tmp.
#   3. Buffer order-excluded Fungi.fa.gz → diamond makedb + blastp.
#   4. Rank top-4 donor species → extract protein_top4.fa.
#
# Prerequisite: slurm_download_odb_fungi_plants.sh (Fungi.fa.gz)
#               slurm_lookup_taxids_fungi_embryophyta.sh (species_order_taxids.tsv)
#
# Output per species (results/fungi_test/<sp>/proteins/):
#   tiberius_peptides.fa
#   diamond_hits.tsv
#   species_rank.tsv
#   top_species.txt
#   protein_top4.fa
#
#SBATCH --job-name=prep_prot_fungi
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%5
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_prot_fungi_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/prep_prot_fungi_%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
module load singularity/3.11.3

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
RESULTS_DIR=${PROJDIR}/results/fungi_test
BENCH=/home/gabriell/tiberius_benchmarking
ODB_GZ=${PROJDIR}/odb/raw/Fungi.fa.gz
NODES_DMP=/home/gabriell/tiberius_proteins_analysis/odb/nodes.dmp
TAXIDS_TSV=${RESULTS_DIR}/species_order_taxids.tsv
SIF=${PROJDIR}/sif/tiberius_2.0.2.sif
TIBERIUS_REPO=/home/gabriell/Tiberius
ANNOT_TAG=annotate_run006_e300
TOP_N=4
CPUS=${SLURM_CPUS_PER_TASK:-16}

mkdir -p "${PROJDIR}/logs"

declare -a SPECIES=(
    Agaricus_bisporus
    Aspergillus_fumigatus
    Cryphonectria_parasitica
    Parastagonospora_nodorum
    Puccinia_striiformis
    Punctularia_strigosozonata
    Tilletiopsis_washingtonensis
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}
GENOME=${RESULTS_DIR}/${species}/assembly/genome.fa
OUT_DIR=${RESULTS_DIR}/${species}/proteins

echo "[$(date -Iseconds)] species=${species}"

for f in "${ODB_GZ}" "${NODES_DMP}" "${TAXIDS_TSV}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
[[ -s "${GENOME}" ]] || { echo "SKIP ${species}: missing genome.fa" >&2; exit 0; }

if [[ -s "${OUT_DIR}/protein_top4.fa" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${species}: protein_top4.fa already exists"; exit 0
fi

# Determine peptide source GTF (Tiberius benchmarking or fall back to ORFs)
TIB_BENCH="${BENCH}/paper/Fungi/${species}/results/predictions/tiberius/tiberius_seqlen.gtf"
ORF_GTF="${RESULTS_DIR}/${species}/${ANNOT_TAG}/orfs.filtered.gtf"
if [[ -s "${TIB_BENCH}" ]]; then
    PEPTIDE_GTF="${TIB_BENCH}"
elif [[ -s "${ORF_GTF}" ]]; then
    echo "[warn] Tiberius GTF missing; using ORF predictions as peptide source"
    PEPTIDE_GTF="${ORF_GTF}"
else
    echo "SKIP ${species}: no peptide source GTF" >&2; exit 0
fi

mkdir -p "${OUT_DIR}"

run_tool() {
    singularity exec \
        --bind /projects/AI-GUSTUS,/home/gabriell \
        "${SIF}" "$@"
}

# ── Read order_taxid ──────────────────────────────────────────────────────────
ORDER_TAXID=$(awk -F'\t' -v sp="${species}" '$1 == sp {print $4; exit}' "${TAXIDS_TSV}")
[[ -n "${ORDER_TAXID}" ]] || { echo "ERROR: ${species} not in ${TAXIDS_TSV}" >&2; exit 2; }
echo "[$(date -Iseconds)] order_taxid=${ORDER_TAXID}"

# ── Build per-job ODB filter script ──────────────────────────────────────────
FILTER_SCRIPT=$(mktemp "${TMPDIR:-/tmp}/filter_odb_XXXXXX.py")
trap "rm -f ${FILTER_SCRIPT}" EXIT

python3 - "${ORDER_TAXID}" "${NODES_DMP}" "${FILTER_SCRIPT}" <<'PYEOF'
import sys, re
from collections import deque

order_taxid = int(sys.argv[1])
nodes_dmp   = sys.argv[2]
out_py      = sys.argv[3]

parent_of = {}
with open(nodes_dmp) as fh:
    for line in fh:
        p = [x.strip() for x in line.rstrip('\n').split('|')]
        if len(p) >= 2:
            parent_of[int(p[0])] = int(p[1])

children_of = {}
for tid, par in parent_of.items():
    children_of.setdefault(par, []).append(tid)

excl = set()
queue = deque([order_taxid])
while queue:
    cur = queue.popleft()
    excl.add(cur)
    for child in children_of.get(cur, []):
        if child not in excl:
            queue.append(child)

print(f'  {len(excl)} taxids in exclusion set for order {order_taxid}', flush=True)

script = f'''import sys, re

EXCL = {repr(excl)}

wanted = None
if len(sys.argv) > 1:
    wanted = set()
    with open(sys.argv[1]) as fh:
        for line in fh:
            line = line.strip()
            if line:
                wanted.add(int(line))

pat = re.compile(r"^>(\\d+)_")
keep = False
for line in sys.stdin:
    if line.startswith(">"):
        m = pat.match(line)
        if m:
            tid = int(m.group(1))
            keep = tid not in EXCL and (wanted is None or tid in wanted)
        else:
            keep = True
        if keep:
            sys.stdout.write(line.replace('\\t', ' '))
    elif keep:
        sys.stdout.write(line)
'''

with open(out_py, 'w') as fh:
    fh.write(script)
PYEOF

# ── Step 1: extract peptides ──────────────────────────────────────────────────
PEPTIDES="${OUT_DIR}/tiberius_peptides.fa"
if [[ ! -s "${PEPTIDES}" ]]; then
    echo "[$(date -Iseconds)] Extracting peptides from ${PEPTIDE_GTF} …"
    run_tool gffread "${PEPTIDE_GTF}" -g "${GENOME}" -y "${PEPTIDES}"
    # gffread -y appends '.' for stop codons; diamond rejects them → strip in-place
    sed -i '/^[^>]/s/\.//g' "${PEPTIDES}"
    N=$(grep -c '^>' "${PEPTIDES}" || true)
    echo "[$(date -Iseconds)]   ${N} peptides"
else
    echo "[$(date -Iseconds)] Reusing ${PEPTIDES}"
fi

# ── Step 2: Diamond db + blastp ───────────────────────────────────────────────
DIAM_DB="${OUT_DIR}/diamond_db"
DIAM_HITS="${OUT_DIR}/diamond_hits.tsv"

if [[ ! -s "${DIAM_HITS}" ]]; then
    FILT_FA=$(mktemp "${TMPDIR:-/tmp}/odb_filtered_XXXXXX.fa.gz")
    trap "rm -f ${FILTER_SCRIPT} ${FILT_FA}" EXIT

    echo "[$(date -Iseconds)] Buffering order-excluded ODB → ${FILT_FA} …"
    zcat "${ODB_GZ}" | python3 "${FILTER_SCRIPT}" | gzip -1 > "${FILT_FA}"
    echo "[$(date -Iseconds)]   $(du -sh "${FILT_FA}" | cut -f1) buffered"

    echo "[$(date -Iseconds)] Building Diamond DB …"
    run_tool diamond makedb \
        --in      "${FILT_FA}" \
        --db      "${DIAM_DB}" \
        --threads "${CPUS}"
    rm -f "${FILT_FA}"

    echo "[$(date -Iseconds)] Running Diamond blastp …"
    run_tool diamond blastp \
        --query           "${PEPTIDES}" \
        --db              "${DIAM_DB}.dmnd" \
        --out             "${DIAM_HITS}" \
        --outfmt 6        qseqid sseqid pident length evalue bitscore qlen slen \
        --evalue          1e-5 \
        --max-target-seqs 200 \
        --very-sensitive \
        --threads         "${CPUS}"
    echo "[$(date -Iseconds)]   $(wc -l < "${DIAM_HITS}") hits"
    rm -f "${DIAM_DB}.dmnd"
else
    echo "[$(date -Iseconds)] Reusing ${DIAM_HITS}"
fi

# ── Step 3: rank top-N donor species ─────────────────────────────────────────
TOP_TXT="${OUT_DIR}/top_species.txt"
if [[ ! -s "${TOP_TXT}" ]]; then
    echo "[$(date -Iseconds)] Ranking species (top ${TOP_N}) …"
    (cd "${OUT_DIR}" && \
     python3 "${TIBERIUS_REPO}/tiberius/scripts/rank_species_from_diamond.py" \
         "${DIAM_HITS}" "${TOP_N}" \
         > "${OUT_DIR}/species_rank.tsv")
    echo "[$(date -Iseconds)] Top ${TOP_N} taxids:"
    cat "${TOP_TXT}"
else
    echo "[$(date -Iseconds)] Reusing ${TOP_TXT}"
fi

# ── Step 4: extract protein_top4.fa ──────────────────────────────────────────
PROTEIN_TOP4="${OUT_DIR}/protein_top4.fa"
if [[ ! -s "${PROTEIN_TOP4}" ]]; then
    echo "[$(date -Iseconds)] Extracting protein_top${TOP_N}.fa …"
    zcat "${ODB_GZ}" \
        | python3 "${FILTER_SCRIPT}" "${TOP_TXT}" \
        > "${PROTEIN_TOP4}"
    N=$(grep -c '^>' "${PROTEIN_TOP4}" || true)
    echo "[$(date -Iseconds)]   ${N} sequences → ${PROTEIN_TOP4}"
fi

echo "[$(date -Iseconds)] done → ${PROTEIN_TOP4}"
