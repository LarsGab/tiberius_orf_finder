#!/bin/bash
# Download PacBio Iso-Seq reads for the 4 long-read benchmark species.
#
# Reads accessions from data/longread/discovery.tsv (written by
# discover_longread_sra.py).
#
# Download method by accession prefix:
#   ERR / DRR (ENA)  : query ENA portal API for the FASTQ FTP path, then
#                      wget stream + gunzip inline. No prefetch needed.
#   SRR (NCBI)       : prefetch --max-size + fastq-dump --stdout.
#
# Subsample fraction is applied via an inline Python sampler for Gallus
# and Bos (very large runs) to keep the final FASTQ around 50 GB.
#
# Output: /projects/AI-GUSTUS/tiberius_orf_finder/data/longread/${sp}/reads.fq.gz
#
# Usage: sbatch scripts/slurm_longread_download.sh
#
#SBATCH --job-name=lr_download
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_download_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_download_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

declare -a SPECIES=(
    "Gallus_gallus"
    "Bos_taurus"
    "Zootoca_vivipara"
    "Betta_splendens"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

# Subsample fraction + number of runs to use per species.
# Gallus/Bos: one large run, subsample to ~50 GB.
# Zootoca/Betta: all available runs, no subsampling.
case "${species}" in
    Gallus_gallus)    N_RUNS=1; SUBSAMPLE_FRAC=0.150 ;;  # ~50 GB out of ~334 GB actual
    Bos_taurus)       N_RUNS=1; SUBSAMPLE_FRAC=0.100 ;;  # ~50 GB out of ~499 GB actual
    Zootoca_vivipara) N_RUNS=5; SUBSAMPLE_FRAC=1.0   ;;
    Betta_splendens)  N_RUNS=5; SUBSAMPLE_FRAC=1.0   ;;
    *) echo "unknown species: ${species}" >&2; exit 2 ;;
esac

DISCOVERY_TSV=${PROJDIR}/data/longread/discovery.tsv
OUTDIR=${PROJDIR}/data/longread/${species}
# SRA cache: large temp area; cleaned per run to avoid orphaned .sra files.
SRA_CACHE=${PROJDIR}/data/longread/sra_cache/${species}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}" "${SRA_CACHE}"

test -s "${DISCOVERY_TSV}" || { echo "missing discovery TSV: ${DISCOVERY_TSV}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Read top N_RUNS accessions for this species from the discovery TSV.
# Awk uses 1-based field indices: $1=species_input, $3=run_accession.
mapfile -t ACCESSIONS < <(
    awk -F'\t' -v sp="${species}" -v n="${N_RUNS}" \
        'NR>1 && $1==sp {print $3; count++; if(count>=n) exit}' \
        "${DISCOVERY_TSV}"
)

if [[ ${#ACCESSIONS[@]} -eq 0 ]]; then
    echo "[skip] ${species}: no accessions found in ${DISCOVERY_TSV}" >&2
    exit 2
fi
echo "[$(date -Iseconds)] species=${species} accessions=(${ACCESSIONS[*]}) subsample=${SUBSAMPLE_FRAC}"

OUT_FQ=${OUTDIR}/reads.fq.gz

# Truncate output so a restart starts fresh.
> "${OUT_FQ}.tmp"

# Inline Python subsampler used when SUBSAMPLE_FRAC < 1.
SUBSAMPLE_PY='
import sys, random
random.seed(42)
frac = float(sys.argv[1])
inp = sys.stdin.buffer
out = sys.stdout.buffer
while True:
    h = inp.readline()
    if not h:
        break
    s = inp.readline()
    p = inp.readline()
    q = inp.readline()
    if frac >= 1.0 or random.random() < frac:
        out.write(h + s + p + q)
'

for acc in "${ACCESSIONS[@]}"; do
    if [[ "${acc}" == ERR* || "${acc}" == DRR* ]]; then
        # ENA/DDBJ accession: download FASTQ directly from EBI FTP via HTTPS.
        # prefetch cannot resolve ERA accessions without extra vdb-config.
        echo "[$(date -Iseconds)] querying ENA API for FASTQ path: ${acc}"
        FTP_PATHS=$(curl -sSf \
            "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${acc}&result=read_run&fields=fastq_ftp&format=tsv" \
            | awk 'NR==2{print $1}')
        if [[ -z "${FTP_PATHS}" ]]; then
            echo "[error] ENA API returned no FASTQ FTP path for ${acc}" >&2
            exit 2
        fi
        # fastq_ftp may be semicolon-separated for paired-end; Iso-Seq is single-end.
        IFS=';' read -ra FTP_FILES <<< "${FTP_PATHS}"
        for ftp_path in "${FTP_FILES[@]}"; do
            [[ -z "${ftp_path}" ]] && continue
            URL="ftp://${ftp_path}"
            echo "[$(date -Iseconds)] wget ${URL} (frac=${SUBSAMPLE_FRAC})"
            wget -q --retry-connrefused --tries=10 --waitretry=60 -O - "${URL}" \
                | gunzip \
                | python3 -c "${SUBSAMPLE_PY}" "${SUBSAMPLE_FRAC}" \
                | gzip -1 >> "${OUT_FQ}.tmp"
        done
    else
        # NCBI SRR accession: use prefetch + fastq-dump.
        echo "[$(date -Iseconds)] prefetch ${acc}"
        prefetch --max-size 600G --output-directory "${SRA_CACHE}" "${acc}"

        # Locate the .sra or .sralite file (prefetch may write either).
        SRA_FILE=$(find "${SRA_CACHE}/${acc}" \( -name "*.sra" -o -name "*.sralite" \) 2>/dev/null | head -1 || true)
        if [[ -z "${SRA_FILE}" ]]; then
            echo "[warn] prefetch produced no .sra file for ${acc}; trying accession directly" >&2
            SRA_FILE="${acc}"
        fi

        echo "[$(date -Iseconds)] fastq-dump + subsample ${acc} (frac=${SUBSAMPLE_FRAC})"
        fastq-dump --readids --stdout "${SRA_FILE}" \
            | python3 -c "${SUBSAMPLE_PY}" "${SUBSAMPLE_FRAC}" \
            | gzip -1 >> "${OUT_FQ}.tmp"

        rm -rf "${SRA_CACHE}/${acc}"
    fi
done

mv "${OUT_FQ}.tmp" "${OUT_FQ}"

SIZE=$(du -sh "${OUT_FQ}" | cut -f1)
NREADS=$(zcat "${OUT_FQ}" | awk 'NR%4==1' | wc -l)
echo "[$(date -Iseconds)] done -> ${OUT_FQ}  size=${SIZE}  reads=${NREADS}"
