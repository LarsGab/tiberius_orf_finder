#!/bin/bash
# Download PacBio Iso-Seq reads for the 4 embryophyta test species that
# have long-read data on ENA/SRA (Arabidopsis, Eschscholzia, Medicago,
# Urochloa). Brachypodium, Freycinetia, Mimulus have no PacBio data.
#
# Reads accessions from data/longread_embryophyta/discovery.tsv.
#
# Download method by accession prefix:
#   ERR (ENA)   : query ENA portal API for FASTQ FTP path, then
#                 wget HTTPS + integrity check.
#   DRR (DDBJ)  : prefetch + fastq-dump --stdout  (ENA API has no fastq_ftp
#                 for DDBJ accessions; they resolve fine via NCBI prefetch).
#   SRR (NCBI)  : prefetch + fastq-dump --stdout.
#
# None of the 4 species requires subsampling (all runs < 70 GB).
#
# Output: ${PROJDIR}/data/longread_embryophyta/${sp}/reads.fq.gz
#
#SBATCH --job-name=lr_emb_dl
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_emb_dl_%A_%a.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/lr_emb_dl_%A_%a.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder

declare -a SPECIES=(
    "Arabidopsis_thaliana"
    "Eschscholzia_californica"
    "Medicago_truncatula"
    "Urochloa_brizantha"
)
species=${SPECIES[$SLURM_ARRAY_TASK_ID]}

# Number of runs to download per species (top-N from discovery TSV).
# Arabidopsis: 1 run (DRR377385, 67.5 GB) is ample for a 135 MB genome.
# Others: use all available runs.
case "${species}" in
    Arabidopsis_thaliana)    N_RUNS=1; SUBSAMPLE_FRAC=1.0 ;;
    Eschscholzia_californica) N_RUNS=5; SUBSAMPLE_FRAC=1.0 ;;
    Medicago_truncatula)     N_RUNS=2; SUBSAMPLE_FRAC=1.0 ;;
    Urochloa_brizantha)      N_RUNS=5; SUBSAMPLE_FRAC=1.0 ;;
    *) echo "unknown species: ${species}" >&2; exit 2 ;;
esac

DISCOVERY_TSV=${PROJDIR}/data/longread_embryophyta/discovery.tsv
OUTDIR=${PROJDIR}/data/longread_embryophyta/${species}
SRA_CACHE=${PROJDIR}/data/longread_embryophyta/sra_cache/${species}

mkdir -p "${PROJDIR}/logs" "${OUTDIR}" "${SRA_CACHE}"

test -s "${DISCOVERY_TSV}" || { echo "missing discovery TSV: ${DISCOVERY_TSV}" >&2; exit 2; }

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

# Awk uses 1-based field indices: $1=species_input, $3=run_accession.
mapfile -t ACCESSIONS < <(
    awk -F'\t' -v sp="${species}" -v n="${N_RUNS}" \
        'NR>1 && $1==sp {print $3; count++; if(count>=n) exit}' \
        "${DISCOVERY_TSV}"
)

if [[ ${#ACCESSIONS[@]} -eq 0 ]]; then
    echo "[skip] ${species}: no accessions in ${DISCOVERY_TSV}" >&2
    exit 2
fi
echo "[$(date -Iseconds)] species=${species} accessions=(${ACCESSIONS[*]}) subsample=${SUBSAMPLE_FRAC}"

OUT_FQ=${OUTDIR}/reads.fq.gz
> "${OUT_FQ}.tmp"

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
    if [[ "${acc}" == ERR* ]]; then
        # ENA (ERR) accession: download gzipped FASTQ from EBI via HTTPS.
        # DRR (DDBJ) and SRR (NCBI) accessions are NOT served by the ENA
        # fastq_ftp endpoint and must go through prefetch below.
        echo "[$(date -Iseconds)] querying ENA API: ${acc}"
        FTP_PATHS=$(curl -sSf \
            "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${acc}&result=read_run&fields=fastq_ftp&format=tsv" \
            | awk 'NR==2{print $1}')
        if [[ -z "${FTP_PATHS}" ]]; then
            echo "[error] ENA API: no fastq_ftp for ${acc}" >&2; exit 2
        fi
        IFS=';' read -ra FTP_FILES <<< "${FTP_PATHS}"
        for ftp_path in "${FTP_FILES[@]}"; do
            [[ -z "${ftp_path}" ]] && continue
            URL="https://${ftp_path}"
            TEMP_GZ="${OUTDIR}/raw_${acc}.fastq.gz"
            echo "[$(date -Iseconds)] wget -c ${URL}"
            wget -q -c --retry-connrefused --tries=10 --waitretry=60 \
                -O "${TEMP_GZ}" "${URL}"
            gzip -t "${TEMP_GZ}" || { echo "[error] gzip check failed: ${TEMP_GZ}" >&2; exit 2; }
            gunzip -c "${TEMP_GZ}" \
                | python3 -c "${SUBSAMPLE_PY}" "${SUBSAMPLE_FRAC}" \
                | gzip -1 >> "${OUT_FQ}.tmp"
            rm -f "${TEMP_GZ}"
        done
    else
        # DRR (DDBJ) or SRR (NCBI) accession: prefetch + fastq-dump.
        echo "[$(date -Iseconds)] prefetch ${acc}"
        prefetch --max-size 600G --output-directory "${SRA_CACHE}" "${acc}"
        SRA_FILE=$(find "${SRA_CACHE}/${acc}" \( -name "*.sra" -o -name "*.sralite" \) 2>/dev/null | head -1 || true)
        [[ -z "${SRA_FILE}" ]] && SRA_FILE="${acc}"
        echo "[$(date -Iseconds)] fastq-dump ${acc} (frac=${SUBSAMPLE_FRAC})"
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
