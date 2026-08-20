#!/bin/bash
# Download OrthoDB12 partitioned protein FASTA files for Fungi and Viridiplantae.
# Destination: /projects/AI-GUSTUS/tiberius_orf_finder/odb/raw/
# (kept on /projects quota, not /home/gabriell which is nearly full)
#
#SBATCH --job-name=download_odb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/download_odb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/download_odb_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
ODB_DIR=${PROJDIR}/odb/raw
ODB_BASE=https://bioinf.uni-greifswald.de/bioinf/partitioned_odb12

mkdir -p "${PROJDIR}/logs" "${ODB_DIR}"

for clade in Fungi Viridiplantae; do
    dest="${ODB_DIR}/${clade}.fa.gz"
    if [[ -s "${dest}" ]]; then
        echo "[$(date -Iseconds)] Already on disk: ${dest} ($(du -sh "${dest}" | cut -f1))"
    else
        echo "[$(date -Iseconds)] Downloading ${clade}.fa.gz ..."
        wget -q --show-progress -c -O "${dest}" "${ODB_BASE}/${clade}.fa.gz"
        echo "[$(date -Iseconds)] Done: ${dest} ($(du -sh "${dest}" | cut -f1))"
    fi
done

echo "[$(date -Iseconds)] All downloads complete"
ls -lh "${ODB_DIR}"
