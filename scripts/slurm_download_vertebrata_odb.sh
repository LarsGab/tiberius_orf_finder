#!/bin/bash
# Re-download the Vertebrata OrthoDB partition into the shared
# tiberius_proteins_analysis staging dir, matching what the neighbour
# project's slurm/00_setup.sh would have produced. nodes.dmp is not
# touched (still on disk from the previous setup).
#
# After this job succeeds, submit
#   sbatch --dependency=afterok:$THIS
#          /home/gabriell/tiberius_proteins/slurm/vertebrates_test/02_filter_odb.sh
# to rebuild the 9 per-species order-excluded ODB fastas.
#
#SBATCH --job-name=dl_odb_vt
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/dl_odb_vt_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/dl_odb_vt_%j.err

set -euo pipefail

WORK_DIR=/home/gabriell/tiberius_proteins_analysis
URL="https://bioinf.uni-greifswald.de/bioinf/partitioned_odb12/Vertebrata.fa.gz"
DEST="${WORK_DIR}/odb/raw/Vertebrata.fa.gz"

mkdir -p "$(dirname "${DEST}")" /projects/AI-GUSTUS/tiberius_orf_finder/logs

if [[ -s "${DEST}" ]]; then
    echo "[dl_odb_vt] already present, skipping: ${DEST}"
    exit 0
fi

echo "[dl_odb_vt] downloading ${URL} -> ${DEST}"
wget -q --show-progress -O "${DEST}" "${URL}"
echo "[dl_odb_vt] done, size=$(du -h "${DEST}" | cut -f1)"
