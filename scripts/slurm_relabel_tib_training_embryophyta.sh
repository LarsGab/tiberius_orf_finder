#!/bin/bash
# Re-run join_tmap_labels.py on all embryophyta training species' Tiberius
# features, after fixing the transcript_id parsing bug that left every
# gffcompare_class = NA on Tiberius outputs (StringTie ORFs had identical
# gene_id / transcript_id so were unaffected).
#
#SBATCH --job-name=relabel_tib_emb
#SBATCH --partition=snowball,pinky,batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/logs/relabel_tib_emb_%j.out
#SBATCH --error=/projects/AI-GUSTUS/tiberius_orf_finder/logs/relabel_tib_emb_%j.err

set -euo pipefail

PROJDIR=/projects/AI-GUSTUS/tiberius_orf_finder
TRAINDIR=${PROJDIR}/results/training_embryophyta_v2

eval "$(micromamba shell hook --shell bash)"
micromamba activate orffinder

cd "${PROJDIR}"

for f in "${TRAINDIR}"/*/tiberius/tib_features.tsv; do
    sp_dir=$(dirname "$(dirname "${f}")")
    sp=$(basename "${sp_dir}")
    tracking="${sp_dir}/tiberius/gffcompare/gffcmp.tracking"
    if [[ ! -s "${tracking}" ]]; then
        echo "SKIP ${sp}: no tracking"
        continue
    fi
    echo "== ${sp} =="
    python scripts/join_tmap_labels.py \
        --tracking "${tracking}" \
        --features "${f}"
done

echo "[$(date -Iseconds)] done"
