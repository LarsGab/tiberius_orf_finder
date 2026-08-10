#!/bin/bash
#SBATCH --job-name=excl_sacch
#SBATCH --partition=snowball,pinky,batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/excl_sacch.log

TAXIDS=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/saccharomycetales_taxids.txt
INPUT=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi.fa
OUTPUT=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi_excl_Saccharomycetales.fa

# Build awk pattern from taxids
python3 -c "
import sys
ids = open('$TAXIDS').read().split()
pat = '|'.join(f'^>{i}_' for i in ids)
print(pat)
" > /tmp/sacch_pattern.txt

PAT=$(cat /tmp/sacch_pattern.txt)

awk -v pat="$PAT" '
  /^>/ { keep = !match($0, pat) }
  keep { print }
' "$INPUT" > "$OUTPUT"

echo "Done. Sequences in output:"
grep -c '^>' "$OUTPUT"
