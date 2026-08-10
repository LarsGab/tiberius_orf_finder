#!/bin/bash
#SBATCH --job-name=excl_hypocr
#SBATCH --partition=snowball,pinky,batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/excl_hypocr.log

TAXIDS=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/hypocreales_taxids.txt
INPUT=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi.fa
OUTPUT=/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi_excl_Hypocreales.fa

echo "Building exclusion pattern from $TAXIDS..."
python3 -c "
ids = open('$TAXIDS').read().split()
# Build alternation for awk: each entry is a complete prefix match
print('|'.join(ids))
" > /tmp/hypocr_taxids_list.txt

echo "Filtering $INPUT -> $OUTPUT"
python3 - <<'PYEOF'
import sys

taxids_file = '/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/hypocreales_taxids.txt'
input_file  = '/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi.fa'
output_file = '/projects/AI-GUSTUS/tiberius_orf_finder/data/orthodb/Fungi_excl_Hypocreales.fa'

exclude = set(open(taxids_file).read().split())
print(f"Loaded {len(exclude)} Hypocreales taxids to exclude", flush=True)

kept = 0
skipped = 0
keep_current = False

with open(input_file) as fin, open(output_file, 'w') as fout:
    for line in fin:
        if line.startswith('>'):
            # header: >TAXID_1:XXXXXX
            tid = line[1:].split('_1:')[0].strip()
            keep_current = (tid not in exclude)
            if keep_current:
                kept += 1
                fout.write(line)
            else:
                skipped += 1
        elif keep_current:
            fout.write(line)

print(f"Kept {kept} sequences, excluded {skipped} sequences", flush=True)
PYEOF

echo "Done. Sequences in output:"
grep -c '^>' "$OUTPUT"
