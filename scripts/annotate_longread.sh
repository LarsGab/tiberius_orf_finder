#!/bin/bash
# Run the ORF finder on a PacBio Iso-Seq (or ONT cDNA/dRNA) library.
#
# Steps:
#   1. minimap2 splice-aware alignment of FASTQ to genome
#   2. samtools sort -> BAM
#   3. scripts/annotate.py --bam ... --longread (StringTie -L + model + HMM)
#
# Required tools on PATH: minimap2, samtools, stringtie, gffread, python.
#
# Example:
#   bash scripts/annotate_longread.sh \
#     --fastq    data/longread_test/Drosophila_melanogaster/SRR31444194.fastq.gz \
#     --genome   data/longread_test/Drosophila_melanogaster/genome.fa \
#     --weights  results/models/cnn_lstm_run006/epoch_157.weights.h5 \
#     --config   configs/cnn_lstm_run006.yaml \
#     --out      results/longread/Drosophila_melanogaster.gtf \
#     --tech     pacbio-isoseq \
#     --threads  8

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 --fastq FQ --genome FA --weights H5 --config YAML --out GTF [options]

Required:
  --fastq FQ        Long-read RNA-seq FASTQ (.fq, .fastq, .fq.gz, .fastq.gz)
  --genome FA       Reference genome FASTA
  --weights H5      Trained model weights (.weights.h5)
  --config YAML     Model config used at training time
  --out GTF         Output annotation GTF

Options:
  --tech STR        pacbio-isoseq (default) | pacbio-hifi | ont-cdna | ont-drna
  --threads N       Threads for minimap2/samtools/stringtie (default: 8)
  --batch-size N    Inference batch size (default: 200)
  --tmp-dir DIR     Workdir for BAM and intermediates (default: temp dir, deleted)
  --keep-tmp        Keep workdir on success
  -h, --help        Show this help
EOF
}

FASTQ=""; GENOME=""; WEIGHTS=""; CONFIG=""; OUT=""
TECH="pacbio-isoseq"
THREADS=8
BATCH=200
TMP_DIR=""
KEEP_TMP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fastq)      FASTQ="$2";   shift 2 ;;
        --genome)     GENOME="$2";  shift 2 ;;
        --weights)    WEIGHTS="$2"; shift 2 ;;
        --config)     CONFIG="$2";  shift 2 ;;
        --out)        OUT="$2";     shift 2 ;;
        --tech)       TECH="$2";    shift 2 ;;
        --threads)    THREADS="$2"; shift 2 ;;
        --batch-size) BATCH="$2";   shift 2 ;;
        --tmp-dir)    TMP_DIR="$2"; shift 2 ;;
        --keep-tmp)   KEEP_TMP=1;   shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

for v in FASTQ GENOME WEIGHTS CONFIG OUT; do
    [[ -n "${!v}" ]] || { echo "Missing --${v,,}" >&2; usage; exit 2; }
done
[[ -s "$FASTQ"   ]] || { echo "Not found: $FASTQ"   >&2; exit 2; }
[[ -s "$GENOME"  ]] || { echo "Not found: $GENOME"  >&2; exit 2; }
[[ -s "$WEIGHTS" ]] || { echo "Not found: $WEIGHTS" >&2; exit 2; }
[[ -s "$CONFIG"  ]] || { echo "Not found: $CONFIG"  >&2; exit 2; }

case "$TECH" in
    pacbio-isoseq|pacbio-hifi) MM2_PRESET="splice:hq"; MM2_EXTRA="" ;;
    ont-cdna)                  MM2_PRESET="splice";    MM2_EXTRA="" ;;
    ont-drna)                  MM2_PRESET="splice";    MM2_EXTRA="-uf -k14" ;;
    *) echo "Unknown --tech: $TECH (use pacbio-isoseq|pacbio-hifi|ont-cdna|ont-drna)" >&2; exit 2 ;;
esac

if [[ -z "$TMP_DIR" ]]; then
    TMP_DIR="$(mktemp -d -t annotate_longread_XXXXXX)"
    CLEANUP=1
else
    mkdir -p "$TMP_DIR"
    CLEANUP=0
fi
[[ "$KEEP_TMP" -eq 1 ]] && CLEANUP=0
trap '[[ "$CLEANUP" -eq 1 ]] && rm -rf "$TMP_DIR"' EXIT

echo "[$(date +%T)] tmp dir: $TMP_DIR"
echo "[$(date +%T)] tech=$TECH  minimap2 preset=$MM2_PRESET  extra='$MM2_EXTRA'"

BAM="$TMP_DIR/aligned.sorted.bam"

echo "[$(date +%T)] minimap2 + samtools sort"
# shellcheck disable=SC2086
minimap2 -t "$THREADS" -ax "$MM2_PRESET" $MM2_EXTRA --secondary=no \
         "$GENOME" "$FASTQ" \
    | samtools sort -@ "$THREADS" -o "$BAM" -
samtools index -@ "$THREADS" "$BAM"

echo "[$(date +%T)] alignment stats:"
samtools flagstat "$BAM" | sed 's/^/  /'

mkdir -p "$(dirname "$OUT")"

echo "[$(date +%T)] running annotate.py (StringTie -L + model + HMM)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/annotate.py" \
    --bam        "$BAM" \
    --genome     "$GENOME" \
    --weights    "$WEIGHTS" \
    --config     "$CONFIG" \
    --out        "$OUT" \
    --batch-size "$BATCH" \
    --threads    "$THREADS" \
    --longread

echo "[$(date +%T)] done -> $OUT"
