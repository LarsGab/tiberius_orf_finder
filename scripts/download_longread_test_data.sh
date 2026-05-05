#!/bin/bash
# Download long-read RNA-seq + reference genome for test species.
#
# Reads scripts/longread_test_species.csv and writes per-species directories:
#   <out_dir>/<Genus_species>/
#       <SRR>.fastq.gz       (long-read FASTQ; subsampled if --max-reads given)
#       genome.fa            (RefSeq genome from NCBI datasets)
#       annotation.gff       (RefSeq GFF, for downstream evaluation)
#
# Required tools on PATH:
#   prefetch, fasterq-dump  (sra-tools)
#   pigz or gzip
#   datasets, unzip          (NCBI CLI for genome download)
#
# Notes:
#   --max-reads N  caps reads per run via fasterq-dump's --maxSpotId. Useful
#                  for the Apis mellifera run (~40M reads, ~62 Gb full).
#   The script is idempotent: it skips outputs that already exist non-empty.
#
# Example:
#   bash scripts/download_longread_test_data.sh \
#     --out-dir data/longread_test \
#     --max-reads 200000 \
#     --threads 8

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--csv FILE] --out-dir DIR [options]

Options:
  --csv FILE       Species/accession CSV (default: scripts/longread_test_species.csv)
  --out-dir DIR    Output root directory (required)
  --species NAME   Only download this species (matches CSV "species" column)
  --max-reads N    Cap reads per run (default: no cap)
  --threads N      Threads for fasterq-dump/gzip (default: 4)
  --skip-genome    Don't download the genome
  --skip-fastq     Don't download the FASTQ
  -h, --help       Show this help
EOF
}

CSV=""
OUT_DIR=""
SPECIES_FILTER=""
MAX_READS=""
THREADS=4
SKIP_GENOME=0
SKIP_FASTQ=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)         CSV="$2";            shift 2 ;;
        --out-dir)     OUT_DIR="$2";        shift 2 ;;
        --species)     SPECIES_FILTER="$2"; shift 2 ;;
        --max-reads)   MAX_READS="$2";      shift 2 ;;
        --threads)     THREADS="$2";        shift 2 ;;
        --skip-genome) SKIP_GENOME=1;       shift ;;
        --skip-fastq)  SKIP_FASTQ=1;        shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$OUT_DIR" ]] || { echo "Missing --out-dir" >&2; usage; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "$CSV" ]] || CSV="$SCRIPT_DIR/longread_test_species.csv"
[[ -s "$CSV" ]] || { echo "Not found: $CSV" >&2; exit 2; }

mkdir -p "$OUT_DIR"

GZIP="gzip"
command -v pigz >/dev/null 2>&1 && GZIP="pigz -p $THREADS"

# Skip header line, iterate rows
tail -n +2 "$CSV" | while IFS=, read -r species sra_run genome_acc instrument selection reads bases study; do
    [[ -z "$species" ]] && continue
    if [[ -n "$SPECIES_FILTER" && "$species" != "$SPECIES_FILTER" ]]; then
        continue
    fi

    underscored="${species// /_}"
    sp_dir="$OUT_DIR/$underscored"
    mkdir -p "$sp_dir"

    echo
    echo "===== $species ($sra_run, genome $genome_acc) -> $sp_dir ====="

    # 1. FASTQ
    fq_gz="$sp_dir/${sra_run}.fastq.gz"
    if [[ "$SKIP_FASTQ" -eq 0 ]]; then
        if [[ -s "$fq_gz" ]]; then
            echo "[skip] $fq_gz exists"
        else
            echo "[fastq] prefetch $sra_run"
            prefetch --output-directory "$sp_dir" "$sra_run"

            echo "[fastq] fasterq-dump $sra_run (threads=$THREADS, max=${MAX_READS:-all})"
            fqd_args=(--threads "$THREADS" --outdir "$sp_dir" --temp "$sp_dir/.fqd_tmp" --concatenate-reads)
            [[ -n "$MAX_READS" ]] && fqd_args+=(--maxSpotId "$MAX_READS")
            fasterq-dump "${fqd_args[@]}" "$sp_dir/$sra_run"

            # fasterq-dump may write either ${sra_run}.fastq or split files; gzip whatever appears.
            if [[ -s "$sp_dir/${sra_run}.fastq" ]]; then
                $GZIP -f "$sp_dir/${sra_run}.fastq"
            else
                echo "[warn] no ${sra_run}.fastq produced — check sra-tools output" >&2
                ls -la "$sp_dir" >&2
                exit 3
            fi
            rm -rf "$sp_dir/$sra_run" "$sp_dir/.fqd_tmp"
        fi
    fi

    # 2. Genome + GFF
    if [[ "$SKIP_GENOME" -eq 0 ]]; then
        if [[ -s "$sp_dir/genome.fa" && -s "$sp_dir/annotation.gff" ]]; then
            echo "[skip] genome.fa + annotation.gff exist"
        else
            echo "[genome] datasets download genome $genome_acc"
            tmpzip="$sp_dir/.ncbi.zip"
            datasets download genome accession "$genome_acc" \
                --include genome,gff3 \
                --filename "$tmpzip"
            unzip -o -q "$tmpzip" -d "$sp_dir/.ncbi"
            cat "$sp_dir/.ncbi/ncbi_dataset/data/$genome_acc"/*_genomic.fna > "$sp_dir/genome.fa"
            cp  "$sp_dir/.ncbi/ncbi_dataset/data/$genome_acc/genomic.gff" "$sp_dir/annotation.gff"
            rm -rf "$sp_dir/.ncbi" "$tmpzip"
        fi
    fi

    echo "[done] $species"
done

echo
echo "All requested species processed. Output root: $OUT_DIR"
