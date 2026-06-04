#!/usr/bin/env bash
# Run the full per-species short-read pipeline (FETCH-was-done → RUNLIST →
# INDEX → VARUS_RUN → STRINGTIE → LABEL → WRITE_TFRECORD) for a single
# species, locally, **without Nextflow / without SLURM**. Each stage skips
# itself if its output already exists, so this also works to resume a
# partial run rsynced off `brain`.
#
# Usage:
#   scripts/run_species_local.sh <Genus_species> <species_dir> [threads]
#
# If the assembly is not already present at:
#   <species_dir>/assembly/genome.fa
#   <species_dir>/assembly/annotation.gff
# the script will fetch it from one of:
#   - rsync from ${REMOTE_HOST}:${REMOTE_PHYTOZOME_DIR}/<Genus_species>/
#     (defaults are brain + the Mesangiospermae Figshare staging dir,
#      so this Just Works for Embryophyta if you can SSH to brain), OR
#   - NCBI `datasets download genome accession ${NCBI_ACCESSION}`
#     (set NCBI_ACCESSION=GCA_... or GCF_... to override the rsync path).
#
# Optional cached state (any subset, rsync from brain to skip a stage):
#   <species_dir>/varus/Runlist.tsv
#   <species_dir>/varus/genome_index/hisatidx.*.{ht2,ht2l}
#   <species_dir>/varus/VARUS.bam
#   <species_dir>/stringtie/stringtie.gtf
#   <species_dir>/stringtie/transcripts.fa
#   <species_dir>/labels/labels.npz
#   <species_dir>/labels/transcripts_labelled.fa
#
# Produces:
#   <species_dir>/tfrecord/data.tfrecords  (the goal)
#   appends a `<Genus_species>\t<abs_path>` line to ${MANIFEST:-tfrecord_manifest_local.tsv}
#
# Tools that must be on $PATH (provide via micromamba `orffinder` env or your
# choice of conda/system install): hisat2, samtools, stringtie, gffread,
# fastq-dump, python (with varus, biopython, numpy, pyyaml, tensorflow).
#
# Tiberius repo Python modules used:
#   tiberius_orf.data.label_transcripts
#   tiberius_orf.data.chunk_tfrecord
# Requires the repo on PYTHONPATH (see PROJDIR below).

set -euo pipefail

SPECIES_UNDER="${1:?usage: $0 <Genus_species> <species_dir> [threads]}"
SPDIR="${2:?usage: $0 <Genus_species> <species_dir> [threads]}"
THREADS="${3:-8}"

# Binomial as VARUS expects it (e.g. "Yucca filamentosa").
SPECIES="${SPECIES_UNDER//_/ }"

# Project paths – override on the env if your layout differs.
PROJDIR="${PROJDIR:-$(cd "$(dirname "$0")/.." && pwd)}"
MANIFEST="${MANIFEST:-${PROJDIR}/tfrecord_manifest_local.tsv}"
CHUNK_LEN="${CHUNK_LEN:-9999}"
BATCH_SIZE="${BATCH_SIZE:-50000}"
MAX_BATCHES="${MAX_BATCHES:-1000}"
MIN_UNIQ_PCT="${MIN_UNIQ_PCT:-5.0}"

export PYTHONPATH="${PROJDIR}/src:${PYTHONPATH:-}"

# Remote rsync source for the Mesangiospermae Figshare-staged genomes
# (override REMOTE_HOST / REMOTE_PHYTOZOME_DIR if you have it elsewhere).
REMOTE_HOST="${REMOTE_HOST:-brain}"
REMOTE_PHYTOZOME_DIR="${REMOTE_PHYTOZOME_DIR:-/home/gabriell/tiberius_mesangiospermae_training_data/by_species}"
NCBI_ACCESSION="${NCBI_ACCESSION:-}"   # optional override; if set, use NCBI datasets instead of rsync

GENOME="${SPDIR}/assembly/genome.fa"
ANNOT="${SPDIR}/assembly/annotation.gff"

mkdir -p "${SPDIR}/assembly" "${SPDIR}/varus" "${SPDIR}/stringtie" "${SPDIR}/labels" "${SPDIR}/tfrecord"

cd "${SPDIR}"
echo "[${SPECIES_UNDER}] working in ${SPDIR}"

# ------- 0) FETCH_ASSEMBLY -------
if [[ -s "${GENOME}" && -s "${ANNOT}" ]]; then
    echo "[${SPECIES_UNDER}] assembly: cached"
elif [[ -n "${NCBI_ACCESSION}" ]]; then
    echo "[${SPECIES_UNDER}] assembly: NCBI datasets download (${NCBI_ACCESSION})"
    pushd assembly > /dev/null
    rm -rf ncbi ncbi.zip
    datasets download genome accession "${NCBI_ACCESSION}" --include genome,gff3 --filename ncbi.zip
    unzip -o -q ncbi.zip -d ncbi
    cat "ncbi/ncbi_dataset/data/${NCBI_ACCESSION}/"*_genomic.fna > genome.fa
    cp "ncbi/ncbi_dataset/data/${NCBI_ACCESSION}/genomic.gff" annotation.gff
    rm -rf ncbi ncbi.zip
    popd > /dev/null
else
    echo "[${SPECIES_UNDER}] assembly: rsync from ${REMOTE_HOST}:${REMOTE_PHYTOZOME_DIR}/${SPECIES_UNDER}/"
    rsync -avL --progress \
        "${REMOTE_HOST}:${REMOTE_PHYTOZOME_DIR}/${SPECIES_UNDER}/genome.fa" \
        "${REMOTE_HOST}:${REMOTE_PHYTOZOME_DIR}/${SPECIES_UNDER}/annotation.gff" \
        assembly/
fi
[[ -s "${GENOME}" ]] || { echo "FETCH failed: missing ${GENOME}" >&2; exit 2; }
[[ -s "${ANNOT}"  ]] || { echo "FETCH failed: missing ${ANNOT}"  >&2; exit 2; }

# ------- 1) VARUS_RUNLIST_SR -------
if [[ -s varus/Runlist.tsv ]]; then
    echo "[${SPECIES_UNDER}] runlist: cached"
else
    echo "[${SPECIES_UNDER}] runlist: querying NCBI Entrez"
    python -m varus.cli runlist "${SPECIES}" --outdir varus --max-runs 0
    [[ -s varus/Runlist.tsv ]] || { echo "no SRA runs for ${SPECIES}; species can't be recovered locally" >&2; exit 3; }
fi

# ------- 2) VARUS_INDEX_SR -------
if compgen -G "varus/genome_index/hisatidx.1.ht2*" > /dev/null; then
    echo "[${SPECIES_UNDER}] index: cached"
else
    echo "[${SPECIES_UNDER}] index: building HISAT2 index"
    mkdir -p varus/genome_index
    python -m varus.cli index "${GENOME}" \
        --outdir varus/genome_index \
        --threads "${THREADS}" \
        --prefix hisatidx
fi

# ------- 3) VARUS_RUN_SR -------
if [[ -s varus/VARUS.bam ]]; then
    echo "[${SPECIES_UNDER}] varus run: cached"
else
    echo "[${SPECIES_UNDER}] varus run: downloading + aligning SRA (this is the slow step)"
    /usr/bin/time -p -o varus/runtime.varus.txt \
      python -m varus.cli run "${SPECIES}" "${GENOME}" \
        --runlist varus/Runlist.tsv \
        --index   varus/genome_index/hisatidx \
        --outdir  varus \
        --batch-size  "${BATCH_SIZE}" \
        --max-batches "${MAX_BATCHES}" \
        --min-uniq-pct "${MIN_UNIQ_PCT}" \
        --threads  "${THREADS}" \
        --seed 1
    [[ -s varus/VARUS.bam ]] || { echo "VARUS produced no BAM for ${SPECIES}" >&2; exit 4; }
fi

# ------- 4) RUN_STRINGTIE -------
if [[ -s stringtie/stringtie.gtf && -s stringtie/transcripts.fa ]]; then
    echo "[${SPECIES_UNDER}] stringtie: cached"
else
    echo "[${SPECIES_UNDER}] stringtie: sort + assemble + extract transcripts"
    samtools sort -@ "${THREADS}" -o stringtie/sorted.bam varus/VARUS.bam
    stringtie stringtie/sorted.bam -o stringtie/stringtie.gtf -p "${THREADS}"
    gffread -w stringtie/transcripts.fa -g "${GENOME}" stringtie/stringtie.gtf
    rm -f stringtie/sorted.bam
fi

# ------- 5) LABEL_TRANSCRIPTS -------
if [[ -s labels/labels.npz && -s labels/transcripts_labelled.fa ]]; then
    echo "[${SPECIES_UNDER}] label: cached"
else
    echo "[${SPECIES_UNDER}] label: projecting reference CDS onto transcripts"
    python3 -m tiberius_orf.data.label_transcripts \
        --stringtie-gtf  stringtie/stringtie.gtf \
        --reference-gff  "${ANNOT}" \
        --transcripts-fa stringtie/transcripts.fa \
        --out-dir        labels
    [[ -s labels/labels.npz && -s labels/transcripts_labelled.fa ]] || { echo "label step produced no output" >&2; exit 5; }
fi

# ------- 6) WRITE_TFRECORD -------
if [[ -s tfrecord/data.tfrecords ]]; then
    echo "[${SPECIES_UNDER}] tfrecord: cached"
else
    echo "[${SPECIES_UNDER}] tfrecord: chunking labelled transcripts"
    python3 -m tiberius_orf.data.chunk_tfrecord \
        --fasta  labels/transcripts_labelled.fa \
        --labels labels/labels.npz \
        --out    tfrecord/data.tfrecords \
        --chunk-len "${CHUNK_LEN}"
    [[ -s tfrecord/data.tfrecords ]] || { echo "tfrecord step produced no output" >&2; exit 6; }
fi

# ------- manifest append -------
abspath="$(readlink -f tfrecord/data.tfrecords)"
mkdir -p "$(dirname "${MANIFEST}")"
# Idempotent append: drop any old line for this species, then add new one.
if [[ -s "${MANIFEST}" ]]; then
    tmp="$(mktemp)"
    awk -F'\t' -v sp="${SPECIES}" '$1!=sp' "${MANIFEST}" > "${tmp}"
    mv "${tmp}" "${MANIFEST}"
fi
printf '%s\t%s\n' "${SPECIES}" "${abspath}" >> "${MANIFEST}"

echo "[${SPECIES_UNDER}] DONE: ${abspath}"
echo "[${SPECIES_UNDER}] manifest: ${MANIFEST}"
