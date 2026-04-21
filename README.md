# tiberius_orf_finder

Nextflow pipeline that generates per-species training data for the Tiberius
deep-learning ORF finder.  For each species in a CSV it fetches the genome and
reference annotation, aligns RNA-seq reads with VARUS+HISAT2, assembles
transcripts with StringTie, projects reference CDS labels onto the assembled
transcripts, and writes the result as TFRecord shards ready for model training.

## Requirements

- Nextflow ≥ 23
- Python ≥ 3.9 with `tiberius_orf_finder` installed (`pip install -e ".[from_source]"`)
- NCBI `datasets` CLI (for RefSeq species)
- VARUS, HISAT2, StringTie, samtools, gffread on `PATH`

## Quick start

```bash
nextflow run nextflow/main.nf \
  --species_csv  nextflow/conf/species_training.csv \
  --braker_data_dir /path/to/braker_data \
  --varus_dir    /path/to/VARUS \
  --varus_impl   /path/to/VARUS/Implementation \
  --hisat_dir    /path/to/hisat2 \
  --outdir       results/training
```

On the group's HPC cluster (`brain`):

```bash
nextflow run nextflow/main.nf -c nextflow/conf/brain.config \
  --species_csv nextflow/conf/species_training.csv \
  --braker_data_dir /home/gabriell/... \
  --varus_dir ...
```

## Pipeline stages

| Module | Process(es) | Key outputs |
|---|---|---|
| `nextflow/modules/fetch.nf` | `FETCH_ASSEMBLY` | `genome.fa`, `annotation.gff` |
| `nextflow/modules/varus.nf` | `MAKE_VARUS_PARAMS`, `RUN_VARUS` | `VARUSparameters.txt`, `VARUS.bam` |
| `nextflow/modules/stringtie.nf` | `RUN_STRINGTIE` | `stringtie.gtf`, `transcripts.fa` |
| `nextflow/modules/label.nf` | `LABEL_TRANSCRIPTS` | `labels.npz`, `transcripts_labelled.fa`, `stats.tsv` |
| `nextflow/modules/tfrecord.nf` | `WRITE_TFRECORD` | `data.tfrecords` |

A `tfrecord_manifest.tsv` listing all shards is written to `--outdir`.

## Species CSV format

```
species,accession,annotation
Drosophila melanogaster,GCF_000001215.4,RefSeq
Bombyx mori,GCF_000151625.1,RefSeq
Apis mellifera,,BRAKER
```

`annotation` is either `RefSeq` (downloaded via NCBI `datasets`) or `BRAKER`
(staged from `--braker_data_dir/<Genus_species>/`).

## Label schema

Per-position integer labels written to `labels.npz`:

| Value | Class | Meaning |
|---|---|---|
| 0 | IR | intergenic / non-coding |
| 1 | START | A of ATG start codon |
| 2 | E1 | coding, frame offset 1 |
| 3 | E2 | coding, frame offset 2 |
| 4 | E0 | coding, frame offset 0 |
| 5 | STOP | last base of stop codon |

Cycle within an ORF: `START E1 E2 E0 E1 E2 E0 … STOP`.

## TFRecord schema

Each example encodes one `--chunk-len`-nt window (default 9999):

| Feature | Type | Shape | Description |
|---|---|---|---|
| `input` | bytes (uint8 tensor) | `[L, 6]` | one-hot A,C,G,T,N + PAD channel |
| `output` | bytes (uint8 tensor) | `[L, 6]` | one-hot label (IR,START,E1,E2,E0,STOP) |
| `tx_id` | bytes (utf-8) | scalar | transcript identifier |
| `chunk_idx` | int64 | scalar | 0-based chunk index within the transcript |

Padded positions (short final chunk) have `input[..., 5] == 1` and
`output == 0`; the training loop should mask these positions in the loss.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--species_csv` | required | CSV with columns `species,accession,annotation` |
| `--braker_data_dir` | required | root dir for BRAKER-annotated species |
| `--varus_dir` | — | VARUS installation root (provides `runVARUS.pl`) |
| `--varus_runpl` | — | explicit path to `runVARUS.pl` (alternative to `--varus_dir`) |
| `--varus_impl` | required | path to VARUS `Implementation/` dir |
| `--hisat_dir` | required | HISAT2 installation dir |
| `--varus_max_batches` | 1000 | max VARUS download batches per species |
| `--threads` | 8 | default CPU threads per process |
| `--chunk_len` | 9999 | TFRecord window length (nt) |
| `--outdir` | `results` | output root |

## Installation

```bash
git clone https://github.com/LarsGab/tiberius_orf_finder
cd tiberius_orf_finder
pip install -e ".[from_source]"   # installs tiberius, tensorflow, etc.
# or for tests only:
pip install -e ".[test]"
pytest tests/
```
