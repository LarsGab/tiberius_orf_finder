# tiberius_orf_finder

Deep-learning ORF finder for assembled transcripts. Given RNA-seq alignments
(or a StringTie GTF) plus the corresponding genome, it predicts the coding
region of every assembled transcript and writes a genomic GTF of CDS lines.

Architecture: CNN stem + BiLSTM (or CNN + Transformer) per-position
classifier over six labels (`IR / START / E1 / E2 / E0 / STOP`), followed by
a 6-state structured HMM that enforces ATG-starts, in-frame stops, and a
single canonical reading frame per ORF (`OrfAnnotationHMM`, built on top of
[bricks2marble](https://github.com/Gaius-Augustus/bricks2marble) and
[hidten](https://github.com/Gaius-Augustus/hidten)).

---

## Install

```bash
pip install -e ".[from_source]"   # tensorflow, biopython, numpy, pyyaml, bricks2marble, hidten
```

For the data-prep pipeline you additionally need on `PATH`: `nextflow`,
`gffread`, `stringtie`, `hisat2`, `samtools`, NCBI `datasets`, and a VARUS
checkout.

---

## Annotate a transcriptome

The main user-facing entry point is `scripts/annotate.py`. It accepts either
a StringTie GTF or a sorted BAM and emits one genomic GTF of predicted CDS.

**From an existing StringTie GTF:**

```bash
python scripts/annotate.py \
  --stringtie-gtf  path/to/stringtie.gtf \
  --genome         path/to/genome.fa \
  --weights        path/to/model.weights.h5 \
  --config         configs/default.yaml \
  --out            results/annotation.gtf \
  --batch-size     200
```

**From a sorted BAM (StringTie is run internally):**

```bash
python scripts/annotate.py \
  --bam            path/to/aligned.sorted.bam \
  --genome         path/to/genome.fa \
  --weights        path/to/model.weights.h5 \
  --config         configs/default.yaml \
  --out            results/annotation.gtf \
  --longread       # add for PacBio Iso-Seq / ONT spliced alignments
```

Under the hood:

1. `stringtie` (if needed) → `gffread` extracts the spliced transcript FASTA.
2. `bricks2marble.tools.annotate.annotate_genome` drives the inference loop:
   chunking, batched NN forward pass, per-chunk HMM Viterbi
   ([`hmm/decode.py`](src/tiberius_orf/hmm/decode.py)), and chunk-boundary
   repredictions.
3. Each detected ORF is projected from transcript coordinates back to
   genomic coordinates via the StringTie exon structure, with correct GTF
   phase per segment.

Output: a single GTF with one `CDS` line per genomic sub-interval of every
predicted ORF (1-based inclusive coordinates, `source = "tiberius_orf"`,
shared `transcript_id` / `gene_id` per ORF).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--batch-size` | 200 | NN inference batch size |
| `--parallel` | 100 | HMM parallel scan factor (chunks padded to a multiple) |
| `--no-codon-emitter` | off | drop hard ATG/STOP codon constraints (debug) |
| `--longread` | off | pass `-L` to StringTie for long-read assemblies |
| `--tmp-dir` | tempdir | keep intermediate StringTie/gffread output here |
| `--keep-tmp` | off | don't delete the tempdir on exit |

### Pre-trained weights

Trained model weights for the published splits live on the group HPC under
`/home/gabriell/tiberius_orf_finder/results/models/`. Pick the architecture
that matches your `--config` (`configs/default.yaml` is `cnn_lstm`).

---

## Train your own model

Training expects per-species TFRecord shards built by the Nextflow data
pipeline (see below).

```bash
python scripts/train.py \
  --train-manifest results/training/tfrecord_manifest.tsv \
  --val-manifest   results/val/tfrecord_manifest.tsv \
  --config         configs/default.yaml \
  --outdir         results/models/run_001
```

Override `--epochs`, `--batch-size`, `--lr` from the CLI; everything else is
in `configs/default.yaml`. Two architectures are supported via the
`model.type` field:

* `cnn_lstm` — CNN stem + reshape-pooled BiLSTM (default)
* `cnn_transformer` — CNN stem + Transformer encoder

Outputs to `--outdir`: `epoch_XX.weights.h5`, `final.weights.h5`,
`train_log.tsv`.

### Evaluate

Per-class precision / recall / F1 on a held-out test split, with an IR-only
baseline:

```bash
python scripts/evaluate.py \
  --test-manifest results/test/tfrecord_manifest.tsv \
  --weights       results/models/run_001/final.weights.h5 \
  --config        configs/default.yaml \
  --out           results/eval/run_001.tsv
```

### Per-species prediction GTFs

To dump reference + prediction GTFs per species (useful for visualising the
model's calls against the projected reference labels):

```bash
python scripts/predict.py \
  --test-manifest results/test/tfrecord_manifest.tsv \
  --weights       results/models/run_001/final.weights.h5 \
  --config        configs/default.yaml \
  --out-dir       results/predictions/run_001
```

---

## Data pipeline (Nextflow)

The Nextflow workflow at `nextflow/main.nf` turns a CSV of species into
training-ready TFRecord shards. For each species:

1. **Fetch** genome + reference annotation (`FETCH_ASSEMBLY`).
2. **Align** RNA-seq with VARUS + HISAT2 (`MAKE_VARUS_PARAMS`, `RUN_VARUS`).
3. **Assemble** transcripts with StringTie + gffread (`RUN_STRINGTIE`).
4. **Label** by projecting reference CDS onto the assembled transcripts
   (`LABEL_TRANSCRIPTS`).
5. **Chunk** into `chunk_len`-nt TFRecord windows (`WRITE_TFRECORD`).

A `tfrecord_manifest.tsv` listing all shards is written to `--outdir`.

### Run locally

```bash
nextflow run nextflow/main.nf \
  --species_csv     nextflow/conf/species_training.csv \
  --braker_data_dir /path/to/braker_data \
  --varus_dir       /path/to/VARUS \
  --varus_impl      /path/to/VARUS/Implementation \
  --hisat_dir       /path/to/hisat2 \
  --outdir          results/training
```

### Run on the brain HPC cluster

Pre-configured submit scripts wrap `nextflow/conf/brain*.config`:

```bash
sbatch nextflow/submit_training.sh
sbatch nextflow/submit_val.sh
sbatch nextflow/submit_test.sh
sbatch nextflow/submit_smoke.sh      # tiny species set, sanity check
```

### Species CSV

```
species,accession,annotation
Drosophila melanogaster,GCF_000001215.4,RefSeq
Apis mellifera,,BRAKER
```

`annotation` is `RefSeq` (downloaded via NCBI `datasets`) or `BRAKER`
(staged from `--braker_data_dir/<Genus_species>/`).

### Pipeline parameters

| Parameter | Default | Description |
|---|---|---|
| `--species_csv` | required | CSV: `species,accession,annotation` |
| `--braker_data_dir` | required | root dir for BRAKER-annotated species |
| `--varus_dir` | — | VARUS root (provides `runVARUS.pl`) |
| `--varus_runpl` | — | explicit path to `runVARUS.pl` (alternative) |
| `--varus_impl` | required | VARUS `Implementation/` dir |
| `--hisat_dir` | required | HISAT2 installation dir |
| `--varus_max_batches` | 1000 | max VARUS download batches per species |
| `--threads` | 8 | default CPU threads per process |
| `--chunk_len` | 9999 | TFRecord window length (nt) |
| `--outdir` | `results` | output root |

---

## Reference

### Label schema

Per-position integer labels (in `labels.npz` and model output):

| Value | Class | Meaning |
|---|---|---|
| 0 | IR | intergenic / non-coding |
| 1 | START | A of ATG start codon |
| 2 | E1 | coding, frame offset 1 |
| 3 | E2 | coding, frame offset 2 |
| 4 | E0 | coding, frame offset 0 |
| 5 | STOP | last base of stop codon |

Cycle within an ORF: `START E1 E2 E0 E1 E2 E0 … STOP`. The HMM enforces
this cycle plus ATG-at-START, stop-at-STOP, and the in-frame-stop check at
E2.

### TFRecord schema

Each example is one `chunk_len`-nt window:

| Feature | Type | Shape | Description |
|---|---|---|---|
| `input` | bytes (uint8 tensor) | `[L, 6]` | one-hot A,C,G,T,N + PAD channel |
| `output` | bytes (uint8 tensor) | `[L, 6]` | one-hot label (IR,START,E1,E2,E0,STOP) |
| `tx_id` | bytes (utf-8) | scalar | transcript identifier |
| `chunk_idx` | int64 | scalar | 0-based chunk index within the transcript |

Padded positions carry `input[..., 5] == 1` and `output == 0`;
`MaskedCategoricalCrossentropy` ignores them.

Load shards with `tiberius_orf.data.dataset.make_dataset(manifest_or_paths)`.

### Repository layout

```
scripts/
  annotate.py             end-to-end ORF annotation from BAM / StringTie GTF
  train.py                training entry point
  evaluate.py             per-class P/R/F1 on a test manifest
  predict.py              per-species reference + prediction GTFs
configs/
  default.yaml            model + training hyperparameters
nextflow/
  main.nf                 short-read data pipeline
  main_longread.nf        long-read (Iso-Seq / ONT) data pipeline
  conf/*.config           SLURM configs for the brain HPC cluster
  conf/*.csv              per-split species lists
  modules/                fetch / varus / stringtie / label / tfrecord
  submit_*.sh             SLURM wrapper scripts
src/tiberius_orf/
  data/                   FASTA/GTF parsing, label projection, TFRecord I/O
  model/                  cnn_lstm, cnn_transformer, masked CE loss
  hmm/
    annotation_hmm.py     6-state OrfAnnotationHMM (bricks2marble + hidten)
    decode.py             Viterbi batch / single-sequence decoders
    viterbi.py            pure-numpy reference Viterbi (debugging)
tests/                    pytest unit tests
```

### Running tests

```bash
pip install -e ".[test]"
pytest tests/
```
