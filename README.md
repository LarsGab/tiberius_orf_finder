# tiberius_orf_finder

Deep-learning ORF finder for assembled transcripts, derived from Tiberius.
The repo contains two things:

1. **Data pipeline** — a Nextflow workflow that generates per-species TFRecord
   training shards from genomes + RNA-seq (VARUS → StringTie → label → TFRecord).
2. **Model + training** — a bidirectional LSTM that predicts per-position ORF
   labels, plus a Viterbi post-processor that enforces biologically valid
   label sequences.

## Requirements

- Nextflow ≥ 23 (data pipeline only)
- Python ≥ 3.9 with `tiberius_orf_finder` installed
- NCBI `datasets` CLI (for RefSeq species)
- VARUS, HISAT2, StringTie, samtools, gffread (data pipeline only)

```bash
pip install -e ".[from_source]"   # tensorflow, biopython, numpy, pyyaml, …
# tests only:
pip install -e ".[test]" && pytest tests/
```

## Repository layout

```
nextflow/               Nextflow data-prep pipeline
  main.nf               workflow entry point
  conf/brain.config     SLURM config for the group's HPC cluster
  conf/species_*.csv    per-split species lists
  modules/              fetch / varus / stringtie / label / tfrecord
  submit_*.sh           SLURM wrapper scripts (training/val/test/smoke)
configs/
  default.yaml          model + training hyperparameters
scripts/
  train.py              training entry point
  evaluate.py           evaluation + Viterbi decoding on a test set
src/tiberius_orf/
  data/
    species_list.py     species CSV helpers
    label_transcripts.py  project reference CDS onto StringTie transcripts
    chunk_tfrecord.py   chunk labelled transcripts into TFRecord windows
    dataset.py          tf.data.Dataset loader for TFRecords
  model/
    model.py            build_model() — conv stem + stacked BiLSTM
    loss.py             MaskedCategoricalCrossentropy
  hmm/
    viterbi.py          viterbi_decode() / viterbi_decode_batch()
tests/                  pytest unit tests for data/ modules
```

## Data pipeline

For each species in a CSV, the pipeline:
1. Fetches or stages the genome + reference annotation (`FETCH_ASSEMBLY`)
2. Aligns RNA-seq reads with VARUS+HISAT2 (`MAKE_VARUS_PARAMS`, `RUN_VARUS`)
3. Assembles transcripts with StringTie + gffread (`RUN_STRINGTIE`)
4. Projects reference CDS labels onto assembled transcripts (`LABEL_TRANSCRIPTS`)
5. Chunks labelled transcripts into 9999-nt TFRecord windows (`WRITE_TFRECORD`)

A `tfrecord_manifest.tsv` listing all shards is written to `--outdir`.

### Quick start (local)

```bash
nextflow run nextflow/main.nf \
  --species_csv  nextflow/conf/species_training.csv \
  --braker_data_dir /path/to/braker_data \
  --varus_dir    /path/to/VARUS \
  --varus_impl   /path/to/VARUS/Implementation \
  --hisat_dir    /path/to/hisat2 \
  --outdir       results/training
```

### HPC (brain cluster)

Use the pre-configured submit scripts which wrap `brain.config`:

```bash
sbatch nextflow/submit_training.sh   # training split
sbatch nextflow/submit_val.sh        # validation split
sbatch nextflow/submit_test.sh       # test split
sbatch nextflow/submit_smoke.sh      # smoke test (tiny species set)
```

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

### Species CSV format

```
species,accession,annotation
Drosophila melanogaster,GCF_000001215.4,RefSeq
Apis mellifera,,BRAKER
```

`annotation` is `RefSeq` (downloaded via NCBI `datasets`) or `BRAKER`
(staged from `--braker_data_dir/<Genus_species>/`).

## Model

### Architecture

`src/tiberius_orf/model/model.py` — `build_model()`

1. Zero-out the PAD channel (index 5) so padded positions carry no signal.
2. Optional 1-D conv stem (`Conv1D`, kernel 11, 64 filters) to capture local
   sequence context before the recurrent layers.
3. Stacked bidirectional LSTM layers (default: 3 × 512 units each direction).
4. Dense output head → 6 logits per position (IR / START / E1 / E2 / E0 / STOP).

### Viterbi decoder

`src/tiberius_orf/hmm/viterbi.py` — `viterbi_decode(log_emission)`

Post-processes model log-probabilities to enforce valid label transitions:

```
IR → IR | START
START → E1 → E2 → E0 → E1 | STOP
STOP → IR
```

Use `viterbi_decode_batch` for batched numpy arrays `[B, L, 6]`.

## Training

```bash
python scripts/train.py \
  --train-manifest results/training/tfrecord_manifest.tsv \
  --val-manifest   results/val/tfrecord_manifest.tsv \
  --config         configs/default.yaml \
  --outdir         results/models/run_001
```

Key CLI overrides: `--epochs`, `--batch-size`, `--lr`.

Default hyperparameters (`configs/default.yaml`):

| Section | Key | Default |
|---|---|---|
| data | chunk_len | 9999 |
| data | batch_size | 32 |
| model | lstm_units | 512 |
| model | lstm_layers | 3 |
| model | dropout | 0.1 |
| model | use_conv_stem | true |
| training | learning_rate | 1e-3 |
| training | epochs | 50 |
| training | class_weights | [1,10,1,1,1,10] |

Class weights up-weight START (1) and STOP (5) by 10× to compensate for the
heavy IR imbalance.

Outputs written to `--outdir`: `checkpoint.weights.h5`, `final.weights.h5`,
`train_log.tsv`.

## Evaluation

```bash
python scripts/evaluate.py \
  --test-manifest results/test/tfrecord_manifest.tsv \
  --weights       results/models/run_001/final.weights.h5 \
  --config        configs/default.yaml \
  --out           results/eval/test_metrics.tsv
```

Reports per-class precision / recall / F1 using Viterbi-decoded predictions,
compared against an IR-only baseline.

## Label schema

Per-position integer labels (used in `labels.npz` and model output):

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

Each example is one `chunk_len`-nt window:

| Feature | Type | Shape | Description |
|---|---|---|---|
| `input` | bytes (uint8 tensor) | `[L, 6]` | one-hot A,C,G,T,N + PAD channel |
| `output` | bytes (uint8 tensor) | `[L, 6]` | one-hot label (IR,START,E1,E2,E0,STOP) |
| `tx_id` | bytes (utf-8) | scalar | transcript identifier |
| `chunk_idx` | int64 | scalar | 0-based chunk index within the transcript |

Padded positions have `input[..., 5] == 1` and `output == 0`.
Mask these in the loss (`MaskedCategoricalCrossentropy` handles this automatically).

Load with `tiberius_orf.data.dataset.make_dataset(manifest_or_paths)`.
