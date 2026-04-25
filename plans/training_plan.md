# Training & Model Implementation Plan

## What exists today

The data pipeline is complete and running on brain. TFRecords are being generated for 115 insect species (100 training, 4 val, 11 test). Each record encodes one 9999-nt chunk of a StringTie transcript:

- `input`     — uint8 tensor [L, 6]: one-hot nucleotide (A/C/G/T/N) + PAD channel
- `output`    — uint8 tensor [L, 6]: one-hot label (IR / START / E1 / E2 / E0 / STOP)
- `tx_id`     — bytes: transcript id
- `chunk_idx` — int64: 0-based chunk index within transcript

PAD positions: `input[..., 5] == 1`. The training loss must ignore these.

Label encoding (from `label_transcripts.py`):
- 0 IR, 1 START, 2 E1, 3 E2, 4 E0, 5 STOP
- Cycle between START and STOP: `START - E1 - E2 - E0 - E1 - E2 - E0 - ... - STOP`

Source files:
- `src/tiberius_orf/data/chunk_tfrecord.py` — TFRecord writer; also has the reader utility `parse_tfrecord_example()`
- `src/tiberius_orf/data/label_transcripts.py` — label projection logic
- `src/tiberius_orf/model/__init__.py` — empty stub, implement here
- `src/tiberius_orf/hmm/__init__.py` — empty stub, implement here

TFRecords land at (on brain):
- `/home/gabriell/tiberius_orf_finder/results/training/<Genus_species>/tfrecord/data.tfrecords`
- `/home/gabriell/tiberius_orf_finder/results/val/<Genus_species>/tfrecord/data.tfrecords`
- `/home/gabriell/tiberius_orf_finder/results/test/<Genus_species>/tfrecord/data.tfrecords`
- Manifests: `results/{training,val,test}/tfrecord_manifest.tsv` (species \t path)

## What needs to be built

### 1. TFRecord dataset loader (`src/tiberius_orf/data/dataset.py`)

- Read from a manifest TSV (species \t path) or a list of `.tfrecords` paths
- Parse each example via `parse_tfrecord_example()` in `chunk_tfrecord.py`
- Decode uint8 tensors to float32
- Mask PAD positions (`input[..., 5] == 1`) — return as a boolean mask alongside the tensors
- Shuffle, batch, prefetch — standard `tf.data` pipeline
- Split by species for leave-one-out validation if needed

### 2. Model architecture (`src/tiberius_orf/model/model.py`)

Follow Tiberius's design (repo at `/home/gabriell/git/Tiberius` on brain or `~/git/Tiberius` locally):
- Input: [batch, L, 6] float32 (nucleotide one-hot + PAD; zero-out PAD before feeding)
- Bidirectional LSTM stack (Tiberius uses 3-4 layers, hidden dim ~512)
- Dense output head: [batch, L, 6] logits over the 6 label classes
- Optional: convolutional stem before the LSTM (Tiberius has this)

Check Tiberius's `src/tiberius/model/` for the exact layer sizes and whether a CRF/HMM decode layer is wired in during training or only at inference.

### 3. HMM decode layer (`src/tiberius_orf/hmm/viterbi.py`)

Tiberius uses a differentiable HMM layer (in `bricks2marble`) to enforce valid label transitions (e.g. STOP must follow the correct reading frame). Check `bricks2marble`'s `hidten` or `hmm` module. The 6 ORF states map cleanly onto a linear-chain HMM:

```
IR -> START -> E1 -> E2 -> E0 -> E1 -> ... -> STOP -> IR
         \___________________________________/  (cycle)
```

For the ORF finder, valid transitions are:
- IR -> IR, IR -> START
- START -> E1
- E1 -> E2, E2 -> E0, E0 -> E1
- E0 -> STOP (only E0 can precede STOP, matching the reading frame)
- STOP -> IR

During training: use soft HMM (forward algorithm for the loss) or just cross-entropy on the logits with masked positions.  
During inference: Viterbi decode to produce a valid label sequence.

### 4. Loss function (`src/tiberius_orf/model/loss.py`)

- Masked cross-entropy: ignore positions where PAD mask is True
- Class weights: IR is likely over-represented (most transcript positions are intergenic); weight START and STOP more heavily
- Optionally: add a sequence-level CTC or HMM loss term

### 5. Training script (`scripts/train.py`)

CLI:
```
python scripts/train.py \
  --train-manifest results/training/tfrecord_manifest.tsv \
  --val-manifest   results/val/tfrecord_manifest.tsv \
  --config         configs/default.yaml \
  --outdir         results/models/run_001
```

Should:
- Load dataset via step 1
- Instantiate model via step 2
- Train with tf.keras `model.fit()` or a custom loop
- Save checkpoints to `--outdir`
- Log metrics (loss, per-class F1 on val set) to a TSV or W&B

### 6. Config system (`configs/default.yaml`)

Hyperparameters to expose:
- `chunk_len: 9999`
- `lstm_units: 512`
- `lstm_layers: 3`
- `dropout: 0.1`
- `batch_size: 32`
- `learning_rate: 1e-3`
- `epochs: 50`
- `class_weights: [1, 10, 1, 1, 1, 10]`  # up-weight START and STOP

### 7. Evaluation script (`scripts/evaluate.py`)

Run Viterbi decode on the test set, report per-class precision/recall/F1. Compare against a naive baseline (predict IR everywhere).

## Suggested implementation order

1. Dataset loader + smoke test loading one species' shard
2. Model architecture (no HMM, just LSTM + dense head)
3. Loss + training script — get a training curve on D. melanogaster first
4. HMM decode layer for inference
5. Full eval on test set

## Key references

- Tiberius repo: `/home/gabriell/git/Tiberius` (check `src/tiberius/model/` for architecture)
- bricks2marble HMM layer: `/home/gabriell/git/bricks2marble` (check `hidten` module)
- TFRecord schema: `src/tiberius_orf/data/chunk_tfrecord.py` lines 1-17 (docstring)
- Label encoding: `src/tiberius_orf/data/label_transcripts.py` lines 35-36
- TF version on brain: 2.17.0 (in micromamba env `storm`)

## Constraints

- TensorFlow 2.17.0 in micromamba env `storm` on brain
- Training runs via `hpc-runner` on brain's SLURM; use `dry-run-first` before every HPC submission
- No commits — user commits manually
- Python package root: `src/tiberius_orf/`; installed via `PYTHONPATH=/home/gabriell/tiberius_orf_finder/src` on brain
