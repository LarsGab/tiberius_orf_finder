"""Run inference on a test set and emit reference + prediction GTFs per species.

For each species in the manifest, writes two files in `<out-dir>/<species>/`:
    * reference.gtf   — derived from the projected labels in labels.npz
    * prediction.gtf  — derived from model + Viterbi decode

Both GTFs are in genomic coordinates (1-based inclusive), with one CDS line
per genomic sub-interval (multi-exon ORFs produce multiple lines per ORF).

CLI::

    python scripts/predict.py \\
      --test-manifest results/test/tfrecord_manifest_available.tsv \\
      --weights       results/models/run_002/epoch_41.weights.h5 \\
      --config        configs/default.yaml \\
      --out-dir       results/predictions/run_002_epoch41
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict ORFs and write GTF.")
    ap.add_argument("--test-manifest", type=Path, required=True,
                    help="TSV manifest of test TFRecords (species<TAB>path).")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Path to model weights (.h5).")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output root; one subdir per species.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Inference batch size (default 8).")
    return ap.parse_args(argv)


def _resolve_species_dirs(tfrec_path: Path) -> tuple[Path, Path]:
    """Given .../<species>/tfrecord/data.tfrecords return (labels_npz, stringtie_gtf)."""
    species_dir = tfrec_path.parent.parent
    return (
        species_dir / "labels" / "labels.npz",
        species_dir / "stringtie" / "stringtie.gtf",
    )


def _read_chunks_from_tfrecord(
    tfrec_path: Path, chunk_len: int
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Return {tx_id: [(chunk_idx, x_chunk), ...]} (unsorted)."""
    import tensorflow as tf

    spec = {
        "input":     tf.io.FixedLenFeature([], tf.string),
        "output":    tf.io.FixedLenFeature([], tf.string),
        "tx_id":     tf.io.FixedLenFeature([], tf.string),
        "chunk_idx": tf.io.FixedLenFeature([], tf.int64),
    }

    chunks: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    raw_ds = tf.data.TFRecordDataset([str(tfrec_path)])
    for serialized in raw_ds:
        parsed = tf.io.parse_single_example(serialized, spec)
        x = tf.cast(
            tf.io.parse_tensor(parsed["input"], out_type=tf.uint8), tf.float32
        )
        x.set_shape([chunk_len, 6])
        tx_id = parsed["tx_id"].numpy().decode("utf-8")
        ci = int(parsed["chunk_idx"].numpy())
        chunks[tx_id].append((ci, x.numpy()))
    return chunks


def _predict_per_tx(
    model,
    chunks_by_tx: dict[str, list[tuple[int, np.ndarray]]],
    chunk_len: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run model + Viterbi-decode the full-transcript logit sequence per tx_id.

    Returns {tx_id: int32 label sequence of length = real transcript length
    (padded positions stripped)}.
    """
    import tensorflow as tf
    from tiberius_orf.hmm.viterbi import viterbi_decode

    out: dict[str, np.ndarray] = {}
    tx_ids = sorted(chunks_by_tx)

    # group chunks across all transcripts into batches for inference
    flat: list[tuple[str, int, np.ndarray]] = []   # (tx_id, ci, x)
    for tid in tx_ids:
        for ci, x in chunks_by_tx[tid]:
            flat.append((tid, ci, x))

    # logits per (tx_id, ci)
    logits_buf: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for i in range(0, len(flat), batch_size):
        batch = flat[i : i + batch_size]
        x_batch = np.stack([t[2] for t in batch])             # [b, L, 6]
        logits = model(x_batch, training=False).numpy()       # [b, L, 6]
        for (tid, ci, _), lg in zip(batch, logits):
            logits_buf[tid][ci] = lg

    # for each transcript, concat logits in chunk order, log-softmax, Viterbi,
    # then truncate to the true (non-padded) length using the input pad channel
    for tid in tx_ids:
        chunks_by_tx[tid].sort(key=lambda t: t[0])
        ordered_x = np.concatenate([x for _, x in chunks_by_tx[tid]], axis=0)  # [N*L, 6]
        ordered_logits = np.concatenate(
            [logits_buf[tid][ci] for ci, _ in chunks_by_tx[tid]], axis=0
        )  # [N*L, 6]
        # log-softmax
        m = ordered_logits.max(axis=-1, keepdims=True)
        log_probs = ordered_logits - m - np.log(
            np.exp(ordered_logits - m).sum(axis=-1, keepdims=True)
        )
        pred_seq = viterbi_decode(log_probs)                 # [N*L]
        valid = ordered_x[..., 5] != 1                       # not pad
        true_len = int(valid.sum())
        out[tid] = pred_seq[:true_len].astype(np.int32)

    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    dc, mc = cfg["data"], cfg["model"]

    import tensorflow as tf  # noqa: F401
    from tiberius_orf.data.dataset import load_manifest
    from tiberius_orf.data.label_transcripts import parse_stringtie_gtf
    from tiberius_orf.data.gtf_writer import write_gtf
    from tiberius_orf.model.model import build_model_from_config

    model = build_model_from_config(cfg, chunk_len=dc["chunk_len"])
    model.load_weights(str(args.weights))
    print(f"Loaded {mc['type']} weights from {args.weights}", flush=True)

    # parse manifest WITH species names (load_manifest only returns paths)
    species_paths: list[tuple[str, Path]] = []
    for line in args.test_manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        species_paths.append((parts[0], Path(parts[-1])))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[str, int, int, int]] = []   # (species, n_tx, ref_lines, pred_lines)

    for species, tfrec_path in species_paths:
        labels_path, stringtie_gtf = _resolve_species_dirs(tfrec_path)
        if not tfrec_path.exists():
            print(f"  [skip] missing tfrecord: {tfrec_path}", flush=True)
            continue
        if not stringtie_gtf.exists():
            print(f"  [skip] missing stringtie.gtf: {stringtie_gtf}", flush=True)
            continue

        print(f"\n=== {species} ===", flush=True)
        print(f"  loading transcripts from {stringtie_gtf}", flush=True)
        transcripts = parse_stringtie_gtf(stringtie_gtf)

        # reference labels
        ref_labels: dict[str, np.ndarray] = {}
        if labels_path.exists():
            with np.load(labels_path, allow_pickle=False) as npz:
                ref_labels = {k: np.asarray(npz[k]) for k in npz.files}

        # inference
        print(f"  reading chunks from {tfrec_path}", flush=True)
        chunks = _read_chunks_from_tfrecord(tfrec_path, dc["chunk_len"])
        print(f"  inference on {len(chunks)} transcripts", flush=True)
        pred_labels = _predict_per_tx(model, chunks, dc["chunk_len"], args.batch_size)

        # write GTFs
        species_out = args.out_dir / species
        species_out.mkdir(parents=True, exist_ok=True)
        n_ref = write_gtf(species_out / "reference.gtf",
                          ref_labels, transcripts, source="reference")
        n_pred = write_gtf(species_out / "prediction.gtf",
                           pred_labels, transcripts, source="tiberius_orf")
        print(f"  wrote {n_ref} ref CDS lines, {n_pred} pred CDS lines", flush=True)
        summary_rows.append((species, len(pred_labels), n_ref, n_pred))

    # final summary table
    summary_path = args.out_dir / "summary.tsv"
    with open(summary_path, "w") as fh:
        fh.write("species\tn_transcripts\tref_cds_lines\tpred_cds_lines\n")
        for row in summary_rows:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"\nSummary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
