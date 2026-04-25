"""Train the tiberius_orf model.

CLI::

    python scripts/train.py \\
      --train-manifest results/training/tfrecord_manifest.tsv \\
      --val-manifest   results/val/tfrecord_manifest.tsv \\
      --config         configs/default.yaml \\
      --outdir         results/models/run_001
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train tiberius_orf model.")
    ap.add_argument("--train-manifest", type=Path, required=True,
                    help="TSV manifest of training TFRecords (species<TAB>path).")
    ap.add_argument("--val-manifest", type=Path, required=True,
                    help="TSV manifest of validation TFRecords.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="YAML config file (default: configs/default.yaml).")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="Output directory for checkpoints and logs.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override batch_size from config.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Override learning_rate from config.")
    return ap.parse_args(argv)


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _pack_y(ds):
    """Reformat (x, y, pad_mask) -> (x, packed_y) for model.fit().

    packed_y has shape [B, L, 7]: first 6 channels = one-hot labels,
    last channel = float32 pad flag (1.0 = padded position).
    """
    import tensorflow as tf
    def _pack(x, y, pad_mask):
        pad_float = tf.cast(pad_mask[..., tf.newaxis], tf.float32)
        packed = tf.concat([y, pad_float], axis=-1)
        packed.set_shape([None, None, 7])
        return x, packed
    return ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config)

    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["training"]
    cc = cfg["checkpointing"]

    if args.epochs is not None:
        tc["epochs"] = args.epochs
    if args.batch_size is not None:
        dc["batch_size"] = args.batch_size
    if args.lr is not None:
        tc["learning_rate"] = args.lr

    args.outdir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from tiberius_orf.data.dataset import make_dataset
    from tiberius_orf.model.model import build_model_from_config
    from tiberius_orf.model.loss import MaskedCategoricalCrossentropy, MaskedAccuracy

    print(f"TF version: {tf.__version__}", flush=True)

    train_ds = _pack_y(make_dataset(
        args.train_manifest,
        chunk_len=dc["chunk_len"],
        batch_size=dc["batch_size"],
        shuffle=True,
        shuffle_buffer=dc["shuffle_buffer"],
        repeat=True,
    ))
    val_ds = _pack_y(make_dataset(
        args.val_manifest,
        chunk_len=dc["chunk_len"],
        batch_size=dc["batch_size"],
        shuffle=False,
        repeat=False,
    ))

    model = build_model_from_config(cfg, chunk_len=dc["chunk_len"])
    print(f"Model type: {mc['type']}", flush=True)
    model.summary()

    loss_fn = MaskedCategoricalCrossentropy(class_weights=tc["class_weights"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=tc["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[MaskedAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.outdir / "epoch_{epoch:02d}.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(args.outdir / "train_log.tsv"),
                                     separator="\t"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    model.fit(
        train_ds,
        epochs=tc["epochs"],
        steps_per_epoch=tc["steps_per_epoch"],
        validation_data=val_ds,
        callbacks=callbacks,
    )

    model.save_weights(str(args.outdir / "final.weights.h5"))
    print(f"Training complete. Outputs in {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
