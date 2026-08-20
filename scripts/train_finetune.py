"""Train the ORFfinder fine-tuning classification head.

Loads a manifest of (scores_tsv, tib_gtf, ref_gff) tuples, assigns TP/FP
labels via gffcompare, and trains a small MLP on top of the pre-computed
ORFfinder score features to classify Tiberius predictions as TP or FP.

The backbone (ORFfinder CNN-LSTM) is NOT loaded here — only the aggregated
feature TSVs produced by score_tiberius.py are used.

Usage
-----
python scripts/train_finetune.py \\
  --manifest   /projects/AI-GUSTUS/tiberius_orf_finder/results/finetune/manifest_vertebrates_train.tsv \\
  --out-dir    /projects/AI-GUSTUS/tiberius_orf_finder/results/finetune/head_v1 \\
  --val-frac   0.1 \\
  --epochs     100 \\
  --batch-size 512

Outputs in --out-dir:
  best_weights.weights.h5   — best validation AUC checkpoint
  final_weights.weights.h5  — weights at last epoch
  training_history.tsv      — per-epoch metrics
  config.yaml               — reproducibility record
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tiberius_orf.data.tiberius_pred_dataset import build_dataset, load_manifest_tsv
from tiberius_orf.model.finetune_head import (
    FEATURE_COLS, FinetuneConfig, build_finetune_head,
)


def _enable_gpu_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train ORFfinder fine-tuning classification head."
    )
    ap.add_argument("--manifest",    type=Path, required=True,
                    help="Manifest TSV from build_finetune_manifest.py.")
    ap.add_argument("--out-dir",     type=Path, required=True)
    ap.add_argument("--val-frac",    type=float, default=0.1,
                    help="Fraction of species held out for validation.")
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--batch-size",  type=int,   default=512)
    ap.add_argument("--hidden",      type=int,   nargs="+", default=[64, 32],
                    help="Hidden layer sizes for the MLP head.")
    ap.add_argument("--dropout",     type=float, default=0.3)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--seed",        type=int,   default=42)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    _enable_gpu_memory_growth()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    manifest = load_manifest_tsv(args.manifest)
    print(f"[train_finetune] {len(manifest)} species in manifest", flush=True)

    # Species-level train/val split so no species leaks between sets
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(manifest))
    n_val = max(1, int(len(manifest) * args.val_frac))
    val_entries   = [manifest[i] for i in idx[:n_val]]
    train_entries = [manifest[i] for i in idx[n_val:]]
    print(f"[train_finetune] train={len(train_entries)} val={len(val_entries)} species",
          flush=True)

    print("[train_finetune] Building training dataset …", flush=True)
    train_ds, X_train, y_train = build_dataset(
        train_entries, feature_cols=FEATURE_COLS,
        shuffle=True, seed=args.seed, batch_size=args.batch_size,
    )
    print("[train_finetune] Building validation dataset …", flush=True)
    val_ds, X_val, y_val = build_dataset(
        val_entries, feature_cols=FEATURE_COLS,
        shuffle=False, batch_size=args.batch_size,
        class_weight_tp=1.0,  # no reweighting for validation
    )

    cfg = FinetuneConfig(
        hidden=args.hidden, dropout=args.dropout, n_features=len(FEATURE_COLS),
    )
    model = build_finetune_head(cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )
    model.summary()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    best_weights = args.out_dir / "best_weights.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_weights), monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5,
            patience=10, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=20,
            restore_best_weights=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=2,
    )

    model.save_weights(str(args.out_dir / "final_weights.weights.h5"))

    # Evaluate final AUC on val set
    val_preds = model.predict(X_val, batch_size=args.batch_size, verbose=0).squeeze()
    val_auc = roc_auc_score(y_val, val_preds) if y_val.sum() > 0 else float("nan")
    print(f"[train_finetune] Final val AUC = {val_auc:.4f}", flush=True)

    # Save history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(args.out_dir / "training_history.tsv", sep="\t", index=False)

    # Save config
    config_record = {
        "manifest":    str(args.manifest),
        "feature_cols": FEATURE_COLS,
        "hidden":      args.hidden,
        "dropout":     args.dropout,
        "lr":          args.lr,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "val_frac":    args.val_frac,
        "seed":        args.seed,
        "n_train_species": len(train_entries),
        "n_val_species":   len(val_entries),
        "final_val_auc":   float(val_auc),
    }
    with open(args.out_dir / "config.yaml", "w") as f:
        yaml.dump(config_record, f, default_flow_style=False)

    print(f"[train_finetune] Outputs -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
