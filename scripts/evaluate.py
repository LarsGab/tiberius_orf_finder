"""Evaluate the tiberius_orf model on a test set using Viterbi decoding.

Reports per-class precision/recall/F1 and compares against an IR-only baseline.

CLI::

    python scripts/evaluate.py \\
      --test-manifest results/test/tfrecord_manifest.tsv \\
      --weights       results/models/run_001/final.weights.h5 \\
      --config        configs/default.yaml \\
      --out           results/eval/test_metrics.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml

CLASS_NAMES = ["IR", "START", "E1", "E2", "E0", "STOP"]
N_CLASSES = 6


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate tiberius_orf model.")
    ap.add_argument("--test-manifest", type=Path, required=True,
                    help="TSV manifest of test TFRecords.")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Path to saved model weights (.h5).")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Write per-class metrics to this TSV (optional).")
    return ap.parse_args(argv)


def _precision_recall_f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _print_and_collect(tp, fp, fn, baseline_tp, baseline_fp, baseline_fn):
    rows = []
    header = f"{'class':<8}  {'prec':>7}  {'rec':>7}  {'f1':>7}  "
    header += f"{'base_prec':>9}  {'base_rec':>8}  {'base_f1':>7}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(CLASS_NAMES):
        p, r, f = _precision_recall_f1(tp[i], fp[i], fn[i])
        bp, br, bf = _precision_recall_f1(
            baseline_tp[i], baseline_fp[i], baseline_fn[i]
        )
        row = f"{name:<8}  {p:>7.4f}  {r:>7.4f}  {f:>7.4f}  "
        row += f"{bp:>9.4f}  {br:>8.4f}  {bf:>7.4f}"
        print(row)
        rows.append((name, p, r, f, bp, br, bf))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    dc = cfg["data"]
    mc = cfg["model"]

    import tensorflow as tf
    from tiberius_orf.data.dataset import make_dataset
    from tiberius_orf.model.model import build_model_from_config
    from tiberius_orf.hmm.viterbi import viterbi_decode_batch

    model = build_model_from_config(cfg, chunk_len=dc["chunk_len"])
    model.load_weights(str(args.weights))
    print(f"Loaded {mc['type']} weights from {args.weights}", flush=True)

    test_ds = make_dataset(
        args.test_manifest,
        chunk_len=dc["chunk_len"],
        batch_size=dc["batch_size"],
        shuffle=False,
        repeat=False,
    )

    # Accumulators: model and IR-only baseline
    tp  = np.zeros(N_CLASSES, dtype=np.int64)
    fp  = np.zeros(N_CLASSES, dtype=np.int64)
    fn  = np.zeros(N_CLASSES, dtype=np.int64)
    btp = np.zeros(N_CLASSES, dtype=np.int64)
    bfp = np.zeros(N_CLASSES, dtype=np.int64)
    bfn = np.zeros(N_CLASSES, dtype=np.int64)

    for x_batch, y_batch, pad_mask in test_ds:
        x_np  = x_batch.numpy()       # [B, L, 6]
        y_np  = y_batch.numpy()       # [B, L, 6]  one-hot
        pm_np = pad_mask.numpy()      # [B, L]  True = PAD

        logits   = model(x_batch, training=False).numpy()  # [B, L, 6]
        log_prob = tf.nn.log_softmax(logits).numpy()
        pred     = viterbi_decode_batch(log_prob)          # [B, L] int

        true_cls = np.argmax(y_np, axis=-1)  # [B, L] int
        valid    = ~pm_np                    # [B, L] bool

        # IR-only baseline: predict 0 everywhere
        baseline = np.zeros_like(true_cls)

        for cls in range(N_CLASSES):
            pred_mask = (pred == cls) & valid
            true_mask = (true_cls == cls) & valid
            base_mask = (baseline == cls) & valid

            tp[cls]  += int(np.sum(pred_mask & true_mask))
            fp[cls]  += int(np.sum(pred_mask & ~true_mask))
            fn[cls]  += int(np.sum(~pred_mask & true_mask))
            btp[cls] += int(np.sum(base_mask & true_mask))
            bfp[cls] += int(np.sum(base_mask & ~true_mask))
            bfn[cls] += int(np.sum(~base_mask & true_mask))

    print("\nPer-class metrics (model vs IR-only baseline):")
    rows = _print_and_collect(tp, fp, fn, btp, bfp, bfn)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("class\tprec\trec\tf1\tbase_prec\tbase_rec\tbase_f1\n")
            for row in rows:
                fh.write("\t".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                                   for v in row) + "\n")
        print(f"\nMetrics written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
