"""Fine-tuning dataset builder: pre-scored Tiberius TSVs with gffcompare TP/FP labels."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tiberius_orf.model.finetune_head import FEATURE_COLS


def _run_gffcompare(tib_gtf: Path, ref_gff: Path, tmpdir: Path) -> set[str]:
    prefix = tmpdir / "gc"
    cmd = [
        "gffcompare", "--strict-match", "-e", "3",
        "-r", str(ref_gff), "-o", str(prefix), str(tib_gtf),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  gffcompare failed:\n{result.stderr[:400]}")
        return set()
    tracking = Path(str(prefix) + ".tracking")
    if not tracking.exists():
        print(f"  WARNING: no .tracking file in {tmpdir}")
        return set()
    tp_ids: set[str] = set()
    for line in tracking.read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5 or parts[3] != "=":
            continue
        try:
            tp_ids.add(parts[4].split(":", 1)[1].split("|")[1])
        except (IndexError, ValueError):
            continue
    return tp_ids


def assign_tp_labels(scores_tsv: Path, tib_gtf: Path, ref_gff: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_tsv, sep="\t", low_memory=False)
    species_name = Path(scores_tsv).parent.name
    if tib_gtf.exists() and ref_gff.exists():
        with tempfile.TemporaryDirectory() as td:
            tp_ids = _run_gffcompare(tib_gtf, ref_gff, Path(td))
        df["label"] = df["transcript_id"].isin(tp_ids).astype(np.float32)
    else:
        print(f"  WARNING: missing GTF or reference for {species_name}; setting all labels to 0.0")
        df["label"] = np.zeros(len(df), dtype=np.float32)
    n_tp = int((df["label"] == 1.0).sum())
    n_fp = int((df["label"] == 0.0).sum())
    print(f"  {species_name}: TP={n_tp}  FP={n_fp}")
    return df


def build_dataset(
    manifest: list[dict],
    feature_cols: list[str] = FEATURE_COLS,
    shuffle: bool = True,
    seed: int = 42,
    batch_size: int = 512,
    class_weight_tp: float | None = None,
) -> tuple[tf.data.Dataset, np.ndarray, np.ndarray]:
    frames: list[pd.DataFrame] = []
    for entry in manifest:
        df = assign_tp_labels(
            Path(entry["scores_tsv"]),
            Path(entry["tib_gtf"]),
            Path(entry["ref_gff"]),
        )
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    missing = [c for c in feature_cols if c not in combined.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}\n"
            "Re-run score_tiberius.py to generate extended features."
        )

    X = combined[feature_cols].fillna(0.0).values.astype(np.float32)
    y = combined["label"].values.astype(np.float32)

    if class_weight_tp is None:
        n_pos = float((y == 1.0).sum())
        n_neg = float((y == 0.0).sum())
        class_weight_tp = (n_neg / n_pos) if n_pos > 0 else 1.0
        print(f"  auto class_weight_tp={class_weight_tp:.4f} (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    weights = np.where(y == 1.0, np.float32(class_weight_tp), np.float32(1.0)).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((X, y, weights))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, X, y


def load_manifest_tsv(manifest_tsv: Path) -> list[dict]:
    df = pd.read_csv(manifest_tsv, sep="\t")
    required = {"species", "scores_tsv", "tib_gtf", "ref_gff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest TSV missing columns: {sorted(missing)}")
    return df.to_dict(orient="records")
