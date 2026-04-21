"""TFRecord dataset loader for tiberius_orf_finder.

Reads manifests of the form:
    species<TAB>path/to/data.tfrecords
or plain lists of .tfrecords paths, and returns a tf.data.Dataset of
(input, output, pad_mask) tuples where:
    input:    float32 [L, 6]  — nucleotide one-hot + PAD channel
    output:   float32 [L, 6]  — label one-hot
    pad_mask: bool    [L]     — True where input[..., 5] == 1 (pad position)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def load_manifest(manifest_path: Path | str) -> list[str]:
    """Return a list of .tfrecords paths from a TSV manifest (species\\tpath)."""
    paths = []
    for line in Path(manifest_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        paths.append(parts[-1])   # last column is always the path
    return paths


def make_dataset(
    sources: Sequence[str] | str | Path,
    chunk_len: int = 9999,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer: int = 2048,
    prefetch: int | None = None,
    repeat: bool = False,
) -> "tf.data.Dataset":
    """Return a batched tf.data.Dataset from tfrecords paths or a manifest TSV.

    Args:
        sources: a manifest .tsv path, a single .tfrecords path, or a list of
            .tfrecords paths.
        chunk_len: window length used when writing the TFRecords (default 9999).
        batch_size: examples per batch.
        shuffle: whether to shuffle the example stream.
        shuffle_buffer: shuffle buffer size (ignored when shuffle=False).
        prefetch: AUTOTUNE if None.
        repeat: whether to repeat the dataset indefinitely (use for training).

    Returns:
        Dataset of (input, output, pad_mask) where each tensor has a leading
        batch dimension. input/output are float32 [B, L, 6]; pad_mask is bool
        [B, L].
    """
    import tensorflow as tf

    if isinstance(sources, (str, Path)):
        src = str(sources)
        if src.endswith(".tsv") or src.endswith(".txt"):
            file_list = load_manifest(src)
        else:
            file_list = [src]
    else:
        file_list = [str(p) for p in sources]

    spec = {
        "input":     tf.io.FixedLenFeature([], tf.string),
        "output":    tf.io.FixedLenFeature([], tf.string),
        "tx_id":     tf.io.FixedLenFeature([], tf.string),
        "chunk_idx": tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse(serialized):
        parsed = tf.io.parse_single_example(serialized, spec)
        x = tf.cast(
            tf.io.parse_tensor(parsed["input"],  out_type=tf.uint8), tf.float32
        )  # [L, 6]
        y = tf.cast(
            tf.io.parse_tensor(parsed["output"], out_type=tf.uint8), tf.float32
        )  # [L, 6]
        pad_mask = tf.cast(x[..., 5], tf.bool)  # True where PAD
        return x, y, pad_mask

    ds = tf.data.TFRecordDataset(file_list, compression_type="")
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(prefetch if prefetch is not None else tf.data.AUTOTUNE)
    return ds
