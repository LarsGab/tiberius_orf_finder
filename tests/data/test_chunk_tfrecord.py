"""Tests for the pure-numpy chunking/encoding layer. TFRecord I/O is covered
only if tensorflow is importable in the current environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiberius_orf.data.chunk_tfrecord import (
    DEFAULT_CHUNK_LEN,
    NUM_CLASSES,
    NUM_INPUT_CHANNELS,
    chunk_all,
    chunk_transcript,
    encode_nucleotides,
    labels_to_onehot,
    read_fasta,
)
from tiberius_orf.data.label_transcripts import E0, E1, E2, IR, START, STOP


def test_encode_nucleotides_onehot():
    arr = encode_nucleotides("AcGtNx")
    expected = np.array([
        [1, 0, 0, 0, 0],   # A
        [0, 1, 0, 0, 0],   # c -> C
        [0, 0, 1, 0, 0],   # G
        [0, 0, 0, 1, 0],   # t -> T
        [0, 0, 0, 0, 1],   # N
        [0, 0, 0, 0, 1],   # unknown -> N
    ], dtype=np.uint8)
    assert np.array_equal(arr, expected)


def test_labels_to_onehot_round_trip():
    labels = np.array([IR, START, E1, E2, E0, STOP, IR], dtype=np.int8)
    oh = labels_to_onehot(labels)
    assert oh.shape == (7, NUM_CLASSES)
    assert np.array_equal(np.argmax(oh, axis=1), labels.astype(np.int64))


def test_chunk_shorter_than_chunk_len_pads():
    seq = "ACGT"
    labels = np.array([IR, START, E1, E2], dtype=np.int8)
    chunks = list(chunk_transcript(seq, labels, chunk_len=8))
    assert len(chunks) == 1

    x, y = chunks[0]
    assert x.shape == (8, NUM_INPUT_CHANNELS)
    assert y.shape == (8, NUM_CLASSES)

    # First 4 positions: real nt one-hot, no pad.
    assert (x[:4, :5].sum(axis=1) == 1).all()
    assert (x[:4, 5] == 0).all()
    # Pad tail: pad channel = 1, all other channels = 0.
    assert (x[4:, 5] == 1).all()
    assert (x[4:, :5] == 0).all()

    # Labels: first 4 are one-hot of IR/START/E1/E2, rest are all zero.
    assert np.array_equal(np.argmax(y[:4], axis=1), [IR, START, E1, E2])
    assert (y[4:] == 0).all()


def test_chunk_exact_multiple_no_padding():
    L = 4
    seq = "ACGTACGT"      # length 8 = 2 * L
    labels = np.zeros(8, dtype=np.int8)
    chunks = list(chunk_transcript(seq, labels, chunk_len=L))
    assert len(chunks) == 2
    for x, y in chunks:
        assert x.shape == (L, NUM_INPUT_CHANNELS)
        assert (x[:, 5] == 0).all()        # no pad anywhere


def test_chunk_longer_than_chunk_len_pads_tail():
    L = 4
    seq = "ACGTAC"        # length 6 -> two chunks: 4 nt + 2 nt (padded)
    labels = np.arange(6, dtype=np.int8) % NUM_CLASSES
    chunks = list(chunk_transcript(seq, labels, chunk_len=L))
    assert len(chunks) == 2

    x2, y2 = chunks[1]
    assert x2.shape == (L, NUM_INPUT_CHANNELS)
    assert (x2[:2, 5] == 0).all()          # real positions
    assert (x2[2:, 5] == 1).all()          # padded tail
    assert (y2[2:] == 0).all()             # all-zero label on pad


def test_chunk_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        list(chunk_transcript("ACGT", np.zeros(3, dtype=np.int8)))


def test_chunk_all_iterates_in_deterministic_order():
    sequences = {"T2": "ACGT", "T1": "GGG"}
    labels = {"T1": np.zeros(3, dtype=np.int8),
              "T2": np.zeros(4, dtype=np.int8)}
    chunks = list(chunk_all(sequences, labels, chunk_len=4))
    # Sorted by tx_id -> T1, T2.
    assert [tid for tid, _, _, _ in chunks] == ["T1", "T2"]


def test_read_fasta(tmp_path):
    p = tmp_path / "x.fa"
    p.write_text(">a extra header\nACGT\nACGT\n>b\nNNN\n", encoding="utf-8")
    seqs = read_fasta(p)
    assert seqs == {"a": "ACGTACGT", "b": "NNN"}


# ---------- optional TFRecord tests (skip if tensorflow is unavailable) ----------

tf_available = False
try:
    import tensorflow as tf  # noqa: F401
    tf_available = True
except Exception:
    pass


@pytest.mark.skipif(not tf_available, reason="tensorflow not installed")
def test_tfrecord_round_trip(tmp_path):
    from tiberius_orf.data.chunk_tfrecord import (
        decode_example,
        write_tfrecord,
    )
    seq = "ACGTACGTAC"
    labels = np.array([IR, START, E1, E2, E0, E1, E2, E0, E1, STOP],
                      dtype=np.int8)
    chunks = list(chunk_transcript(seq, labels, chunk_len=DEFAULT_CHUNK_LEN))
    examples = [("tx1", ci, x, y) for ci, (x, y) in enumerate(chunks)]

    path = tmp_path / "out.tfrecords"
    n = write_tfrecord(path, examples)
    assert n == len(examples)

    import tensorflow as tf
    ds = tf.data.TFRecordDataset([str(path)])
    decoded = [decode_example(rec.numpy()) for rec in ds]
    assert len(decoded) == n
    x0, y0, tid0, ci0 = decoded[0]
    assert tid0 == "tx1"
    assert ci0 == 0
    assert x0.shape == (DEFAULT_CHUNK_LEN, NUM_INPUT_CHANNELS)
    assert y0.shape == (DEFAULT_CHUNK_LEN, NUM_CLASSES)
    # First 10 positions should match our input sequence/labels.
    assert np.array_equal(x0[:10, :5], encode_nucleotides(seq))
    assert np.array_equal(np.argmax(y0[:10], axis=1), labels)
