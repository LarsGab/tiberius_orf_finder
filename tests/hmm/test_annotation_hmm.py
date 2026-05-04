"""Tests for the simplified 6-state OrfAnnotationHMM Keras layer.

These tests exercise both the pure-numpy emission/transition tables and
end-to-end decoding through ``hidten.tf.TFHMM`` on a tiny synthetic CDS.
They are skipped if ``tensorflow``, ``hidten``, or ``bricks2marble`` is
not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if any heavy dep is missing.
tf = pytest.importorskip("tensorflow")
pytest.importorskip("hidten")
pytest.importorskip("bricks2marble")

from hidten import HMMMode  # noqa: E402

from tiberius_orf.hmm.annotation_hmm import (  # noqa: E402
    ALLOWED_TRANSITIONS,
    DEFAULT_STOP_CODONS,
    E0, E1, E2, IR, N_STATES, START, STOP,
    OrfAnnotationHMM,
    _codon_probs_64,
    codon_emissions,
    eye_emission,
    state_start_dist,
    state_transitions,
)


# ---------- codon-probs builder ----------

def test_codon_probs_64_round_trips_through_bricks2marble():
    """`_codon_probs_64` is just a numpy view onto bricks2marble's
    `make_codon_probs`. We verify (a) shape (b) total mass and (c) that
    a single-codon distribution puts a single 1.0 at SOME index, and
    that index is consistent across the two pivot directions for codons
    that are palindromic on the b2 base.
    """
    from bricks2marble.tf.hmm.tools import make_codon_probs

    for codon in ("ATG", "TAA", "TAG", "TGA", "AAA", "TTT", "GCG"):
        for pivot_left in (True, False):
            probs = _codon_probs_64([(codon, 1.0)], pivot_left)
            assert probs.shape == (64,)
            assert probs.sum() == pytest.approx(1.0)
            # exactly one position should hold all the mass
            assert (probs > 0).sum() == 1
            # and it must agree with the underlying make_codon_probs
            expected = make_codon_probs([(codon, 1.0)], pivot_left).numpy().flatten()
            np.testing.assert_allclose(probs, expected, atol=0)


# ---------- transition table ----------

def test_state_transitions_logit_shape_and_order():
    allow, init = state_transitions(initial_ir_len=300, initial_exon_len=600)
    assert len(allow) == len(ALLOWED_TRANSITIONS) == 8
    assert init.shape == (8,)
    assert init.dtype == np.float32
    # E1 -> STOP must be present (the bug-fix)
    assert (E1, STOP) in [tuple(p) for p in allow]
    # E0 -> STOP must NOT be present
    assert (E0, STOP) not in [tuple(p) for p in allow]


def test_state_start_dist_uniform():
    allow_start, init = state_start_dist()
    assert allow_start == list(range(N_STATES))
    np.testing.assert_allclose(init, np.full(N_STATES, 1.0 / N_STATES))


# ---------- codon emission tables ----------

def _idx_left(codon):
    """Flat 64-index for a codon under bricks2marble's left-pivot encoding."""
    from bricks2marble.tf.hmm.tools import make_codon_probs
    probs = make_codon_probs([(codon, 1.0)], pivot_left=True).numpy().flatten()
    return int(np.argmax(probs))


def _idx_right(codon):
    """Flat 64-index for a codon under bricks2marble's right-pivot encoding."""
    from bricks2marble.tf.hmm.tools import make_codon_probs
    probs = make_codon_probs([(codon, 1.0)], pivot_left=False).numpy().flatten()
    return int(np.argmax(probs))


def test_codon_emissions_start_codon_at_start():
    left, right = codon_emissions([("ATG", 1.0)], DEFAULT_STOP_CODONS)
    assert left.shape == (1, 6, 65)
    assert right.shape == (1, 6, 65)

    # START's left-pivot row puts all mass on ATG (under bricks2marble's encoding)
    atg = _idx_left("ATG")
    assert left[0, START, atg] == pytest.approx(1.0)
    assert left[0, START, :64].sum() == pytest.approx(1.0)
    # All other codons forbidden at START
    for k in range(64):
        if k != atg:
            assert left[0, START, k] == 0.0
    # START's right-pivot row is unrestricted (all mass on column 64)
    assert right[0, START, 64] == pytest.approx(1.0)


def test_codon_emissions_stop_codon_at_stop():
    _, right = codon_emissions([("ATG", 1.0)], DEFAULT_STOP_CODONS)
    stop_idxs = {_idx_right(c) for c, _ in DEFAULT_STOP_CODONS}
    # Stop row has mass only on stop codons
    for k in range(64):
        if k in stop_idxs:
            assert right[0, STOP, k] > 0.0
        else:
            assert right[0, STOP, k] == 0.0
    # And mass sums to 1 over the 64 codons
    assert right[0, STOP, :64].sum() == pytest.approx(1.0, abs=1e-6)


def test_codon_emissions_in_frame_stop_check_at_e2():
    _, right = codon_emissions([("ATG", 1.0)], DEFAULT_STOP_CODONS)
    stop_idxs = {_idx_right(c) for c, _ in DEFAULT_STOP_CODONS}
    # E2 right-pivot row forbids stop codons (in-frame stop check)
    for k in stop_idxs:
        assert right[0, E2, k] == 0.0
    # And distributes uniformly over the remaining 61 codons
    non_stop_vals = [right[0, E2, k] for k in range(64) if k not in stop_idxs]
    expected = 1.0 / 61.0
    for v in non_stop_vals:
        assert v == pytest.approx(expected, rel=1e-6)


def test_codon_emissions_unrestricted_states():
    """IR / E1 / E0 must have all mass on the 65th (constant) channel for
    both pivots — i.e. they contribute a flat 1/64 baseline regardless of
    nucleotide context."""
    left, right = codon_emissions([("ATG", 1.0)], DEFAULT_STOP_CODONS)
    for state in (IR, E1, E0):
        assert left[0, state, 64] == pytest.approx(1.0)
        assert left[0, state, :64].sum() == 0.0
        assert right[0, state, 64] == pytest.approx(1.0)
        assert right[0, state, :64].sum() == 0.0


# ---------- eye emission ----------

def test_eye_emission_diag_and_offdiag():
    eps = 0.01
    eye = eye_emission(eps)
    assert eye.shape == (1, 6, 6)
    for q in range(N_STATES):
        assert eye[0, q, q] == pytest.approx(1.0 - eps)
        for k in range(N_STATES):
            if k != q:
                assert eye[0, q, k] == pytest.approx(eps / (N_STATES - 1))


# ---------- end-to-end layer ----------

def _peaked_softmax(labels: list[int], sharpness: float = 50.0) -> np.ndarray:
    """Build a (1, T, 6) softmax-like emission peaked on `labels`."""
    L = len(labels)
    logits = np.zeros((1, L, N_STATES), dtype=np.float32)
    for t, q in enumerate(labels):
        logits[0, t, q] = sharpness
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def _make_nuc(seq: str) -> np.ndarray:
    """Encode an ACGT(N) string as a (1, T, 5) one-hot array."""
    base_to_col = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    L = len(seq)
    arr = np.zeros((1, L, 5), dtype=np.float32)
    for t, b in enumerate(seq):
        arr[0, t, base_to_col[b.upper()]] = 1.0
    return arr


def test_layer_forward_runs_and_shape():
    """Smoke test: the layer runs in POSTERIOR mode on a length-12
    sequence and returns the expected shape."""
    layer = OrfAnnotationHMM(mode=HMMMode.POSTERIOR, parallel=1)
    seq = "AAATGAAAAAAA"
    labels = [IR, IR, START, E1, E2, E0, E1, E2, E0, E1, STOP, IR]
    x = _peaked_softmax(labels)
    nuc = _make_nuc(seq)
    out = layer(tf.constant(x), tf.constant(nuc))
    # POSTERIOR returns (B, T, H, Q)
    assert tuple(out.shape) == (1, 12, 1, N_STATES)


def test_layer_viterbi_recovers_canonical_orf():
    """End-to-end: with strongly-peaked NN logits and a synthetic ORF
    nucleotide sequence (ATG + middle codon + TAA), Viterbi must
    recover the exact label sequence.

    The codon emitters add a hard-zero at any codon position that
    violates the genetic code, so this also implicitly tests that the
    ATG-at-START / TAA-at-STOP / no-in-frame-stop constraints are
    consistent with the chosen transition table.
    """
    seq = "AAATGAAATAAA"   # IR IR ATG AAA TAA A  -> START at pos 2, STOP at pos 10
    labels = [IR, IR, START, E1, E2, E0, E1, E2, E0, E1, STOP, IR]

    layer = OrfAnnotationHMM(mode=HMMMode.VITERBI, parallel=1)
    x = _peaked_softmax(labels)
    nuc = _make_nuc(seq)
    out = layer(tf.constant(x), tf.constant(nuc)).numpy()
    # Viterbi returns (B, T, H)
    assert out.shape == (1, 12, 1)
    np.testing.assert_array_equal(out[0, :, 0], np.array(labels))
