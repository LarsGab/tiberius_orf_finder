"""Tests for the pure-numpy viterbi decoder."""

from __future__ import annotations

import numpy as np

from tiberius_orf.hmm.viterbi import (
    E0, E1, E2, IR, N_STATES, START, STOP,
    viterbi_decode, viterbi_decode_batch,
)


def _peaked_log_emission(labels: list[int], peak: float = 50.0) -> np.ndarray:
    """Build a [L, 6] log-emission array peaked on ``labels``."""
    L = len(labels)
    log_em = np.zeros((L, N_STATES), dtype=np.float64)
    for t, q in enumerate(labels):
        log_em[t, q] = peak
    return log_em


def test_viterbi_recovers_canonical_orf_sequence():
    """`build_labels(12, orf=(2, 10))` produces this label sequence; with a
    peaked emission Viterbi must reproduce it exactly. This is the test
    case that the old `E0 -> STOP` typo fails: it would either drop the
    STOP or shift the cycle by one near the end of the ORF.
    """
    labels = [IR, IR, START, E1, E2, E0, E1, E2, E0, E1, STOP, IR]
    log_em = _peaked_log_emission(labels)
    decoded = viterbi_decode(log_em)
    np.testing.assert_array_equal(decoded, np.array(labels, dtype=np.int32))


def test_viterbi_allows_e1_to_stop_only():
    """Hard-encode an emission that strongly prefers STOP at position t=1
    and either E1 or E0 at position t=0. With the corrected transitions
    Viterbi must pick E1 at t=0 (E1 -> STOP allowed) and reject E0 at t=0
    (E0 -> STOP forbidden).
    """
    log_em = np.full((2, N_STATES), -1e3, dtype=np.float64)
    log_em[0, E1] = 0.0
    log_em[0, E0] = 0.0   # equal emission for E1 and E0 at position 0
    log_em[1, STOP] = 0.0
    decoded = viterbi_decode(log_em)
    assert decoded[0] == E1, f"expected E1 before STOP, got state {int(decoded[0])}"
    assert decoded[1] == STOP


def test_viterbi_batch_shapes():
    log_em = _peaked_log_emission([IR, START, E1, E2, E0, E1, STOP])
    batch = np.stack([log_em, log_em], axis=0)
    decoded = viterbi_decode_batch(batch)
    assert decoded.shape == (2, 7)
    np.testing.assert_array_equal(decoded[0], decoded[1])


def test_viterbi_empty_input():
    out = viterbi_decode(np.zeros((0, N_STATES), dtype=np.float64))
    assert out.shape == (0,)
    assert out.dtype == np.int32
