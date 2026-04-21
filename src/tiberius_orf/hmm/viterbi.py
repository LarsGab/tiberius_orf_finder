"""Viterbi decoder for the 6-state ORF label HMM.

States (matching label_transcripts.py):
    0 IR    1 START    2 E1    3 E2    4 E0    5 STOP

Valid transitions (all others have -inf log-probability):
    IR    -> IR, START
    START -> E1
    E1    -> E2
    E2    -> E0
    E0    -> E1, STOP
    STOP  -> IR

Usage::

    log_probs = log_softmax(model(input))   # [L, 6]
    labels    = viterbi_decode(log_probs)   # [L] int array 0-5
"""

from __future__ import annotations

import numpy as np

# State indices
IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5
N_STATES = 6

_NEG_INF = -1e30

# Transition matrix in log-space: log_trans[from, to]
# -inf means forbidden; 0.0 means allowed (uniform prior over valid successors).
_TRANS_ALLOWED: list[tuple[int, int]] = [
    (IR,    IR),
    (IR,    START),
    (START, E1),
    (E1,    E2),
    (E2,    E0),
    (E0,    E1),
    (E0,    STOP),
    (STOP,  IR),
]

_LOG_TRANS = np.full((N_STATES, N_STATES), _NEG_INF, dtype=np.float64)
for _f, _t in _TRANS_ALLOWED:
    _LOG_TRANS[_f, _t] = 0.0   # log(1) — will be superseded by emission scores


def viterbi_decode(log_emission: np.ndarray) -> np.ndarray:
    """Run Viterbi decoding on per-position log-probabilities.

    Args:
        log_emission: float array of shape [L, 6] — log-probability (or logit)
            for each state at each position. Typically `log_softmax(logits)`.

    Returns:
        int32 array of shape [L] with the most probable valid label sequence.
    """
    log_emission = np.asarray(log_emission, dtype=np.float64)
    L = log_emission.shape[0]
    if L == 0:
        return np.empty(0, dtype=np.int32)

    # viterbi[t, s] = best log-prob of any path ending in state s at position t
    viterbi = np.full((L, N_STATES), _NEG_INF, dtype=np.float64)
    backptr = np.zeros((L, N_STATES), dtype=np.int32)

    # initialise: allow any state at position 0
    viterbi[0] = log_emission[0]

    for t in range(1, L):
        # _LOG_TRANS[prev, cur] + viterbi[t-1, prev]  -> [N_STATES, N_STATES]
        scores = viterbi[t - 1, :, None] + _LOG_TRANS   # [prev, cur]
        backptr[t] = np.argmax(scores, axis=0)
        viterbi[t] = scores[backptr[t], np.arange(N_STATES)] + log_emission[t]

    # traceback
    path = np.empty(L, dtype=np.int32)
    path[L - 1] = int(np.argmax(viterbi[L - 1]))
    for t in range(L - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    return path


def viterbi_decode_batch(log_emission: np.ndarray) -> np.ndarray:
    """Viterbi decode a batch of sequences.

    Args:
        log_emission: float array [B, L, 6].

    Returns:
        int32 array [B, L].
    """
    log_emission = np.asarray(log_emission, dtype=np.float64)
    B = log_emission.shape[0]
    return np.stack([viterbi_decode(log_emission[b]) for b in range(B)])
