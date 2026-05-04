"""Simplified 6-state ORF HMM, modelled on Tiberius's gene-prediction HMM.

This is the intron- / splice-site-free counterpart to
``tiberius/hmm.py:HMMBlock``. State indices match the project label order
used in ``data/label_transcripts.py`` and ``hmm/viterbi.py``::

    0 IR    1 START    2 E1    3 E2    4 E0    5 STOP

A canonical CDS labelled by ``build_labels`` is
``START E1 E2 E0 E1 E2 E0 ... E1 STOP``, so codons in the cycle are
``(E0, E1, E2)`` and the codon containing the stop is ``(E0, E1, STOP)``.

Hard constraints (mirrored from ``state_transitions_simple`` /
``get_nuc_emission_distribution`` in ``bricks2marble.tf.hmm.tools``):

* allowed transitions:
  ``IR->IR, IR->START, START->E1, E1->E2, E1->STOP, E2->E0, E0->E1, STOP->IR``
* left-pivoted 3-mer at ``START`` must be a start codon (default ATG)
* right-pivoted 3-mer at ``E2`` must NOT be a stop codon (in-frame stop check)
* right-pivoted 3-mer at ``STOP`` must be a stop codon (default TAA/TAG/TGA)

The layer wraps ``hidten.tf.TFHMM`` directly rather than going through
``bricks2marble.tf.AnnotationHMM`` because the latter hardcodes 12 + 3*isc
states, leaving no path to a 6-state graph.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from bricks2marble.tf.hmm.tools import left_right_3mers
from hidten import HMMMode
from hidten.tf import TFHMM, TFCategoricalEmitter


IR, START, E1, E2, E0, STOP = 0, 1, 2, 3, 4, 5
N_STATES = 6
STATE_NAMES: tuple[str, ...] = ("IR", "START", "E1", "E2", "E0", "STOP")

ALLOWED_TRANSITIONS: tuple[tuple[int, int], ...] = (
    (IR,    IR),
    (IR,    START),
    (START, E1),
    (E1,    E2),
    (E1,    STOP),
    (E2,    E0),
    (E0,    E1),
    (STOP,  IR),
)

DEFAULT_START_CODONS: tuple[tuple[str, float], ...] = (("ATG", 1.0),)
DEFAULT_STOP_CODONS: tuple[tuple[str, float], ...] = (
    ("TAG", 0.34), ("TAA", 0.33), ("TGA", 0.33),
)


# ---------- transitioner setup ----------

def state_transitions(
    initial_ir_len: float,
    initial_exon_len: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Build (allow, init) for the OrfAnnotationHMM transitioner (1 head).

    ``initial_*_len`` is interpreted exactly as in
    ``bricks2marble.tf.hmm.tools.state_transitions_simple``: a value > 1 is
    the geometric scale parameter directly; a value in (0, 1) is treated
    as a per-step stay-probability and converted via ``1/(1-p)``.

    Per-row ``zero_row_softmax`` re-normalises the kernel at runtime, so
    only the within-row ratios of the values matter.
    """
    p_IR = (
        float(initial_ir_len)
        if initial_ir_len > 1
        else 1.0 / (1.0 - float(initial_ir_len))
    )
    p_exon = (
        float(initial_exon_len)
        if initial_exon_len > 1
        else 1.0 / (1.0 - float(initial_exon_len))
    )
    if p_IR <= 1.0 or p_exon <= 1.0:
        raise ValueError("initial_ir_len and initial_exon_len must yield p > 1")

    # Logits per allowed transition (same order as ALLOWED_TRANSITIONS).
    # These are intentionally in the same shape as Tiberius's
    # state_transitions_simple after introns / splice-sites are removed:
    # the only non-zero logits are the geometric self-loop / cycle scales.
    logits = np.array([
        np.log(p_IR - 1.0),     # IR    -> IR
        0.0,                     # IR    -> START
        0.0,                     # START -> E1
        np.log(p_exon - 1.0),    # E1    -> E2
        0.0,                     # E1    -> STOP
        0.0,                     # E2    -> E0
        0.0,                     # E0    -> E1
        0.0,                     # STOP  -> IR
    ], dtype=np.float32)

    # The transitioner's initializer setter applies safe_log internally,
    # so we pass exp(logits) here. Per-row softmax at runtime turns these
    # into proper transition probabilities.
    init = np.exp(logits).astype(np.float32)
    return list(ALLOWED_TRANSITIONS), init


def state_start_dist() -> tuple[list[int], np.ndarray]:
    """Allow start in any state with uniform initial probability.

    Mirrors Tiberius's ``state_start_dist`` (which uses near-uniform random
    init with ``train_start_dist=False``); this matters because the layer
    is applied to chunked sequences whose first position is rarely IR.
    """
    allow_start = list(range(N_STATES))
    init = np.full(N_STATES, 1.0 / N_STATES, dtype=np.float32)
    return allow_start, init


# ---------- codon emitter setup ----------

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _codon_index(triplet: str) -> int:
    """Index of a codon under bricks2marble's collapsed-3-mer encoding.

    For both pivot directions ``make_kmer(..., collapse_pivot=True)`` packs
    the three bases as ``b1 * 16 + b2 * 4 + b3`` (5'->3' order). The
    "pivot" only changes how the k-mer is *aligned* to the time axis
    (left- vs right-padded by ``left_right_3mers``), not the encoding
    within the 64-vector.
    """
    b1, b2, b3 = (_BASE_IDX[c.upper()] for c in triplet)
    return b1 * 16 + b2 * 4 + b3


def _codon_probs_64(codons: list[tuple[str, float]] | tuple) -> np.ndarray:
    probs = np.zeros(64, dtype=np.float32)
    for triplet, p in codons:
        probs[_codon_index(triplet)] += float(p)
    return probs


def _not_stop_probs_64(stop_codons: list[tuple[str, float]] | tuple) -> np.ndarray:
    """Uniform distribution over codons that are not stops."""
    stop_probs = _codon_probs_64(stop_codons)
    not_stop = (stop_probs == 0).astype(np.float32)
    not_stop /= not_stop.sum()
    return not_stop


def codon_emissions(
    start_codons: list[tuple[str, float]] | tuple,
    stop_codons: list[tuple[str, float]] | tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (left, right) codon emission probability matrices.

    Each output has shape ``(1, 6, 65)`` — one head, 6 states, 64 codons +
    a 65th "unrestricted" channel that ``left_right_3mers`` wires to the
    constant ``1/64`` so unrestricted states emit a flat baseline.

    Per-state restrictions:
        IR     left=any,        right=any
        START  left=start_codon right=any
        E1     left=any,        right=any
        E2     left=any,        right=not_stop  (in-frame stop check)
        E0     left=any,        right=any
        STOP   left=any,        right=stop_codon
    """
    start_left = _codon_probs_64(start_codons)
    not_stop_right = _not_stop_probs_64(stop_codons)
    stop_right = _codon_probs_64(stop_codons)

    unrestricted = np.zeros(65, dtype=np.float32)
    unrestricted[64] = 1.0

    def _pad(p64: np.ndarray) -> np.ndarray:
        return np.concatenate([p64, np.zeros(1, dtype=np.float32)])

    left = np.stack([
        unrestricted,        # IR
        _pad(start_left),    # START -> ATG (left-pivot)
        unrestricted,        # E1
        unrestricted,        # E2
        unrestricted,        # E0
        unrestricted,        # STOP
    ], axis=0)[np.newaxis, ...]

    right = np.stack([
        unrestricted,             # IR
        unrestricted,             # START
        unrestricted,             # E1
        _pad(not_stop_right),     # E2 -> not stop
        unrestricted,             # E0
        _pad(stop_right),         # STOP -> stop codon
    ], axis=0)[np.newaxis, ...]

    return left, right


# ---------- stream emitter setup (eye matrix) ----------

def eye_emission(epsilon: float) -> np.ndarray:
    """Build the (1, 6, 6) emission probability matrix for the stream emitter.

    Mirrors ``emission_parameters_eye``: each state ``i`` emits class ``i``
    with probability ``1 - epsilon`` and any other class with
    ``epsilon / (S-1)``. Feeding the NN's per-position softmax as the
    categorical observation then yields an emission score that is, up to
    smoothing, the NN's predicted probability of the corresponding state.
    """
    S = N_STATES
    eye = np.eye(S, dtype=np.float32)
    eye += epsilon / (S - 1)
    eye[np.diag_indices(S)] -= epsilon * (1.0 + 1.0 / (S - 1))
    return eye[np.newaxis, ...]   # (1, 6, 6)


# ---------- optional codon-hint emitter setup ----------

def codon_hint_emission(f: float) -> np.ndarray:
    """Build a (1, 6, 7) emission matrix for the optional codon-hint emitter.

    Mirrors ``get_codon_emission_distribution`` from bricks2marble for the
    6-state simplified HMM. The 7-channel one-hot is

        0 = outside any hint
        1 = start codon position 1   -> boosts START
        2 = start codon position 2   -> boosts E1
        3 = start codon position 3   -> boosts E2
        4 = stop  codon position 1   -> boosts E0
        5 = stop  codon position 2   -> boosts E1
        6 = stop  codon position 3   -> boosts STOP

    The "outside" column is row-compensated so that, after the row-softmax
    applied by ``TFCategoricalEmitter``, every state emits the outside
    class with the same probability ``1/7`` (otherwise boosted rows would
    silently depress their states everywhere outside hint regions). E1 is
    boosted twice (positions 2 of both start and stop codons) so it gets
    a different compensation than the singly-boosted rows.
    """
    emission = np.ones((N_STATES, 7), dtype=np.float32)
    emission[START, 1] = f
    emission[E1,    2] = f
    emission[E2,    3] = f
    emission[E0,    4] = f
    emission[E1,    5] = f
    emission[STOP,  6] = f

    c1 = (f + 5.0) / 6.0
    c2 = (f + 2.0) / 3.0
    for s in (START, E2, E0, STOP):
        emission[s, 0] = c1
    emission[E1, 0] = c2

    return emission[np.newaxis, ...]


# ---------- the layer ----------

class OrfAnnotationHMM(tf.keras.Layer):
    """6-state HMM head for the ORF finder.

    Inputs to ``call``:
        x: float tensor of shape ``(B, T, 6)`` — NN per-position state
            probabilities (softmax already applied) over IR, START, E1,
            E2, E0, STOP.
        nuc: float tensor of shape ``(B, T, nbases [+ 6 hints])`` — first
            ``nbases`` channels are the nucleotide one-hot for ACGT(N);
            optional 6 trailing channels are codon-hint one-hots used iff
            ``codon_hint_emitter`` is set.

    Output: tensor whose shape depends on ``mode``:
        * POSTERIOR / FORWARD_LOG / BACKWARD_LOG: ``(B, T, 1, 6)``
        * VITERBI: ``(B, T, 1)``
        * LIKELIHOOD_LOG: ``(B, 1, 1)``

    Args mirror ``tiberius/hmm.py:HMMBlock`` where applicable. Defaults
    freeze every learnable parameter (the HMM is a hard-constrained
    structured layer on top of the NN), and the start distribution is
    uniform — chunked sequences rarely begin in IR.
    """

    def __init__(
        self,
        mode: HMMMode = HMMMode.POSTERIOR,
        parallel: int = 1,
        training: bool = False,
        emitter_epsilon: float = 0.01,
        initial_exon_len: float = 600.0,
        initial_ir_len: float = 300.0,
        codon_hint_emitter: float | None = None,
        start_codons: list[tuple[str, float]] | None = None,
        stop_codons: list[tuple[str, float]] | None = None,
        train_emitter: bool = False,
        train_transitions: bool = False,
        train_start_dist: bool = False,
        uniform_N: bool = False,
        use_codon_emitter: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.parallel = parallel
        self.training_flag = training
        self.emitter_epsilon = emitter_epsilon
        self.initial_exon_len = float(initial_exon_len)
        self.initial_ir_len = float(initial_ir_len)
        self.codon_hint_factor = codon_hint_emitter
        self.start_codons = (
            list(start_codons) if start_codons is not None
            else list(DEFAULT_START_CODONS)
        )
        self.stop_codons = (
            list(stop_codons) if stop_codons is not None
            else list(DEFAULT_STOP_CODONS)
        )
        self._train_emitter = train_emitter
        self._train_transitions = train_transitions
        self._train_start_dist = train_start_dist
        self.uniform_N = uniform_N
        self.use_codon_emitter = use_codon_emitter

        self.hmm = TFHMM(states=N_STATES, heads=1)

        allow, t_init = state_transitions(
            self.initial_ir_len, self.initial_exon_len,
        )
        allow_start, s_init = state_start_dist()
        self.hmm.transitioner.allow = allow
        self.hmm.transitioner.initializer = t_init
        self.hmm.transitioner.allow_start = allow_start
        self.hmm.transitioner.initializer_start = s_init
        self.hmm.transitioner.train_transitions = train_transitions
        self.hmm.transitioner.train_start_dist = train_start_dist

        self.stream_emitter = TFCategoricalEmitter()
        self.stream_emitter.initializer = eye_emission(emitter_epsilon).flatten()
        self.stream_emitter.clip_min = 1e-7
        self.stream_emitter.trainable = train_emitter
        self.hmm.add_emitter(self.stream_emitter)

        if use_codon_emitter:
            left_probs, right_probs = codon_emissions(
                self.start_codons, self.stop_codons,
            )
            self.nuc_emitter_left = TFCategoricalEmitter()
            self.nuc_emitter_left.initializer = left_probs.flatten()
            self.nuc_emitter_left.trainable = False
            self.hmm.add_emitter(self.nuc_emitter_left)

            self.nuc_emitter_right = TFCategoricalEmitter()
            self.nuc_emitter_right.initializer = right_probs.flatten()
            self.nuc_emitter_right.trainable = False
            self.hmm.add_emitter(self.nuc_emitter_right)
        else:
            self.nuc_emitter_left = None
            self.nuc_emitter_right = None

        if codon_hint_emitter is not None:
            self.codon_hint_emitter_layer = TFCategoricalEmitter()
            self.codon_hint_emitter_layer.initializer = (
                codon_hint_emission(codon_hint_emitter).flatten()
            )
            self.codon_hint_emitter_layer.trainable = False
            self.hmm.add_emitter(self.codon_hint_emitter_layer)
        else:
            self.codon_hint_emitter_layer = None

    def build(self, input_shape) -> None:
        # When called as `layer(x, nuc=nuc)`, Keras passes the x shape only.
        # When called as `layer((x, nuc))`, it passes a tuple of shapes.
        if (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) > 0
            and isinstance(input_shape[0], (list, tuple))
        ):
            x_shape = tuple(input_shape[0])
        else:
            x_shape = tuple(input_shape)
        stream_shape = x_shape
        build_shapes: list = [stream_shape]
        if self.use_codon_emitter:
            codon_shape = stream_shape[:-1] + (65,)
            build_shapes.extend([codon_shape, codon_shape])
        if self.codon_hint_emitter_layer is not None:
            build_shapes.append(stream_shape[:-1] + (7,))
        self.hmm.build(tuple(build_shapes))
        super().build(input_shape)

    def _split_nuc(
        self, nuc: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor | None]:
        nbases = 4 if self.uniform_N else 5
        if self.codon_hint_emitter_layer is not None:
            ch = nuc[..., nbases:nbases + 6]
            outside = 1.0 - tf.reduce_sum(ch, axis=-1, keepdims=True)
            ch = tf.concat([outside, ch], axis=-1)
            return nuc[..., :nbases], ch
        return nuc[..., :nbases], None

    def call(self, x: tf.Tensor, nuc: tf.Tensor) -> tf.Tensor:
        nuc_raw, codon_hint = self._split_nuc(nuc)
        emissions: tuple[tf.Tensor, ...] = (x,)
        if self.use_codon_emitter:
            nuc_left, nuc_right = left_right_3mers(
                nuc_raw, uniform_N=self.uniform_N,
            )
            emissions = emissions + (nuc_left, nuc_right)
        if codon_hint is not None:
            emissions = emissions + (codon_hint,)
        return self.hmm(*emissions, mode=self.mode, parallel=self.parallel)

    def state_names(self) -> list[str]:
        return list(STATE_NAMES)
