"""Masked categorical cross-entropy loss for ORF label prediction.

Padded positions (where input[..., 5] == 1) must be excluded from the loss.
Class weights can compensate for the heavy IR imbalance: START and STOP are
rare but high-value signals, so up-weighting them stabilises training.
"""

from __future__ import annotations

import tensorflow as tf


# Default weights: [IR, START, E1, E2, E0, STOP]
DEFAULT_CLASS_WEIGHTS = [1.0, 10.0, 1.0, 1.0, 1.0, 10.0]


def masked_crossentropy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    pad_mask: tf.Tensor,
    class_weights: list[float] | None = None,
) -> tf.Tensor:
    """Compute mean cross-entropy, ignoring padded positions.

    Args:
        y_true:       float32 [B, L, 6] one-hot label tensor.
        y_pred:       float32 [B, L, 6] logits.
        pad_mask:     bool    [B, L]    True where position is PAD.
        class_weights: per-class multipliers; defaults to DEFAULT_CLASS_WEIGHTS.

    Returns:
        Scalar mean loss over all non-padded positions in the batch.
    """
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    weights_t = tf.constant(class_weights, dtype=tf.float32)  # [6]

    # per-position cross-entropy  [B, L]
    per_pos = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # per-position class weight: dot(y_true, weights_t) -> [B, L]
    pos_weight = tf.reduce_sum(y_true * weights_t, axis=-1)

    valid_mask = tf.cast(~pad_mask, tf.float32)  # 1.0 where not PAD
    weighted = per_pos * pos_weight * valid_mask

    n_valid = tf.reduce_sum(valid_mask) + 1e-8
    return tf.reduce_sum(weighted) / n_valid


class MaskedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Keras Loss wrapper around masked_crossentropy.

    Expects y_true to be a float32 tensor of shape [B, L, 7] where the last
    channel encodes the PAD mask (1.0 = padded position).  This packing trick
    lets it work seamlessly with model.compile() and model.fit().
    """

    def __init__(
        self,
        class_weights: list[float] | None = None,
        name: str = "masked_crossentropy",
    ):
        super().__init__(name=name)
        self.class_weights = class_weights or DEFAULT_CLASS_WEIGHTS

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # unpack: first 6 channels = one-hot, last channel = pad flag
        labels = y_true[..., :6]
        pad_mask = tf.cast(y_true[..., 6], tf.bool)
        return masked_crossentropy(labels, y_pred, pad_mask, self.class_weights)

    def get_config(self):
        cfg = super().get_config()
        cfg["class_weights"] = self.class_weights
        return cfg
