"""Bidirectional-LSTM model for per-position ORF label prediction.

Architecture:
    1. Zero-out the PAD channel (index 5) so it carries no signal into the LSTM.
    2. Optional 1-D convolutional stem to extract local sequence features.
    3. Stacked bidirectional LSTM layers with recurrent dropout.
    4. Dense output head -> 6-class logits per position (IR/START/E1/E2/E0/STOP).

Input shape:  [batch, L, 6]  float32  (nucleotide one-hot + PAD channel)
Output shape: [batch, L, 6]  float32  (logits; apply softmax for probabilities)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


NUM_INPUT_CHANNELS = 6
NUM_CLASSES = 6


def build_model(
    chunk_len: int = 9999,
    lstm_units: int = 512,
    lstm_layers: int = 3,
    dropout: float = 0.1,
    use_conv_stem: bool = True,
    conv_filters: int = 64,
    conv_kernel: int = 11,
) -> keras.Model:
    """Build and return the BiLSTM ORF-finder model.

    Args:
        chunk_len: sequence window length (number of positions).
        lstm_units: units per LSTM direction (total hidden dim = 2 * lstm_units).
        lstm_layers: number of stacked BiLSTM layers.
        dropout: dropout rate applied between BiLSTM layers.
        use_conv_stem: prepend a 1-D conv stem before the LSTMs.
        conv_filters: output channels for the conv stem.
        conv_kernel: kernel size for the conv stem.
    """
    inp = keras.Input(shape=(chunk_len, NUM_INPUT_CHANNELS), name="input")

    # zero out PAD channel so padded positions contribute nothing
    pad_zero = keras.layers.Lambda(
        lambda x: x * tf.concat(
            [tf.ones_like(x[..., :5]), tf.zeros_like(x[..., 5:])], axis=-1
        ),
        name="zero_pad_channel",
    )(inp)

    x = pad_zero

    if use_conv_stem:
        x = keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=conv_kernel,
            padding="same",
            activation="relu",
            name="conv_stem",
        )(x)

    for i in range(lstm_layers):
        return_seq = True   # always return sequences for per-position output
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                lstm_units,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=0.0,  # recurrent dropout is slow on GPU
            ),
            name=f"bilstm_{i}",
        )(x)
        if dropout > 0.0 and i < lstm_layers - 1:
            x = keras.layers.Dropout(dropout, name=f"drop_{i}")(x)

    logits = keras.layers.Dense(NUM_CLASSES, name="logits")(x)  # [B, L, 6]

    return keras.Model(inputs=inp, outputs=logits, name="tiberius_orf")
