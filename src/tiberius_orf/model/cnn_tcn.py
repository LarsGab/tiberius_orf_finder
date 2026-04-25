"""CNN + dilated Temporal Convolutional Network ORF model.

Stacks N residual TCN blocks with exponentially increasing dilation
(1, 2, 4, ..., 2^(N-1)). Each block:
    LayerNorm -> Conv1D(dilated, padding=same) -> ReLU -> Dropout
                -> Conv1D(dilated, padding=same) -> + residual

Operates at full sequence length — no reshape pooling, no recurrence.
Receptive field grows as sum of dilations × (kernel - 1) + 1; with kernel=5
and 8 layers, RF ≈ 2 * 255 * 4 + 1 ≈ 2k positions.
"""

from __future__ import annotations

from tensorflow import keras

from .common import NUM_INPUT_CHANNELS, NUM_CLASSES, zero_pad_channel, conv_stem


def _tcn_block(x, filters: int, kernel: int, dilation: int,
               dropout: float, idx: int):
    h = keras.layers.LayerNormalization(name=f"tcn_norm_{idx}")(x)
    h = keras.layers.Conv1D(
        filters, kernel_size=kernel, dilation_rate=dilation,
        padding="same", activation="relu",
        name=f"tcn_conv1_{idx}",
    )(h)
    h = keras.layers.Dropout(dropout, name=f"tcn_drop_{idx}")(h)
    h = keras.layers.Conv1D(
        filters, kernel_size=kernel, dilation_rate=dilation,
        padding="same",
        name=f"tcn_conv2_{idx}",
    )(h)
    # match channel count for residual if needed
    if x.shape[-1] != filters:
        x = keras.layers.Conv1D(filters, kernel_size=1,
                                name=f"tcn_proj_{idx}")(x)
    return keras.layers.Add(name=f"tcn_add_{idx}")([x, h])


def build_cnn_tcn(
    chunk_len: int = 9999,
    conv_filters: int = 64,
    conv_kernel: int = 9,
    conv_layers: int = 2,
    tcn_filters: int = 64,
    tcn_kernel: int = 5,
    tcn_layers: int = 8,
    dropout: float = 0.1,
) -> keras.Model:
    inp = keras.Input(shape=(chunk_len, NUM_INPUT_CHANNELS), name="input")
    x = zero_pad_channel(inp)
    x = conv_stem(x, conv_filters, conv_kernel, conv_layers)

    for i in range(tcn_layers):
        x = _tcn_block(x, tcn_filters, tcn_kernel,
                       dilation=2 ** i, dropout=dropout, idx=i + 1)

    x = keras.layers.LayerNormalization(name="tcn_norm_final")(x)
    logits = keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return keras.Model(inputs=inp, outputs=logits, name="tiberius_orf_cnn_tcn")
