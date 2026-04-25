"""CNN + U-Net encoder-decoder ORF model.

Symmetric U-Net with `depth` downsampling stages, each:
    [Conv1D x2 (relu)] -> save skip -> MaxPool1D(pool_size)
Bottleneck: Conv1D x2 + dropout
Decoder mirrors the encoder; each stage:
    UpSampling1D(pool_size) -> concat skip -> Conv1D x2

chunk_len must be divisible by pool_size**depth
(default 9999 with pool_size=3, depth=2 -> 9999 -> 3333 -> 1111).
"""

from __future__ import annotations

from tensorflow import keras

from .common import NUM_INPUT_CHANNELS, NUM_CLASSES, zero_pad_channel, conv_stem


def _conv_pair(x, filters: int, kernel: int, name: str):
    x = keras.layers.Conv1D(filters, kernel, padding="same",
                            activation="relu", name=f"{name}_a")(x)
    x = keras.layers.Conv1D(filters, kernel, padding="same",
                            activation="relu", name=f"{name}_b")(x)
    return x


def build_cnn_unet(
    chunk_len: int = 9999,
    conv_filters: int = 64,
    conv_kernel: int = 9,
    conv_layers: int = 2,
    base_filters: int = 64,
    unet_kernel: int = 5,
    depth: int = 2,
    pool_size: int = 3,
    bottleneck_dropout: float = 0.1,
) -> keras.Model:
    if chunk_len % (pool_size ** depth) != 0:
        raise ValueError(
            f"chunk_len ({chunk_len}) must be divisible by pool_size**depth "
            f"({pool_size}**{depth} = {pool_size ** depth})"
        )

    inp = keras.Input(shape=(chunk_len, NUM_INPUT_CHANNELS), name="input")
    x = zero_pad_channel(inp)
    x = conv_stem(x, conv_filters, conv_kernel, conv_layers)

    # Encoder
    skips = []
    f = base_filters
    for i in range(depth):
        x = _conv_pair(x, f, unet_kernel, name=f"enc_{i+1}")
        skips.append(x)
        x = keras.layers.MaxPool1D(pool_size=pool_size, name=f"pool_{i+1}")(x)
        f *= 2

    # Bottleneck
    x = _conv_pair(x, f, unet_kernel, name="bottleneck")
    if bottleneck_dropout > 0.0:
        x = keras.layers.Dropout(bottleneck_dropout, name="bottleneck_drop")(x)

    # Decoder
    for i in reversed(range(depth)):
        f //= 2
        x = keras.layers.UpSampling1D(size=pool_size, name=f"up_{i+1}")(x)
        x = keras.layers.Concatenate(name=f"concat_{i+1}")([x, skips[i]])
        x = _conv_pair(x, f, unet_kernel, name=f"dec_{i+1}")

    logits = keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return keras.Model(inputs=inp, outputs=logits, name="tiberius_orf_cnn_unet")
