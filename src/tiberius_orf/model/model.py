"""Dispatcher for the four ORF-finder architectures."""

from __future__ import annotations

from tensorflow import keras

from .cnn_lstm import build_cnn_lstm
from .cnn_transformer import build_cnn_transformer
from .cnn_tcn import build_cnn_tcn
from .cnn_unet import build_cnn_unet


_BUILDERS = {
    "cnn_lstm":        build_cnn_lstm,
    "cnn_transformer": build_cnn_transformer,
    "cnn_tcn":         build_cnn_tcn,
    "cnn_unet":        build_cnn_unet,
}


def build_model(model_type: str = "cnn_lstm", **kwargs) -> keras.Model:
    if model_type not in _BUILDERS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; choose from {sorted(_BUILDERS)}"
        )
    return _BUILDERS[model_type](**kwargs)


_SHARED_KEYS = ("conv_filters", "conv_kernel", "conv_layers")


def build_model_from_config(cfg: dict, chunk_len: int) -> keras.Model:
    mc = cfg["model"]
    mt = mc["type"]
    shared = {k: mc[k] for k in _SHARED_KEYS}
    type_kwargs = mc[mt]
    return build_model(model_type=mt, chunk_len=chunk_len, **shared, **type_kwargs)
