"""Small MLP head over frozen ORFfinder features that emits a TP probability."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tensorflow import keras


FEATURE_COLS: list[str] = [
    "mean_coding_prob",
    "max_coding_prob",
    "min_coding_prob",
    "p10_coding_prob",
    "p90_coding_prob",
    "std_coding_prob",
    "coding_first10pct",
    "coding_mid80pct",
    "coding_last10pct",
    "codon1_mean_coding",
    "codon2_mean_coding",
    "start_prob",
    "stop_prob",
    "mean_ir_prob",
    "max_ir_prob",
    "p90_ir_prob",
    "frac_argmax_coding",
    "frac_argmax_ir",
]


@dataclass
class FinetuneConfig:
    hidden: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.3
    n_features: int = 18


def build_finetune_head(cfg: FinetuneConfig) -> keras.Model:
    inp = keras.Input(shape=(cfg.n_features,), name="features")
    x = inp
    for i, h in enumerate(cfg.hidden):
        x = keras.layers.Dense(h, activation="relu", name=f"dense_{i+1}")(x)
        x = keras.layers.Dropout(cfg.dropout, name=f"drop_{i+1}")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="tp_prob")(x)
    return keras.Model(inputs=inp, outputs=out, name="finetune_head")


def load_finetune_head(path: Path, cfg: FinetuneConfig | None = None) -> keras.Model:
    """Load a saved fine-tuning head from a weights .h5 file.

    Requires cfg to rebuild the architecture before loading weights.
    If cfg is None, a default FinetuneConfig() is used.
    """
    if cfg is None:
        cfg = FinetuneConfig()
    model = build_finetune_head(cfg)
    model.load_weights(str(path))
    return model
