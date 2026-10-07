"""f0 baselines: PESTO and CREPE (torchcrepe port). Both take 16 kHz mono and use a 10 ms hop.

Training-data overlap with our gold sets (reported alongside the scores):
- PESTO's bundled checkpoints (mir-1k, mir-1k_g7) are trained on MIR-1K.
- CREPE's released model was trained on a mix that includes MIR-1K and MDB-stem-synth.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np

HOP_S = 0.010


@dataclass(frozen=True)
class F0Baseline:
    name: str
    description: str
    predict: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    voicing_threshold: float
    trained_on: tuple[str, ...] = ()  # gold corpora seen in training
    version: str = "1"


def _device():
    from ..device import setup_device

    return setup_device()


PESTO_MODEL = "mir-1k_g7"  # package default


@functools.lru_cache(maxsize=4)
def _pesto_model(sr: int):
    """pesto.predict() reloads the checkpoint and rebuilds the CQT on every call; load once instead."""
    import pesto as pesto_pkg

    return pesto_pkg.load_model(PESTO_MODEL, HOP_S * 1000, sampling_rate=sr).to(_device()).eval()


def pesto(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same computation as pesto.predict(x, sr, step_size=10, model_name=PESTO_MODEL)."""
    import torch

    model = _pesto_model(sr)
    with torch.inference_mode():
        f0, conf, _vol = model(
            torch.from_numpy(np.ascontiguousarray(x)).to(_device()), sr=sr, convert_to_freq=True, return_activations=False
        )
    t = np.arange(f0.shape[-1]) * model.hop_size / 1000.0
    return t, f0.cpu().numpy().astype(float), conf.cpu().numpy().astype(float)


def crepe(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torchcrepe

    dev = _device()
    hop = int(round(HOP_S * sr))
    f0, periodicity = torchcrepe.predict(
        torch.from_numpy(np.ascontiguousarray(x))[None],
        sr,
        hop_length=hop,
        fmin=50.0,
        fmax=1600.0,
        model="full",
        return_periodicity=True,
        batch_size=2048,
        device=str(dev),
        pad=True,
    )
    f0, periodicity = f0[0].cpu().numpy().astype(float), periodicity[0].cpu().numpy().astype(float)
    return np.arange(len(f0)) * hop / sr, f0, periodicity


PRAAT_FLOOR_HZ, PRAAT_CEILING_HZ = 50.0, 1000.0  # one untuned range covering every gold set


def praat_ac(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Praat autocorrelation (Boersma 1993), defaults except the pitch range, sampled on the 10 ms grid.

    Unvoiced frames have no estimate (NaN), so they count as misses in RPA; confidence is 1/0 voicing.
    """
    import parselmouth

    pitch = parselmouth.Sound(x.astype(np.float64), sr).to_pitch_ac(
        time_step=HOP_S, pitch_floor=PRAAT_FLOOR_HZ, pitch_ceiling=PRAAT_CEILING_HZ
    )
    t = np.arange(int(len(x) / sr / HOP_S) + 1) * HOP_S
    f0 = np.array([pitch.get_value_at_time(float(ti)) for ti in t], float)  # NaN when unvoiced
    return t, f0, np.isfinite(f0).astype(float)


F0_BASELINES: dict[str, F0Baseline] = {
    b.name: b
    for b in (
        F0Baseline(
            "pesto",
            f"PESTO {PESTO_MODEL} (pesto-pitch 2.0.1), 10 ms hop; voiced if confidence >= 0.5",
            pesto,
            voicing_threshold=0.5,
            trained_on=("mir1k",),
        ),
        F0Baseline(
            "crepe",
            "CREPE full (torchcrepe 0.0.24), Viterbi decoding, 50-1600 Hz, 10 ms hop; voiced if periodicity >= 0.21",
            crepe,
            voicing_threshold=0.21,
            trained_on=("mir1k", "mdb_stem_synth"),
        ),
        F0Baseline(
            "praat_ac",
            f"Praat autocorrelation (parselmouth), defaults except range {PRAAT_FLOOR_HZ:g}-{PRAAT_CEILING_HZ:g} Hz, "
            "10 ms grid; unvoiced frames have no estimate",
            praat_ac,
            voicing_threshold=0.5,
        ),
    )
}
