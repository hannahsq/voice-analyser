"""Baselines (plan §6.5), each scored by the shared metrics via `va baselines`.

Formant baselines:  predict(x, sr, times, meta) -> (T, 4) Hz, NaN = no estimate
f0 baselines:       predict(x, sr) -> (times_s, f0_hz, voicing_prob)
"""

from __future__ import annotations

from .f0 import F0_BASELINES
from .formants import FORMANT_BASELINES

__all__ = ["F0_BASELINES", "FORMANT_BASELINES"]
