"""The noise sources and their components, shared by the three synthesisers (D-064).

Aspiration, whisper (and the /h/ onset, which is whisper) were built by the same recipe three
times over, in `voice.py`, `interactive.py` and `kl_set.py`: draw white noise, shape it into the
engine's source domain, multiply by the envelope, run it through the tract, and give it a level.
The six copies had drifted only in ways the physics explains, so what belongs here is the recipe
and the level policy; each engine passes its tract as a closure because the tract is the one thing
that genuinely differs (an all-pole cascade's radiated filter, the interactive tube's flow
responses, the KL waveguide's radiated flow).

Two levels are in play, and they are not the same rule:

- aspiration is defined *against the voiced component* it competes with: the HNR sets the ratio
  after the tract, so a change to the aspiration's spectrum moves its shape, not its level
  (D-048; `tests/test_synth.py::test_aspiration_tilt_changes_only_the_aspiration`).
- whisper has no voiced partner in its frames - it is a separate phonation mode (D-012) - so its
  level was an arbitrary `* 0.05`, shared with frication, and a tilt redistributed its energy with
  nothing holding the level (D-064: at -12 dB/oct the 1-3.5 kHz level fell about 18 dB while the
  low frequencies rose, and with the pre-D-063 source normalisation the /h/ came out louder than
  the vowel). `level="as-is"` renders the un-tilted source of the same clip through the same
  envelope and tract and scales the tilted one to that render's power, so the tilt moves only the
  shape and `h_amp` stays the level control; `level="legacy"` is the old behaviour, kept for the
  sets that were generated before D-064.
  The anchor is broadband, the same invariant the aspiration's HNR holds: anchoring in 1-3.5 kHz
  instead leaves the total power swinging with the tilt (cascade: +16 dB at -12 dB/oct), which is
  the same objection in another form.

Tilt conventions: `tilt_db_oct` means the *cascade's* source tilt, i.e. the tilt of the radiated
source (0 = white, None = the first difference of white noise, about +6 dB/oct). The interactive
tube's noise is unradiated flow into U_lips / U_glottis, which matches the cascade's radiated
source, so it uses the tilt as given; the KL waveguide radiates (differentiates) its flow itself,
so its flow noise takes `offset_db=-6.0` (D-062).
"""

from __future__ import annotations

import numpy as np

# the /h/ source stops tilting below this: -12 dB/oct with a 50 Hz floor put +52 dB at 50 Hz re
# 1 kHz and made /h/ louder than the vowel it led into (D-064)
WHISPER_FLAT_BELOW_HZ = 500.0
ASPIRATION_FLAT_BELOW_HZ = 50.0

# 1-3.5 kHz: the band the D-063 targets and the probes care about, kept as the optional anchor for
# `level_gain` (the whisper level itself is anchored broadband; see `whisper_component`)
REFERENCE_BAND_HZ = (1000.0, 3500.0)
REFERENCE_NPERSEG = 512


def tilted_noise(white: np.ndarray, tilt_db_oct: float, fs: float = 16000.0,
                 flat_below_hz: float = ASPIRATION_FLAT_BELOW_HZ) -> np.ndarray:
    """White noise given a spectral tilt (dB per octave re 1 kHz, flat below flat_below_hz), by FFT;
    same length."""
    n = len(white)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    gain = 10 ** (tilt_db_oct * np.log2(np.maximum(f, flat_below_hz) / 1000.0) / 20)
    return np.fft.irfft(np.fft.rfft(white) * gain, n)


def noise_source(white: np.ndarray, tilt_db_oct: float | None, fs: float = 16000.0, *,
                 flat_below_hz: float = ASPIRATION_FLAT_BELOW_HZ,
                 match_first_difference_power: bool = False) -> np.ndarray:
    """The noise source for `white` (n + 1 samples, so the first difference is n).

    tilt_db_oct None: the first difference of white noise (every set before D-048), else tilted
    white noise flat below `flat_below_hz`. `match_first_difference_power`: rescale the tilted noise
    to the first difference's power (variance 2), so the tilt moves the shape only - the /h/ and
    whisper convention since D-063."""
    if tilt_db_oct is None:
        return np.diff(white)
    x = tilted_noise(white[:-1], tilt_db_oct, fs, flat_below_hz)
    if match_first_difference_power:
        return x * np.sqrt(2.0 / max(np.mean(x * x), 1e-30))
    return x


def whisper_noise(white: np.ndarray, tilt_db_oct: float | None, fs: float = 16000.0) -> np.ndarray:
    """The whisper / /h/ source: `noise_source` with the /h/ tilt floor and its power convention."""
    return noise_source(white, tilt_db_oct, fs, flat_below_hz=WHISPER_FLAT_BELOW_HZ,
                        match_first_difference_power=True)


def band_energy(x: np.ndarray, fs: float = 16000.0, band: tuple[float, float] = REFERENCE_BAND_HZ,
                nperseg: int = REFERENCE_NPERSEG) -> float:
    """Welch power in `band` (Hann, `nperseg` capped at the length); the scale of a comparison
    between two renders, not an absolute level."""
    from scipy.signal import welch

    seg = int(min(nperseg, len(x)))
    if seg < 8:
        return float(np.sum(np.asarray(x) ** 2))
    f, p = welch(np.asarray(x, float), fs=fs, nperseg=seg, window="hann", detrend=False)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.sum(p[m]) * (f[1] - f[0]))


def level_gain(x: np.ndarray, ref: np.ndarray, fs: float = 16000.0,
               band: tuple[float, float] | None = None) -> float:
    """sqrt(E_ref / E_x), or 1.0 when either side is silent. Broadband (mean square) by default - the
    power the aspiration's HNR holds - or in `band` (Welch) when one is given."""
    if band is None:
        ex, er = float(np.mean(np.asarray(x, float) ** 2)), float(np.mean(np.asarray(ref, float) ** 2))
    else:
        ex, er = band_energy(x, fs, band), band_energy(ref, fs, band)
    if ex <= 0.0 or er <= 0.0:
        return 1.0
    return float(np.sqrt(er / ex))


def aspiration_component(white: np.ndarray, mod: np.ndarray, va: np.ndarray, voiced: np.ndarray, *,
                         tilt_db_oct: float | None, tract, hnr_db: float, fs: float = 16000.0,
                         offset_db: float = 0.0, flat_below_hz: float = ASPIRATION_FLAT_BELOW_HZ) -> np.ndarray:
    """The aspiration component: open-phase modulated noise through `tract` (`tract(x) -> x`), then
    scaled to `hnr_db` against `voiced` over the loud frames. Same draws whatever the tilt.
    `offset_db` is the engine's fixed tilt between its source domain and the cascade's radiated one."""
    tilt = None if tilt_db_oct is None else tilt_db_oct + offset_db
    src = noise_source(white, tilt, fs, flat_below_hz=flat_below_hz) * mod * va
    asp = tract(src)
    on = va > 0.5 * va.max() if va.max() > 0 else np.zeros(len(va), bool)
    if on.any():
        ev, ea = np.mean(voiced[on] ** 2), np.mean(asp[on] ** 2)
        if ea > 0:
            asp *= np.sqrt(ev / ea / 10 ** (hnr_db / 10))
    return asp


def whisper_component(white: np.ndarray, wa: np.ndarray, *, tilt_db_oct: float | None, tract,
                      fs: float = 16000.0, offset_db: float = 0.0, level: str = "legacy",
                      band: tuple[float, float] | None = None) -> np.ndarray:
    """The whisper / /h/ component: `whisper_noise` through `tract` (`tract(x) -> x`), times `wa`,
    at a level of 0.05 of the clip's full scale.

    `level="legacy"`: the tilt shapes the component and moves its level with it. `level="as-is"`:
    the same envelope and tract render the un-tilted source too, and the tilted component is scaled
    to that render's power - broadband by default, the invariant the aspiration's HNR holds, so the
    tilt moves only the shape and `h_amp` stays the level control; `band` anchors it in that band
    instead. Only a set tilt takes that path, so the un-tilted output is identical either way."""
    tilt = None if tilt_db_oct is None else tilt_db_oct + offset_db
    y = tract(whisper_noise(white, tilt, fs) * wa)
    if level == "as-is" and tilt is not None:
        ref = tract(whisper_noise(white, None, fs) * wa)
        y = y * level_gain(y, ref, fs, band)
    elif level not in ("legacy", "as-is"):
        raise ValueError(f"unknown whisper level policy: {level!r}")
    return y * 0.05
