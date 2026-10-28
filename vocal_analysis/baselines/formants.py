"""Formant baselines: Praat defaults, Praat multi-ceiling sweep (FastTrack-style), TVLP, QCP-FB."""

from __future__ import annotations

import functools
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

N_FORMANTS = 4
REPO_ROOT = Path(__file__).resolve().parents[2]
TVLP_PATH = REPO_ROOT / "external" / "voice-dataset-labeler" / "extractors" / "tvlp.py"


@dataclass(frozen=True)
class FormantBaseline:
    name: str
    description: str
    predict: Callable[[np.ndarray, int, np.ndarray, dict], np.ndarray]
    version: str = "1"


def _sample_track(formant, times: np.ndarray) -> np.ndarray:
    out = np.full((len(times), N_FORMANTS), np.nan)
    for i, t in enumerate(times):
        for k in range(N_FORMANTS):
            v = formant.get_value_at_time(k + 1, float(t))
            out[i, k] = v if v == v else np.nan  # Praat returns NaN when undefined
    return out


def praat_default(x: np.ndarray, sr: int, times: np.ndarray, meta: dict) -> np.ndarray:
    """Burg with Praat's defaults: 5 formants below 5500 Hz, 25 ms window, for every speaker."""
    import parselmouth

    formant = parselmouth.Sound(x.astype(np.float64), sr).to_formant_burg()
    return _sample_track(formant, times)


# FastTrack-style selection (Barreda 2021): analyse the vowel at many ceilings,
# fit a smooth (DCT) model to each formant track, and pick the ceiling whose
# tracks are best explained by the smooth model.
SWEEP_CEILINGS = np.arange(4000.0, 8000.0 + 1, 250.0)
SWEEP_STEP_S = 0.002
SWEEP_N_COEFS = 5
SWEEP_SELECT_FORMANTS = 3


def _dct_residual(track: np.ndarray, n_coefs: int) -> float:
    """Mean |residual| (Hz) of a least-squares DCT-II fit; inf if too few valid points."""
    ok = np.isfinite(track)
    n = len(track)
    if ok.sum() < max(n_coefs + 2, int(0.67 * n)):
        return np.inf
    t = (np.arange(n) + 0.5) / n
    basis = np.cos(np.pi * np.outer(t, np.arange(n_coefs)))
    coef, *_ = np.linalg.lstsq(basis[ok], track[ok], rcond=None)
    return float(np.mean(np.abs(basis[ok] @ coef - track[ok])))


def praat_sweep(x: np.ndarray, sr: int, times: np.ndarray, meta: dict) -> np.ndarray:
    """Praat Burg at ceilings 4000-8000 Hz (step 250), ceiling chosen by DCT smoothness of F1-F3.

    Needs the vowel span (meta start_s/end_s, oracle segmentation from the corpus);
    falls back to the whole file when it is missing.
    """
    import parselmouth

    snd = parselmouth.Sound(x.astype(np.float64), sr)
    start = meta.get("start_s", 0.0)
    end = meta.get("end_s", len(x) / sr)
    grid = np.arange(start, end, SWEEP_STEP_S)
    best, best_err = None, np.inf
    for ceiling in SWEEP_CEILINGS:
        formant = snd.to_formant_burg(time_step=SWEEP_STEP_S, max_number_of_formants=5.5, maximum_formant=float(ceiling))
        tracks = _sample_track(formant, grid)
        err = sum(_dct_residual(tracks[:, k], SWEEP_N_COEFS) for k in range(SWEEP_SELECT_FORMANTS))
        if err < best_err:
            best, best_err = formant, err
    if best is None:
        return np.full((len(times), N_FORMANTS), np.nan)
    return _sample_track(best, times)


# FastTrack's documented recommendation (functions/file_1_trackAutoselect.praat):
# the ceiling range depends on talker size.
FASTTRACK_RANGES = {"m": (4500.0, 6500.0), "w": (5000.0, 7000.0), "b": (5500.0, 7500.0), "g": (5500.0, 7500.0)}
FASTTRACK_N_COS = 5  # intercept + 5 cosine terms, as in FastTrack's findError.praat


def _fasttrack_error(tracks: np.ndarray) -> float:
    """FastTrack's selection error: sum over F1-F3 of mean |residual| (Hz) from an
    intercept + cos(pi*c*t), c = 1..5 regression, on frames where >= 4 formants were found."""
    rows = tracks[np.all(np.isfinite(tracks[:, :4]), axis=1)]
    n = len(rows)
    if n < FASTTRACK_N_COS + 3:
        return np.inf
    t = np.arange(1, n + 1) / n
    basis = np.column_stack([np.ones(n)] + [np.cos(np.pi * c * t) for c in range(1, FASTTRACK_N_COS + 1)])
    err = 0.0
    for k in range(3):
        coef, *_ = np.linalg.lstsq(basis, rows[:, k], rcond=None)
        err += float(np.mean(np.abs(rows[:, k] - basis @ coef)))
    return err


def _fasttrack_select(x: np.ndarray, sr: int, meta: dict):
    """FastTrack's chosen analysis: (Praat Formant object or None, grid times over the vowel span)."""
    import parselmouth

    lo, hi = FASTTRACK_RANGES.get(meta.get("group"), (4500.0, 7500.0))
    snd = parselmouth.Sound(x.astype(np.float64), sr)
    start = meta.get("start_s", 0.0)
    end = meta.get("end_s", len(x) / sr)
    grid = np.arange(start, end, SWEEP_STEP_S)
    best, best_err = None, np.inf
    for ceiling in np.arange(lo, hi + 1, 250.0):
        formant = snd.to_formant_burg(time_step=SWEEP_STEP_S, max_number_of_formants=5.5, maximum_formant=float(ceiling))
        err = _fasttrack_error(_sample_track(formant, grid))
        if err < best_err:
            best, best_err = formant, err
    return best, grid


def praat_fasttrack(x: np.ndarray, sr: int, times: np.ndarray, meta: dict) -> np.ndarray:
    """FastTrack-faithful selection: per-talker-group ceiling range (250 Hz steps),
    FastTrack's error. Uses the speaker group (m/w/b/g) and the oracle vowel span."""
    best, _ = _fasttrack_select(x, sr, meta)
    if best is None:
        return np.full((len(times), N_FORMANTS), np.nan)
    return _sample_track(best, times)


# Seeds snapped to QCP-FB peaks (Hannah, 2026-10-29; report-only): FastTrack decides which peak is which formant,
# QCP-FB (D-046 arm B settings) re-measures each one at its nearest closed-phase peak within SNAP_WINDOW, the seed is
# kept when no peak is that close. The smoothing constants are fixed in advance, not tuned on any gold set.
SNAP_WINDOW = 0.15  # relative
SMOOTH_MEDIAN = 5  # frames of FastTrack's 2 ms grid
SMOOTH_ALPHA = 0.15  # leaky integrator, run forwards and backwards (zero phase)


def _interp_nan(y: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    return np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])


def _smooth_seed(best, grid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """FastTrack's tracks on its grid, in log2 Hz: a running median (jumps to a neighbouring peak), then a zero-phase
    leaky integrator (jitter), sampled at `times`."""
    from scipy.ndimage import median_filter
    from scipy.signal import filtfilt

    tr = _sample_track(best, grid)
    out = np.full((len(times), N_FORMANTS), np.nan)
    for k in range(N_FORMANTS):
        y = np.log2(_interp_nan(tr[:, k]))
        if not np.isfinite(y).sum() >= 2 * SMOOTH_MEDIAN:
            continue
        y = median_filter(y, SMOOTH_MEDIAN, mode="nearest")
        y = filtfilt([SMOOTH_ALPHA], [1.0, SMOOTH_ALPHA - 1.0], y, padlen=min(len(y) - 1, 3 * int(1 / SMOOTH_ALPHA)))
        out[:, k] = 2.0 ** np.interp(times, grid, y)
    return out


def _dct_seed(best, grid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """FastTrack's own smoother: its intercept + 5-cosine regression of each track over the span, at `times`."""
    tr = _sample_track(best, grid)
    n = len(grid)
    span = lambda t: (np.asarray(t) - grid[0]) / max(grid[-1] - grid[0], 1e-9)  # noqa: E731
    basis = lambda u: np.column_stack([np.ones(len(u))] + [np.cos(np.pi * c * u) for c in range(1, FASTTRACK_N_COS + 1)])  # noqa: E731
    out = np.full((len(times), N_FORMANTS), np.nan)
    u = span(grid)
    for k in range(N_FORMANTS):
        ok = np.isfinite(tr[:, k])
        if ok.sum() < FASTTRACK_N_COS + 3 or n < 2:
            continue
        coef, *_ = np.linalg.lstsq(basis(u[ok]), tr[ok, k], rcond=None)
        out[:, k] = basis(np.clip(span(times), 0, 1)) @ coef
    return out


def _snap(seed: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Each seed formant moved to the nearest candidate within SNAP_WINDOW of it, else kept."""
    out = seed.copy()
    for i in range(len(seed)):
        c = cands[i][np.isfinite(cands[i])]
        if not len(c):
            continue
        for k in range(seed.shape[1]):
            s = seed[i, k]
            if not np.isfinite(s):
                continue
            j = np.argmin(np.abs(c - s))
            if abs(c[j] - s) <= SNAP_WINDOW * s:
                out[i, k] = c[j]
    return out


def _fasttrack_snap(seed_kind: str):
    def predict(x: np.ndarray, sr: int, times: np.ndarray, meta: dict) -> np.ndarray:
        from ..eval import qcp

        best, grid = _fasttrack_select(x, sr, meta)
        if best is None:
            return np.full((len(times), N_FORMANTS), np.nan)
        times = np.asarray(times, float)
        seed = {"raw": lambda: _sample_track(best, times), "smooth": lambda: _smooth_seed(best, grid, times),
                "dct": lambda: _dct_seed(best, grid, times)}[seed_kind]()
        cands = qcp.analyse(x, sr, times, f0_mean=_f0_mean(x, sr), fs=10000.0, dq=0.3, n_formants=8)["F"]
        return _snap(seed, cands)

    return predict


@functools.lru_cache(maxsize=1)
def _tvlp_module():
    if not TVLP_PATH.exists():
        raise FileNotFoundError(f"{TVLP_PATH} missing; run `git submodule update --init`")
    spec = importlib.util.spec_from_file_location("vdl_tvlp", TVLP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TVLP_WIN_S = 0.15


def tvlp(x: np.ndarray, sr: int, times: np.ndarray, meta: dict) -> np.ndarray:
    """TVLP (voice-dataset-labeler @ 0ab5568) on a 150 ms window centred on each frame, its defaults otherwise."""
    mod = _tvlp_module()
    half = int(TVLP_WIN_S * sr / 2)
    out = np.full((len(times), N_FORMANTS), np.nan)
    for i, t in enumerate(times):
        c = int(round(t * sr))
        lo, hi = max(0, c - half), min(len(x), c + half)
        freqs, _, _ = mod.extract_formants(x[lo:hi], sr, n_formants=N_FORMANTS, win_s=(hi - lo) / sr)
        out[i] = freqs
    return out


def _f0_mean(x: np.ndarray, sr: int) -> float:
    """The recording's median f0 over voiced frames (Praat autocorrelation, 60-800 Hz): SEDREAMS' window length."""
    import parselmouth

    f = parselmouth.Sound(x.astype(np.float64), sr).to_pitch_ac(pitch_floor=60.0, pitch_ceiling=800.0).selected_array["frequency"]
    f = f[f > 0]
    return float(np.median(f)) if len(f) else 150.0


def _qcp(fs: float, dq: float):
    def predict(x: np.ndarray, sr: int, times: np.ndarray, meta: dict) -> np.ndarray:
        from ..eval import qcp

        out = np.full((len(times), N_FORMANTS), np.nan)
        r = qcp.analyse(x, sr, np.asarray(times, float), f0_mean=_f0_mean(x, sr), fs=fs, dq=dq)
        out[:, : r["F"].shape[1]] = r["F"][:, :N_FORMANTS]
        return out

    return predict


FORMANT_BASELINES: dict[str, FormantBaseline] = {
    b.name: b
    for b in (
        FormantBaseline("praat_default", "Praat Burg, defaults (5 formants, 5500 Hz ceiling)", praat_default),
        FormantBaseline(
            "praat_sweep",
            "Praat Burg, 4000-8000 Hz ceiling sweep, FastTrack-style DCT selection (oracle vowel span)",
            praat_sweep,
        ),
        FormantBaseline(
            "praat_fasttrack",
            "Praat Burg, FastTrack's per-talker-group ceiling ranges (men 4500-6500, women 5000-7000, "
            "children 5500-7500 Hz) and selection error (speaker group + oracle vowel span)",
            praat_fasttrack,
        ),
        FormantBaseline("tvlp", "TVLP, voice-dataset-labeler @ 0ab5568, 150 ms window", tvlp),
        FormantBaseline("qcp_fb_published", "QCP-FB as published (Gowda et al. 2017): duration quotient 0.7, 8 kHz, "
                        "SEDREAMS GCIs (D-046 arm A; not adopted)", _qcp(8000.0, 0.7)),
        FormantBaseline("qcp_fb", "QCP-FB, duration quotient 0.3, 10 kHz, SEDREAMS GCIs (D-046 arm B; narrowly failed "
                        "its gate, not adopted; report-only)", _qcp(10000.0, 0.3)),
        FormantBaseline("fasttrack_qcp_snap", "FastTrack (as praat_fasttrack), each formant snapped to the nearest QCP-FB "
                        "(arm B) peak within 15 %", _fasttrack_snap("raw")),
        FormantBaseline("fasttrack_smooth_qcp_snap", "FastTrack smoothed (median 5 frames, then a zero-phase leaky integrator, "
                        "alpha 0.15, log Hz), snapped as fasttrack_qcp_snap", _fasttrack_snap("smooth")),
        FormantBaseline("fasttrack_dct_qcp_snap", "FastTrack's own DCT fit (intercept + 5 cosines) as the seed, snapped as "
                        "fasttrack_qcp_snap", _fasttrack_snap("dct")),
    )
}
