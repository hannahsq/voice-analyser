"""Re-filtering, QC and label export (plan §5.2 steps 5-6).

`source_from_audio` runs once per recording (inverse filtering, source harmonics, residual ripple);
`render_variant` re-filters that source through one set of perturbed poles and cuts the result into
clips with the synthetic sets' label keys, so `synth.shards` and `train.data` read them unchanged.

Labels: formants and bandwidths are exact (they are the re-filter's poles). f0, voicing and the
sonorant mask are the tracker's estimates of the original voice, which re-filtering leaves alone;
`meta["labels"]` says so. source_h_db is measured on the recovered source (dB on the recording's own
scale, so only differences such as H1-H2 mean anything); rd and length_cm are NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..backend.harmonics import harmonic_amplitudes
from ..synth import filters
from . import inverse
from .tracks import N_LABEL_FORMANTS, PerturbConfig, perturb, smooth_tracks

FS = inverse.FS
HOP = inverse.HOP
N_SOURCE_HARMONICS = 8


@dataclass
class ResynthConfig:
    # "iaif", "tracks" (anti-resonators at the smoothed tracker formants) or "auto": IAIF on frames with
    # f0 up to iaif_max_f0_hz, tracks above, where LPC latches onto harmonics
    method: str = "auto"
    iaif_max_f0_hz: float = 250.0
    order_vt: int | None = None  # IAIF tract order; default fs / 1000 + 2
    smooth_window: int = 7  # Savitzky-Golay frames
    voiced_threshold: float = 0.5
    sonorant_threshold: float = 0.5
    perturb: PerturbConfig = field(default_factory=PerturbConfig)
    keep_original_outside: bool = True  # non-sonorant stretches keep the original audio (cross-faded)
    ramp_s: float = 0.02
    qc_f_max_hz: float = 4000.0  # the F1-F3 region; above it, upper-pole errors dominate
    # QC: reject a clip whose median voiced-frame |tract_left| exceeds this. On synthetic vowels: untouched speech
    # 0.97-1.23, exact inverse filtering ~0, IAIF below 250 Hz -0.19..0.06, IAIF above 300 Hz -0.3..-1.2
    max_tract_left: float = 0.25
    accept_unmeasured: bool = True  # QC: keep clips where it can't be measured (f0 above 800 Hz)
    min_voiced_frac: float = 0.2  # QC: reject a clip with fewer voiced frames
    clip_s: tuple[float, float] = (3.0, 6.0)  # clips are cut at quiet frames within this range
    variants: int = 2  # perturbations per recording


_RD_FIT = np.round(np.arange(0.3, 2.7001, 0.05), 2)


def source_qc(source: np.ndarray, f0: np.ndarray, voiced: np.ndarray, formants: np.ndarray, f_max_hz: float = 4000.0,
              min_harmonics: int = 5, fs: int = FS, hop: int = HOP) -> dict[str, np.ndarray]:
    """Residual-ripple QC per frame (plan §5.2): is the original tract still in the recovered source?

    The source's harmonic amplitudes below f_max (dB) are fitted with an LF spectrum (best Rd) plus a
    gain, a spectral tilt (channel) and beta x the tracked tract's response 20 log10 |H(k f0)|: the
    tracked formants plus uniform-tube poles above them, Hawks & Miller bandwidths (the tract
    `inverse.track_inverse` removes). Returns:
    - tract_left: beta, the fraction of the tracked tract still in the source: 0 when inverse
      filtering removed it, 1 when it did nothing, negative when it removed too much. Aspiration noise
      and jitter scatter the harmonics but don't correlate with the tract, so they leave beta alone.
    - ripple_db: RMS of what that fit leaves (noise, and tract errors the tracks don't describe).
    - h_db: the source harmonics, (T, 40).
    NaN on unvoiced frames and where fewer than min_harmonics lie below f_max (f0 above 800 Hz with
    the defaults).
    """
    from ..backend.corrections import upper_poles
    from ..synth.glottal import source_h_db
    from ..synth.tract import bandwidth_prior

    formants = np.asarray(formants, float)
    tract = np.concatenate([formants, upper_poles(formants, fs)], 1)
    times = np.arange(len(f0)) * hop / fs
    f0v = np.where(voiced, f0, np.nan)
    h = harmonic_amplitudes(source, fs, times, f0v, k_max=40)
    lf = source_h_db(_RD_FIT, h.shape[1])  # (R, K)
    beta, rms = np.full(len(f0), np.nan), np.full(len(f0), np.nan)
    k = np.arange(1, h.shape[1] + 1)
    for i in np.flatnonzero(voiced & np.isfinite(f0)):
        ok = np.isfinite(h[i]) & (f0[i] * k < f_max_hz)
        fn = tract[i][np.isfinite(tract[i])]
        if ok.sum() < min_harmonics or not np.all(np.isfinite(formants[i])):
            continue
        fk = f0[i] * k[ok]
        g = 20 * np.log10(np.abs(filters.response(fk, fn[None, :], bandwidth_prior(fn, f0[i])[None, :], fs)))
        basis = np.stack([np.ones(ok.sum()), np.log2(k[ok]), g], 1)
        d = h[i, ok][None, :] - lf[:, ok]  # (R, K'): what gain, tilt and beta must explain
        coef, *_ = np.linalg.lstsq(basis, d.T, rcond=None)
        err = np.mean((d - (basis @ coef).T) ** 2, axis=1)
        best = int(np.argmin(err))
        beta[i], rms[i] = coef[2, best], np.sqrt(err[best])
    return {"tract_left": beta, "ripple_db": rms, "h_db": h}


def source_from_audio(audio: np.ndarray, analysis: dict, cfg: ResynthConfig) -> dict:
    """Everything that does not depend on the perturbation. analysis: the tracker's per-frame
    f0, voiced_p, sonorant_p, formants (T, >=3), bandwidths (frame i at i * 10 ms)."""
    n_frames = len(audio) // HOP + 1
    a = {k: np.asarray(v)[:n_frames] for k, v in analysis.items()}
    if len(a["f0"]) < n_frames:
        raise ValueError(f"analysis has {len(a['f0'])} frames, audio needs {n_frames}")
    voiced = a["voiced_p"] > cfg.voiced_threshold
    sonorant = (a["sonorant_p"] > cfg.sonorant_threshold) | voiced
    tracks = smooth_tracks(a["formants"], sonorant, cfg.smooth_window)
    if cfg.method not in ("iaif", "tracks", "auto"):
        raise ValueError(cfg.method)
    if cfg.method in ("iaif", "auto"):
        s_iaif, _ = inverse.iaif(audio, FS, cfg.order_vt)
    if cfg.method in ("tracks", "auto"):
        s_tracks = inverse.track_inverse(audio, tracks, a["bandwidths"])
    if cfg.method == "auto":  # IAIF where f0 is low enough for LPC, tracks above (cross-faded per sample)
        low = ~(voiced & (a["f0"] > cfg.iaif_max_f0_hz))
        w = _ramp_mask(low, len(audio), int(cfg.ramp_s * FS))
        source = w * s_iaif + (1 - w) * s_tracks
    else:
        source = s_iaif if cfg.method == "iaif" else s_tracks
    qc = source_qc(source, a["f0"], voiced, tracks, cfg.qc_f_max_hz)
    return {"source": source, "f0": a["f0"].astype(float), "voiced": voiced, "sonorant": sonorant, "tracks": tracks,
            "tract_left": qc["tract_left"], "ripple_db": qc["ripple_db"], "source_h_db": qc["h_db"][:, :N_SOURCE_HARMONICS]}


def refilter(source: np.ndarray, formants: np.ndarray, bandwidths: np.ndarray, extras: dict | None = None,
             fs: int = FS, hop: int = HOP) -> np.ndarray:
    """Time-varying resonator cascade at per-frame poles (T, P), interpolated per sample in (F, B)
    (plan §5.2 step 5), plus constant extra pole-zero pairs."""
    n = len(source)
    f = inverse.frames_to_samples(formants, n, hop)
    b = inverse.frames_to_samples(bandwidths, n, hop)
    zero = np.zeros(len(f), bool)
    if extras:
        ef = np.r_[extras["extra_pole_f"], extras["extra_zero_f"]]
        eb = np.r_[extras["extra_pole_b"], extras["extra_zero_b"]]
        f = np.concatenate([f, np.repeat(ef[:, None], n, 1)])
        b = np.concatenate([b, np.repeat(eb[:, None], n, 1)])
        zero = np.r_[zero, np.zeros(len(extras["extra_pole_f"]), bool), np.ones(len(extras["extra_zero_f"]), bool)]
    return filters.filter_tv(np.asarray(source, float), f, b, fs, zero)


def _ramp_mask(mask: np.ndarray, n: int, ramp: int, hop: int = HOP) -> np.ndarray:
    """Per-sample weight: 1 inside mask frames, cosine ramps of `ramp` samples into the gaps."""
    w = np.repeat(mask.astype(float), hop)[:n]
    w = np.pad(w, (0, n - len(w)))
    if ramp > 1:
        k = np.hanning(2 * ramp + 1)
        w = np.convolve(w, k / k.sum(), "same")
    return np.clip(w, 0.0, 1.0)


def cut_points(audio: np.ndarray, n_frames: int, clip_s: tuple[float, float], hop: int = HOP) -> list[int]:
    """Frame indices [0, ..., n_frames] splitting a recording into clips of clip_s[0]..clip_s[1]
    seconds, each cut at the quietest frame in range. A recording shorter than clip_s[1] is one clip."""
    lo, hi = (int(round(s * FS / hop)) for s in clip_s)
    frame_e = np.array([np.sum(audio[i * hop : (i + 1) * hop] ** 2) for i in range(n_frames)])
    cuts = [0]
    while n_frames - cuts[-1] > hi:
        a, b = cuts[-1] + lo, min(cuts[-1] + hi, n_frames - lo)
        cuts.append(a + int(np.argmin(frame_e[a : b + 1])) if b >= a else cuts[-1] + hi)
    return cuts + [n_frames]


def render_variant(audio: np.ndarray, src: dict, rng: np.random.Generator, cfg: ResynthConfig) -> tuple[list, dict]:
    """One perturbation of one recording -> (clips [(audio float32, labels, meta)], info)."""
    p = perturb(src["tracks"], src["f0"], src["voiced"], rng, cfg.perturb)
    y = refilter(src["source"], p["formants"], p["bandwidths"], p["extras"])
    n_frames = len(src["f0"])
    son = src["sonorant"]
    ws = _ramp_mask(son, len(y), int(cfg.ramp_s * FS))
    on = ws > 0.99
    if on.any():
        y *= np.sqrt(np.sum(audio[on] ** 2) / max(np.sum(y[on] ** 2), 1e-20))
    if cfg.keep_original_outside:
        y = ws * y + (1 - ws) * audio
        full = np.array([on[i * HOP : (i + 1) * HOP].all() if i * HOP < len(on) else False for i in range(n_frames)])
        son = son & full  # cross-faded frames are not pure re-filtered signal: no formant labels there
    voiced = src["voiced"] & son if cfg.keep_original_outside else src["voiced"]
    labels = {
        "t": np.arange(n_frames) * HOP / FS,
        "f0": np.where(src["voiced"], src["f0"], np.nan).astype(np.float32),
        "voiced": src["voiced"].copy(),
        "sonorant": son,
        "whisper": np.zeros(n_frames, bool),
        "formants": p["formants"].astype(np.float32),
        "bandwidths": p["bandwidths"].astype(np.float32),
        "rd": np.full(n_frames, np.nan, np.float32),
        "source_h_db": np.where(voiced[:, None], src["source_h_db"], np.nan).astype(np.float32),
        "length_cm": np.full(n_frames, np.nan, np.float32),
        "tract_left": src["tract_left"].astype(np.float32),
        "ripple_db": src["ripple_db"].astype(np.float32),
    }
    clips, rejected = [], []
    cuts = cut_points(audio, n_frames, cfg.clip_s)
    med = lambda v: float(np.median(v)) if len(v) else None  # noqa: E731
    for a, b in zip(cuts[:-1], cuts[1:]):
        lab = {k: v[a:b] for k, v in labels.items()}
        lab["t"] = lab["t"] - lab["t"][0]
        seg = y[a * HOP : b * HOP]
        m = lab["voiced"] & np.isfinite(lab["tract_left"])
        qc = {"voiced_frac": float(lab["voiced"].mean()), "measured_frac": float(m.mean()),
              "tract_left_median": med(lab["tract_left"][m]), "ripple_median_db": med(lab["ripple_db"][m]),
              "start_s": a * HOP / FS}
        reason = ("few_voiced" if qc["voiced_frac"] < cfg.min_voiced_frac
                  else "unmeasured" if qc["tract_left_median"] is None and not cfg.accept_unmeasured
                  else "tract_left" if abs(qc["tract_left_median"] or 0.0) > cfg.max_tract_left
                  else "nonfinite" if not np.all(np.isfinite(seg)) else None)
        if reason:
            rejected.append(qc | {"reason": reason})
            continue
        peak = np.max(np.abs(seg))
        meta = {"fs": FS, "duration_s": len(seg) / FS, "qc": qc, "perturb": p["info"],
                "extras": {k: v.tolist() for k, v in p["extras"].items()},
                "labels": {"formants": "exact", "bandwidths": "exact", "f0": "tracker", "voiced": "tracker",
                           "sonorant": "tracker", "source_h_db": "measured"}}
        clips.append(((seg / peak * 0.9 if peak > 0 else seg).astype(np.float32), lab, meta))
    return clips, {"perturb": p["info"], "n_clips": len(clips), "rejected": rejected}
