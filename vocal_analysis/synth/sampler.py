"""Random clip sampler: trajectories, audio and per-frame labels (plan §5.1).

`generate(index, cfg, seed)` is deterministic: the RNG is seeded from (seed, index), so any
clip can be regenerated on its own. Labels are emitted per 10 ms frame (frame i at
t = i * hop), read off the same control trajectories the audio was rendered from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import tract, tube
from .glottal import RD_MAX, RD_MIN, source_h_db
from .voice import CTRL, FS, Trajectories, synthesize

N_LABEL_FORMANTS = tract.N_POLES  # all oral poles are labelled (NaN where absent); training uses F1-F6 at most
N_SOURCE_HARMONICS = 8


@dataclass
class SynthConfig:
    duration_s: tuple[float, float] = (2.0, 4.0)
    hop_s: float = 0.010
    length_cm: tuple[float, float] = (10.0, 20.0)
    p_length_from_f0: float = 0.5  # otherwise L independent of f0 (decorrelated coverage)
    p_high_f0: float = 0.45  # centre f0 log-uniform in high_f0_hz with this probability
    low_f0_hz: tuple[float, float] = (55.0, 400.0)
    high_f0_hz: tuple[float, float] = (300.0, 1300.0)
    f0_limits_hz: tuple[float, float] = (50.0, 1600.0)
    rd: tuple[float, float] = (RD_MIN, RD_MAX)
    rd_wander_sd: tuple[float, float] = (0.0, 0.25)
    loudness_sd_db: tuple[float, float] = (0.0, 3.0)
    jitter: tuple[float, float] = (0.0, 0.02)
    shimmer_db: tuple[float, float] = (0.0, 1.0)
    hnr_db: tuple[float, float] | None = (5.0, 40.0)  # None: no aspiration
    vibrato_rate_hz: tuple[float, float] = (4.0, 8.0)
    vibrato_extent_cents: tuple[float, float] = (40.0, 100.0)  # bulk; see p_vibrato_wide
    p_vibrato_wide: float = 0.3  # extent from (0, 150) instead
    voiced_seg_s: tuple[float, float] = (0.3, 2.0)
    gap_s: tuple[float, float] = (0.05, 0.5)
    gap_kinds: dict = field(default_factory=lambda: {"silence": 0.5, "whisper": 0.25, "frication": 0.25})
    target_every_s: tuple[float, float] = (0.08, 0.4)
    transition_s: tuple[float, float] = (0.03, 0.15)
    p_tube: float = 0.6  # per clip: Story area function + lossy tube (plan §5.1 preferred), else the resonator-cascade prior
    p_tube_constrict: float = 0.15  # per tube target: approximant-like tier II constriction
    p_uniform_formants: float = 0.3
    p_cluster: float = 0.1
    p_nasal: float = 0.15
    p_subglottal: float = 0.2
    contour_probs_low: dict = field(default_factory=lambda: {"speech": 0.5, "singing": 0.3, "glide": 0.2})
    contour_probs_high: dict = field(default_factory=lambda: {"singing": 0.7, "glide": 0.2, "speech": 0.1})
    p_register_step: float = 0.15
    p_vibrato: float = 0.8  # in singing contours
    # Hard-case hooks: None, "f1_near_h1", "f1_near_h2" (F1 within 10 % of f0 / 2 f0) or "f2_f3_close" (|F3 - F2| < 300 Hz)
    formant_constraint: str | None = None


@dataclass
class SynthClip:
    audio: np.ndarray  # float32, FS
    labels: dict[str, np.ndarray]  # per frame
    meta: dict


# ----------------------------------------------------------------------------- helpers


def _choice(rng: np.random.Generator, probs: dict) -> str:
    keys = list(probs)
    p = np.array([probs[k] for k in keys], float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _smooth_noise(rng: np.random.Generator, n: int, corr_pts: float, std: float) -> np.ndarray:
    """Gaussian-smoothed white noise with the given SD (correlation length in control points)."""
    if std == 0 or n == 0:
        return np.zeros(n)
    pad = int(4 * corr_pts) + 1
    w = rng.standard_normal(n + 2 * pad)
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / max(corr_pts, 1e-3)) ** 2)
    s = np.convolve(w, k / k.sum(), mode="same")[pad : pad + n]
    return s / (s.std() + 1e-12) * std


def _raised_cosine_steps(n: int, times: np.ndarray, values: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Piecewise-constant values with raised-cosine transitions centred on `times` (control points)."""
    out = np.full(n, values[0], float)
    idx = np.arange(n)
    for t, v_prev, v, w in zip(times, values[:-1], values[1:], widths):
        x = np.clip((idx - (t - w / 2)) / max(w, 1e-9), 0.0, 1.0)
        out += (v - v_prev) * (0.5 - 0.5 * np.cos(np.pi * x))
    return out


# ----------------------------------------------------------------------------- contours


def _f0_contour(rng, cfg: SynthConfig, n: int, centre: float, kind: str) -> tuple[np.ndarray, dict]:
    """log2 f0 per control point, plus a description."""
    t = np.arange(n) / 1000.0
    base = np.log2(centre)
    info: dict = {"kind": kind, "centre_hz": centre}
    if kind == "flat":
        lf = np.full(n, base)
    elif kind == "speech":
        decl = rng.uniform(-2.0, 0.0) / 12.0  # octaves per second
        wander = _smooth_noise(rng, n, rng.uniform(80, 250), rng.uniform(1.0, 4.0) / 12.0)
        lf = base + decl * (t - t.mean()) + wander
        info.update(declination_st_s=decl * 12)
    elif kind == "glide":
        k = rng.integers(1, 4)
        times = np.sort(rng.uniform(0, n, k))
        vals = base + np.r_[0.0, np.cumsum(rng.uniform(-1.0, 1.0, k))] * rng.uniform(0.2, 1.0)
        lf = _raised_cosine_steps(n, times, vals - vals.mean() + base, rng.uniform(200, 1500, k))
    else:  # singing: notes, portamento, vibrato
        times, vals = [], [base]
        pos = rng.uniform(150, 800)
        while pos < n:
            times.append(pos)
            vals.append(vals[-1] + rng.integers(-7, 8) / 12.0)
            pos += rng.uniform(150, 800)
        vals = np.array(vals)
        vals += base - vals.mean()
        lf = _raised_cosine_steps(n, np.array(times), vals, rng.uniform(30, 150, len(times)))
        if rng.random() < cfg.p_vibrato:
            rate = rng.uniform(*cfg.vibrato_rate_hz) * (1 + _smooth_noise(rng, n, 300, 0.05))
            ext = (rng.uniform(0, 150) if rng.random() < cfg.p_vibrato_wide else rng.uniform(*cfg.vibrato_extent_cents))
            ext_t = ext * (1 + _smooth_noise(rng, n, 400, 0.15))
            onsets = np.r_[0.0, times]
            since = t * 1000 - onsets[np.searchsorted(onsets, t * 1000, side="right") - 1]
            delay = rng.uniform(100, 400)
            ramp = np.clip((since - delay) / 150.0, 0.0, 1.0)
            phase = 2 * np.pi * np.cumsum(rate) / 1000.0 + rng.uniform(0, 2 * np.pi)
            lf = lf + ramp * ext_t / 1200.0 * np.sin(phase)
            info.update(vibrato_rate_hz=float(np.mean(rate)), vibrato_extent_cents=float(ext))
    lo, hi = np.log2(cfg.f0_limits_hz[0]), np.log2(cfg.f0_limits_hz[1])
    return np.clip(lf, lo, hi), info


def _constrain(rng, f: np.ndarray, f0: float, how: str) -> np.ndarray:
    """Apply a hard-case formant constraint to one F1..F8 target (then restore ordering)."""
    f = f.copy()
    if how in ("f1_near_h1", "f1_near_h2"):
        f[0] = (1 if how == "f1_near_h1" else 2) * f0 * rng.uniform(0.9, 1.1)
    elif how == "f2_f3_close":
        f[2] = f[1] + rng.uniform(tract.MIN_SPACING_HZ, 300.0)
    else:
        raise ValueError(f"unknown formant constraint {how!r}")
    head = f[:3] if how == "f2_f3_close" else f[:1]
    rest = f[len(head):]
    for i in range(len(rest)):  # keep ascending above the constrained formants
        lower = head[-1] if i == 0 else rest[i - 1]
        if np.isfinite(rest[i]) and rest[i] < lower + tract.MIN_SPACING_HZ:
            rest[i] = lower + tract.MIN_SPACING_HZ
    out = np.r_[head, rest]
    out[out > tract.F_MAX_HZ] = np.nan
    return out


# ----------------------------------------------------------------------------- clip


def generate(index: int, cfg: SynthConfig | None = None, seed: int = 0) -> SynthClip:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng([seed, index])
    dur = rng.uniform(*cfg.duration_s)
    n = int(round(dur * 1000)) + 1  # control points (1 ms)

    # Voice
    high = rng.random() < cfg.p_high_f0
    centre = _loguniform(rng, *(cfg.high_f0_hz if high else cfg.low_f0_hz))
    kind = _choice(rng, cfg.contour_probs_high if centre >= 300 else cfg.contour_probs_low)
    log2f0, contour = _f0_contour(rng, cfg, n, centre, kind)
    rd = np.full(n, rng.uniform(*cfg.rd)) + _smooth_noise(rng, n, 300, rng.uniform(*cfg.rd_wander_sd))
    register_steps = []
    if rng.random() < cfg.p_register_step:
        tstep = int(rng.uniform(0.2, 0.8) * n)
        jump = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)  # octaves
        width = rng.uniform(5, 40)
        log2f0 = log2f0 + _raised_cosine_steps(n, np.array([tstep]), np.array([0.0, jump]), np.array([width]))
        rd = rd + _raised_cosine_steps(n, np.array([tstep]), np.array([0.0, rng.uniform(-0.6, 0.6)]), np.array([width]))
        register_steps.append({"t_s": tstep / 1000, "octaves": float(jump)})
        log2f0 = np.clip(log2f0, np.log2(cfg.f0_limits_hz[0]), np.log2(cfg.f0_limits_hz[1]))
    rd = np.clip(rd, RD_MIN, RD_MAX)
    f0 = 2.0 ** log2f0

    if rng.random() < cfg.p_length_from_f0:
        length = float(np.clip(17.5 - 4.0 * np.log2(centre / 110.0) + rng.normal(0, 1.0), *cfg.length_cm))
    else:
        length = rng.uniform(*cfg.length_cm)

    # Phrasing: voiced segments separated by gaps of different kinds
    voice_env = np.zeros(n)
    whisper = np.zeros(n)
    fric = np.zeros(n)
    segments = []
    pos = 0.0 if rng.random() < 0.5 else rng.uniform(*cfg.gap_s) * 1000
    while pos < n:
        seg = rng.uniform(*cfg.voiced_seg_s) * 1000
        a, b = int(pos), int(min(pos + seg, n))
        att, dec = rng.uniform(10, 60), rng.uniform(10, 60)
        idx = np.arange(a, b)
        env = np.minimum(np.clip((idx - a) / att, 0, 1), np.clip((b - idx) / dec, 0, 1))
        voice_env[a:b] = np.maximum(voice_env[a:b], 0.5 - 0.5 * np.cos(np.pi * env))
        segments.append({"kind": "voiced", "start_s": a / 1000, "end_s": b / 1000})
        pos = b + rng.uniform(*cfg.gap_s) * 1000
        g0, g1 = b, int(min(pos, n))
        if g1 > g0:
            gk = _choice(rng, cfg.gap_kinds)
            segments.append({"kind": gk, "start_s": g0 / 1000, "end_s": g1 / 1000})
            gi = np.arange(g0, g1)
            genv = np.sin(np.pi * (gi - g0) / max(g1 - g0, 1)) ** 2
            if gk == "whisper":
                whisper[g0:g1] = genv
            elif gk == "frication":
                fric[g0:g1] = genv
    loud = 10 ** (_smooth_noise(rng, n, 300, rng.uniform(*cfg.loudness_sd_db)) / 20)

    # Articulation: targets with cosine transitions in log frequency
    tt, pos = [0.0], 0.0
    while pos < n:
        pos += rng.uniform(*cfg.target_every_s) * 1000
        tt.append(pos)
    use_tube = cfg.p_tube > 0 and rng.random() < cfg.p_tube  # no draw at p_tube = 0, so cascade-only seeds reproduce
    tube_info = None
    if use_tube:
        spk = tube.sample_speaker(rng, length)
        drawn = [tube.sample_target(rng, spk, tract.N_POLES, tract.F_MAX_HZ, cfg.p_tube_constrict) for _ in tt]
        targets = np.array([d[0] for d in drawn])
        tube_bws = np.array([d[1] for d in drawn])
        tube_info = {"speaker": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(spk).items()},
                     "targets": [d[2] for d in drawn]}
    else:
        targets = np.array([tract.sample_formants(rng, length, cfg.p_uniform_formants, cfg.p_cluster) for _ in tt])
    if cfg.formant_constraint:
        f0_at = f0[np.clip(np.array(tt).astype(int), 0, n - 1)]
        targets = np.array([_constrain(rng, t, f, cfg.formant_constraint) for t, f in zip(targets, f0_at)])
    targets = np.log(targets)
    widths = rng.uniform(*cfg.transition_s, len(tt) - 1) * 1000
    logf = np.stack([_raised_cosine_steps(n, np.array(tt[1:]), targets[:, k], widths) for k in range(tract.N_POLES)], 1)
    formants = np.exp(logf)  # NaN poles stay NaN (log(NaN))
    if use_tube:  # the tube's own bandwidths (constraints move only frequencies)
        bw_targets = tube_bws
    else:
        bw_targets = np.stack([tract.sample_bandwidths(rng, np.exp(targets[i]), centre) for i in range(len(tt))])
    logb = np.stack([_raised_cosine_steps(n, np.array(tt[1:]), np.log(np.nan_to_num(bw_targets[:, k], nan=100.0)), widths)
                     for k in range(tract.N_POLES)], 1)
    bws = np.where(np.isfinite(formants), np.exp(logb), np.nan)
    extras = tract.sample_extras(rng, length, cfg.p_nasal, cfg.p_subglottal)
    n_ep, n_ez = len(extras["extra_pole_f"]), len(extras["extra_zero_f"])

    traj = Trajectories(
        f0=f0, rd=rd, voice_amp=voice_env * loud,
        whisper_amp=whisper * loud * rng.uniform(0.3, 1.5), fric_amp=fric * loud * rng.uniform(0.3, 1.5),
        poles_f=np.concatenate([formants, np.tile(extras["extra_pole_f"], (n, 1))], 1),
        poles_b=np.concatenate([bws, np.tile(extras["extra_pole_b"], (n, 1))], 1),
        zeros_f=np.tile(extras["extra_zero_f"], (n, 1)).reshape(n, n_ez),
        zeros_b=np.tile(extras["extra_zero_b"], (n, 1)).reshape(n, n_ez),
    )
    jitter, shimmer = rng.uniform(*cfg.jitter), rng.uniform(*cfg.shimmer_db)
    hnr = None if cfg.hnr_db is None else rng.uniform(*cfg.hnr_db)
    comps = synthesize(traj, rng, jitter=jitter, shimmer_db=shimmer, hnr_db=hnr)
    audio = comps["total"]
    peak = np.max(np.abs(audio)) + 1e-12
    gain = 0.5 / peak
    audio = audio * gain

    # Frame labels (frame i at t = i * hop)
    hop_ctrl = int(round(cfg.hop_s * 1000))
    fi = np.arange(0, n - 1, hop_ctrl)
    voiced = voice_env[fi] >= 0.5
    labels = {
        "t": fi / 1000.0,
        "f0": f0[fi],
        "voiced": voiced,
        "sonorant": voiced,
        "whisper": whisper[fi] >= 0.5,
        "formants": formants[fi, :N_LABEL_FORMANTS],
        "bandwidths": bws[fi, :N_LABEL_FORMANTS],
        "rd": rd[fi],
        "source_h_db": source_h_db(rd[fi], N_SOURCE_HARMONICS),
        "length_cm": np.full(len(fi), length),
    }
    meta = {
        "index": index, "seed": seed, "fs": FS, "duration_s": len(audio) / FS, "length_cm": length,
        "contour": contour, "register_steps": register_steps, "segments": segments,
        "jitter": jitter, "shimmer_db": shimmer, "hnr_db": hnr, "gain": gain,
        "tract": "tube" if use_tube else "cascade", "tube": tube_info,
        "extras": {k: v.tolist() for k, v in extras.items()}, "n_extra_poles": n_ep, "n_extra_zeros": n_ez,
        "config": asdict(cfg),
    }
    return SynthClip(audio.astype(np.float32), labels, meta)
