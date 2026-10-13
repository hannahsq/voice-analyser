"""Random clip sampler: trajectories, audio and per-frame labels (plan §5.1).

`generate(index, cfg, seed)` is deterministic: the RNG is seeded from (seed, index), so any
clip can be regenerated on its own. Labels are emitted per 10 ms frame (frame i at
t = i * hop), read off the same control trajectories the audio was rendered from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import coupling, tract, tube
from . import interactive as interactive_source
from .glottal import RD_MAX, RD_MIN, source_h_db
from .voice import CTRL, FS, Trajectories, synthesize

N_LABEL_FORMANTS = tract.N_POLES  # all oral poles are labelled (NaN where absent); training uses F1-F6 at most
N_SOURCE_HARMONICS = 8


@dataclass
class SynthConfig:
    duration_s: tuple[float, float] = (2.0, 4.0)
    hop_s: float = 0.010
    length_cm: tuple[float, float] = (10.0, 20.0)
    p_length_from_f0: float = 0.5  # otherwise L uniform over length_cm whatever the f0 (broad, partly unphysiological coverage)
    # f0-coupled L = 17.5 - slope * log2(min(centre, cap) / 110) + N(0, sd), redrawn (not clipped) into length_cm
    length_f0_slope_cm: float = 2.5
    length_f0_sd_cm: float = 1.5
    length_f0_cap_hz: float | None = 350.0  # above this f0 the coupled length stops shrinking (adult singers)
    length_redraw: bool = True
    # When L is not drawn from f0: "uniform" over length_cm, or "population" (D-022: LENGTH_POPULATION, a mixture of
    # adult women, adult men and children from imaging data plus a uniform coverage share; length_cm then unused)
    length_dist: str = "uniform"
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
    p_whisper_clip: float = 0.06  # whole clip whispered: every voiced segment becomes tract-excited noise
    p_h_onset: float = 0.25  # per voiced segment: /h/-like aspiration (40-120 ms) leading into it
    h_amp: tuple[float, float] = (1.5, 5.0)  # /h/ level, ~5-15 dB below the vowel
    sonorant_includes_whisper: bool = True  # formants are defined on whispered frames too (plan §4; D-012)
    target_every_s: tuple[float, float] = (0.08, 0.4)
    transition_s: tuple[float, float] = (0.03, 0.15)
    p_tube: float = 0.6  # per clip: Story area function + lossy tube (plan §5.1 preferred), else the resonator-cascade prior
    p_tube_constrict: float = 0.15  # per tube target: approximant-like tier II constriction
    # Vowel targets: "box" (tube q1, q2 uniform over Story's box; cascade PB vowels equally likely) or "conversational"
    # (D-022: by conversational vowel frequency around tube targets fitted to PB's men, women and children and
    # interpolated by length, 30 % uniform coverage around their weighted centre; cascade: PB's per-group tables
    # interpolated by length, with schwa, weighted the same way)
    vowel_dist: str = "box"
    p_uniform_formants: float = 0.3
    # Cascade bandwidths: Hawks & Miller's f0 scaling uses the clip's centre f0 (None), or this fixed f0, which keeps
    # tract bandwidths independent of the source (D-020; 132 Hz is where the scaling is 1)
    bandwidth_f0_hz: float | None = None
    # cascade clips with centre f0 in f1_band_f0_hz: per target, F1 = f0 x log-uniform(f1_band_ratio) with this probability
    p_f1_band: float = 0.3
    f1_band_f0_hz: tuple[float, float] = (150.0, 350.0)
    f1_band_ratio: tuple[float, float] = (1.5, 4.5)
    p_cluster: float = 0.1
    p_nasal: float = 0.15
    p_subglottal: float = 0.2
    contour_probs_low: dict = field(default_factory=lambda: {"speech": 0.5, "singing": 0.3, "glide": 0.2})
    contour_probs_high: dict = field(default_factory=lambda: {"singing": 0.7, "glide": 0.2, "speech": 0.1})
    p_register_step: float = 0.15
    p_vibrato: float = 0.8  # in singing contours
    # Hard-case hooks: None, "f1_near_h1", "f1_near_h2" (F1 within 10 % of f0 / 2 f0) or "f2_f3_close" (|F3 - F2| < 300 Hz)
    formant_constraint: str | None = None
    # Glottal coupling within the cycle (D-023), per tube clip: F1-F5 and their bandwidths move towards their
    # open-glottis values as the glottis opens (time-domain voiced path); labels stay the closed-glottis poles.
    # Glottal sizes and the closed-phase gap from Holmberg et al. (1988) and Patel et al. (2026) (`coupling.sample_glottis`).
    p_coupling: float = 0.0
    glottal_peak_area_cm2: tuple[float, float] = (0.05, 0.22)  # at 17.5 cm; scaled by (length / 17.5)^2
    glottal_gap_offset: tuple[float, float] = (-0.3, 0.1)  # closed-phase gap: follows the open quotient, shifted per clip
    # per clip, every shift is multiplied by this: the model's open-glottis shifts are uncertain (D-023: its F2-F3
    # shifts exceed what LPC studies find), and the labels don't depend on them, so the teacher sees the whole range
    coupling_scale: tuple[float, float] = (0.0, 1.0)
    # the lowest this many coupled poles take the model's full shift (scale 1) instead: F1 and B1's are checked against
    # Barney et al. (2007) and Hanson's B1 increases, unlike F2-F3's (D-024, train_v2c2); 0 = all scaled (train_v2c)
    coupling_full_poles: int = 0
    # Interactive source (D-026), per tube clip: glottal flow from the glottal area and the tract's load
    # (`interactive`), so the tract shapes the source. Replaces the quasi-static coupling on those clips. Labels are
    # the tube's closed-glottis poles, with the area function interpolated through transitions (keyframes at quarter
    # steps) so labels stay exact there; label bandwidths are the tube's own (the glottal losses the speaker's
    # bw_scale stands in for now come from the interaction). Not with formant_constraint (it would move labels
    # off the tube's poles).
    p_interactive: float = 0.0
    interactive_sq: tuple[float, float] = (0.9, 2.0)  # Rosenberg area pulse speed quotient, uniform per clip
    interactive_coupled: bool = True  # False: the same clips with the load's feedback off (the control arm)


# How train_v1 and suite_v1 were generated, for fields added since (D-011, D-012). Pinning these
# reproduces those sets exactly; they are also the values to assume when an older index lacks a field.
V1_SETTINGS = {
    "p_tube": 0.0, "p_tube_constrict": 0.15,
    "length_f0_slope_cm": 4.0, "length_f0_sd_cm": 1.0, "length_f0_cap_hz": None, "length_redraw": False,
    "p_f1_band": 0.0, "f1_band_f0_hz": (150.0, 350.0), "f1_band_ratio": (1.5, 4.5),
    "p_whisper_clip": 0.0, "p_h_onset": 0.0, "h_amp": (1.5, 5.0), "sonorant_includes_whisper": False,
}


# D-022: vocal tract length population (cm). Adults: Story, Vorperian et al. (2018, JASA 143(5)), CT, ages 18-25:
# men 17.6 (SD 0.89), women 15.6 (SD 1.14). Children: Vorperian et al.'s growth curve (same paper, Table I), ages 4-12,
# 11.4-14.5 cm. Groups weighted 45 : 45 : 10 (balanced by sex, so ambiguous frames are not pulled towards either);
# a 20 % coverage share is uniform over 12 cm centred on the groups' weighted mean (16.3 cm).
LENGTH_POPULATION = {
    "groups": (("women", 0.45, "normal", 15.6, 1.14), ("men", 0.45, "normal", 17.6, 0.89), ("children", 0.10, "uniform", 11.4, 14.5)),
    "p_cover": 0.2, "cover_cm": (10.3, 22.3),
}


def _population_length(rng: np.random.Generator) -> tuple[float, str]:
    pop = LENGTH_POPULATION
    lo, hi = pop["cover_cm"]
    if rng.random() < pop["p_cover"]:
        return float(rng.uniform(lo, hi)), "cover"
    w = np.array([g[1] for g in pop["groups"]])
    group, _, kind, a, b = pop["groups"][rng.choice(len(w), p=w / w.sum())]
    if kind == "uniform":
        return float(rng.uniform(a, b)), group
    for _ in range(100):  # redrawn into the coverage range (the tails are far inside it)
        length = rng.normal(a, b)
        if lo <= length <= hi:
            return float(length), group
    return float(np.clip(length, lo, hi)), group


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


def _coupled_length(rng, cfg: SynthConfig, centre: float) -> float:
    c = centre if cfg.length_f0_cap_hz is None else min(centre, cfg.length_f0_cap_hz)
    mean = 17.5 - cfg.length_f0_slope_cm * np.log2(c / 110.0)
    if not cfg.length_redraw:  # v1: clip, which piles lengths up at the ends of the range
        return float(np.clip(mean + rng.normal(0, cfg.length_f0_sd_cm), *cfg.length_cm))
    for _ in range(100):
        length = mean + rng.normal(0, cfg.length_f0_sd_cm)
        if cfg.length_cm[0] <= length <= cfg.length_cm[1]:
            return float(length)
    return float(np.clip(length, *cfg.length_cm))


def _constrain(rng, f: np.ndarray, f0: float, how: str, ratio: tuple[float, float] = (1.5, 4.5)) -> np.ndarray:
    """Apply a hard-case formant constraint to one F1..F8 target (then restore ordering)."""
    f = f.copy()
    if how == "f1_band":
        f[0] = f0 * np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
    elif how in ("f1_near_h1", "f1_near_h2"):
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


KEYFRAMES_PER_STEP = 4  # interactive tract: keyframes at quarter steps between articulatory targets


def _target_shape(spk: tube.TubeSpeaker, info: dict) -> tube.TubeShape:
    """The tube shape `tube.sample_target` drew (its info) for this speaker."""
    return tube.build_shape(info["q1"], info["q2"], spk.length_cm, pharynx_ratio=spk.pharynx_ratio,
                            area_scale=spk.area_scale, omega_warp=spk.omega_warp, constrict=info.get("constrict"),
                            pg=info["pg_cm"], pm=info["pm_cm"], perimeter_factor=spk.perimeter_factor)


def _interactive_tract(spk, drawn, tt, widths, n, glottis):
    """Keyframed tract for the interactive source. The articulatory path (the fractional target index per control
    point, with the formants' raised-cosine transitions) is sampled at quarter steps; each keyframe is the area
    function blended between its two targets (`interactive.blend_shapes`), with its own impulse responses and poles.
    Labels interpolate the keyframes' poles in log frequency; the source blends their responses linearly, so the
    two agree to within a quarter step. Returns responses, per-control-point keyframe index and weight, and label
    formants and bandwidths (n, N_POLES) (the tube's own bandwidths)."""
    T = len(drawn)
    shapes = [_target_shape(spk, d[2]) for d in drawn]
    path = _raised_cosine_steps(n, np.array(tt[1:]), np.arange(T, dtype=float), widths) * KEYFRAMES_PER_STEP
    n_kf = int(np.floor(path.max() + 1e-9)) + 2
    n_kf = min(n_kf, (T - 1) * KEYFRAMES_PER_STEP + 1)
    resp, kf_f, kf_b = [], [], []
    for q in range(n_kf):
        j, r = divmod(q, KEYFRAMES_PER_STEP)
        if r == 0:
            shape = shapes[j]
            f, b = drawn[j][0], drawn[j][1] / spk.bw_scale
        else:
            shape = interactive_source.blend_shapes(shapes[j], shapes[j + 1], r / KEYFRAMES_PER_STEP)
            f, b = np.full(tract.N_POLES, np.nan), np.full(tract.N_POLES, np.nan)
            try:
                pf, pb = tube.poles(shape, tract.F_MAX_HZ)
                k = min(len(pf), tract.N_POLES)
                f[:k], b[:k] = pf[:k], pb[:k]
            except RuntimeError:  # rare: this keyframe's labels stay NaN (masked in the loss)
                pass
        resp.append(interactive_source.tract_responses(shape, FS, sub_scale=glottis.sub_scale))
        kf_f.append(f)
        kf_b.append(b)
    kf_f, kf_b = np.log(np.array(kf_f)), np.log(np.array(kf_b))
    block = np.minimum(np.floor(path).astype(np.int64), n_kf - 1)
    w = np.where(block < n_kf - 1, path - block, 0.0)
    nxt = np.minimum(block + 1, n_kf - 1)
    formants = np.exp((1 - w)[:, None] * kf_f[block] + w[:, None] * kf_f[nxt])
    bws = np.exp((1 - w)[:, None] * kf_b[block] + w[:, None] * kf_b[nxt])
    return resp, block, w, formants, bws


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

    length_group = None
    if rng.random() < cfg.p_length_from_f0:
        length = _coupled_length(rng, cfg, centre)
    elif cfg.length_dist == "population":
        length, length_group = _population_length(rng)
    else:
        length = rng.uniform(*cfg.length_cm)

    # Phrasing: voiced segments separated by gaps of different kinds
    voice_env = np.zeros(n)
    whisper = np.zeros(n)
    fric = np.zeros(n)
    h_env = np.zeros(n)
    segments = []
    whisper_clip = cfg.p_whisper_clip > 0 and rng.random() < cfg.p_whisper_clip
    pos = 0.0 if rng.random() < 0.5 else rng.uniform(*cfg.gap_s) * 1000
    while pos < n:
        seg = rng.uniform(*cfg.voiced_seg_s) * 1000
        a, b = int(pos), int(min(pos + seg, n))
        att, dec = rng.uniform(10, 60), rng.uniform(10, 60)
        idx = np.arange(a, b)
        env = np.minimum(np.clip((idx - a) / att, 0, 1), np.clip((b - idx) / dec, 0, 1))
        target = whisper if whisper_clip else voice_env
        target[a:b] = np.maximum(target[a:b], 0.5 - 0.5 * np.cos(np.pi * env))
        segments.append({"kind": "whispered" if whisper_clip else "voiced", "start_s": a / 1000, "end_s": b / 1000})
        if cfg.p_h_onset > 0 and not whisper_clip and rng.random() < cfg.p_h_onset:
            h0 = max(int(a - rng.uniform(40, 120)), 0)
            if h0 < a:  # rises within 20 ms, holds, then fades out under the vowel's attack
                h1 = min(int(a + att), n)
                hi = np.arange(h0, h1)
                rise = max(min(20.0, (a - h0) / 2), 1.0)
                h_env[h0:h1] = np.maximum(h_env[h0:h1], np.minimum(np.clip((hi - h0) / rise, 0, 1),
                                                                  np.clip((h1 - hi) / max(h1 - a, 1), 0, 1)))
                segments.append({"kind": "h_onset", "start_s": h0 / 1000, "end_s": a / 1000})
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
    tube_info, glottis, c_scale = None, None, None
    interactive, area_sq = False, None
    if use_tube:
        spk = tube.sample_speaker(rng, length)
        # no draw at p_interactive = 0 or p_coupling = 0, so earlier sets reproduce
        if cfg.p_interactive > 0 and cfg.formant_constraint is None and rng.random() < cfg.p_interactive:
            interactive = True
            glottis = coupling.sample_glottis(rng, length, cfg.glottal_peak_area_cm2, cfg.glottal_gap_offset)
            area_sq = float(rng.uniform(*cfg.interactive_sq))
        elif cfg.p_coupling > 0 and rng.random() < cfg.p_coupling:
            glottis = coupling.sample_glottis(rng, length, cfg.glottal_peak_area_cm2, cfg.glottal_gap_offset)
            c_scale = float(rng.uniform(*cfg.coupling_scale))
        q_draw = tube.vowel_q_sampler(tube.CONVERSATIONAL_VOWELS, tube.vowel_centres(length)) if cfg.vowel_dist == "conversational" else None
        drawn = [tube.sample_target(rng, spk, tract.N_POLES, tract.F_MAX_HZ, cfg.p_tube_constrict, q_draw=q_draw,
                                    glottis=None if interactive else glottis) for _ in tt]
        targets = np.array([d[0] for d in drawn])
        tube_bws = np.array([d[1] for d in drawn])
        tube_info = {"speaker": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(spk).items()},
                     "targets": [d[2] for d in drawn], "glottis": asdict(glottis) if glottis else None,
                     "coupling_scale": c_scale, "interactive": interactive, "area_sq": area_sq}
    else:
        targets = np.array([tract.sample_formants(rng, length, cfg.p_uniform_formants, cfg.p_cluster,
                                                  conversational=cfg.vowel_dist == "conversational") for _ in tt])
        if cfg.p_f1_band > 0 and cfg.f1_band_f0_hz[0] <= centre <= cfg.f1_band_f0_hz[1]:
            f0_at = f0[np.clip(np.array(tt).astype(int), 0, n - 1)]
            targets = np.array([_constrain(rng, t, f, "f1_band", cfg.f1_band_ratio) if rng.random() < cfg.p_f1_band else t
                                for t, f in zip(targets, f0_at)])
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
        bw_f0 = centre if cfg.bandwidth_f0_hz is None else cfg.bandwidth_f0_hz
        bw_targets = np.stack([tract.sample_bandwidths(rng, np.exp(targets[i]), bw_f0) for i in range(len(tt))])
    logb = np.stack([_raised_cosine_steps(n, np.array(tt[1:]), np.log(np.nan_to_num(bw_targets[:, k], nan=100.0)), widths)
                     for k in range(tract.N_POLES)], 1)
    bws = np.where(np.isfinite(formants), np.exp(logb), np.nan)
    if interactive:
        kf_shapes, kf_block, kf_w, formants, bws = _interactive_tract(spk, drawn, tt, widths, n, glottis)
        tube_info["n_keyframes"] = len(kf_shapes)
    extras = tract.sample_extras(rng, length, cfg.p_nasal, cfg.p_subglottal)
    n_ep, n_ez = len(extras["extra_pole_f"]), len(extras["extra_zero_f"])

    traj = Trajectories(
        f0=f0, rd=rd, voice_amp=voice_env * loud,
        whisper_amp=(whisper * loud * rng.uniform(0.3, 1.5)
                     + (h_env * loud * rng.uniform(*cfg.h_amp) if h_env.any() else 0.0)), fric_amp=fric * loud * rng.uniform(0.3, 1.5),
        poles_f=np.concatenate([formants, np.tile(extras["extra_pole_f"], (n, 1))], 1),
        poles_b=np.concatenate([bws, np.tile(extras["extra_pole_b"], (n, 1))], 1),
        zeros_f=np.tile(extras["extra_zero_f"], (n, 1)).reshape(n, n_ez),
        zeros_b=np.tile(extras["extra_zero_b"], (n, 1)).reshape(n, n_ez),
    )
    coupled = None
    if glottis is not None and not interactive:  # per-target open-glottis shifts, with the same transitions as the formants
        steps = lambda key: np.stack([_raised_cosine_steps(n, np.array(tt[1:]), np.array([d[2][key][j] for d in drawn]), widths)  # noqa: E731
                                      for j in range(coupling.N_COUPLED)], 1)
        sc = c_scale
        if cfg.coupling_full_poles > 0:
            sc = np.full(coupling.N_COUPLED, c_scale)
            sc[:cfg.coupling_full_poles] = 1.0
        coupled = {"df": steps("open_df") * sc, "bm": 1.0 + (steps("open_bm") - 1.0) * sc, "gap_offset": glottis.gap_offset}
    jitter, shimmer = rng.uniform(*cfg.jitter), rng.uniform(*cfg.shimmer_db)
    hnr = None if cfg.hnr_db is None else rng.uniform(*cfg.hnr_db)
    if interactive:
        comps = interactive_source.synthesize(traj, kf_shapes, kf_block, kf_w, glottis, area_sq, rng, jitter=jitter,
                                              shimmer_db=shimmer, hnr_db=hnr, extras=extras,
                                              coupled=cfg.interactive_coupled)
    else:
        comps = synthesize(traj, rng, jitter=jitter, shimmer_db=shimmer, hnr_db=hnr, coupling=coupled)
    audio = comps["total"]
    peak = np.max(np.abs(audio)) + 1e-12
    gain = 0.5 / peak
    audio = audio * gain

    # Frame labels (frame i at t = i * hop)
    hop_ctrl = int(round(cfg.hop_s * 1000))
    fi = np.arange(0, n - 1, hop_ctrl)
    voiced = voice_env[fi] >= 0.5
    whispered = np.maximum(whisper, h_env)[fi] >= 0.5
    labels = {
        "t": fi / 1000.0,
        "f0": f0[fi],
        "voiced": voiced,
        "sonorant": voiced | whispered if cfg.sonorant_includes_whisper else voiced,
        "whisper": whispered,
        "formants": formants[fi, :N_LABEL_FORMANTS],
        "bandwidths": bws[fi, :N_LABEL_FORMANTS],
        "rd": rd[fi],
        # LF harmonics; an interactive clip's source is not LF (and is shaped by the tract), so unknown there
        "source_h_db": (np.full((len(fi), N_SOURCE_HARMONICS), np.nan) if interactive
                        else source_h_db(rd[fi], N_SOURCE_HARMONICS)),
        "length_cm": np.full(len(fi), length),
    }
    meta = {
        "index": index, "seed": seed, "fs": FS, "duration_s": len(audio) / FS, "length_cm": length, "length_group": length_group,
        "contour": contour, "register_steps": register_steps, "segments": segments,
        "jitter": jitter, "shimmer_db": shimmer, "hnr_db": hnr, "gain": gain,
        "tract": "tube" if use_tube else "cascade", "tube": tube_info,
        "extras": {k: v.tolist() for k, v in extras.items()}, "n_extra_poles": n_ep, "n_extra_zeros": n_ez,
        "config": asdict(cfg),
    }
    return SynthClip(audio.astype(np.float32), labels, meta)
