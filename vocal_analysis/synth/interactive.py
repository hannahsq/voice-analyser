"""Interactive glottal source (D-026): glottal flow computed from the glottal area and the acoustic load, so the tract
feeds back into the source (Titze's level-1 interaction: formant ripple in the flow, flow skewing, open-phase F1 and B1
changes), instead of a fixed LF pulse through a filter.

Per sample n, with glottal area A_n (cm^2) and lung pressure P_L (dyn/cm^2), the transglottal pressure drop is
    P_L - p_load,n = k_t rho / (2 A_n^2) U_n |U_n| + R_v,n U_n + (rho d / A_n) dU/dt
(kinetic, viscous and inertive terms of the glottal slit, as `coupling.source_impedance` linearises them; Flanagan 1972,
Ishizaka & Flanagan 1972). The load is the tract's input impedance at the glottis in series with the subglottal system,
as impulse responses: p_load,n = sum_k z_k U_{n-k}. Moving z_0 U_n and the inertance's U_n to the left leaves a
quadratic in U_n, solved in closed form each sample. The lip flow is U_g through the tract's transfer
(U_lips / U_glottis = 1 / (C Z_r + D)); the radiated pressure is its first difference, as the LF derivative is in the
other paths.

The tract is the closed-glottis lossy tube of `tube` (Story's area functions, wall and boundary-layer losses, lip
radiation), so labels stay that tube's poles (`tube.poles`): the same resonances, now heard through a real glottis.
Impulse responses are sampled from the analytic frequency responses (`tract_responses`); a slowly changing tract is
rendered by interpolating the responses of neighbouring articulatory targets per control block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .coupling import SUBGLOTTAL, Glottis
from .tube import MU, RHO, TubeShape, _radiation, chain_matrix

KT = 1.0  # transglottal kinetic pressure coefficient (Bernoulli; Titze 2006 uses about 1)
N_IR = 1024  # impulse-response length at 16 kHz (64 ms: B1 >= 25 Hz decays by >= 60 dB)
N_FFT = 2048  # frequency grid for the responses (7.8 Hz; 8192 changes them by ~1e-4 of the peak)


def area_pulse(cycles: np.ndarray, te: np.ndarray | float, sq: np.ndarray | float = 1.3,
               gap: np.ndarray | float = 0.0) -> np.ndarray:
    """Glottal area per sample as a fraction of the peak: Rosenberg's (1971) type-C pulse over the open phase
    (opening 0.5 (1 - cos(pi t / Tp)), closing cos(pi (t - Tp) / (2 Tn)), Tp / Tn = sq, Tp + Tn = te in cycles) on top
    of the closed-phase gap. The closing branch ends with a finite slope, a corner at closure as the membranous folds
    meet: the excitation that a smooth (sin^2) pulse lacks. With a sin^2 pulse and a gap the source spectrum had nulls
    and fell 16-18 dB/oct, and every model read F1 25-40 Hz low (D-026); this pulse falls 7-11 dB/oct like the LF
    derivative, and a gap raises H1-H2 as glottal chinks do."""
    fr = np.asarray(cycles) % 1.0
    te = np.broadcast_to(np.asarray(te, float), fr.shape)
    sq = np.broadcast_to(np.asarray(sq, float), fr.shape)
    tp = te * sq / (1.0 + sq)
    tn = te - tp
    a = np.where(fr < tp, 0.5 * (1.0 - np.cos(np.pi * fr / tp)),
                 np.where(fr < te, np.cos(np.pi * (fr - tp) / (2.0 * tn)), 0.0))
    return gap + (1.0 - gap) * a


@dataclass
class TractResponses:
    """Impulse responses of one tract shape at fs: z (N_IR,) load impedance at the glottis, tract plus subglottal
    (cgs acoustic ohm; z[0] includes the instantaneous part), h (N_IR,) U_lips / U_glottis."""

    z: np.ndarray
    h: np.ndarray


def subglottal_impedance(s: np.ndarray, sub_scale: float = 1.0, sub_gain: float = 1.0) -> np.ndarray:
    """Subglottal input impedance: parallel resonances at Ishizaka, Matsudaira & Kaneko's (1976) maxima, as
    `coupling.source_impedance` uses them, scaled in frequency by sub_scale and in magnitude by sub_gain."""
    z = np.zeros_like(s)
    for f, r, b in SUBGLOTTAL:
        w0, bw = 2 * np.pi * f * sub_scale, 2 * np.pi * b * sub_scale
        z = z + sub_gain * r * bw * s / (s * s + bw * s + w0 * w0)
    return z


def tract_frequency_responses(shape: TubeShape, f_hz: np.ndarray, sub_scale: float | None = 1.0,
                              sub_gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Load impedance at the glottis (tract input impedance plus subglottal; sub_scale None = no subglottal system)
    and U_lips / U_glottis, at real frequencies f_hz (Hz)."""
    s = 2j * np.pi * np.maximum(np.asarray(f_hz, float), 1e-3)
    A, B, C, D = chain_matrix(s, shape)
    zr = _radiation(s, shape.areas[-1])
    den = C * zr + D
    z = (A * zr + B) / den
    if sub_scale is not None:
        z = z + subglottal_impedance(s, sub_scale, sub_gain)
    return z, 1.0 / den


def tract_responses(shape: TubeShape, fs: float = 16000.0, sub_scale: float | None = 1.0, n_ir: int = N_IR,
                    n_fft: int = N_FFT, sub_gain: float = 1.0) -> TractResponses:
    """Impulse responses from the frequency responses sampled on an n_fft grid up to fs / 2. The tract and
    subglottal responses are causal and decay with their losses, so the periodic inverse transform is the
    (band-limited) impulse response to within its decay over n_fft samples."""
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    zf, hf = tract_frequency_responses(shape, f, sub_scale, sub_gain)
    # the Nyquist bin of a real sequence must be real: use its real part
    zf[-1], hf[-1] = zf[-1].real, hf[-1].real
    return TractResponses(np.fft.irfft(zf, n_fft)[:n_ir].copy(), np.fft.irfft(hf, n_fft)[:n_ir].copy())


@njit(cache=True)
def _render(area, p_lung, z, h, block, w, ctrl, kt, depth, length_g, rho, mu, fs, coupled):
    """Per-sample flow and lip flow. z, h: (T, K) responses per target; block: target index of each control block's
    start, w: (blocks,) weight of the next target (linear between targets). Returns (U_glottis, U_lips)."""
    n = len(area)
    K = z.shape[1]
    ug = np.zeros(n)
    ul = np.zeros(n)
    zc = np.empty(K)
    hc = np.empty(K)
    for i in range(n):
        b = i // ctrl
        if i % ctrl == 0:
            t0 = block[b]
            t1 = min(t0 + 1, z.shape[0] - 1)
            wb = w[b]
            for k in range(K):
                zc[k] = (1.0 - wb) * z[t0, k] + wb * z[t1, k]
                hc[k] = (1.0 - wb) * h[t0, k] + wb * h[t1, k]
        kmax = min(K, i + 1)
        hist = 0.0
        if coupled:
            for k in range(1, kmax):
                hist += zc[k] * ug[i - k]
        a = area[i]
        if a > 1e-7:
            prev = ug[i - 1] if i > 0 else 0.0
            inert = rho * depth / a * fs  # inertance times fs (backward difference)
            qa = kt * rho / (2.0 * a * a)
            qb = 12.0 * mu * depth * length_g * length_g / (a * a * a) + inert + (zc[0] if coupled else 0.0)
            c = p_lung[i] - hist + inert * prev
            u = 2.0 * c / (qb + np.sqrt(qb * qb + 4.0 * qa * abs(c)))
        else:
            u = 0.0
        ug[i] = u
        s = 0.0
        for k in range(kmax):
            s += hc[k] * ug[i - k]
        ul[i] = s
    return ug, ul


@njit(cache=True)
def _convolve_tv(x, h, block, w, ctrl):
    """x through the blended responses h (T, K) as in `_render` (no feedback): noise sources at the glottis."""
    n = len(x)
    K = h.shape[1]
    y = np.zeros(n)
    hc = np.empty(K)
    for i in range(n):
        b = i // ctrl
        if i % ctrl == 0:
            t0 = block[b]
            t1 = min(t0 + 1, h.shape[0] - 1)
            for k in range(K):
                hc[k] = (1.0 - w[b]) * h[t0, k] + w[b] * h[t1, k]
        s = 0.0
        for k in range(min(K, i + 1)):
            s += hc[k] * x[i - k]
        y[i] = s
    return y


def render(area_cm2: np.ndarray, responses: list[TractResponses], block_target: np.ndarray, block_w: np.ndarray,
           glottis: Glottis, ctrl: int = 16, fs: float = 16000.0, coupled: bool = True,
           p_lung: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Glottal flow and radiated pressure for a glottal area waveform (cm^2 per sample) through a tract that moves
    between `responses` (one per articulatory target): control block b uses target block_target[b], blended
    towards the next by block_w[b]. coupled=False drops the load (flow set by the glottis and lung pressure alone),
    the reference for interaction effects. Returns 'flow' (U_g, cm^3/s), 'lip_flow' and 'pressure' (first difference
    of the lip flow, the radiated signal)."""
    z = np.stack([r.z for r in responses])
    h = np.stack([r.h for r in responses])
    pl = np.full(len(area_cm2), glottis.pressure) if p_lung is None else np.asarray(p_lung, float)
    ug, ul = _render(np.asarray(area_cm2, float), pl, z, h, np.asarray(block_target, np.int64), np.asarray(block_w, float),
                     ctrl, KT, glottis.depth_cm, glottis.length_cm, RHO, MU, float(fs), coupled)
    return {"flow": ug, "lip_flow": ul, "pressure": np.diff(ul, prepend=0.0)}


def synthesize(traj, responses: list[TractResponses], block_target: np.ndarray, block_w: np.ndarray, glottis: Glottis,
               sq: float, rng: np.random.Generator, jitter: float = 0.0, shimmer_db: float = 0.0,
               hnr_db: float | None = None, extras: dict | None = None, fs: float = 16000.0,
               coupled: bool = True) -> dict[str, np.ndarray]:
    """The components of `voice.synthesize` with the interactive source (same Trajectories; its poles are ignored:
    the tract is `responses`, blended per control block). The area is the Rosenberg pulse at the open quotient of
    the clip's Rd (LF te), the closed-phase gap following it (`coupling.gap_from_open_quotient`), scaled by the
    voice envelope (onsets and loudness open the glottis less). Aspiration (open-phase modulated) and whisper are
    flow noise at the glottis through the same responses; frication is as in `voice.synthesize`. `extras` (nasal
    and other pole-zero pairs, `tract.sample_extras`) filter every component afterwards."""
    from scipy.signal import butter, sosfilt

    from . import filters
    from .coupling import gap_from_open_quotient
    from .glottal import open_phase
    from .voice import CTRL, _glottal_phase, _per_sample

    n = traj.n_samples
    f0_s = _per_sample(traj.f0)
    n_periods = int(np.sum(f0_s) / fs * 1.1) + 16
    jit = np.exp(rng.normal(0.0, jitter, n_periods)) if jitter > 0 else np.ones(n_periods)
    shim = 10 ** (rng.normal(0.0, shimmer_db, n_periods) / 20) if shimmer_db > 0 else np.ones(n_periods)
    cycles, period = _glottal_phase(f0_s, jit, float(fs))
    rd = np.round(traj.rd, 2)
    te_tab = {r: open_phase(float(r)) for r in np.unique(rd)}
    te_s = _per_sample(np.array([te_tab[r] for r in rd]))
    gap = gap_from_open_quotient(te_s, glottis.gap_offset)
    va = _per_sample(traj.voice_amp)
    va_n = va / va.max() if va.max() > 0 else va
    pulse = area_pulse(cycles, te_s, sq, 0.0)
    area = glottis.peak_area_cm2 * (gap + (1.0 - gap) * pulse) * va_n
    bt, bw = np.asarray(block_target, np.int64), np.asarray(block_w, float)
    out = {"voiced": render(area, responses, bt, bw, glottis, CTRL, fs, coupled)["pressure"] * shim[period], "cycles": cycles}
    h = np.stack([r.h for r in responses])
    if hnr_db is not None:
        src = np.diff(rng.standard_normal(n + 1)) * (0.25 + 0.75 * pulse) * va
        asp = _convolve_tv(src, h, bt, bw, CTRL)
        on = va > 0.5 * va.max() if va.max() > 0 else np.zeros(n, bool)
        if on.any():
            ev, ea = np.mean(out["voiced"][on] ** 2), np.mean(asp[on] ** 2)
            if ea > 0:
                asp *= np.sqrt(ev / ea / 10 ** (hnr_db / 10))
        out["aspiration"] = asp
    wa = _per_sample(traj.whisper_amp)
    if wa.any():
        out["whisper"] = _convolve_tv(np.diff(rng.standard_normal(n + 1)) * wa, h, bt, bw, CTRL) * 0.05
    fa = _per_sample(traj.fric_amp)
    if fa.any():
        sos = butter(4, rng.uniform(2000, 5000), "highpass", fs=fs, output="sos")
        out["frication"] = sosfilt(sos, rng.standard_normal(n)) * fa * 0.05
    if extras is not None and len(extras["extra_pole_f"]) + len(extras["extra_zero_f"]) > 0:
        f = np.r_[extras["extra_pole_f"], extras["extra_zero_f"]]
        b = np.r_[extras["extra_pole_b"], extras["extra_zero_b"]]
        is_zero = np.r_[np.zeros(len(extras["extra_pole_f"]), bool), np.ones(len(extras["extra_zero_f"]), bool)]
        for k in [k for k in out if k != "cycles"]:
            out[k] = filters.filter_tv(out[k], np.repeat(f[:, None], n, 1), np.repeat(b[:, None], n, 1), fs, is_zero)
    out["total"] = sum(v for k, v in out.items() if k != "cycles")
    return out


def blend_shapes(a: TubeShape, b: TubeShape, w: float) -> TubeShape:
    """Area function between two articulations: diameters and tubelet lengths linear in w (Story's model is linear in
    diameter), perimeter factor from a (the same speaker)."""
    d = (1.0 - w) * np.sqrt(a.areas) + w * np.sqrt(b.areas)
    return TubeShape(d * d, (1.0 - w) * a.lengths + w * b.lengths, a.perimeter_factor)
