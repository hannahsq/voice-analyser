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
N_FFT = 16384  # frequency grid for the responses (fine enough that the periodic responses don't wrap)


@dataclass
class TractResponses:
    """Impulse responses of one tract shape at fs: z (N_IR,) load impedance at the glottis, tract plus subglottal
    (cgs acoustic ohm; z[0] includes the instantaneous part), h (N_IR,) U_lips / U_glottis."""

    z: np.ndarray
    h: np.ndarray


def subglottal_impedance(s: np.ndarray, sub_scale: float = 1.0) -> np.ndarray:
    """Subglottal input impedance: parallel resonances at Ishizaka, Matsudaira & Kaneko's (1976) maxima, as
    `coupling.source_impedance` uses them, scaled in frequency by sub_scale."""
    z = np.zeros_like(s)
    for f, r, b in SUBGLOTTAL:
        w0, bw = 2 * np.pi * f * sub_scale, 2 * np.pi * b * sub_scale
        z = z + r * bw * s / (s * s + bw * s + w0 * w0)
    return z


def tract_frequency_responses(shape: TubeShape, f_hz: np.ndarray, sub_scale: float | None = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Load impedance at the glottis (tract input impedance plus subglottal; sub_scale None = no subglottal system)
    and U_lips / U_glottis, at real frequencies f_hz (Hz)."""
    s = 2j * np.pi * np.maximum(np.asarray(f_hz, float), 1e-3)
    A, B, C, D = chain_matrix(s, shape)
    zr = _radiation(s, shape.areas[-1])
    den = C * zr + D
    z = (A * zr + B) / den
    if sub_scale is not None:
        z = z + subglottal_impedance(s, sub_scale)
    return z, 1.0 / den


def tract_responses(shape: TubeShape, fs: float = 16000.0, sub_scale: float | None = 1.0, n_ir: int = N_IR,
                    n_fft: int = N_FFT) -> TractResponses:
    """Impulse responses from the frequency responses sampled on an n_fft grid up to fs / 2. The tract and
    subglottal responses are causal and decay with their losses, so the periodic inverse transform is the
    (band-limited) impulse response to within its decay over n_fft samples."""
    f = np.fft.rfftfreq(n_fft, 1.0 / fs)
    zf, hf = tract_frequency_responses(shape, f, sub_scale)
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
