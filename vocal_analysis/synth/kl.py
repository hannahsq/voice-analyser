"""Kelly-Lochbaum waveguide vocal tract with a coupled glottis: a time-domain acoustic path beside the resonator cascade.

Ported from voice-dataset-labeler's `synth/kl.py` (MIT, Hannah Querengasser; spec KL_SYNTH_SPEC.md on its
feat_bayesian_vtl branch) and rewritten here, with three fixes found on 2026-10-30 / 31 (`docs/considering.md`):
- the glottal boundary (the spec's GL-3): from Ug = A0 (p+ - p-) / (rho c), the launched wave is
  p+ = rho c Ug / A0 + p-; the original halved both terms, so a closed glottis reflected only half the wave;
- the section length: one section per sample of delay is dx = c / fs; the original used 2 L fs / c sections, a tube
  twice as long acoustically (its R11 check found the resonances an octave low);
- the flow is solved with the load, not against last sample's pressure: Bernoulli plus the viscous term (the
  interactive source's law, `interactive.lf_area_pulse`) with the glottal-end pressure rho c Ug / A0 + 2 p-, a
  quadratic in Ug per sample.

Geometry and physics follow `tube.py`, so the same Story area functions (all tiers) drive both: the area function is
resampled onto equal sections of dx = c / (OVERSAMPLE x 16 kHz) (0.27 cm at 8x, about 65 sections for 17.5 cm);
yielding walls are a mass-damper-spring per junction (`tube.WALL_*` per unit area, the wall area of one section);
viscous and heat-conduction losses are a per-section attenuation from `tube.py`'s line constants at LOSS_REF_HZ
(frequency-independent here, unlike `tube.py`'s sqrt(f): the labels below are measured, so they are exact anyway);
lip radiation is `tube._radiation`'s piston load as a first-order reflection filter (bilinear). Areas move linearly
between 10 ms control frames. The output is the radiated pressure's shape, d/dt of the lip volume velocity,
decimated to 16 kHz.

Labels (`frame_poles`): each control frame's closed-glottis resonances, measured from this waveguide itself (its
impulse response with the glottis rigid, FFT, then the poles of U_lips / U_glottis by vector fitting), so they are
exact for what is rendered, discretisation and loss model included.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import resample_poly

from . import tube
from .interactive import KT
from .tube import C_CM_S, ETA, LAMBDA_OVER_CP, MU, RHO, WALL_B, WALL_K, WALL_M, TubeShape, _radiation

FS = 16000
HOP = 160
OVERSAMPLE = 8
LOSS_REF_HZ = 1000.0
LABEL_IR_SAMPLES = 1 << 14  # at 8 x 16 kHz: 0.128 s (a 20 Hz bandwidth decays by about 70 dB); vector fitting needs no fine bins
LABEL_F_MAX_HZ = 7500.0
SEED_MARGIN_HZ = 1000.0  # poles fitted above the labelled band, so its top edge is not distorted
FIT_MIN_HZ = 50.0


def section_length(oversample: int = OVERSAMPLE) -> float:
    return C_CM_S / (FS * oversample)


def uniform_areas(shape: TubeShape, oversample: int = OVERSAMPLE) -> np.ndarray:
    """The area function on equal sections of `section_length`: each section's mean area over its span (the
    tubelets' lengths may differ: Tier III). n = round(L / dx), so the tube is within dx / 2 of shape's length."""
    dx = section_length(oversample)
    edges = np.r_[0.0, np.cumsum(shape.lengths)]
    n = max(2, int(round(edges[-1] / dx)))
    vol = np.r_[0.0, np.cumsum(shape.areas * shape.lengths)]  # cumulative volume at the tubelet edges
    x = np.linspace(0.0, edges[-1], n + 1)
    return np.diff(np.interp(x, edges, vol)) / np.diff(x)


def _loss_factor(areas: np.ndarray, perimeter_factor: float, dx: float) -> np.ndarray:
    """Per-section, per-traversal amplitude factor exp(-alpha dx) from viscous and heat-conduction losses at
    LOSS_REF_HZ (`tube.chain_matrix`'s series and shunt terms, walls left out: they are modelled separately)."""
    s = 2j * np.pi * LOSS_REF_HZ
    perim = perimeter_factor * 2.0 * np.sqrt(np.pi * areas)
    z = s * RHO / areas + perim / areas**2 * np.sqrt(RHO * MU) * np.sqrt(s)
    y = s * areas / (RHO * C_CM_S**2) + perim * (ETA - 1) / (RHO * C_CM_S**2) * np.sqrt(LAMBDA_OVER_CP / RHO) * np.sqrt(s)
    return np.exp(-np.sqrt(z * y).real * dx)


def _lip_filter(lip_area: float, fs: float) -> tuple[float, float, float]:
    """(b0, b1, a1) of the lip reflection p- = G(z) p+, G = (Zr - Z0) / (Zr + Z0), bilinear. G(DC) = -1 (open end)."""
    rr = 128.0 * RHO * C_CM_S / (9.0 * np.pi**2 * lip_area)
    lr = 8.0 * RHO / (3.0 * np.pi * np.sqrt(np.pi * lip_area))
    tau, z0, k = lr / rr, RHO * C_CM_S / lip_area, 2.0 * fs
    n0, n1 = tau * k * (rr - z0) - z0, -tau * k * (rr - z0) - z0
    d0, d1 = tau * k * (rr + z0) + z0, -tau * k * (rr + z0) + z0
    return n0 / d0, n1 / d0, d1 / d0


@njit(cache=True)
def _run(areas, loss, lip, step, ag, p_sub, kv, wall, dt, dx, impulse, coupled=True):
    """The waveguide at the internal rate (time step dt, section length dx). areas (n_ctrl, N) per control frame, linear
    in between over `step` samples; loss (n_ctrl, N) per-traversal factors; lip (n_ctrl, 3) reflection filter
    coefficients; ag glottal area per sample (cm^2); kv the viscous flow coefficient (12 mu d lg^2); wall (m, b, k, on)
    per unit area; impulse: glottis rigid, a unit volume velocity at sample 0 (for the labels); coupled False: the
    flow from the glottis and subglottal pressure alone (no feedback from the tract; the control for interaction).
    Returns (lip volume velocity, glottal volume velocity) per sample."""
    n_ctrl, N = areas.shape
    M = len(ag)
    fw = np.zeros(N)  # right-going wave arriving at section i's lip end
    bw = np.zeros(N)  # left-going wave arriving at section i's glottal end
    nf, nb = np.zeros(N), np.zeros(N)
    wx, wv = np.zeros(N - 1), np.zeros(N - 1)  # wall displacement and velocity per junction
    a = np.empty(N)
    lf_x1 = 0.0  # lip filter state: previous input and output
    lf_y1 = 0.0
    u_lip = np.zeros(M)
    u_g = np.zeros(M)
    rc = RHO * C_CM_S
    wm, wb, wk, use_wall = wall[0], wall[1], wall[2], wall[3] > 0.0
    for t in range(M):
        c = t // step
        if c >= n_ctrl - 1:
            c, w = n_ctrl - 1, 0.0
        else:
            w = (t - c * step) / step
        for i in range(N):
            a[i] = areas[c, i] if w == 0.0 else (1.0 - w) * areas[c, i] + w * areas[c + 1, i]
        b0, b1, a1 = lip[c, 0], lip[c, 1], lip[c, 2]
        # glottis: KT rho U^2 / (2 Ag^2) + (kv / Ag^3 + rho c / A0) U = p_sub - 2 p-  (Bernoulli + viscous, with the load)
        p_in = bw[0]
        if impulse:
            ug = 1.0 if t == 0 else 0.0
        else:
            g = ag[t]
            rhs = p_sub - 2.0 * p_in if coupled else p_sub
            ug = 0.0
            if g > 1e-9 and rhs != 0.0:
                qa = KT * RHO / 2.0 / (g * g)
                qb = kv / (g * g * g) + (rc / a[0] if coupled else 0.0)
                r = abs(rhs)
                ug = 2.0 * r / (qb + np.sqrt(qb * qb + 4.0 * qa * r))
                if rhs < 0.0:
                    ug = -ug
        u_g[t] = ug
        nf[0] = (rc * ug / a[0] + p_in) * loss[c, 0]  # GL-3, corrected: a closed glottis (ug = 0) reflects fully
        # junction i: continuity Y_i (f - nb) = Y_j (nf - b) + u_wall with P = f + nb = nf + b, Y = A / (rho c)
        for i in range(N - 1):
            yi, yj = a[i] / rc, a[i + 1] / rc
            uw = 0.0
            if use_wall:
                uw = wv[i] * 2.0 * np.sqrt(np.pi * 0.5 * (a[i] + a[i + 1])) * dx
            p = (2.0 * (yi * fw[i] + yj * bw[i + 1]) - uw) / (yi + yj)
            if use_wall:  # m dv/dt = P - b v - k x per unit area (semi-implicit Euler)
                wv[i] += dt * (p - wb * wv[i] - wk * wx[i]) / wm
                wx[i] += dt * wv[i]
            nf[i + 1] = (p - bw[i + 1]) * loss[c, i + 1]
            nb[i] = (p - fw[i]) * loss[c, i]
        # lips: the arriving wave through the reflection filter
        x0 = fw[N - 1]
        y0 = b0 * x0 + b1 * lf_x1 - a1 * lf_y1
        lf_x1, lf_y1 = x0, y0
        nb[N - 1] = y0 * loss[c, N - 1]
        u_lip[t] = a[N - 1] * (x0 - y0) / rc
        for i in range(N):
            fw[i], bw[i] = nf[i], nb[i]
    return u_lip, u_g


def glottal_area(f0_hz: np.ndarray, n_samples: int, fs: float, peak_cm2: float, oq: float = 0.6, sq: float = 1.3,
                 gap: float = 0.0, closure: float = 1.0) -> np.ndarray:
    """Glottal area per sample (cm^2) from a per-frame f0 track (HOP frames at 16 kHz): Rosenberg's pulse
    (`interactive.area_pulse`) with open quotient oq, speed quotient sq and a closed-phase gap (fraction of the peak)."""
    from .interactive import area_pulse

    t_frames = (np.arange(len(f0_hz)) * HOP + 0.5 * HOP) / FS
    f0 = np.interp(np.arange(n_samples) / fs, t_frames, f0_hz)
    cycles = np.cumsum(f0 / fs)
    return peak_cm2 * area_pulse(cycles, oq, sq, gap=gap, closure=closure)


def _controls(shapes: list[TubeShape], oversample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per control frame: areas on a common section count (the clip's mean length), loss factors, lip filters."""
    dx = section_length(oversample)
    n = max(2, int(round(np.mean([s.length_cm for s in shapes]) / dx)))
    areas = np.empty((len(shapes), n))
    loss = np.empty((len(shapes), n))
    lip = np.empty((len(shapes), 3))
    for c, s in enumerate(shapes):
        ua = uniform_areas(s, oversample)
        areas[c] = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ua)), ua) if len(ua) != n else ua
        areas[c] = np.maximum(areas[c], 1e-3)
        loss[c] = _loss_factor(areas[c], s.perimeter_factor, dx)
        lip[c] = _lip_filter(areas[c, -1], FS * oversample)
    return areas, loss, lip, n * dx


def render(shapes: list[TubeShape], ag_cm2: np.ndarray, pressure: float, glottis_depth_cm: float, glottis_length_cm: float,
           oversample: int = OVERSAMPLE, walls: bool = True, coupled: bool = True) -> dict:
    """Synthesise through the waveguide. shapes: one TubeShape per HOP frame (areas move linearly between them);
    ag_cm2: glottal area per sample at FS; pressure: subglottal pressure, dyn/cm^2; coupled False: no feedback from the
    tract into the flow. Returns 16 kHz `audio` (d/dt of
    the lip volume velocity, unnormalised), `ug` (glottal volume velocity), and the tube's `length_cm`."""
    areas, loss, lip, length = _controls(shapes, oversample)
    fs_in = FS * oversample
    m = len(ag_cm2) * oversample
    ag = np.interp(np.arange(m) / fs_in, np.arange(len(ag_cm2)) / FS, ag_cm2)
    kv = 12.0 * MU * glottis_depth_cm * glottis_length_cm**2
    wall = np.array([WALL_M, WALL_B, WALL_K, 1.0 if walls else 0.0])
    u_lip, u_g = _run(areas, loss, lip, HOP * oversample, ag, float(pressure), kv, wall, 1.0 / fs_in,
                      section_length(oversample), False, coupled)
    rad = np.diff(u_lip, prepend=0.0) * fs_in
    audio = resample_poly(rad, 1, oversample)[: len(ag_cm2)]
    return {"audio": audio.astype(np.float32), "ug": resample_poly(u_g, 1, oversample)[: len(ag_cm2)].astype(np.float32),
            "length_cm": length}


def _vector_fit(h: np.ndarray, f: np.ndarray, seed_f: np.ndarray, seed_b: np.ndarray, n_iter: int = 8
                ) -> tuple[np.ndarray, np.ndarray]:
    """Poles of a real system's response h at frequencies f (Hz) by real vector fitting (Gustavsen & Semlyen 1999):
    h ~ sum over conjugate pole pairs + d + e s, the poles relocated as the zeros of a fitted weighting function,
    seeded with (seed_f, seed_b) and weighted by 1 / |h| (relative error). Separates close pairs that share one
    spectral peak (F2 and F3 of /i/). -> (F, B) in Hz, sorted."""
    s = 2j * np.pi * f
    p = -np.pi * np.asarray(seed_b, float) + 2j * np.pi * np.asarray(seed_f, float)
    w = 1.0 / np.maximum(np.abs(h), 1e-30)

    def basis(p):
        cols = []
        for pk in p:
            a, b = 1.0 / (s - pk), 1.0 / (s - np.conj(pk))
            cols += [a + b, 1j * a - 1j * b]
        return np.stack(cols, 1)

    for _ in range(n_iter):
        phi = basis(p)
        ext = np.concatenate([phi, np.ones((len(s), 1)), s[:, None], -h[:, None] * phi], 1) * w[:, None]
        rhs = h * w
        sol, *_ = np.linalg.lstsq(np.concatenate([ext.real, ext.imag]), np.concatenate([rhs.real, rhs.imag]), rcond=None)
        ct = sol[2 * len(p) + 2:]
        A = np.zeros((2 * len(p), 2 * len(p)))
        bvec = np.zeros(2 * len(p))
        for k, pk in enumerate(p):
            A[2 * k: 2 * k + 2, 2 * k: 2 * k + 2] = [[pk.real, pk.imag], [-pk.imag, pk.real]]
            bvec[2 * k] = 2.0
        z = np.linalg.eigvals(A - np.outer(bvec, ct))
        z = z[z.imag > 0]
        z = np.where(z.real > 0, -z.real + 1j * z.imag, z)  # keep stable
        if len(z) != len(p):
            break
        p = np.sort_complex(z)
    F, B = p.imag / (2 * np.pi), -p.real / np.pi
    order = np.argsort(F)
    return F[order], B[order]


def frame_poles(shapes: list[TubeShape], oversample: int = OVERSAMPLE, walls: bool = True, n_ir: int = LABEL_IR_SAMPLES,
                f_max_hz: float = LABEL_F_MAX_HZ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed-glottis resonances (F, B per frame, Hz) of the waveguide as `render` builds it: the impulse response of
    U_lips / U_glottis with the glottis rigid and the frame's shape held, FFT, `_vector_fit` seeded by `tube.poles` of
    the same discretised tube (so close pairs stay apart and none is missed). Frames with the same
    areas share one computation."""
    areas, loss, lip, _ = _controls(shapes, oversample)
    fs_in = FS * oversample
    wall = np.array([WALL_M, WALL_B, WALL_K, 1.0 if walls else 0.0])
    f = np.fft.rfftfreq(n_ir, 1.0 / fs_in)
    out, cache = [], {}
    for c in range(len(shapes)):
        key = areas[c].tobytes()
        if key not in cache:
            u, _ = _run(areas[c : c + 1], loss[c : c + 1], lip[c : c + 1], n_ir, np.zeros(n_ir), 0.0, 0.0, wall,
                        1.0 / fs_in, section_length(oversample), True)
            dx = section_length(oversample)
            seed_f, seed_b = tube.poles(TubeShape(areas[c], np.full(len(areas[c]), dx), shapes[c].perimeter_factor),
                                        f_max_hz + SEED_MARGIN_HZ)  # same discretised tube, continuous physics
            band = (f >= FIT_MIN_HZ) & (f <= f_max_hz + SEED_MARGIN_HZ)
            F, B = _vector_fit(np.fft.rfft(u)[band], f[band], seed_f, seed_b)
            cache[key] = (F[F <= f_max_hz], B[F <= f_max_hz])
        out.append(cache[key])
    return out
