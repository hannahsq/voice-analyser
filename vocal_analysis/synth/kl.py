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
viscous and heat-conduction losses follow `tube.py`'s line constants, whose attenuation is exactly proportional to
sqrt(f): each section's traversal goes through a gain and a cascade of first-order shelves with fixed poles
(LOSS_CORNERS_HZ) whose zeros move with the section's loss, the shelves' weights fitted once so the log magnitude
follows -eps sqrt(f / LOSS_REF_HZ) over 60 Hz - 8 kHz (eps the section's attenuation at LOSS_REF_HZ; zero loss is an
exact identity);
lip radiation is `tube._radiation`'s piston load as a first-order reflection filter (bilinear). Areas move linearly
between 10 ms control frames. The output is the radiated pressure's shape, d/dt of the lip volume velocity,
decimated to 16 kHz.

Labels (`frame_poles`): each control frame's closed-glottis resonances, measured from this waveguide itself (its
impulse response with the glottis rigid, FFT, then the poles of U_lips / U_glottis by vector fitting), so they are
exact for what is rendered, discretisation and loss model included. F2-F4 agree with `tube.poles` of the same
geometry within 1 %; a closed vowel's broad F1 within about 3 % (the root of a non-rational response and a rational
fit of it differ there; `docs/considering.md`), so labels for scoring should use `tube.poles`, the training definition.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.signal import resample_poly

from . import tube
from .branches import PORT_CLOSED_CM2, Nasal, Subglottal, branched_poles, port_junction
from .interactive import KT
from .tube import C_CM_S, ETA, LAMBDA_OVER_CP, MU, RHO, WALL_B, WALL_K, WALL_M, TubeShape, _radiation

FS = 16000
HOP = 160
OVERSAMPLE = 8
LOSS_REF_HZ = 1000.0
LOSS_CORNERS_HZ = (100.0, 400.0, 1600.0, 6400.0)
LOSS_FIT_HZ = (60.0, 8000.0)
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


def _loss_eps(areas: np.ndarray, perimeter_factor: float, dx: float) -> np.ndarray:
    """Per-section, per-traversal attenuation alpha dx (nepers) from viscous and heat-conduction losses at
    LOSS_REF_HZ (`tube.chain_matrix`'s series and shunt terms, walls left out: they are modelled separately). It
    scales as sqrt(f) to within 0.1 % over 100 Hz - 7 kHz at any area."""
    s = 2j * np.pi * LOSS_REF_HZ
    perim = perimeter_factor * 2.0 * np.sqrt(np.pi * areas)
    z = s * RHO / areas + perim / areas**2 * np.sqrt(RHO * MU) * np.sqrt(s)
    y = s * areas / (RHO * C_CM_S**2) + perim * (ETA - 1) / (RHO * C_CM_S**2) * np.sqrt(LAMBDA_OVER_CP / RHO) * np.sqrt(s)
    return np.sqrt(z * y).real * dx


def _loss_template(oversample: int = OVERSAMPLE) -> tuple[np.ndarray, np.ndarray]:
    """(poles, weights): weights[0] the log gain and weights[1:] each shelf's zero offset, per neper at LOSS_REF_HZ.
    Shelf k is (1 - z_k / q) / (1 - p_k / q) with z_k = p_k + eps w_k; to first order in eps its log magnitude is
    eps w_k Re(-e^-jw / (1 - p_k e^-jw)), so the weights are a linear least-squares fit (log-spaced frequencies) of
    -sqrt(f / LOSS_REF_HZ)."""
    fs = FS * oversample
    f = np.geomspace(*LOSS_FIT_HZ, 400)
    e = np.exp(-2j * np.pi * f / fs)
    p = np.exp(-2.0 * np.pi * np.asarray(LOSS_CORNERS_HZ) / fs)
    basis = np.stack([np.ones_like(f)] + [np.real(-e / (1.0 - pk * e)) for pk in p], 1)
    w, *_ = np.linalg.lstsq(basis, -np.sqrt(f / LOSS_REF_HZ), rcond=None)
    return p, w


def _loss_filters(eps: np.ndarray, oversample: int = OVERSAMPLE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gain, zeros, poles) of each section's loss filter for attenuations eps (any shape): gain eps.shape, zeros
    eps.shape + (K,), poles (K,)."""
    p, w = _loss_template(oversample)
    return np.exp(w[0] * eps), p + eps[..., None] * w[1:], p


def _lip_filter(lip_area: float, fs: float) -> tuple[float, float, float]:
    """(b0, b1, a1) of the lip reflection p- = G(z) p+, G = (Zr - Z0) / (Zr + Z0), bilinear. G(DC) = -1 (open end)."""
    rr = 128.0 * RHO * C_CM_S / (9.0 * np.pi**2 * lip_area)
    lr = 8.0 * RHO / (3.0 * np.pi * np.sqrt(np.pi * lip_area))
    tau, z0, k = lr / rr, RHO * C_CM_S / lip_area, 2.0 * fs
    n0, n1 = tau * k * (rr - z0) - z0, -tau * k * (rr - z0) - z0
    d0, d1 = tau * k * (rr + z0) + z0, -tau * k * (rr + z0) + z0
    return n0 / d0, n1 / d0, d1 / d0


@njit(cache=True)
def _loss(x, c, i, d, loss_g, loss_z, loss_p, sx, sy):
    """One sample through section i's loss filter in direction d (control frame c): the shelves, then the gain."""
    for k in range(len(loss_p)):
        y = x - loss_z[c, i, k] * sx[d, i, k] + loss_p[k] * sy[d, i, k]
        sx[d, i, k], sy[d, i, k] = x, y
        x = y
    return x * loss_g[c, i]


@njit(cache=True)
def _run(areas, loss_g, loss_z, loss_p, lip, step, ag, p_sub, kv, wall, dt, dx, impulse, coupled,
         sg_areas, sg_g, sg_z, lung, nj, n_areas, n_g, n_z, n_lip, helm):
    """The waveguide at the internal rate (time step dt, section length dx). areas (n_ctrl, N) per control frame, linear
    in between over `step` samples; loss_g (n_ctrl, N), loss_z (n_ctrl, N, K), loss_p (K,): each section's
    per-traversal loss filter (`_loss_filters`), one state per direction; lip (n_ctrl, 3) reflection filter
    coefficients; ag glottal area per sample (cm^2); kv the viscous flow coefficient (12 mu d lg^2); wall (m, b, k, on)
    per unit area; impulse: glottis rigid, the volume velocity prescribed as ag (a unit impulse for the labels, flow
    noise for aspiration); coupled False: the
    flow from the glottis and subglottal pressure alone (no feedback from the tract; the control for interaction).

    Subglottal system (used when coupled and not impulse, and len(sg_areas) > 0): sections glottis first, loss filters
    sg_g (1, Ns), sg_z (1, Ns, K); the lungs a pressure source p_sub whose reflection is -1 at DC and -lung[0] above a
    one-pole crossover (coefficient lung[1]). Without it, p_sub acts directly below the glottis, as before.
    Nasal tract (nj >= 0): joined at oral junction nj by a three-way junction; n_areas (n_ctrl, Nn) port section first
    (its area the per-frame port opening), n_g / n_z its loss filters, n_lip (n_ctrl, 3) the nostrils' reflection
    filter; helm (junction, M, R, 1/C): the sinus resonator's neck volume velocity leaves at that nasal junction.
    Returns (lip volume velocity, glottal volume velocity, nostril volume velocity) per sample."""
    n_ctrl, N = areas.shape
    M = len(ag)
    K = len(loss_p)
    rc = RHO * C_CM_S
    fw, bw, nf, nb = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)  # fw arrives at a section's lip end, bw at its glottal end
    sx, sy = np.zeros((2, N, K)), np.zeros((2, N, K))
    wx, wv = np.zeros(N - 1), np.zeros(N - 1)  # wall displacement and velocity per junction
    a = np.empty(N)
    lf_x1, lf_y1 = 0.0, 0.0  # lip filter state: previous input and output
    Ns = len(sg_areas)
    has_sg = Ns > 0 and coupled and not impulse
    Ns1 = max(Ns, 1)
    tf, tb, ntf, ntb = np.zeros(Ns1), np.zeros(Ns1), np.zeros(Ns1), np.zeros(Ns1)  # tf down-going (lung end), tb up-going
    tsx, tsy = np.zeros((2, Ns1, K)), np.zeros((2, Ns1, K))
    lung_lp = 0.0
    has_n = nj >= 0
    Nn = n_areas.shape[1]
    gf, gb, ngf, ngb = np.zeros(Nn), np.zeros(Nn), np.zeros(Nn), np.zeros(Nn)  # gf away from the port, gb towards it
    gsx, gsy = np.zeros((2, Nn, K)), np.zeros((2, Nn, K))
    an = np.empty(Nn)
    nl_x1, nl_y1 = 0.0, 0.0
    h_at, h_m, h_r, h_cinv = int(helm[0]), helm[1], helm[2], helm[3]
    hu, hq = 0.0, 0.0  # sinus neck volume velocity and displaced volume
    u_lip, u_g, u_nose = np.zeros(M), np.zeros(M), np.zeros(M)
    wm, wb, wk, use_wall = wall[0], wall[1], wall[2], wall[3] > 0.0
    for t in range(M):
        c = t // step
        if c >= n_ctrl - 1:
            c, w = n_ctrl - 1, 0.0
        else:
            w = (t - c * step) / step
        for i in range(N):
            a[i] = areas[c, i] if w == 0.0 else (1.0 - w) * areas[c, i] + w * areas[c + 1, i]
        if has_n:
            for k in range(Nn):
                an[k] = n_areas[c, k] if w == 0.0 else (1.0 - w) * n_areas[c, k] + w * n_areas[c + 1, k]
        b0, b1, a1 = lip[c, 0], lip[c, 1], lip[c, 2]
        # glottis: KT rho U^2 / (2 Ag^2) + (kv / Ag^3 + rho c / A0 [+ rho c / At]) U = p_below - p_above, the pressures
        # written with the arriving waves: p_sub - 2 p- without a trachea, 2 a_t - 2 p- with one
        p_in = bw[0]
        a_t = tb[0]
        zt = rc / sg_areas[0] if has_sg else 0.0
        if impulse:
            ug = ag[t]
        else:
            g = ag[t]
            if not coupled:
                rhs = p_sub
            elif has_sg:
                rhs = 2.0 * a_t - 2.0 * p_in
            else:
                rhs = p_sub - 2.0 * p_in
            ug = 0.0
            if g > 1e-9 and rhs != 0.0:
                qa = KT * RHO / 2.0 / (g * g)
                qb = kv / (g * g * g) + (rc / a[0] + zt if coupled else 0.0)
                r = abs(rhs)
                ug = 2.0 * r / (qb + np.sqrt(qb * qb + 4.0 * qa * r))
                if rhs < 0.0:
                    ug = -ug
        u_g[t] = ug
        nf[0] = _loss(rc * ug / a[0] + p_in, c, 0, 0, loss_g, loss_z, loss_p, sx, sy)  # GL-3: a closed glottis reflects
        if has_sg:  # trachea: the wave launched down from the glottis, two-port junctions, the lungs
            ntf[0] = _loss(a_t - zt * ug, 0, 0, 0, sg_g, sg_z, loss_p, tsx, tsy)
            for i in range(Ns - 1):
                yi, yj = sg_areas[i], sg_areas[i + 1]
                p = 2.0 * (yi * tf[i] + yj * tb[i + 1]) / (yi + yj)
                ntf[i + 1] = _loss(p - tb[i + 1], 0, i + 1, 0, sg_g, sg_z, loss_p, tsx, tsy)
                ntb[i] = _loss(p - tf[i], 0, i, 1, sg_g, sg_z, loss_p, tsx, tsy)
            x = tf[Ns - 1]
            lung_lp += lung[1] * (x - lung_lp)
            ntb[Ns - 1] = _loss(p_sub - (lung[0] * x + (1.0 - lung[0]) * lung_lp), 0, Ns - 1, 1, sg_g, sg_z, loss_p,
                                tsx, tsy)
            for i in range(Ns):
                tf[i], tb[i] = ntf[i], ntb[i]
        # junction i: continuity Y_i (f - nb) = Y_j (nf - b) [+ Y_n (out - bn)] + u_wall, with one pressure P there
        for i in range(N - 1):
            yi, yj = a[i] / rc, a[i + 1] / rc
            uw = 0.0
            if use_wall:
                uw = wv[i] * 2.0 * np.sqrt(np.pi * 0.5 * (a[i] + a[i + 1])) * dx
            if has_n and i == nj:  # the velopharyngeal port
                yn = an[0] / rc
                p = (2.0 * (yi * fw[i] + yj * bw[i + 1] + yn * gb[0]) - uw) / (yi + yj + yn)
                ngf[0] = _loss(p - gb[0], c, 0, 0, n_g, n_z, loss_p, gsx, gsy)
            else:
                p = (2.0 * (yi * fw[i] + yj * bw[i + 1]) - uw) / (yi + yj)
            if use_wall:  # m dv/dt = P - b v - k x per unit area (semi-implicit Euler)
                wv[i] += dt * (p - wb * wv[i] - wk * wx[i]) / wm
                wx[i] += dt * wv[i]
            nf[i + 1] = _loss(p - bw[i + 1], c, i + 1, 0, loss_g, loss_z, loss_p, sx, sy)
            nb[i] = _loss(p - fw[i], c, i, 1, loss_g, loss_z, loss_p, sx, sy)
        # lips: the arriving wave through the reflection filter
        x0 = fw[N - 1]
        y0 = b0 * x0 + b1 * lf_x1 - a1 * lf_y1
        lf_x1, lf_y1 = x0, y0
        nb[N - 1] = _loss(y0, c, N - 1, 1, loss_g, loss_z, loss_p, sx, sy)
        u_lip[t] = a[N - 1] * (x0 - y0) / rc
        for i in range(N):
            fw[i], bw[i] = nf[i], nb[i]
        if has_n:  # nasal passage: two-port junctions (areas for admittances, so the sinus flow is scaled by rho c)
            for k in range(Nn - 1):
                yi, yj = an[k], an[k + 1]
                uh = hu * rc if k == h_at else 0.0
                p = (2.0 * (yi * gf[k] + yj * gb[k + 1]) - uh) / (yi + yj)
                if k == h_at:  # M dU/dt = P - R U - q / C, dq/dt = U (semi-implicit Euler)
                    hu += dt * (p - h_r * hu - h_cinv * hq) / h_m
                    hq += dt * hu
                ngf[k + 1] = _loss(p - gb[k + 1], c, k + 1, 0, n_g, n_z, loss_p, gsx, gsy)
                ngb[k] = _loss(p - gf[k], c, k, 1, n_g, n_z, loss_p, gsx, gsy)
            x0 = gf[Nn - 1]
            y0 = n_lip[c, 0] * x0 + n_lip[c, 1] * nl_x1 - n_lip[c, 2] * nl_y1
            nl_x1, nl_y1 = x0, y0
            ngb[Nn - 1] = _loss(y0, c, Nn - 1, 1, n_g, n_z, loss_p, gsx, gsy)
            u_nose[t] = an[Nn - 1] * (x0 - y0) / rc
            for k in range(Nn):
                gf[k], gb[k] = ngf[k], ngb[k]
    return u_lip, u_g, u_nose


def glottal_area(f0_hz: np.ndarray, n_samples: int, fs: float, peak_cm2: float, oq: float = 0.6, sq: float = 1.3,
                 gap: float = 0.0, closure: float = 1.0) -> np.ndarray:
    """Glottal area per sample (cm^2) from a per-frame f0 track (HOP frames at 16 kHz): Rosenberg's pulse
    (`interactive.area_pulse`) with open quotient oq, speed quotient sq and a closed-phase gap (fraction of the peak)."""
    from .interactive import area_pulse

    t_frames = (np.arange(len(f0_hz)) * HOP + 0.5 * HOP) / FS
    f0 = np.interp(np.arange(n_samples) / fs, t_frames, f0_hz)
    cycles = np.cumsum(f0 / fs)
    return peak_cm2 * area_pulse(cycles, oq, sq, gap=gap, closure=closure)


def n_sections(shapes: list[TubeShape], oversample: int = OVERSAMPLE) -> int:
    """The clip's section count: its mean length in sections (fixed within a clip)."""
    return max(2, int(round(np.mean([s.length_cm for s in shapes]) / section_length(oversample))))


def _controls(shapes: list[TubeShape], oversample: int, n: int | None = None
              ) -> tuple[np.ndarray, tuple, np.ndarray, float]:
    """Per control frame: areas on a common section count n (default `n_sections`), loss filters
    (`_loss_filters`), lip filters."""
    dx = section_length(oversample)
    n = n or n_sections(shapes, oversample)
    areas = np.empty((len(shapes), n))
    eps = np.empty((len(shapes), n))
    lip = np.empty((len(shapes), 3))
    for c, s in enumerate(shapes):
        ua = uniform_areas(s, oversample)
        areas[c] = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(ua)), ua) if len(ua) != n else ua
        areas[c] = np.maximum(areas[c], 1e-3)
        eps[c] = _loss_eps(areas[c], s.perimeter_factor, dx)
        lip[c] = _lip_filter(areas[c, -1], FS * oversample)
    return areas, _loss_filters(eps, oversample), lip, n * dx


def _no_branches(n_ctrl: int, oversample: int) -> tuple:
    """_run's subglottal and nasal arguments with both branches off."""
    k = len(LOSS_CORNERS_HZ)
    return (np.zeros(0), np.ones((1, 0)), np.zeros((1, 0, k)), np.zeros(2), -1, np.zeros((n_ctrl, 0)),
            np.ones((n_ctrl, 0)), np.zeros((n_ctrl, 0, k)), np.zeros((n_ctrl, 3)), np.array([-1.0, 1.0, 0.0, 0.0]))


def _sub_args(sub: Subglottal | None, oversample: int) -> tuple:
    if sub is None:
        return _no_branches(1, oversample)[:4]
    dx = section_length(oversample)
    if abs(sub.dx - dx) > 1e-9:
        raise ValueError("subglottal sections must be the waveguide's (branches.subglottal(..., kl.section_length()))")
    g, z, _ = _loss_filters(_loss_eps(sub.areas, sub.perimeter_factor, dx)[None, :], oversample)
    alpha = 1.0 - np.exp(-2.0 * np.pi * sub.lung_fc_hz / (FS * oversample))
    return sub.areas.astype(float), g, z, np.array([sub.lung_r, alpha])


def _nasal_args(nas: Nasal | None, nj: int, port_cm2: np.ndarray | None, n_ctrl: int, oversample: int) -> tuple:
    """Nasal arguments: n_areas (n_ctrl, 1 + len(nas.areas)) with the port section first (port_cm2 per frame)."""
    if nas is None:
        return _no_branches(n_ctrl, oversample)[4:]
    dx = section_length(oversample)
    if abs(nas.dx - dx) > 1e-9:
        raise ValueError("nasal sections must be the waveguide's (branches.nasal(..., kl.section_length()))")
    port = np.broadcast_to(np.asarray(port_cm2, float), (n_ctrl,))
    n_areas = np.concatenate([np.maximum(port, 0.0)[:, None], np.broadcast_to(nas.areas, (n_ctrl, len(nas.areas)))], 1)
    eps = _loss_eps(np.maximum(n_areas, 1e-3), nas.perimeter_factor, dx)
    g, z, _ = _loss_filters(eps, oversample)
    n_lip = np.broadcast_to(np.array(_lip_filter(nas.areas[-1], FS * oversample)), (n_ctrl, 3)).copy()
    m, r, cinv = nas.helmholtz()
    return int(nj), n_areas, g, z, n_lip, np.array([nas.sinus_at + 1.0, m, r, cinv])  # +1: the port section is first


def render(shapes: list[TubeShape], ag_cm2: np.ndarray, pressure: float, glottis_depth_cm: float, glottis_length_cm: float,
           oversample: int = OVERSAMPLE, walls: bool = True, coupled: bool = True, subglottal: Subglottal | None = None,
           nasal: Nasal | None = None, port_cm: float | None = None, port_cm2: np.ndarray | float = 0.0,
           n_sect: int | None = None, flow: np.ndarray | None = None) -> dict:
    """Synthesise through the waveguide. shapes: one TubeShape per HOP frame (areas move linearly between them);
    ag_cm2: glottal area per sample at FS; pressure: subglottal (lung) pressure, dyn/cm^2; coupled False: no feedback
    from the tract into the flow (the subglottal system is then unused). subglottal: a `branches.Subglottal` below the
    glottis. nasal: a `branches.Nasal` joined port_cm above the glottis (`branches.port_position_cm`), opened by
    port_cm2 (per frame or one value; 0 = velum raised). Returns 16 kHz `audio` (d/dt of the lips' plus the nostrils'
    volume velocity, unnormalised), `ug` (glottal volume velocity), the tube's `length_cm` and the port's junction `nj`
    (-1 without a nose). n_sect: the section count (default `n_sections`). flow: a prescribed glottal volume velocity
    per sample (cm^3/s) through a rigid glottis instead of the flow from ag_cm2 (aspiration and whisper sources).
    """
    areas, loss, lip, length = _controls(shapes, oversample, n_sect)
    fs_in = FS * oversample
    m = len(ag_cm2) * oversample
    ag = np.interp(np.arange(m) / fs_in, np.arange(len(ag_cm2)) / FS, ag_cm2 if flow is None else flow)
    kv = 12.0 * MU * glottis_depth_cm * glottis_length_cm**2
    wall = np.array([WALL_M, WALL_B, WALL_K, 1.0 if walls else 0.0])
    nj = -1 if nasal is None else port_junction(port_cm, areas.shape[1], section_length(oversample))
    u_lip, u_g, u_nose = _run(areas, *loss, lip, HOP * oversample, ag, float(pressure), kv, wall, 1.0 / fs_in,
                              section_length(oversample), flow is not None, coupled, *_sub_args(subglottal, oversample),
                              *_nasal_args(nasal, nj, port_cm2, len(shapes), oversample))
    rad = np.diff(u_lip + u_nose, prepend=0.0) * fs_in
    audio = resample_poly(rad, 1, oversample)[: len(ag_cm2)]
    return {"audio": audio.astype(np.float32), "ug": resample_poly(u_g, 1, oversample)[: len(ag_cm2)].astype(np.float32),
            "length_cm": length, "nj": nj}


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
                f_max_hz: float = LABEL_F_MAX_HZ, nasal: Nasal | None = None, port_cm: float | None = None,
                port_cm2: np.ndarray | float = 0.0, n_sect: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Closed-glottis resonances (F, B per frame, Hz) of the waveguide as `render` builds it: the impulse response of
    (U_lips + U_nostrils) / U_glottis with the glottis rigid and the frame's shape held, FFT, `_vector_fit` seeded by
    the continuous model of the same discretised tube (`tube.poles`; with the port open, every pole of
    `branches.branched_poles`, oral and nasal), so close pairs stay apart and none is missed. With the port open
    this returns all poles, unnumbered: the scoring labels are `frame_labels`. Frames with the same areas share one
    computation."""
    areas, loss, lip, _ = _controls(shapes, oversample, n_sect)
    fs_in = FS * oversample
    dx = section_length(oversample)
    wall = np.array([WALL_M, WALL_B, WALL_K, 1.0 if walls else 0.0])
    nj = -1 if nasal is None else port_junction(port_cm, areas.shape[1], dx)
    nargs = _nasal_args(nasal, nj, port_cm2, len(shapes), oversample)
    f = np.fft.rfftfreq(n_ir, 1.0 / fs_in)
    band = (f >= FIT_MIN_HZ) & (f <= f_max_hz + SEED_MARGIN_HZ)
    out, cache = [], {}
    for c in range(len(shapes)):
        port = float(nargs[1][c, 0]) if nasal is not None else 0.0
        key = (areas[c].tobytes(), port)
        if key not in cache:
            na = (nargs[0], nargs[1][c : c + 1], nargs[2][c : c + 1], nargs[3][c : c + 1], nargs[4][c : c + 1], nargs[5])
            u, _, un = _run(areas[c : c + 1], loss[0][c : c + 1], loss[1][c : c + 1], loss[2], lip[c : c + 1], n_ir,
                            np.r_[1.0, np.zeros(n_ir - 1)], 0.0, 0.0, wall, 1.0 / fs_in, dx, True, True,
                            *_sub_args(None, oversample), *na)
            oral = TubeShape(areas[c], np.full(len(areas[c]), dx), shapes[c].perimeter_factor)
            if nasal is not None and port > PORT_CLOSED_CM2:
                fo, bo, fn, bn, _ = branched_poles(oral, nj, port, nasal, f_max_hz + SEED_MARGIN_HZ)
                seed_f, seed_b = np.r_[fo, fn], np.r_[bo, bn]
            else:
                seed_f, seed_b = tube.poles(oral, f_max_hz + SEED_MARGIN_HZ)  # same discretised tube, continuous physics
            F, B = _vector_fit(np.fft.rfft(u + un)[band], f[band], seed_f, seed_b)
            cache[key] = (F[F <= f_max_hz], B[F <= f_max_hz])
        out.append(cache[key])
    return out


def frame_labels(shapes: list[TubeShape], oversample: int = OVERSAMPLE, f_max_hz: float = LABEL_F_MAX_HZ,
                 nasal: Nasal | None = None, port_cm: float | None = None, port_cm2: np.ndarray | float = 0.0,
                 n_sect: int | None = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Scoring labels per frame (`docs/considering.md`): F, B of the oral formants by the training definition,
    `tube.poles` of the same discretised tube, and with the port open `branches.branched_poles` (the oral roots
    followed as the port opens; the nose's own poles are not numbered), plus `mixed` per formant (an avoided crossing
    with a nasal pole: mask that label). The waveguide's measured poles (`frame_poles`) are the consistency check."""
    areas, _, _, _ = _controls(shapes, oversample, n_sect)
    dx = section_length(oversample)
    nj = -1 if nasal is None else port_junction(port_cm, areas.shape[1], dx)
    port = np.broadcast_to(np.asarray(port_cm2, float), (len(shapes),))
    out, cache = [], {}
    for c in range(len(shapes)):
        p = float(port[c]) if nasal is not None else 0.0
        key = (areas[c].tobytes(), p)
        if key not in cache:
            oral = TubeShape(areas[c], np.full(len(areas[c]), dx), shapes[c].perimeter_factor)
            if nasal is not None and p > PORT_CLOSED_CM2:
                F, B, _, _, mixed = branched_poles(oral, nj, p, nasal, f_max_hz)
            else:
                F, B = tube.poles(oral, f_max_hz)
                mixed = np.zeros(len(F), bool)
            cache[key] = (F, B, mixed)
        out.append(cache[key])
    return out
