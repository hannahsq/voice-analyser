"""Lossy-tube vocal tract driven by Story's (2005) parametric area function (plan §5.1, preferred tract).

Area function (Story 2005, JASA 117(5), 3231-3254):
- Tier I, vowel substrate (Eq. 3): A(i) = pi/4 [Omega(i) + q1 phi1(i) + q2 phi2(i)]^2, i = 1..44 from
  glottis to lips, with Omega, phi1 and phi2 from Table III (one adult male, MRI; Story et al. 1996).
- Tier II, constriction (Eqs. 5-9): A(i) <- A(i) C(i), a skewed Gaussian dip of location lc, range rc,
  skew sc and target area ac. Used here for sonorant (approximant-like) constrictions only, so ac stays open.
- Tier III, length (Eqs. 10-13): tubelet lengths change near the glottis (larynx height, pg) and the lips
  (protrusion, pm), each spread over rg = rm sections with K = 2 ln(10 000).
Tier IV (nasal coupling) is not modelled here; the sampler's nasal pole-zero pairs stand in for it.

Acoustics: each tubelet is a lossy transmission line, glottis closed (volume-velocity source), with
- viscous and heat-conduction boundary-layer losses (Flanagan 1972), written as analytic functions of s
  so that complex roots can be found;
- yielding walls (Ishizaka, French & Flanagan 1975 soft-tissue values) as a shunt admittance;
- lip radiation as a piston in an infinite baffle, Z_r = s L_r R_r / (R_r + s L_r) (Sondhi & Schroeter 1987).
The transfer function U_lips / U_glottis = 1 / (C Z_r + D) for the chain matrix [[A, B], [C, D]] has unity
gain at DC, like the resonator cascade. Its poles s_n = -pi B_n + j 2 pi F_n (the resonator convention of
`filters.coefficients`) are found by Newton's method, continued from the lossless tube's real-axis roots
as the losses are switched on. They are exact roots of the tube's denominator, to ~1e-6 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Story (2005) Table III: neutral diameter function Omega(i) and modes phi1(i), phi2(i), cm, i = 1 (glottis)..44 (lips).
OMEGA = np.array([
    0.636, 0.561, 0.561, 0.550, 0.598, 0.895, 1.187, 1.417, 1.380, 1.273, 1.340, 1.399, 1.433, 1.506, 1.493,
    1.473, 1.499, 1.529, 1.567, 1.601, 1.591, 1.547, 1.570, 1.546, 1.532, 1.496, 1.429, 1.425, 1.496, 1.608,
    1.668, 1.757, 1.842, 1.983, 2.073, 2.123, 2.194, 2.175, 2.009, 1.785, 1.675, 1.539, 1.405, 1.312])
PHI1 = np.array([
    0.018, 0.001, -0.013, -0.025, -0.036, -0.048, -0.062, -0.076, -0.093, -0.111, -0.130, -0.149, -0.167, -0.183,
    -0.196, -0.204, -0.207, -0.203, -0.193, -0.175, -0.151, -0.119, -0.082, -0.041, 0.004, 0.051, 0.097, 0.141,
    0.181, 0.214, 0.240, 0.257, 0.264, 0.260, 0.246, 0.224, 0.194, 0.159, 0.122, 0.087, 0.057, 0.038, 0.034, 0.048])
PHI2 = np.array([
    -0.013, -0.007, -0.029, -0.059, -0.088, -0.108, -0.120, -0.123, -0.118, -0.107, -0.092, -0.075, -0.056, -0.035,
    -0.014, 0.008, 0.032, 0.057, 0.084, 0.111, 0.138, 0.164, 0.188, 0.206, 0.218, 0.221, 0.214, 0.195, 0.164,
    0.121, 0.070, 0.013, -0.046, -0.100, -0.143, -0.167, -0.165, -0.132, -0.066, 0.031, 0.148, 0.264, 0.346, 0.338])
N_SECTIONS = 44
STORY_TUBELET_CM = 0.396825  # Story's footnote 5: c / (2 Fs) with c = 35 000 cm/s, Fs = 44.1 kHz
STORY_LENGTH_CM = N_SECTIONS * STORY_TUBELET_CM  # 17.46 cm
PHARYNX_SECTIONS = 22  # Story's Fig. 2: the velopharyngeal port sits at about section 22 (8.7 cm)

# Story (2009) erratum to Table IV (JASA 125(2), 1248): [q1, q2] reconstructing his speaker's vowels with
# Eq. 3. The 2005 printing scrambled these values. tests/test_tube.py checks the vowel space they produce.
STORY_VOWELS = {
    "i": (-5.176, 0.981), "I": (-2.556, 1.043), "E": (-1.092, 1.331), "ae": (0.677, 2.315), "V": (2.561, 0.128),
    "A": (3.849, 1.356), "O": (3.471, -0.493), "U": (1.729, -1.91), "o": (0.102, -2.685), "u": (-3.565, -2.065),
}
# Sampling box: the extremes of Table IV, as in Story's Fig. 10(a).
Q1_RANGE = (min(q[0] for q in STORY_VOWELS.values()), max(q[0] for q in STORY_VOWELS.values()))
Q2_RANGE = (min(q[1] for q in STORY_VOWELS.values()), max(q[1] for q in STORY_VOWELS.values()))

# Physical constants, CGS (Flanagan 1972, Table 3.1; body-temperature moist air).
RHO = 1.14e-3  # g/cm^3
C_CM_S = 35000.0  # same as tract.C_CM_S
MU = 1.86e-4  # viscosity, poise
ETA = 1.4  # adiabatic constant
LAMBDA_OVER_CP = 5.5e-5 / 0.24  # thermal conductivity / specific heat, g/(cm s)
# Yielding walls (Ishizaka, French & Flanagan 1975): mass, resistance, stiffness per unit area.
WALL_M, WALL_B, WALL_K = 1.5, 1600.0, 3.0e5

F_SEARCH_MAX_HZ = 9000.0
GRID_HZ = 2.0


@dataclass
class TubeShape:
    areas: np.ndarray  # (N,) cm^2, glottis to lips
    lengths: np.ndarray  # (N,) cm per tubelet
    perimeter_factor: float = 1.0  # perimeter / circular perimeter (cross-section shape; scales wall and boundary losses)

    @property
    def length_cm(self) -> float:
        return float(self.lengths.sum())


# ----------------------------------------------------------------------------- Story's area-function tiers


def vowel_areas(q1: float, q2: float, omega: np.ndarray = OMEGA) -> np.ndarray:
    """Tier I (Eq. 3). Diameters below zero are clipped to zero before squaring (a mode sum can't fold a closed tract open)."""
    d = np.maximum(omega + q1 * PHI1 + q2 * PHI2, 0.0)
    return np.pi / 4.0 * d * d


def constriction(n: int, area_at_lc: float, lc: float, ac: float, rc: float, sc: float = 1.0, mc: float = 1.0) -> np.ndarray:
    """Tier II superposition function C(i) (Eqs. 5-9), i = 1..n; lc and rc in sections. Multiply the areas by it."""
    i = np.arange(1, n + 1, dtype=float)
    dc = 1.0 - ac / area_at_lc
    rcf = sc * rc / (1.0 + sc)  # downstream (lips) share of the range, Eq. 7
    rcb = rc / (1.0 + sc)  # upstream (glottis) share, Eq. 8
    r = np.where(i < lc, rcb, rcf)
    c = 1.0 - mc * dc * np.exp(-np.log(16.0) * ((i - lc) / r) ** 2)
    return np.maximum(c, 0.0)


def length_function(base: np.ndarray, pg: float = 0.0, pm: float = 0.0, rg: float = 8.0, rm: float = 8.0,
                    lg: float = 1.0, lm: float | None = None) -> np.ndarray:
    """Tier III (Eqs. 10-13) on a base length vector: adds pg cm near the glottis (larynx lowering > 0) and
    pm cm near the lips (protrusion > 0), each spread over about rg / rm sections. Story's denominator
    (Delta x sum of the Gaussian) is generalised to sum(base x Gaussian) so that unequal tubelets still
    gain exactly pg and pm cm."""
    n = len(base)
    i = np.arange(1, n + 1, dtype=float)
    lm = float(n) if lm is None else lm
    k = 2.0 * np.log(10000.0)
    eg = np.exp(-k * ((i - lg) / (2.0 * rg)) ** 2)
    em = np.exp(-k * ((i - lm) / (2.0 * rm)) ** 2)
    return base * (1.0 + pg * eg / np.sum(base * eg)) * (1.0 + pm * em / np.sum(base * em))


# ----------------------------------------------------------------------------- acoustics


def _chain_cd(s: np.ndarray, shape: TubeShape, loss: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """C and D of the glottis-to-lips chain matrix at complex frequencies s (rad/s), any shape.

    loss scales every loss term and the radiation load (0 = lossless tube, open at the lips).
    Sections use cosh(gl) and l sinhc(gl) forms, which are entire in gamma^2 = Z Y, so no branch choice is needed.
    """
    s = np.asarray(s, complex)
    rs = np.sqrt(s)  # principal branch; the cut (negative real axis) is far from the formant poles
    A, B = np.ones_like(s), np.zeros_like(s)  # running product [[A, B], [C, D]], rows for P and U at the glottis
    C, D = np.zeros_like(s), np.ones_like(s)
    for area, l in zip(shape.areas, shape.lengths):
        perim = shape.perimeter_factor * 2.0 * np.sqrt(np.pi * area)
        z = s * RHO / area + loss * perim / area**2 * np.sqrt(RHO * MU) * rs
        y = (s * area / (RHO * C_CM_S**2) + loss * perim * (ETA - 1) / (RHO * C_CM_S**2) * np.sqrt(LAMBDA_OVER_CP / RHO) * rs
             + loss * perim / (WALL_B + s * WALL_M + WALL_K / s))
        g2l2 = z * y * l * l
        g = np.sqrt(g2l2)
        ch = np.cosh(g)
        with np.errstate(invalid="ignore", divide="ignore"):
            shc = np.where(np.abs(g) > 1e-8, np.sinh(g) / g, 1.0 + g2l2 / 6.0)
        sb, sc = z * l * shc, y * l * shc  # section [[ch, sb], [sc, ch]]
        A, B, C, D = A * ch + B * sc, A * sb + B * ch, C * ch + D * sc, C * sb + D * ch
    return C, D


def _radiation(s: np.ndarray, lip_area: float) -> np.ndarray:
    rr = 128.0 * RHO * C_CM_S / (9.0 * np.pi**2 * lip_area)
    lr = 8.0 * RHO / (3.0 * np.pi * np.sqrt(np.pi * lip_area))
    return s * lr * rr / (rr + s * lr)


def _denominator(s: np.ndarray, shape: TubeShape, loss: float = 1.0) -> np.ndarray:
    C, D = _chain_cd(s, shape, loss)
    return C * loss * _radiation(s, shape.areas[-1]) + D


def transfer(f_hz: np.ndarray, shape: TubeShape) -> np.ndarray:
    """U_lips / U_glottis at real frequencies (Hz); 1 at DC."""
    s = 2j * np.pi * np.maximum(np.asarray(f_hz, float), 1e-3)
    return 1.0 / _denominator(s, shape)


def poles(shape: TubeShape, f_max_hz: float = F_SEARCH_MAX_HZ, n_loss_steps: int = 4, tol_hz: float = 1e-6
          ) -> tuple[np.ndarray, np.ndarray]:
    """Formant frequencies and bandwidths (Hz, ascending) of the lossy tube below f_max_hz.

    Lossless roots (real zero crossings of D(j w) with an open lip end) seed Newton's method, which follows
    each root as the losses and radiation load are ramped up to their full values. Intermediate steps only
    need to stay on their root (0.5 Hz); the last is solved to tol_hz. If a root is lost (two converge on
    the same root; heavily damped roots move far), the continuation is retried with twice as many steps, up
    to 64. Roots are returned sorted by frequency.
    """
    f = np.arange(GRID_HZ, f_max_hz + 400.0, GRID_HZ)
    d0 = _denominator(2j * np.pi * f, shape, 0.0).real
    idx = np.nonzero(np.sign(d0[:-1]) * np.sign(d0[1:]) < 0)[0]
    roots = f[idx] - d0[idx] * GRID_HZ / (d0[idx + 1] - d0[idx])  # linear interpolation
    steps = n_loss_steps
    while True:
        try:
            F, Bw = _continue_roots(shape, 2j * np.pi * roots, steps, tol_hz)
            break
        except RuntimeError:
            if steps >= 64:
                raise
            steps *= 2
    keep = F <= f_max_hz
    return F[keep], Bw[keep]


def _continue_roots(shape: TubeShape, s: np.ndarray, n_loss_steps: int, tol_hz: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(s)
    for loss in np.linspace(0.0, 1.0, n_loss_steps + 1)[1:]:
        tol = 2 * np.pi * (tol_hz if loss == 1.0 else 0.5)
        for _ in range(40):
            h = 1e-5 * np.abs(s)
            d = _denominator(np.concatenate([s, s + h, s - h]), shape, loss)  # one pass for value and derivative
            step = d[:n] / ((d[n : 2 * n] - d[2 * n :]) / (2 * h))
            s = s - step
            if np.all(np.abs(step) < tol):
                break
        else:
            raise RuntimeError(f"tube pole search did not converge at loss={loss:.2f}")
    F, Bw = s.imag / (2 * np.pi), -s.real / np.pi
    # A heavily damped root can pass a sharp one in frequency while staying far from it in the complex
    # plane, so frequency order is not preserved; losing track means two roots converged to the same one.
    gap = np.abs(s[:, None] - s[None, :]) + np.diag(np.full(n, np.inf))
    if np.any(Bw <= 0) or (n > 1 and gap.min() < 2 * np.pi * 1.0):
        raise RuntimeError("tube pole continuation lost track of a root")
    order = np.argsort(F)
    return F[order], Bw[order]


# ----------------------------------------------------------------------------- shape construction


def build_shape(q1: float, q2: float, length_cm: float, *, pharynx_ratio: float = 1.0, area_scale: float = 1.0,
                omega_warp: np.ndarray | None = None, constrict: dict | None = None, pg: float = 0.0, pm: float = 0.0,
                perimeter_factor: float = 1.0) -> TubeShape:
    """Story's tract scaled to length_cm (the neutral length, before pg / pm are applied).

    pharynx_ratio > 1 lengthens the pharynx (sections 1-22) relative to the oral cavity at a fixed total
    length; omega_warp multiplies Omega (another speaker's neutral shape); area_scale multiplies every area.
    """
    omega = OMEGA * (1.0 if omega_warp is None else omega_warp)
    areas = vowel_areas(q1, q2, omega) * area_scale
    if constrict:
        lc = constrict["lc"]
        a_lc = float(np.interp(lc, np.arange(1, N_SECTIONS + 1), areas))
        if constrict["ac"] < a_lc:
            areas = areas * constriction(N_SECTIONS, a_lc, lc, constrict["ac"], constrict["rc"], constrict.get("sc", 1.0))
    seg = np.where(np.arange(N_SECTIONS) < PHARYNX_SECTIONS, pharynx_ratio, 1.0)
    lengths = length_function(seg * length_cm / seg.sum(), pg, pm)
    return TubeShape(areas, lengths, perimeter_factor)


# ----------------------------------------------------------------------------- random speakers and targets


@dataclass
class TubeSpeaker:
    """Per-clip anatomy: Story's tract warped to another speaker and scaled to length_cm."""

    length_cm: float
    omega_warp: np.ndarray  # multiplies Omega
    pharynx_ratio: float
    area_scale: float
    perimeter_factor: float
    bw_scale: float  # glottal and other losses the closed-glottis tube leaves out


def sample_speaker(rng: np.random.Generator, length_cm: float) -> TubeSpeaker:
    """Shorter tracts get a proportionally shorter pharynx (Fitch & Giedd 1999), via (L / 17.46)^beta."""
    i = (np.arange(N_SECTIONS) + 0.5) / N_SECTIONS
    warp = np.exp(sum(rng.normal(0.0, 0.06) * np.cos(np.pi * k * i) for k in range(1, 5)))
    rel = length_cm / STORY_LENGTH_CM
    return TubeSpeaker(
        length_cm=length_cm, omega_warp=warp,
        pharynx_ratio=float(rel ** rng.uniform(0.0, 0.6) * np.exp(rng.normal(0.0, 0.05))),
        area_scale=float(rel**2 * np.exp(rng.uniform(np.log(0.75), np.log(1.33)))),
        perimeter_factor=float(np.exp(rng.uniform(0.0, np.log(1.8)))),
        bw_scale=float(np.exp(rng.uniform(0.0, np.log(2.0)))),
    )


def sample_target(rng: np.random.Generator, spk: TubeSpeaker, n_poles: int, f_max_hz: float, p_constrict: float = 0.15,
                  p_larynx: float = 0.3, p_lips: float = 0.3, min_area_cm2: float = 0.1) -> tuple[np.ndarray, np.ndarray, dict]:
    """One articulatory target: F1..F_n_poles and bandwidths (NaN above f_max_hz) plus the parameters drawn.

    q1, q2 are uniform over the Table IV box (Story's Fig. 10); shapes whose substrate narrows below min_area_cm2 (at Story's
    size) are redrawn, so the tract stays open (sonorant). Optional tier II constriction (approximant-like,
    0.15-0.8 cm^2 at Story's size) and tier III larynx / lip length changes.
    """
    rel = spk.length_cm / STORY_LENGTH_CM
    for _ in range(100):
        q1, q2 = rng.uniform(*Q1_RANGE), rng.uniform(*Q2_RANGE)
        if vowel_areas(q1, q2, OMEGA * spk.omega_warp).min() >= min_area_cm2:
            break
    else:
        raise RuntimeError("no open tract shape found")
    info: dict = {"q1": float(q1), "q2": float(q2)}
    constrict = None
    if rng.random() < p_constrict:
        constrict = {"lc": float(rng.uniform(14, 42)), "rc": float(rng.uniform(4, 12)),
                     "sc": float(np.exp(rng.uniform(np.log(1 / 3), np.log(3)))),
                     "ac": float(rng.uniform(0.15, 0.8) * spk.area_scale)}
        info["constrict"] = constrict
    pg = float(rng.uniform(-0.5, 0.5) * rel) if rng.random() < p_larynx else 0.0
    pm = float(rng.uniform(-0.3, 1.0) * rel) if rng.random() < p_lips else 0.0
    shape = build_shape(q1, q2, spk.length_cm, pharynx_ratio=spk.pharynx_ratio, area_scale=spk.area_scale,
                        omega_warp=spk.omega_warp, constrict=constrict, pg=pg, pm=pm, perimeter_factor=spk.perimeter_factor)
    f, b = poles(shape, f_max_hz)
    info.update(pg_cm=pg, pm_cm=pm, length_cm=shape.length_cm)
    freqs, bws = np.full(n_poles, np.nan), np.full(n_poles, np.nan)
    k = min(len(f), n_poles)
    freqs[:k], bws[:k] = f[:k], b[:k] * spk.bw_scale
    return freqs, bws, info
