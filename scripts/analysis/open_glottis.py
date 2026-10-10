"""Open-glottis resonances of the lossy tube (quasi-static): how far each formant and bandwidth moves from its
closed-glottis value when the glottis is open with area Ag, with and without a subglottal system.

Source side, in Norton form at the glottis: a volume-velocity source in parallel with Zs = Zg + Zsub, so the tube's
denominator (C Zr + D) becomes (C Zr + D) + (A Zr + B) / Zs (closed glottis: Zs -> inf).
- Zg: glottal slit (Flanagan 1972; Ishizaka & Flanagan 1972): kinetic resistance linearised about the mean flow,
  rho U / Ag^2 with U = Ag sqrt(2 dP / rho), plus viscous 12 mu d lg^2 / Ag^3, plus inertance rho d / Ag.
- Zsub: parallel resonances at the subglottal impedance maxima of Ishizaka, Matsudaira & Kaneko (1976): 640, 1400,
  2100 Hz, magnitudes within their 2.5-5 MPa s m^-3, bandwidths ~150-300 Hz; scaled by tract length for women.
Roots are continued from the closed-glottis poles as the coupling is ramped in, so each stays the same formant.
Args: out file. Run from the m3 worktree.
"""
import sys

import numpy as np

from vocal_analysis.synth import tube
from vocal_analysis.synth.tube import C_CM_S, ETA, LAMBDA_OVER_CP, MU, RHO, WALL_B, WALL_K, WALL_M

SPEAKERS = {  # tract length, area scale, pharynx ratio (the population fit's typical speakers), glottis length & depth (cm)
    "men": dict(L=17.5, area=1.0, phar=1.0, lg=1.4, d=0.3, sub=1.0),
    "women": dict(L=15.0, area=(15.0 / 17.46) ** 2, phar=(15.0 / 17.46) ** 0.3, lg=1.0, d=0.2, sub=17.5 / 15.0),
}
SUB = [(640.0, 45.0, 160.0), (1400.0, 35.0, 200.0), (2100.0, 25.0, 300.0)]  # Hz, peak |Z| (cgs ohm), bandwidth Hz
DP = 8000.0  # transglottal pressure, dyn/cm^2 (800 Pa)


def chain(s, shape):
    rs = np.sqrt(s)
    A, B, C, D = np.ones_like(s), np.zeros_like(s), np.zeros_like(s), np.ones_like(s)
    for area, l in zip(shape.areas, shape.lengths):
        perim = shape.perimeter_factor * 2.0 * np.sqrt(np.pi * area)
        z = s * RHO / area + perim / area**2 * np.sqrt(RHO * MU) * rs
        y = (s * area / (RHO * C_CM_S**2) + perim * (ETA - 1) / (RHO * C_CM_S**2) * np.sqrt(LAMBDA_OVER_CP / RHO) * rs
             + perim / (WALL_B + s * WALL_M + WALL_K / s))
        g2l2 = z * y * l * l
        g = np.sqrt(g2l2)
        ch = np.cosh(g)
        shc = np.where(np.abs(g) > 1e-8, np.sinh(g) / np.where(g == 0, 1, g), 1.0 + g2l2 / 6.0)
        sb, sc = z * l * shc, y * l * shc
        A, B, C, D = A * ch + B * sc, A * sb + B * ch, C * ch + D * sc, C * sb + D * ch
    return A, B, C, D


def z_source(s, ag, spk, flow=True, sub=True):
    zg = 12 * MU * spk["d"] * spk["lg"] ** 2 / ag**3 + s * RHO * spk["d"] / ag
    if flow:
        zg = zg + RHO * np.sqrt(2 * DP / RHO) / ag
    if sub:
        for f, r, b in SUB:
            w0, bw = 2 * np.pi * f * spk["sub"], 2 * np.pi * b * spk["sub"]
            zg = zg + r * bw * s / (s * s + bw * s + w0 * w0)
    return zg


def den(s, shape, ag, spk, alpha, **kw):
    A, B, C, D = chain(s, shape)
    zr = tube._radiation(s, shape.areas[-1])
    return (C * zr + D) + alpha * (A * zr + B) / z_source(s, ag, spk, **kw)


def open_poles(shape, ag, spk, n=3, steps=40, **kw):
    F, Bw = tube.poles(shape)
    s = -np.pi * Bw[:n] + 2j * np.pi * F[:n]
    for alpha in np.linspace(0, 1, steps + 1)[1:]:
        for _ in range(50):
            h = 1e-5 * np.abs(s)
            d = den(np.concatenate([s, s + h, s - h]), shape, ag, spk, alpha, **kw)
            step = d[:n] / ((d[n:2 * n] - d[2 * n:]) / (2 * h))
            s = s - step
            if np.all(np.abs(step) < 2 * np.pi * 0.01):
                break
    return (F[:n], Bw[:n]), (s.imag / (2 * np.pi), -s.real / np.pi)


if __name__ == "__main__":
    out = open(sys.argv[1], "w", encoding="utf-8")
    p = lambda *a: (print(*a, file=out), print(*a))  # noqa: E731
    p("Open-glottis (quasi-static) resonances of the lossy tube vs closed glottis. dF in %, B in Hz (closed -> open).")
    p(f"Transglottal pressure {DP / 10:.0f} Pa; Zsub = Ishizaka et al. 1976 maxima (scaled for women).\n")
    # check against Barney et al. 2007: static 51 mm2 opening, 17 x 17 mm duct, no flow -> fR1 +13 %
    duct = tube.TubeShape(np.full(44, 2.89), np.full(44, 17.5 / 44))
    for name, kw in (("no flow, no subglottal", dict(flow=False, sub=False)), ("no flow, subglottal", dict(flow=False))):
        (F, _), (Fo, _) = open_poles(duct, 0.51, SPEAKERS["men"], **kw)
        p(f"uniform 2.89 cm2 duct, 17.5 cm, static Ag 0.51 cm2 ({name}): F1-F3 shift "
          + " ".join(f"{100 * (b / a - 1):+.1f} %" for a, b in zip(F, Fo)) + "   [Barney et al. 2007: F1 +13 %]")
    for variant, kw in (("glottis + subglottal", {}), ("glottis only", dict(sub=False))):
        for sex, spk in SPEAKERS.items():
            p(f"\n{variant}, {sex}")
            p(f"{'vowel':6s}{'fR1':>6s}{'fR2':>6s}{'fR3':>6s} | " + " | ".join(
                f"Ag {ag:.2f}: dF1  dF2  dF3   B1      B2      B3     " for ag in (0.05, 0.1, 0.2)))
            for v, (q1, q2) in tube.STORY_VOWELS.items():
                shape = tube.build_shape(q1, q2, spk["L"], pharynx_ratio=spk["phar"], area_scale=spk["area"])
                cells = []
                for ag in (0.05, 0.1, 0.2):
                    (F, B), (Fo, Bo) = open_poles(shape, ag, spk, **kw)
                    d = 100 * (Fo / F - 1)
                    cells.append(f"        {d[0]:+4.1f} {d[1]:+4.1f} {d[2]:+4.1f}  " + " ".join(
                        f"{b:3.0f}>{bo:3.0f}" for b, bo in zip(B, Bo)))
                p(f"{v:6s}{F[0]:6.0f}{F[1]:6.0f}{F[2]:6.0f} | " + " | ".join(cells))
