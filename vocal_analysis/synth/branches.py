"""Side branches of the vocal tract for the Kelly-Lochbaum waveguide (`kl.py`): the subglottal system below the glottis
and the nasal tract off the velopharyngeal port, with a continuous-model counterpart for the labels.

Subglottal: a trachea (TRACHEA_CM at TRACHEA_AREA_CM2) and a bronchial horn (BRONCHI_CM, area widening
BRONCHI_EXPANSION-fold, standing in for the branching tree's growing total cross-section) ending in the lungs, a
pressure-release end whose reflection loses LUNG_R above LUNG_FC_HZ but passes DC whole (the lung pressure drives the
flow). The geometry is calibrated, not measured: at Story's speaker's size it gives subglottal resonances of about 620,
1390 and 2210 Hz (bandwidths about 100, 270, 280), the adult men's values reported by Chi & Sonderegger (2007) and
Lulich (2010); it scales with the speaker's tract length (women's are about 15 % higher in those reports, as their
tracts are shorter). Cartilaginous: no yielding walls. The subglottal system only reaches the output through the open
glottis, so it leaves the closed-glottis labels (D-023) unchanged.

Nasal: a tube from the port to the nostrils with the maxillary sinuses as one Helmholtz resonator part way along. The
port opens at the pharynx / oral boundary (Story's section PHARYNX_SECTIONS of 44), which the speaker's pharynx ratio
and larynx height already move, plus PORT_JITTER_CM per speaker (the port is a 1-2 cm opening, not a point;
`docs/considering.md`). PROVISIONAL: the nasal area function is a smooth parametric profile with ranges set from the
general magnitudes in the MRI literature (Dang, Honda & Suzuki 1994: a passage about 11 cm long, the two sides'
combined area largest in the mid nasal cavity, narrowing at the nostrils; convoluted turbinates, so large perimeter
and losses), not from tabulated values; individual variation is drawn wider than the sex difference. Port area is a
per-frame control (0 = velum raised). Bony: no yielding walls.

Labels with the port open (`branched_poles`): F1-F4 are the oral tract's poles with the nasal load, followed by
continuation from the closed port as the port opens to its area; the nose's own poles are returned separately and are
not numbered. The continuous model is `tube.chain_matrix`'s, on the same uniform sections the waveguide uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tube import C_CM_S, PHARYNX_SECTIONS, RHO, STORY_LENGTH_CM, TubeShape, _radiation, chain_matrix, poles

TRACHEA_CM = 9.0
TRACHEA_AREA_CM2 = 2.5
BRONCHI_CM = 12.0
BRONCHI_EXPANSION = 10.0
SUBGLOTTAL_PERIMETER = 2.0  # boundary losses relative to a smooth circular tube (bronchial walls and branching)
LUNG_R = 0.3
LUNG_FC_HZ = 50.0

NASAL_CM = 11.0
PORT_JITTER_CM = 0.5
PORT_CLOSED_CM2 = 1e-6  # continuation starts here: even a near-degenerate oral / nasal pair has barely split
SINUS_Q = 6.0


def _size(length_cm: float) -> float:
    return length_cm / STORY_LENGTH_CM


@dataclass
class Subglottal:
    areas: np.ndarray  # (N,) cm^2 on sections of dx, glottis first
    dx: float
    perimeter_factor: float = SUBGLOTTAL_PERIMETER
    lung_r: float = LUNG_R
    lung_fc_hz: float = LUNG_FC_HZ


def subglottal(length_cm: float, dx: float, rng: np.random.Generator | None = None) -> Subglottal:
    """The subglottal system for a speaker whose neutral tract is length_cm, scaled with it (lengths by size, areas by
    size^2), with per-speaker jitter (length sd 5 %, area sd 10 %) when rng is given."""
    k = _size(length_cm)
    lk, ak = (np.exp(rng.normal(0.0, 0.05)), np.exp(rng.normal(0.0, 0.1))) if rng is not None else (1.0, 1.0)
    nt, nb = max(1, round(TRACHEA_CM * k * lk / dx)), max(1, round(BRONCHI_CM * k * lk / dx))
    a0 = TRACHEA_AREA_CM2 * k * k * ak
    return Subglottal(np.r_[np.full(nt, a0), a0 * BRONCHI_EXPANSION ** (np.arange(1, nb + 1) / nb)], dx)


def lung_reflection(f_hz: np.ndarray, sub: Subglottal) -> np.ndarray:
    """Pressure reflection at the lung end: -1 at DC, -lung_r above lung_fc_hz (one-pole crossover)."""
    lp = 1.0 / (1.0 + 1j * np.asarray(f_hz) / sub.lung_fc_hz)
    return -(sub.lung_r + (1.0 - sub.lung_r) * lp)


def subglottal_impedance(f_hz: np.ndarray, sub: Subglottal) -> np.ndarray:
    """Input impedance looking down from the glottis (dyn s / cm^5)."""
    s = 2j * np.pi * np.asarray(f_hz, float)
    A, B, C, D = chain_matrix(s, TubeShape(sub.areas, np.full(len(sub.areas), sub.dx), sub.perimeter_factor),
                              walls=False)
    R = lung_reflection(f_hz, sub)
    zl = RHO * C_CM_S / sub.areas[-1] * (1 + R) / (1 - R)
    return (A * zl + B) / (C * zl + D)


def subglottal_resonances(sub: Subglottal, f_max_hz: float = 3500.0, grid_hz: float = 2.0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Peaks of the subglottal input impedance and their -3 dB widths (Hz): Sg1, Sg2, ..."""
    f = np.arange(50.0, f_max_hz, grid_hz)
    m = np.abs(subglottal_impedance(f, sub))
    idx = np.nonzero((m[1:-1] > m[:-2]) & (m[1:-1] > m[2:]))[0] + 1
    F, B = [], []
    for j in idx:
        half = m[j] / np.sqrt(2.0)
        lo, hi = j, j
        while lo > 0 and m[lo] > half:
            lo -= 1
        while hi < len(m) - 1 and m[hi] > half:
            hi += 1
        F.append(f[j])
        B.append(f[hi] - f[lo])
    return np.array(F), np.array(B)


@dataclass
class Nasal:
    areas: np.ndarray  # (N,) cm^2 on sections of dx, from just above the port to the nostrils
    dx: float
    perimeter_factor: float
    sinus_at: int  # the Helmholtz resonator joins after this section
    sinus_volume_cm3: float
    sinus_neck_cm2: float
    sinus_neck_cm: float

    def helmholtz(self) -> tuple[float, float, float]:
        """(M, R, 1/C) of the sinus resonator per unit volume velocity: neck inertance with end corrections, a
        resistance giving Q = SINUS_Q, cavity compliance V / (rho c^2)."""
        r = np.sqrt(self.sinus_neck_cm2 / np.pi)
        m = RHO * (self.sinus_neck_cm + 1.7 * r) / self.sinus_neck_cm2
        cinv = RHO * C_CM_S**2 / self.sinus_volume_cm3
        return m, np.sqrt(cinv * m) / SINUS_Q, cinv

    def sinus_hz(self) -> float:
        m, _, cinv = self.helmholtz()
        return float(np.sqrt(cinv / m) / (2 * np.pi))


def nasal(length_cm: float, dx: float, rng: np.random.Generator | None = None) -> Nasal:
    """A nasal tract for a speaker whose neutral oral tract is length_cm (PROVISIONAL ranges, module docstring):
    nasopharynx ~2 cm^2 rising to ~4 cm^2 (both sides) in the mid cavity, ~1.2 cm^2 at the nostrils; perimeter 2-4x a
    circle's; sinus 10-30 cm^3 through a 0.03-0.3 cm^2 ostium. rng None gives the central values."""
    k = _size(length_cm)
    if rng is None:
        lk, a_np, a_c, a_n, pf, vol, neck = 1.0, 2.0, 4.0, 1.2, 3.0, 20.0, 0.1
    else:
        lk = np.exp(rng.normal(0.0, 0.07))
        a_np, a_c, a_n = (v * np.exp(rng.uniform(np.log(0.7), np.log(1.4))) for v in (2.0, 4.0, 1.2))
        pf = rng.uniform(2.0, 4.0)
        vol, neck = rng.uniform(10.0, 30.0), np.exp(rng.uniform(np.log(0.03), np.log(0.3)))
    n = max(4, round(NASAL_CM * k * lk / dx))
    x = (np.arange(n) + 0.5) / n
    prof = np.interp(x, [0.0, 0.25, 0.55, 0.85, 1.0], [a_np, a_np, a_c, a_c * 0.6, a_n])
    return Nasal(prof * k * k, dx, pf, int(0.5 * n), vol * k**3, neck * k * k, 0.5 * k)


def port_position_cm(shape: TubeShape, rng: np.random.Generator | None = None) -> float:
    """Distance of the velopharyngeal port above the glottis: the end of Story's pharyngeal sections in this shape
    (moved by the pharynx ratio and larynx height), plus a per-speaker jitter of sd PORT_JITTER_CM / 2."""
    x = float(np.sum(shape.lengths[:PHARYNX_SECTIONS]))
    if rng is not None:
        x += float(np.clip(rng.normal(0.0, PORT_JITTER_CM / 2), -PORT_JITTER_CM, PORT_JITTER_CM))
    return x


def port_junction(port_cm: float, n_sections: int, dx: float) -> int:
    """The waveguide junction (between sections j and j + 1, at (j + 1) dx) nearest the port."""
    return int(np.clip(round(port_cm / dx) - 1, 1, n_sections - 3))


# ----------------------------------------------------------------------------- continuous model


def _mul(a, b):
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3])


def _nasal_load(s: np.ndarray, nas: Nasal, port_cm2: float | None, loss: float) -> tuple[np.ndarray, np.ndarray]:
    """(num, den) of the nasal tract's input admittance at the port, Y = num / den, both entire in s: the port
    section (port_cm2, one dx; None leaves it out), the passage, the sinus shunt, the nostrils' radiation load."""
    head = nas.areas[: nas.sinus_at + 1] if port_cm2 is None else np.r_[port_cm2, nas.areas[: nas.sinus_at + 1]]
    first = TubeShape(head, np.full(len(head), nas.dx), nas.perimeter_factor)
    rest = TubeShape(nas.areas[nas.sinus_at + 1:], np.full(len(nas.areas) - nas.sinus_at - 1, nas.dx),
                     nas.perimeter_factor)
    m, r, cinv = nas.helmholtz()
    zh = loss * r + s * m + cinv / s  # shunt [[1, 0], [1 / zh, 1]], scaled by zh to stay entire
    one = np.ones_like(s)
    M = _mul(_mul(chain_matrix(s, first, loss, walls=False), (zh, 0 * s, one, zh)),
             chain_matrix(s, rest, loss, walls=False))
    zr = loss * _radiation(s, nas.areas[-1])
    return M[2] * zr + M[3], M[0] * zr + M[1]


def branched_denominator(s: np.ndarray, oral: TubeShape, j: int, port_cm2: float, nas: Nasal, loss: float = 1.0
                         ) -> np.ndarray:
    """Characteristic function of the oral tract (uniform sections, glottis closed) with the nasal tract joined at
    junction j: its zeros are the coupled system's poles (oral and nasal). Entire in s."""
    s = np.asarray(s, complex)
    lo = TubeShape(oral.areas[: j + 1], oral.lengths[: j + 1], oral.perimeter_factor)
    hi = TubeShape(oral.areas[j + 1:], oral.lengths[j + 1:], oral.perimeter_factor)
    num, den = _nasal_load(s, nas, port_cm2, loss)
    A, B, C, D = _mul(_mul(chain_matrix(s, lo, loss), (den, 0 * s, num, den)), chain_matrix(s, hi, loss))
    return C * loss * _radiation(s, oral.areas[-1]) + D


def _newton(fun, s: np.ndarray, tol_hz: float, max_iter: int = 60) -> np.ndarray:
    n = len(s)
    for _ in range(max_iter):
        h = 1e-6 * np.abs(s)
        d = fun(np.concatenate([s, s + h, s - h]))
        step = d[:n] / ((d[n: 2 * n] - d[2 * n:]) / (2 * h))
        big = np.abs(step) > 0.2 * np.abs(s)
        step[big] *= 0.2 * np.abs(s[big]) / np.abs(step[big])  # damped
        s = s - step
        if np.all(np.abs(step) < 2 * np.pi * tol_hz):
            return s
    raise RuntimeError("branched pole search did not converge")


def _nasal_seeds(nas: Nasal, f_max_hz: float) -> np.ndarray:
    """The nose's own poles with the port closed (the passage closed at its port end, open at the nostrils, with the
    sinus): zeros of its input admittance's numerator, from minima of |num| on a real-frequency grid refined by
    Newton's method."""
    f = np.arange(40.0, f_max_hz + 400.0, 4.0)
    q = np.abs(_nasal_load(2j * np.pi * f, nas, None, 1.0)[0])
    idx = np.nonzero((q[1:-1] < q[:-2]) & (q[1:-1] < q[2:]))[0] + 1
    out = []
    for z in -np.pi * 100.0 + 2j * np.pi * f[idx]:
        try:
            r = _newton(lambda x: _nasal_load(x, nas, None, 1.0)[0], np.array([z]), 1e-6)[0]
        except RuntimeError:
            continue
        if r.real < 0 and all(abs(r - o) > 2 * np.pi * 1.0 for o in out):
            out.append(r)
    return np.array(out, complex)


def _continue_port(s: np.ndarray, oral: TubeShape, j: int, nas: Nasal, port_cm2: float, n_steps: int, tol_hz: float,
                   strict: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Follow the roots s as the port opens from PORT_CLOSED_CM2 to port_cm2 (log-spaced). A step is accepted only if
    Newton converges, every root stays stable and distinct, and (strict) identities are kept both ways: each new root
    is nearest its own old root and each old root nearest its own new one (two roots near an avoided crossing come close
    without meeting, so a fixed fraction of their gap would stall the steps); otherwise the step is halved (down to
    2^-16 of the range). strict False accepts identity swaps and reports them. -> (roots, swapped per root)."""
    lo, hi = np.log(PORT_CLOSED_CM2), np.log(port_cm2)
    t, h = 0.0, 1.0 / n_steps
    idx = np.arange(len(s))
    swapped = np.zeros(len(s), bool)
    while t < 1.0:
        h = min(h, 1.0 - t)
        a = float(np.exp(lo + (t + h) * (hi - lo)))
        try:
            new = _newton(lambda x: branched_denominator(x, oral, j, a, nas), s, tol_hz if t + h >= 1.0 else 0.5)
            d = np.abs(new[:, None] - s[None, :])
            keeps = (d.argmin(1) == idx) & (d.argmin(0) == idx)
            gap = np.abs(new[:, None] - new[None, :]) + np.diag(np.full(len(s), np.inf))
            ok = np.all(new.real < 0) and (len(s) < 2 or gap.min() > 2 * np.pi * 0.5) and (np.all(keeps) or not strict)
        except RuntimeError:
            ok = False
        if ok:
            swapped |= ~keeps
            s, t, h = new, t + h, h * 1.5
        else:
            h /= 2
            if h < 2.0**-16:
                raise RuntimeError("branched continuation lost track of a root")
    return s, swapped


ORAL_CLEAR = 0.6  # a label's root must carry at least this share of its energy in the oral tract,
SHAPE_CLEAR = 0.8  # its oral field must match the port-closed formant's this well (MAC),
LABEL_MARGIN = 0.2  # and its score (MAC x oral share) must beat the label's next best root by this; else mixed


def _section_mats(s: complex, areas: np.ndarray, dx: float, pf: float, walls: bool) -> list:
    return [chain_matrix(np.array([s]), TubeShape(np.array([a]), np.array([dx]), pf), walls=walls) for a in areas]


def _field(mats: list, areas: np.ndarray, dx: float, p: complex, u: complex) -> tuple[np.ndarray, complex, complex]:
    """A mode's field through a run of sections from (p, u) at the near end, each section's [[A, B], [C, D]] mapping
    its far end to its near end (so the far end is the inverse: AD - BC = 1). -> (per section the energy-normalised
    pair [p sqrt(A dx / (rho c^2)), u sqrt(rho dx / A)], whose squared norm is the acoustic energy; p and u at the far
    end)."""
    out = np.empty((len(areas), 2), complex)
    for n, ((A, B, C, D), a) in enumerate(zip(mats, areas)):
        out[n] = p * np.sqrt(a * dx / (RHO * C_CM_S**2)), u * np.sqrt(RHO * dx / a)
        p, u = complex(D[0] * p - B[0] * u), complex(-C[0] * p + A[0] * u)
    return out, p, u


def mode(s: complex, oral: TubeShape, j: int, port_cm2: float, nas: Nasal) -> tuple[float, np.ndarray]:
    """A coupled mode at complex frequency s: the closed-glottis field (p = 1, u = 0) carried section by section to the
    port, split there by the nasal input admittance (none with port_cm2 0) and on through both tubes. -> (share of
    its energy in the oral tract, the oral tract's energy-normalised field, flattened)."""
    dx = float(oral.lengths[0])
    a = np.asarray(oral.areas, float)
    lo, p, u = _field(_section_mats(s, a[: j + 1], dx, oral.perimeter_factor, True), a[: j + 1], dx, 1.0 + 0j, 0j)
    if port_cm2 > 0:
        num, den = _nasal_load(np.array([s]), nas, port_cm2, 1.0)
        un = complex(num[0] / den[0] * p)
    else:
        un = 0j
    hi, _, _ = _field(_section_mats(s, a[j + 1:], dx, oral.perimeter_factor, True), a[j + 1:], dx, p, u - un)
    vec = np.r_[lo.ravel(), hi.ravel()]
    eo = float(np.sum(np.abs(vec) ** 2))
    en = 0.0
    if port_cm2 > 0:
        na = np.r_[port_cm2, nas.areas]
        m, r, cinv = nas.helmholtz()
        head = nas.sinus_at + 2  # the port section and the passage up to the sinus
        n1, pn, qn = _field(_section_mats(s, na[:head], nas.dx, nas.perimeter_factor, False), na[:head], nas.dx, p, un)
        qn -= pn / (r + s * m + cinv / s)
        n2, _, _ = _field(_section_mats(s, na[head:], nas.dx, nas.perimeter_factor, False), na[head:], nas.dx, pn, qn)
        en = float(np.sum(np.abs(n1) ** 2) + np.sum(np.abs(n2) ** 2))
    return eo / (eo + en), vec


def oral_fraction(s: complex, oral: TubeShape, j: int, port_cm2: float, nas: Nasal) -> float:
    """Share of a coupled mode's energy in the oral tract (`mode`)."""
    return mode(s, oral, j, port_cm2, nas)[0]


def branched_poles(oral: TubeShape, j: int, port_cm2: float, nas: Nasal, f_max_hz: float = 5000.0,
                   n_steps: int = 24, tol_hz: float = 1e-4
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(F, B) of the oral formants with the nasal load, (F, B) of the nose's own poles (Hz), and `mixed` per oral
    formant: True when its identity is a convention, so a scored label should be masked.

    Continuation finds every root but its identities are arbitrary where an oral and a nasal root interact: they cross
    (an avoided crossing), start degenerate (Newton may follow either), or exchange character as the port opens
    (nasalised /u/'s F2 continues to a root with 21 % of its energy in the oral tract while a 98 % oral one sits beside
    it). So identity is by mode shape (the modal assurance criterion): each root's field along the oral tract is
    compared, in an energy norm, with each port-closed formant's (`mode`), scored MAC x oral energy share, and labels
    assigned to roots by maximum total score (Hungarian). A label is mixed unless its root's oral share is at least
    ORAL_CLEAR, its MAC at least SHAPE_CLEAR, and its score LABEL_MARGIN above its next best root's.

    Each oral root of the port-closed tract (`tube.poles`) and each nasal root (the nose with the port all but closed)
    is followed by Newton's method as the port opens geometrically from PORT_CLOSED_CM2 to port_cm2; a root keeps the
    identity it started with, so where an oral and a nasal pole meet (an avoided crossing) the label follows the
    branch continuous with the oral one. Oral F are returned in their port-closed order."""
    fo, bo = poles(oral, f_max_hz + 1000.0)
    so = -np.pi * bo + 2j * np.pi * fo
    sn = _nasal_seeds(nas, f_max_hz + 1000.0) if port_cm2 > PORT_CLOSED_CM2 else np.zeros(0, complex)
    s = np.r_[so, sn]
    no = len(so)
    swapped = np.zeros(len(s), bool)
    if port_cm2 > PORT_CLOSED_CM2:
        try:
            s, swapped = _continue_port(s, oral, j, nas, port_cm2, n_steps, tol_hz)
        except RuntimeError:
            s, swapped = _continue_port(s, oral, j, nas, port_cm2, n_steps, tol_hz, strict=False)
    F, B = s.imag / (2 * np.pi), -s.real / np.pi
    mixed = np.zeros(no, bool)
    if port_cm2 > PORT_CLOSED_CM2 and len(s) > no:
        from scipy.optimize import linear_sum_assignment

        ref = [mode(z, oral, j, 0.0, nas)[1] for z in so]
        modes = [mode(z, oral, j, port_cm2, nas) for z in s]
        phi = np.array([m[0] for m in modes])
        mac = np.array([[abs(np.vdot(r, m[1])) ** 2 / (np.vdot(r, r).real * np.vdot(m[1], m[1]).real) for m in modes]
                        for r in ref])
        score = mac * phi[None, :]
        rows, cols = linear_sum_assignment(-score)
        pick = np.empty(no, int)
        pick[rows] = cols
        rest = np.setdiff1d(np.arange(len(s)), pick)
        for i in range(no):
            c = pick[i]
            runner = np.max(np.delete(score[i], c)) if len(s) > 1 else 0.0
            mixed[i] = phi[c] < ORAL_CLEAR or mac[i, c] < SHAPE_CLEAR or score[i, c] - runner < LABEL_MARGIN
        rest = rest[np.argsort(s[rest].imag)]
        s = np.r_[s[pick], s[rest]]
        F, B = s.imag / (2 * np.pi), -s.real / np.pi
        down = np.nonzero(np.diff(F[:no]) <= 0)[0]  # labels out of frequency order: neither is clear
        mixed[down] = mixed[down + 1] = True
    fn = F[no:]
    keep_o, keep_n = F[:no] <= f_max_hz, fn <= f_max_hz
    return F[:no][keep_o], B[:no][keep_o], fn[keep_n], B[no:][keep_n], mixed[keep_o]
