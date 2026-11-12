"""M4's reconstruction loss (plan section 5.3): a differentiable cascade renderer compared with the observed harmonics.

Renderer: a cascade of digital two-pole resonators at 16 kHz with unit gain at 0 Hz (the synthesiser's own sections,
`synth.filters`), for F1-F4 and nuisance poles, divided by nuisance anti-resonators (zeros), times the radiation
(a first difference). Its magnitude is rendered in dB at the frame's harmonic frequencies k f0.

Targets: the harmonic amplitudes of the audio at k f0, from a least-squares harmonic fit around each frame
(`harmonic_targets`, offline, numpy). The renderer is always compared at the harmonics, never with an estimated
envelope, at every f0: D-101 found WORLD's CheapTrick envelope misplacing F1 by about a semitone at 150-250 Hz,
which is where envelope estimation fails, and at low f0 harmonics are dense enough that the two agree anyway.

Source: the glottal source's level and spectral slope are not predicted; they are profiled out per frame (the
weighted least-squares offset and slope in log frequency of target minus render, removed in closed form), so
they cannot be traded against F1 by the network. `profile="quadratic"` also removes a curvature term (the source's
H1-H2 bend) for an ablation. The open-phase coupling is not rendered: the reconstruction's F1 is the acoustic (whole-
cycle) F1, which D-091 made the reported F1. `considering.md`'s gate on renders with known labels still applies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FS = 16000
HOP = 160
LN10 = float(np.log(10.0))


# --------------------------------------------------------------------------------------------- renderer (torch)
def resonator_db(f: torch.Tensor, freq: torch.Tensor, bw: torch.Tensor, fs: float = FS) -> torch.Tensor:
    """20 log10 |H(f)| of a unit-DC-gain two-pole resonator (synth.filters.coefficients); broadcasts f against
    freq / bw. An anti-resonator (zero) is its negative."""
    r = torch.exp(-np.pi * bw / fs)
    b1 = 2.0 * r * torch.cos(2.0 * np.pi * freq / fs)
    b2 = -r * r
    a = 1.0 - b1 - b2
    w = 2.0 * np.pi * f / fs
    re = 1.0 - b1 * torch.cos(w) - b2 * torch.cos(2.0 * w)
    im = b1 * torch.sin(w) + b2 * torch.sin(2.0 * w)
    return 10.0 * (torch.log(a * a) - torch.log(re * re + im * im + 1e-20)) / LN10


def radiation_db(f: torch.Tensor, fs: float = FS) -> torch.Tensor:
    """20 log10 |1 - e^{-j w}|: the first difference that stands for lip radiation"""
    return 10.0 * torch.log(2.0 - 2.0 * torch.cos(2.0 * np.pi * f / fs) + 1e-20) / LN10


_LF_DB: dict = {}
RD_SEARCH = np.round(np.arange(0.3, 2.7 + 1e-9, 0.05), 2)


def lf_source_db(rd: torch.Tensor, k_max: int) -> torch.Tensor:
    """The LF flow derivative's harmonic levels in dB (re Ee = 1) for harmonics 1..k_max at Rd (any shape), linear in
    Rd on synth.glottal's 0.01 grid, so differentiable in rd (D-103). Shape rd.shape + (k_max,)."""
    from ..synth import glottal

    key = (k_max, rd.dtype, rd.device)
    if key not in _LF_DB:
        logmag, _ = glottal.harmonic_table()
        _LF_DB[key] = torch.tensor(20.0 * logmag[:, :k_max] / LN10, dtype=rd.dtype, device=rd.device)
    tab = _LF_DB[key]
    pos = (rd.clamp(glottal.RD_MIN, glottal.RD_MAX) - glottal.RD_MIN) / 0.01
    i0 = pos.floor().clamp(0, tab.shape[0] - 2)
    w = (pos - i0).unsqueeze(-1)
    i0 = i0.long()
    return (1 - w) * tab[i0] + w * tab[i0 + 1]


def render_db(f: torch.Tensor, poles_f: torch.Tensor, poles_b: torch.Tensor, zeros_f: torch.Tensor | None = None,
              zeros_b: torch.Tensor | None = None, radiation: bool = True) -> torch.Tensor:
    """Cascade magnitude in dB at f (..., K). poles_f / poles_b (..., P) and zeros (..., Z) share f's leading dims;
    NaN sections are skipped."""
    def sections(F, B):
        ok = torch.isfinite(F) & torch.isfinite(B)
        d = resonator_db(f.unsqueeze(-1), torch.where(ok, F, 1000.0).unsqueeze(-2), torch.where(ok, B, 100.0).unsqueeze(-2))
        return (d * ok.unsqueeze(-2)).sum(-1)

    out = sections(poles_f, poles_b)
    if zeros_f is not None:
        out = out - sections(zeros_f, zeros_b)
    if radiation:
        out = out + radiation_db(f)
    return out


# --------------------------------------------------------------------------------------------- targets (numpy)
def harmonic_targets(audio: np.ndarray, f0: np.ndarray, voiced: np.ndarray, fmax: float = 5000.0, k_max: int = 64,
                     min_win_s: float = 0.025, periods: float = 3.0, floor_db: float = 60.0,
                     snr_min_db: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per 10 ms frame (frame i centred at i * HOP): the amplitudes in dB of harmonics k f0 (k = 1.. up to fmax and
    k_max), by a Hann-weighted least-squares fit of cosines and sines at k f0 over max(min_win_s, periods / f0).
    Returns (amp_db (T, k_max), valid (T, k_max)); invalid where unvoiced, f0 missing, k f0 > fmax, or more than
    floor_db below the frame's strongest harmonic (the noise floor). f0 is a fixed estimate (the labeller's), not the
    model's, so the target doesn't move with the model.

    snr_min_db (D-104, amendment 1): also invalid where the harmonic's power a_k^2 is less than this far above the local
    noise: the mean power of the four nearest sinusoids at half-integer multiples (k +- 0.5, k +- 1.5) f0, fitted jointly
    with the harmonics in the same window, so the noise is measured by the same estimator. Aspiration noise
    otherwise sets the level of weak high harmonics (D-103: up to +22 dB at 4-5 kHz at low HNR)."""
    x = np.asarray(audio, np.float64)
    T = len(f0)
    amp = np.full((T, k_max), np.nan)
    for i in range(T):
        f = f0[i]
        if not (voiced[i] and np.isfinite(f) and f > 0):
            continue
        K = int(min(k_max, np.floor(fmax / f)))
        if K < 1:
            continue
        half = int(round(max(min_win_s, periods / f) * FS / 2))
        c = i * HOP
        a, b = c - half, c + half + 1
        if a < 0 or b > len(x):
            continue
        n = np.arange(a, b) - c
        w = np.hanning(b - a)
        ph = 2 * np.pi * f * n[:, None] * np.arange(1, K + 1)[None] / FS
        A = np.concatenate([np.cos(ph), np.sin(ph)], 1) * w[:, None]
        coef = np.linalg.lstsq(A, x[a:b] * w, rcond=None)[0]
        mag = np.hypot(coef[:K], coef[K:])
        amp[i, :K] = 20 * np.log10(mag + 1e-12)
        if snr_min_db is not None:  # noise: sinusoids at (k + 1/2) f0, fitted jointly with the harmonics in the same window
            kh = np.arange(K + 1) + 0.5
            phh = 2 * np.pi * f * n[:, None] * kh[None] / FS
            Aj = np.concatenate([A, np.cos(phh) * w[:, None], np.sin(phh) * w[:, None]], 1)
            cj = np.linalg.lstsq(Aj, x[a:b] * w, rcond=None)[0]
            bh = cj[2 * K:2 * K + K + 1] ** 2 + cj[2 * K + K + 1:] ** 2  # (K + 1,) noise powers at 0.5 .. K + 0.5
            nz = np.array([bh[max(0, j - 1):j + 3].mean() for j in range(K)])  # the four nearest half-integers
            snr = 10 * np.log10(mag ** 2 + 1e-30) - 10 * np.log10(nz + 1e-30)
            amp[i, :K] = np.where(snr >= snr_min_db, amp[i, :K], np.nan)
    valid = np.isfinite(amp)
    peak = np.nanmax(np.where(valid, amp, -np.inf), 1, keepdims=True)
    valid &= amp >= peak - floor_db
    return np.where(valid, amp, 0.0).astype(np.float32), valid


# --------------------------------------------------------------------------------------------- loss
def profile_out(resid: torch.Tensor, f: torch.Tensor, w: torch.Tensor, profile: str = "tilt") -> torch.Tensor:
    """resid (..., K) minus its weighted least-squares fit by an offset (+ slope, + curvature) in log2 f per frame;
    w (..., K) weights (0 = ignore). The fitted terms stand for the source's level and spectral shape."""
    lf = torch.log2(f.clamp_min(1.0)) - 10.0  # centred near 1 kHz, for conditioning
    cols = [torch.ones_like(lf)]
    if profile in ("tilt", "quadratic"):
        cols.append(lf)
    if profile == "quadratic":
        cols.append(lf * lf)
    X = torch.stack(cols, -1)  # (..., K, p)
    Xw = X * w.unsqueeze(-1)
    G = Xw.transpose(-1, -2) @ X
    ridge = 1e-9 * G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)  # only keeps empty frames solvable
    G = G + ridge[..., None, None] * torch.eye(X.shape[-1], dtype=X.dtype, device=X.device)
    beta = torch.linalg.solve(G, (Xw.transpose(-1, -2) @ resid.unsqueeze(-1))).squeeze(-1)
    return resid - (X @ beta.unsqueeze(-1)).squeeze(-1)


def reconstruction_loss(target_db: torch.Tensor, valid: torch.Tensor, f0: torch.Tensor, poles_f: torch.Tensor,
                        poles_b: torch.Tensor, zeros_f: torch.Tensor | None = None, zeros_b: torch.Tensor | None = None,
                        frame_mask: torch.Tensor | None = None, profile: str = "tilt", delta_db: float = 6.0,
                        min_harmonics: int = 4, radiation: bool = True, source: str = "profile",
                        rd: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Huber loss (dB) between the observed harmonics and the render sampled at k f0, after profiling the source out.

    target_db / valid (B, T, K) from `harmonic_targets`; f0 (B, T) the same fixed f0 the targets used; poles / zeros
    (B, T, P) / (B, T, Z) in Hz; frame_mask (B, T) e.g. sonorant frames. Frames with fewer than min_harmonics valid
    harmonics carry no loss (too few points to profile a slope out). Returns (mean over frames, per-frame loss).

    source "profile": the source is the profiled offset / slope / curvature (`profile`). source "rd" (D-103): the LF
    flow derivative at Rd is added to the render, the radiation is off (the derivative carries it), and `profile`
    ("offset" or "tilt") is profiled on top; rd (B, T) given, or None to search RD_SEARCH per frame and keep each
    frame's best (the gradient then flows through the chosen Rd's loss)."""
    K = target_db.shape[-1]
    f = f0.unsqueeze(-1) * torch.arange(1, K + 1, dtype=f0.dtype, device=f0.device)
    w = valid.to(target_db.dtype)
    n = w.sum(-1)

    def frame_loss(src_db):
        resid = profile_out((target_db - rend - src_db) * w, f, w, profile) * w
        per = torch.nn.functional.huber_loss(resid, torch.zeros_like(resid), reduction="none", delta=delta_db)
        return (per * w).sum(-1) / n.clamp_min(1.0)

    if source == "rd":
        rend = render_db(torch.where(valid, f, torch.full_like(f, 1000.0)), poles_f, poles_b, zeros_f, zeros_b, False)
        if rd is not None:
            per_frame = frame_loss(lf_source_db(rd, K))
        else:
            cand = torch.stack([frame_loss(lf_source_db(torch.full_like(f0, float(r)), K)) for r in RD_SEARCH], 0)
            per_frame = cand.min(0).values
    else:
        rend = render_db(torch.where(valid, f, torch.full_like(f, 1000.0)), poles_f, poles_b, zeros_f, zeros_b, radiation)
        per_frame = frame_loss(torch.zeros_like(rend))
    use = n >= min_harmonics
    if frame_mask is not None:
        use = use & frame_mask.bool()
    loss = (per_frame * use).sum() / use.sum().clamp_min(1)
    return loss, torch.where(use, per_frame, torch.full_like(per_frame, float("nan")))


# --------------------------------------------------------------------------------------------- nuisance head
class NuisanceHead(nn.Module):
    """Per-frame nuisance sections for the renderer only (plan section 7.4): n_poles poles and n_zeros zeros, each a
    frequency inside its own band (sigmoid in log frequency) and a bandwidth in [bw_min, bw_max] (sigmoid in log).
    Zeros are confined to 250-4000 Hz (SoFilNet's zeros drifted to the top of the band without a limit,
    `considering.md`); the poles to 200-7500 Hz. Never read by any other head."""

    def __init__(self, dim: int, n_poles: int = 2, n_zeros: int = 2, pole_band=(200.0, 7500.0),
                 zero_band=(250.0, 4000.0), bw_band=(40.0, 1000.0)):
        super().__init__()
        self.n_poles, self.n_zeros = n_poles, n_zeros
        self.bands = {"pole": pole_band, "zero": zero_band}
        self.bw_band = bw_band
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 2 * (n_poles + n_zeros)))

    @staticmethod
    def _squash(x, lo, hi):
        return torch.exp(np.log(lo) + torch.sigmoid(x) * (np.log(hi) - np.log(lo)))

    def forward(self, h: torch.Tensor) -> dict:
        y = self.proj(h)
        p, z = self.n_poles, self.n_zeros
        out = {"pole_f": self._squash(y[..., :p], *self.bands["pole"]),
               "pole_b": self._squash(y[..., p:2 * p], *self.bw_band)}
        if z:
            out["zero_f"] = self._squash(y[..., 2 * p:2 * p + z], *self.bands["zero"])
            out["zero_b"] = self._squash(y[..., 2 * p + z:], *self.bw_band)
        return out
