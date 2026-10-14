"""Per-formant observability (D-021 item 3, D-027): how much the audio's evidence can say about each formant.

For one frame, an analyst sees the log spectrum at the harmonics (voiced) or on a dense grid (noise-excited frames),
each sample with an uncertainty set by its signal-to-noise ratio. The spectrum is modelled as

    L(f) = E(log2 f) + sum_k R(f; F_k, B_k)        (dB)

with R the all-pole resonance pair (unit gain at 0 Hz) and E a smooth envelope (a cosine series in log frequency,
n_env terms over 50 Hz - f_max: level, tilt and broad source / radiation / higher-pole shape). E and every bandwidth
B_k are unknown to the analyst (nuisance parameters), so the information they absorb is not credited to the formants;
in particular a formant's far skirts, which an exact all-pole model would let distant harmonics read, are mostly
absorbed by E, leaving the evidence near the peak. With
weak priors (formant frequency SD a fraction of F, log-bandwidth SD, level and tilt), the posterior covariance is the
inverse Fisher information plus prior precision (a Cramer-Rao / Laplace bound). A sample's sensitivity to the
parameters is scaled by snr / (1 + snr): the analyst sees signal plus noise, so a sample under the noise floor says
nothing. Observability of formant k, from the posterior SD in cents,

    o_k = 1 / (1 + (sd_cents_k / ref_cents)^2)     in (0, 1],

is 0.5 when the evidence pins the formant to ref_cents (100: about a semitone), near 1 when well determined, near 0
when unobservable (above a channel's cut-off, under the noise floor, F1 between sparse harmonics at a high f0). The four D-021 cases fall out of
one calculation: harmonic spacing sets how many samples there are, SNR their weight, a nasal zero or a low-pass their
level, and two close formants share information (the joint bound). The bound is what an ideal analyst could reach,
not what a given model does; the model's own sigma should approach sqrt(var_post) where it is efficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LN10_20 = 20.0 / np.log(10.0)


@dataclass
class ObservabilityConfig:
    model_db: float = 2.0  # irreducible dB scatter per sample (source fine structure beyond level and tilt, jitter)
    ref_cents: float = 100.0  # posterior SD at which o = 0.5
    prior_f_rel: float = 0.35  # formant frequency prior SD, as a fraction of F (about half an octave)
    prior_log_bw: float = 0.7  # natural-log bandwidth prior SD
    n_env: int = 6  # smooth-envelope nuisance terms (cosines over log2 f; the first is the level)
    prior_env_db: float = 30.0
    f_max_hz: float = 7600.0  # samples above this are not used (anti-alias region at 16 kHz)
    noise_grid_hz: float = 40.0  # sample spacing on noise-excited frames (1 / 25 ms window)
    noise_sample_db: float = 5.6  # periodogram scatter of one noise-excited sample (chi-squared, 2 dof)
    # temporal pooling: a frame also counts its neighbours' information about each formant, discounted by
    # exp(-|dt| / context_s) (formants move slowly while f0 sweeps the harmonics across them); 0 = frame alone
    context_s: float = 0.05
    hop_s: float = 0.010


def resonance_db(f: np.ndarray, F: np.ndarray, B: np.ndarray) -> np.ndarray:
    """dB gain of the all-pole pair at F with bandwidth B (unit gain at 0 Hz); broadcasts."""
    p = -np.pi * B + 2j * np.pi * F
    s = 2j * np.pi * f
    return 10.0 * np.log10(np.abs(p) ** 4 / (np.abs(s - p) ** 2 * np.abs(s - np.conj(p)) ** 2))


def frame_bound(freqs: np.ndarray, snr_db: np.ndarray, sample_db_sd: np.ndarray, formants: np.ndarray,
                bandwidths: np.ndarray, cfg: ObservabilityConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One frame. freqs: sample frequencies (Hz); snr_db: signal over noise there; sample_db_sd: their dB SD;
    formants / bandwidths: (K,) with NaN for absent formants (left out). Returns (o, sd_hz): observability and
    posterior SD per formant (NaN where absent)."""
    cfg = cfg or ObservabilityConfig()
    K = len(formants)
    o, sd = np.full(K, np.nan), np.full(K, np.nan)
    present = np.isfinite(formants) & np.isfinite(bandwidths) & (formants > 0)
    idx = np.flatnonzero(present)
    if not len(idx):
        return o, sd
    F, B = formants[idx], bandwidths[idx]
    keep = (freqs > 0) & (freqs < cfg.f_max_hz) & np.isfinite(sample_db_sd) & np.isfinite(snr_db)
    rho = 10 ** (snr_db[keep] / 10.0)
    f, w = freqs[keep], (rho / (1.0 + rho)) ** 2 / sample_db_sd[keep] ** 2
    n = len(idx)
    m = cfg.n_env
    prior_var = np.concatenate([np.full(m, cfg.prior_env_db ** 2), (cfg.prior_f_rel * F) ** 2,
                                np.full(n, cfg.prior_log_bw ** 2)])
    J = np.diag(1.0 / prior_var)
    if len(f):
        G = np.empty((len(f), m + 2 * n))
        lo, hi = np.log2(50.0), np.log2(cfg.f_max_hz)
        x = (np.log2(f) - lo) / (hi - lo)
        for j in range(m):
            G[:, j] = np.cos(np.pi * j * x)
        for j in range(n):  # central differences: d/dF (Hz), d/d ln B
            dF = 0.5
            G[:, m + j] = (resonance_db(f, F[j] + dF, B[j]) - resonance_db(f, F[j] - dF, B[j])) / (2 * dF)
            dl = 1e-3
            G[:, m + n + j] = (resonance_db(f, F[j], B[j] * np.exp(dl)) - resonance_db(f, F[j], B[j] * np.exp(-dl))) / (2 * dl)
        J = J + (G * w[:, None]).T @ G
    cov = np.linalg.inv(J)
    var_f = np.diag(cov)[m : m + n]
    sd[idx] = np.sqrt(var_f)
    cents = 1200.0 / np.log(2.0) * sd[idx] / F
    o[idx] = 1.0 / (1.0 + (cents / cfg.ref_cents) ** 2)
    return o, sd


def sample_sd_db(snr_db: np.ndarray, cfg: ObservabilityConfig, base_db: float | None = None) -> np.ndarray:
    """dB SD of a spectral sample at the given SNR (signal over noise power at that frequency): the irreducible part
    (cfg.model_db, or base_db) plus additive-noise scatter 20 / ln 10 / sqrt(2 (1 + snr)) (the noise's own scatter
    when the sample is mostly noise)."""
    base = cfg.model_db if base_db is None else base_db
    noise = LN10_20 / np.sqrt(2.0 * (1.0 + 10 ** (np.asarray(snr_db, float) / 10.0)))
    return np.sqrt(base ** 2 + noise ** 2)


def observability(f0: np.ndarray, voiced: np.ndarray, sonorant: np.ndarray, formants: np.ndarray,
                  bandwidths: np.ndarray, snr_db_at, cfg: ObservabilityConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frames (T,), formants / bandwidths (T, K). snr_db_at(t, freqs) -> SNR (dB) of the frame's tract-filtered
    excitation over everything else (aspiration and frication on voiced frames, channel noise) at those frequencies;
    it carries the channel (a low-pass makes the SNR above the cut-off fall). Voiced frames are sampled at the
    harmonics; sonorant unvoiced (whispered) frames on a noise_grid_hz grid; other frames are NaN (absent, not
    unobservable). Returns (o, sd_hz), each (T, K)."""
    cfg = cfg or ObservabilityConfig()
    T, K = formants.shape
    o, sd = np.full((T, K), np.nan), np.full((T, K), np.nan)
    grid = np.arange(cfg.noise_grid_hz, cfg.f_max_hz, cfg.noise_grid_hz)
    for t in range(T):
        if voiced[t] and np.isfinite(f0[t]) and f0[t] > 0:
            freqs = f0[t] * np.arange(1, int(cfg.f_max_hz // f0[t]) + 1)
            snr = np.asarray(snr_db_at(t, freqs), float)
            sdb = sample_sd_db(snr, cfg)
        elif sonorant[t]:
            freqs = grid
            snr = np.asarray(snr_db_at(t, freqs), float)
            sdb = sample_sd_db(snr, cfg, base_db=np.hypot(cfg.model_db, cfg.noise_sample_db))
        else:
            continue
        o[t], sd[t] = frame_bound(freqs, snr, sdb, formants[t], bandwidths[t], cfg)
    if cfg.context_s > 0:
        # information beyond the prior, per frame and formant (frames without a value add nothing), pooled with
        # neighbours of the same sonorant run
        prior_info = 1.0 / (cfg.prior_f_rel * formants) ** 2
        gain = np.where(np.isfinite(sd), np.maximum(1.0 / sd ** 2 - prior_info, 0.0), 0.0)
        r = int(np.ceil(4 * cfg.context_s / cfg.hop_s))
        lag = np.arange(-r, r + 1)
        kern = np.exp(-np.abs(lag) * cfg.hop_s / cfg.context_s)
        run = np.cumsum(np.r_[0, np.diff(sonorant.astype(int)) != 0])
        pooled = np.zeros_like(gain)
        for j, kw in zip(lag, kern):
            src = np.clip(np.arange(T) + j, 0, T - 1)
            same = (run[src] == run) & (np.arange(T) + j >= 0) & (np.arange(T) + j < T)
            pooled += np.where(same[:, None], kw * gain[src], 0.0)
        has = np.isfinite(sd)
        sd = np.where(has, 1.0 / np.sqrt(prior_info + pooled), np.nan)
        cents = 1200.0 / np.log(2.0) * sd / formants
        o = np.where(has, 1.0 / (1.0 + (cents / cfg.ref_cents) ** 2), np.nan)
    return o, sd


def component_snr(comps: dict[str, np.ndarray], frame_t: np.ndarray, voiced: np.ndarray, fs: float = 16000.0,
                  win_s: float = 0.025, n_fft: int = 1024, floor_db: float = 60.0, smooth_bins: int = 3):
    """snr_db_at(t, freqs) for `observability`, from a clip's rendered components (`voice.synthesize` /
    `interactive.synthesize`). Per label frame (centred at frame_t[t] s, Hann window win_s): on voiced frames the
    signal is the voiced component, read at the spectral peak within one bin of each frequency, and the noise is
    everything else (aspiration, whisper, frication), smoothed over +-smooth_bins; on unvoiced frames the signal is
    the tract-filtered noise (aspiration, whisper) and the noise frication. An analysis floor floor_db below the
    frame's strongest bin is added to the noise (dynamic range; no channel). The channel's own gain and noise are
    applied later, at training time, where the channel is drawn."""
    n = len(comps["total"])
    w = int(round(win_s * fs))
    win = np.hanning(w)
    k = np.arange(-(w // 2), w - w // 2)
    centres = np.clip(np.round(frame_t * fs).astype(int), 0, n - 1)
    idx = np.clip(centres[:, None] + k[None, :], 0, n - 1)

    def power(x):
        return np.abs(np.fft.rfft(x[idx] * win, n_fft, axis=1)) ** 2

    zero = np.zeros(n)
    V = power(comps.get("voiced", zero))
    tract_noise = power(comps.get("aspiration", zero) + comps.get("whisper", zero))
    Fr = power(comps.get("frication", zero))
    sig = np.where(voiced[:, None], V, tract_noise)
    noise = np.where(voiced[:, None], tract_noise + Fr, Fr + V)
    kern = np.ones(2 * smooth_bins + 1) / (2 * smooth_bins + 1)
    noise = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, noise)
    peak = np.maximum(np.max(sig + noise, axis=1, keepdims=True), 1e-30)
    noise = noise + peak * 10 ** (-floor_db / 10)
    sig_peak = np.maximum(np.maximum(sig, np.roll(sig, 1, axis=1)), np.roll(sig, -1, axis=1))
    sig = np.where(voiced[:, None], sig_peak, sig)
    snr = 10 * np.log10(np.maximum(sig, 1e-30) / noise)
    bin_hz = fs / n_fft

    def snr_db_at(t, freqs):
        return np.interp(freqs / bin_hz, np.arange(snr.shape[1]), snr[t])

    return snr_db_at
