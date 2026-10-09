"""Log-frequency heat-map axes, blurred targets and decoding (plan §7.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Axis:
    f_min: float
    n_bins: int
    cents_per_bin: float = 20.0

    @property
    def centres_hz(self) -> np.ndarray:
        return self.f_min * 2.0 ** (np.arange(self.n_bins) * self.cents_per_bin / 1200.0)

    @property
    def f_max(self) -> float:
        return float(self.centres_hz[-1])

    def to_bin(self, hz):
        """Fractional bin index (NaN for missing/non-positive)."""
        hz = np.asarray(hz, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(hz > 0, 1200.0 * np.log2(hz / self.f_min) / self.cents_per_bin, np.nan)

    def to_hz(self, b):
        return self.f_min * 2.0 ** (np.asarray(b, float) * self.cents_per_bin / 1200.0)


F0_AXIS = Axis(40.0, 320)  # 40-1594 Hz
FORMANT_AXIS = Axis(100.0, 368)  # 100-6938 Hz
TARGET_SIGMA_CENTS = 35.0
DECODE_HALF_WIDTH = 4  # bins either side of the mode (+-80 cents)


def gaussian_targets(torch, values_hz, axis: Axis, sigma_cents: float = TARGET_SIGMA_CENTS):
    """(…, n_bins) normalised Gaussian targets around each value; all-zero where the value is
    missing or outside the axis (those positions should also be masked out of the loss)."""
    b = torch.arange(axis.n_bins, device=values_hz.device, dtype=torch.float32)
    pos = 1200.0 * torch.log2(values_hz.clamp_min(1e-3) / axis.f_min) / axis.cents_per_bin
    sig = sigma_cents / axis.cents_per_bin
    g = torch.exp(-0.5 * ((b - pos[..., None]) / sig) ** 2)
    valid = torch.isfinite(values_hz) & (values_hz > 0) & (pos > -0.5) & (pos < axis.n_bins - 0.5)
    g = torch.where(valid[..., None], g, torch.zeros_like(g))  # NaN-safe (NaN * 0 would stay NaN)
    return g / g.sum(-1, keepdim=True).clamp_min(1e-12), valid


def decode(probs: np.ndarray, axis: Axis, half_width: int = DECODE_HALF_WIDTH,
           blur_cents: float = TARGET_SIGMA_CENTS) -> tuple[np.ndarray, np.ndarray]:
    """Local weighted mean around the mode -> (Hz, sigma in cents). probs: (..., n_bins).

    sigma is the RMS distance of the whole distribution from the decoded value, with the targets' own blur removed
    in quadrature (training predicts the posterior convolved with it): an uncertainty, near 0 for a confident
    estimate and large for a broad or multimodal one. It used to be measured within the +-80 cent window only,
    which capped it near 52 cents however unsure the model was.
    """
    p = np.asarray(probs, np.float64)
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    mode = p.argmax(-1)
    offs = np.arange(-half_width, half_width + 1)
    idx = np.clip(mode[..., None] + offs, 0, axis.n_bins - 1)
    w = np.take_along_axis(p, idx, -1)
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
    mean_bin = (w * idx).sum(-1)
    msd = (p * (np.arange(axis.n_bins) - mean_bin[..., None]) ** 2).sum(-1) * axis.cents_per_bin**2
    return axis.to_hz(mean_bin), np.sqrt(np.maximum(msd - blur_cents**2, 0.0))
