"""Teacher v1 (plan §7.2, §7.4): offline, bidirectional.

Stage 1 (f0 net) predicts an f0 heat-map and voicing from the VQT. A harmonic comb is
rendered from that f0 (plan §7.1) and stacked with the VQT for stage 2, which predicts
F1-F4 as cascaded heat-maps (head n sees the expected log-frequencies of heads < n and is
softly masked below head n-1), log-bandwidths and a sonorant logit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..features import vqt
from .bins import F0_AXIS, FORMANT_AXIS

N_FORMANTS = 4
LOG2_VQT = torch.log2(torch.tensor(vqt.bin_frequencies(), dtype=torch.float32))


@dataclass
class TeacherConfig:
    # Native convs (MIOpen off, D-002) run at ~1 TFLOPS here and dominate the step time, so channels at full
    # frequency resolution are kept small; resolution itself is not reduced early (plan §7.2).
    channels: tuple[int, int, int] = (24, 48, 64)
    full_res_dilations: tuple[int, ...] = (1, 2, 4)
    f0_channels: int = 16
    embed: int = 384
    gru_hidden: int = 256
    gru_layers: int = 2
    f0_gru_hidden: int = 128
    comb: bool = True
    comb_harmonics: int = 40
    comb_sigma_cents: float = 20.0
    mask_margin_bins: float = 10.0  # 200 cents below the previous formant
    mask_strength: float = 8.0


def normalise(logpow: torch.Tensor) -> torch.Tensor:
    """(B, F, T) dB -> 2 channels: per-frame spectral shape (gain-invariant) and frame level."""
    peak = logpow.amax(1, keepdim=True)
    shape = ((logpow - peak) / 20.0).clamp_min(-5.0)
    level = ((peak + 40.0) / 30.0).expand_as(logpow)
    return torch.stack([shape, level], 1)


class Block(nn.Module):
    def __init__(self, c_in: int, c_out: int, dil: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, (3, 3), padding=(dil, 1), dilation=(dil, 1))
        self.norm = nn.GroupNorm(8, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.skip(x) + Fn.gelu(self.norm(self.conv(x)))


class Down(nn.Module):
    """Halve frequency resolution (stride 2 along frequency only)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, (3, 1), stride=(2, 1), padding=(1, 0))
        self.norm = nn.GroupNorm(8, c_out)

    def forward(self, x):
        return Fn.gelu(self.norm(self.conv(x)))


def _freq_after(n: int, downs: int) -> int:
    for _ in range(downs):
        n = (n - 1) // 2 + 1
    return n


class Encoder(nn.Module):
    """Dilated 2-D conv stack (full frequency resolution first), flatten, BiGRU."""

    def __init__(self, c_in: int, chans: tuple[int, ...], dil_sets: tuple[tuple[int, ...], ...], embed: int,
                 hidden: int, layers: int):
        super().__init__()
        mods: list[nn.Module] = []
        c = c_in
        for i, (ch, dils) in enumerate(zip(chans, dil_sets)):
            if i > 0:
                mods.append(Down(c, ch))
                c = ch
            for d in dils:
                mods.append(Block(c, ch, d))
                c = ch
        self.convs = nn.Sequential(*mods)
        self.n_freq = _freq_after(vqt.N_BINS, len(chans) - 1)
        self.proj = nn.Linear(c * self.n_freq, embed)
        self.gru = nn.GRU(embed, hidden, layers, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, x):  # x: (B, C, F, T)
        h = self.convs(x)
        b, c, f, t = h.shape
        h = self.proj(h.permute(0, 3, 1, 2).reshape(b, t, c * f))
        h, _ = self.gru(Fn.gelu(h))
        return h  # (B, T, 2 * hidden)


class Teacher(nn.Module):
    def __init__(self, cfg: TeacherConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or TeacherConfig()
        c1 = cfg.f0_channels
        self.f0_net = Encoder(2, (c1, c1), ((1, 2, 4), (1, 2)), 256, cfg.f0_gru_hidden, 1)
        self.f0_head = nn.Linear(self.f0_net.out_dim, F0_AXIS.n_bins)
        self.voicing_head = nn.Linear(self.f0_net.out_dim, 1)
        c_in = 3 if cfg.comb else 2
        self.enc = Encoder(c_in, cfg.channels, (cfg.full_res_dilations, (1, 2, 4, 8), (1, 2)), cfg.embed, cfg.gru_hidden, cfg.gru_layers)
        d = self.enc.out_dim
        self.formant_heads = nn.ModuleList([nn.Linear(d + n, FORMANT_AXIS.n_bins) for n in range(N_FORMANTS)])
        self.bw_head = nn.Linear(d, 2 * N_FORMANTS)  # mean and log-variance of log B per formant
        self.sonorant_head = nn.Linear(d, 1)
        self.register_buffer("log2_vqt", LOG2_VQT.clone(), persistent=False)
        self.register_buffer("f0_log2", torch.log2(torch.tensor(F0_AXIS.centres_hz, dtype=torch.float32)), persistent=False)
        self.register_buffer("fm_bins", torch.arange(FORMANT_AXIS.n_bins, dtype=torch.float32), persistent=False)

    def comb(self, f0_hz: torch.Tensor, voiced_p: torch.Tensor) -> torch.Tensor:
        """(B, T) f0 and voicing -> (B, 1, F, T) Gaussian bumps at k f0 on the VQT axis."""
        k = torch.arange(1, self.cfg.comb_harmonics + 1, device=f0_hz.device, dtype=torch.float32)
        hk = torch.log2(f0_hz.clamp_min(20.0))[..., None] + torch.log2(k)  # (B, T, K)
        d = (self.log2_vqt[None, None, :, None] - hk[:, :, None, :]) * 1200.0 / self.cfg.comb_sigma_cents
        c = torch.exp(-0.5 * d * d).amax(-1) * voiced_p[..., None]  # (B, T, F); max, so close high harmonics don't sum
        return c.transpose(1, 2)[:, None]

    def expected_f0(self, f0_logits: torch.Tensor) -> torch.Tensor:
        return torch.exp2((f0_logits.softmax(-1) * self.f0_log2).sum(-1))

    def forward(self, logpow: torch.Tensor, comb_f0: torch.Tensor | None = None, comb_voiced: torch.Tensor | None = None) -> dict:
        """logpow: (B, F, T) VQT log power. comb_f0/comb_voiced override the comb source (teacher forcing)."""
        x = normalise(logpow)
        h0 = self.f0_net(x)
        out = {"f0_logits": self.f0_head(h0), "voicing_logit": self.voicing_head(h0)[..., 0]}
        if self.cfg.comb:
            with torch.no_grad():
                f0 = comb_f0 if comb_f0 is not None else self.expected_f0(out["f0_logits"].float())
                vp = comb_voiced if comb_voiced is not None else torch.sigmoid(out["voicing_logit"].float())
            x = torch.cat([x, self.comb(f0, vp).to(x.dtype)], 1)
        h = self.enc(x)
        logits, means = [], []
        for n, head in enumerate(self.formant_heads):
            inp = torch.cat([h] + [(m / FORMANT_AXIS.n_bins)[..., None].to(h.dtype) for m in means], -1) if means else h
            lg = head(inp).float()
            if means:  # soft mask below the previous formant
                below = (means[-1][..., None] - self.cfg.mask_margin_bins - self.fm_bins).clamp_min(0.0)
                lg = lg - self.cfg.mask_strength * torch.tanh(below / self.cfg.mask_margin_bins)
            logits.append(lg)
            means.append((lg.softmax(-1) * self.fm_bins).sum(-1))  # expected bin index
        bw = self.bw_head(h).float()
        out.update(formant_logits=torch.stack(logits, 2),  # (B, T, N, bins)
                   logb_mean=bw[..., :N_FORMANTS] + 4.5, logb_logvar=bw[..., N_FORMANTS:],
                   sonorant_logit=self.sonorant_head(h)[..., 0].float())
        return out


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
