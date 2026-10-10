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
    # Temporal modules (D-021 item 1): "gru" = BiGRUs (gru_hidden, gru_layers, f0_gru_hidden); "conformer" = the
    # main encoder's BiGRU becomes Conformer blocks and the f0 branch's a dilated 1-D conv stack.
    temporal: str = "gru"
    conformer_dim: int = 128  # Conformer width (projected from `embed`); sized so S2's temporal modules stay close to the GRUs'
    conformer_layers: int = 2
    conformer_heads: int = 4
    conformer_kernel: int = 15
    conformer_ff_mult: int = 2
    f0_conv_dim: int = 128
    f0_conv_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)  # receptive field +-63 frames


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


def _rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding over time for (B, H, T, D): attention depends only on relative position, so any
    crop length works."""
    d, t = x.shape[-1], x.shape[-2]
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class ConformerBlock(nn.Module):
    """Gulati et al. (2020): half feed-forward, self-attention (rotary positions), convolution module, half
    feed-forward, LayerNorm. LayerNorm instead of BatchNorm in the convolution module (batch-size independent)."""

    def __init__(self, d: int, heads: int, kernel: int, ff_mult: int):
        super().__init__()
        ff = lambda: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff_mult * d), nn.SiLU(), nn.Linear(ff_mult * d, d))  # noqa: E731
        self.ff1, self.ff2 = ff(), ff()
        self.heads = heads
        self.att_norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.att_out = nn.Linear(d, d)
        self.conv_norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.dw_norm = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, x):  # (B, T, d)
        x = x + 0.5 * self.ff1(x)
        b, t, d = x.shape
        q, k, v = self.qkv(self.att_norm(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = Fn.scaled_dot_product_attention(_rope(q), _rope(k), v)
        x = x + self.att_out(a.transpose(1, 2).reshape(b, t, d))
        h = Fn.glu(self.pw1(self.conv_norm(x)), -1)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        x = x + self.pw2(Fn.silu(self.dw_norm(h)))
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class DilatedConv1d(nn.Module):
    """Residual depthwise-separable dilated 1-D convolutions over time (the f0 branch's temporal module in the
    conformer variant)."""

    def __init__(self, d: int, dilations: tuple[int, ...]):
        super().__init__()
        self.dw = nn.ModuleList([nn.Conv1d(d, d, 3, padding=r, dilation=r, groups=d) for r in dilations])
        self.pw = nn.ModuleList([nn.Conv1d(d, d, 1) for _ in dilations])
        self.norms = nn.ModuleList([nn.GroupNorm(8, d) for _ in dilations])

    def forward(self, x):  # (B, T, d)
        h = x.transpose(1, 2)
        for dw, pw, norm in zip(self.dw, self.pw, self.norms):
            h = h + Fn.gelu(norm(pw(dw(h))))
        return h.transpose(1, 2)


class Encoder(nn.Module):
    """Dilated 2-D conv stack (full frequency resolution first), flatten, then a temporal module: a BiGRU, Conformer
    blocks or dilated 1-D convolutions."""

    def __init__(self, c_in: int, chans: tuple[int, ...], dil_sets: tuple[tuple[int, ...], ...], embed: int,
                 hidden: int, layers: int, temporal: str = "gru", cfg: TeacherConfig | None = None):
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
        self.temporal = temporal
        if temporal == "gru":
            self.gru = nn.GRU(embed, hidden, layers, batch_first=True, bidirectional=True)
            self.out_dim = 2 * hidden
        elif temporal == "conformer":
            d = cfg.conformer_dim
            self.t_in = nn.Linear(embed, d)
            self.blocks = nn.Sequential(*[ConformerBlock(d, cfg.conformer_heads, cfg.conformer_kernel, cfg.conformer_ff_mult)
                                          for _ in range(cfg.conformer_layers)])
            self.out_dim = d
        elif temporal == "conv":
            self.t_in = nn.Linear(embed, cfg.f0_conv_dim)
            self.tconv = DilatedConv1d(cfg.f0_conv_dim, cfg.f0_conv_dilations)
            self.out_dim = cfg.f0_conv_dim
        else:
            raise ValueError(f"unknown temporal module {temporal!r}")

    def forward(self, x):  # x: (B, C, F, T)
        h = self.convs(x)
        b, c, f, t = h.shape
        h = Fn.gelu(self.proj(h.permute(0, 3, 1, 2).reshape(b, t, c * f)))
        if self.temporal == "gru":
            return self.gru(h)[0]  # (B, T, 2 * hidden)
        if self.temporal == "conformer":
            return self.blocks(self.t_in(h))  # (B, T, conformer_dim)
        return self.tconv(self.t_in(h))


class Teacher(nn.Module):
    def __init__(self, cfg: TeacherConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or TeacherConfig()
        c1 = cfg.f0_channels
        conformer = cfg.temporal == "conformer"
        self.f0_net = Encoder(2, (c1, c1), ((1, 2, 4), (1, 2)), 256, cfg.f0_gru_hidden, 1, "conv" if conformer else "gru", cfg)
        self.f0_head = nn.Linear(self.f0_net.out_dim, F0_AXIS.n_bins)
        self.voicing_head = nn.Linear(self.f0_net.out_dim, 1)
        c_in = 3 if cfg.comb else 2
        self.enc = Encoder(c_in, cfg.channels, (cfg.full_res_dilations, (1, 2, 4, 8), (1, 2)), cfg.embed, cfg.gru_hidden,
                           cfg.gru_layers, cfg.temporal, cfg)
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
