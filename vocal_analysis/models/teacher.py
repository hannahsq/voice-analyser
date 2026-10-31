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
    # D-027: per-formant observability head (logit of o, the evidence-only precision label): reads the formant
    # trunk and the formant heads' expected positions (no gradient into those heads). Off = earlier models.
    obs_head: bool = False
    # D-036: anatomical vocal-tract-length head (D-035's per-frame design): mean and log variance of log2 length (cm)
    # from the formant trunk, behind a gradient gate (Teacher.vtl_gate: 0 = no gradient into the trunk, 1 = full).
    vtl_head: bool = False
    # D-051: instantaneous tract length head (larynx height and lip protrusion included), built as vtl_head and fed
    # through the same gate; needs vtl_head
    vtl_inst_head: bool = False
    # D-054: how the formant and bandwidth heads read the trunk. "flat": linear heads on the pooled per-frame vector h
    # (every model to D-051); "mlp": the same with one hidden layer of readout_hidden (a capacity control); "film":
    # frequency-resolved (`FreqReadout`): the encoder's conv maps, modulated by h, upsampled back to the VQT's bins, one
    # logit per bin per formant, and each bandwidth read at its formant's position
    readout: str = "flat"
    readout_channels: int = 32
    readout_hidden: int = 512


class _GradGate(torch.autograd.Function):
    """Identity forward; the backward gradient multiplied by `alpha`."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.alpha, None


def vtl_head_module(d: int) -> nn.Module:
    """D-035's head: LayerNorm -> 256 -> GELU -> 256 -> GELU -> 2 (index-compatible with its saved `net.*` weights)."""
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))


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

    def forward(self, x, maps: bool = False):  # x: (B, C, F, T)
        """maps: also return the conv maps at the end of each frequency resolution, full resolution first."""
        kept = []
        h = x
        for m in self.convs:
            if isinstance(m, Down):
                kept.append(h)
            h = m(h)
        kept.append(h)
        b, c, f, t = h.shape
        h = Fn.gelu(self.proj(h.permute(0, 3, 1, 2).reshape(b, t, c * f)))
        if self.temporal == "gru":
            h = self.gru(h)[0]  # (B, T, 2 * hidden)
        elif self.temporal == "conformer":
            h = self.blocks(self.t_in(h))  # (B, T, conformer_dim)
        else:
            h = self.tconv(self.t_in(h))
        return (h, kept) if maps else h


def _interp_matrix(src_log2: torch.Tensor, dst_log2: torch.Tensor) -> torch.Tensor:
    """(len(dst), len(src)) linear interpolation in log frequency (clamped at the ends)."""
    pos = torch.searchsorted(src_log2, dst_log2).clamp(1, len(src_log2) - 1)
    lo, hi = src_log2[pos - 1], src_log2[pos]
    w = ((dst_log2 - lo) / (hi - lo)).clamp(0.0, 1.0)
    m = torch.zeros(len(dst_log2), len(src_log2))
    m[torch.arange(len(dst_log2)), pos - 1] = 1.0 - w
    m[torch.arange(len(dst_log2)), pos] = w
    return m


class FreqReadout(nn.Module):
    """D-054: formant logits read per frequency bin, and bandwidths read at each formant's position.

    The coarsest conv map (64 ch x 91 bins) is FiLM-modulated by the temporal module's output h (one scale and shift
    per channel and frame, the same at every bin), then upsampled along frequency twice, adding the finer conv maps
    (U-Net skips) through 1x1 convs, with a residual conv Block at each resolution. A 1x1 conv gives one logit per VQT
    bin per formant, interpolated onto the formant axis; the cascade's soft mask is the flat heads'. Bandwidth n reads
    the full-resolution features at formant n's expected position (no gradient through the position) beside h.
    """

    def __init__(self, d: int, chans: tuple[int, ...], c: int):
        super().__init__()
        self.film = nn.Linear(d, 2 * chans[-1])
        self.lat = nn.ModuleList([nn.Conv2d(ch, c, 1) for ch in chans])
        self.blocks = nn.ModuleList([Block(c, c, 1) for _ in chans])
        self.out = nn.Conv2d(c, N_FORMANTS, 1)
        self.bw = nn.ModuleList([nn.Sequential(nn.Linear(c + d, 64), nn.GELU(), nn.Linear(64, 2)) for _ in range(N_FORMANTS)])
        fm_log2 = torch.log2(torch.tensor(FORMANT_AXIS.centres_hz, dtype=torch.float32))
        self.register_buffer("to_formant", _interp_matrix(LOG2_VQT.clone(), fm_log2), persistent=False)

    def maps(self, kept: list, h: torch.Tensor) -> torch.Tensor:
        """-> full-resolution features (B, c, F_vqt, T)."""
        g, b = self.film(h).transpose(1, 2)[:, :, None, :].chunk(2, 1)  # (B, C, 1, T) each
        x = self.blocks[-1](self.lat[-1](kept[-1] * (1.0 + g) + b))
        for i in range(len(kept) - 2, -1, -1):
            x = Fn.interpolate(x, size=kept[i].shape[-2:], mode="bilinear", align_corners=True)
            x = self.blocks[i](x + self.lat[i](kept[i]))
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, T, formant bins) before the cascade's mask."""
        return torch.einsum("bnft,kf->bntk", self.out(x).float(), self.to_formant)

    def bandwidths(self, x: torch.Tensor, h: torch.Tensor, pos_hz: torch.Tensor) -> torch.Tensor:
        """pos_hz (B, T, N) -> (B, T, 2N): mean of log B per formant, then log-variances (the flat head's layout)."""
        f = x.shape[2]
        lv = LOG2_VQT.to(x.device)
        idx = ((torch.log2(pos_hz.clamp_min(1.0)) - lv[0]) / ((lv[-1] - lv[0]) / (f - 1))).clamp(0, f - 1.001)
        i0 = idx.floor().long()
        w = (idx - i0)[:, None].to(x.dtype)  # (B, 1, T, N)
        xt = x.permute(0, 1, 3, 2)  # (B, c, T, F)
        g0 = torch.gather(xt, 3, i0[:, None].expand(-1, xt.shape[1], -1, -1))
        g1 = torch.gather(xt, 3, (i0 + 1)[:, None].expand(-1, xt.shape[1], -1, -1))
        local = ((1 - w) * g0 + w * g1).permute(0, 2, 3, 1)  # (B, T, N, c)
        v = torch.stack([self.bw[n](torch.cat([local[:, :, n], h.to(local.dtype)], -1)) for n in range(N_FORMANTS)], 2).float()
        return torch.cat([v[..., 0], v[..., 1]], -1)


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
        if cfg.readout == "flat":
            self.formant_heads = nn.ModuleList([nn.Linear(d + n, FORMANT_AXIS.n_bins) for n in range(N_FORMANTS)])
            self.bw_head = nn.Linear(d, 2 * N_FORMANTS)  # mean and log-variance of log B per formant
        elif cfg.readout == "mlp":
            hid = cfg.readout_hidden
            self.formant_heads = nn.ModuleList([nn.Sequential(nn.Linear(d + n, hid), nn.GELU(), nn.Linear(hid, FORMANT_AXIS.n_bins))
                                                for n in range(N_FORMANTS)])
            self.bw_head = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 2 * N_FORMANTS))
        elif cfg.readout == "film":
            self.readout = FreqReadout(d, cfg.channels, cfg.readout_channels)
        else:
            raise ValueError(f"unknown readout {cfg.readout!r}")
        self.sonorant_head = nn.Linear(d, 1)
        if cfg.obs_head:
            self.obs_head = nn.Linear(d + N_FORMANTS, N_FORMANTS)
        if cfg.vtl_head:
            self.vtl_head = vtl_head_module(d)
            self.vtl_gate = 1.0  # set by the trainer each step
        if cfg.vtl_inst_head:
            self.vtl_inst_head = vtl_head_module(d)
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
        film = self.cfg.readout == "film"
        h, kept = self.enc(x, maps=True) if film else (self.enc(x), None)
        if film:
            fx = self.readout.maps(kept, h)
            raw = self.readout.logits(fx)
        logits, means = [], []
        for n in range(N_FORMANTS):
            if film:
                lg = raw[:, n]
            else:
                inp = torch.cat([h] + [(m / FORMANT_AXIS.n_bins)[..., None].to(h.dtype) for m in means], -1) if means else h
                lg = self.formant_heads[n](inp).float()
            if means:  # soft mask below the previous formant
                below = (means[-1][..., None] - self.cfg.mask_margin_bins - self.fm_bins).clamp_min(0.0)
                lg = lg - self.cfg.mask_strength * torch.tanh(below / self.cfg.mask_margin_bins)
            logits.append(lg)
            means.append((lg.softmax(-1) * self.fm_bins).sum(-1))  # expected bin index
        if film:
            pos = FORMANT_AXIS.f_min * torch.exp2(torch.stack(means, -1).detach() * FORMANT_AXIS.cents_per_bin / 1200.0)
            bw = self.readout.bandwidths(fx, h, pos)
        else:
            bw = self.bw_head(h).float()
        out.update(formant_logits=torch.stack(logits, 2),  # (B, T, N, bins)
                   logb_mean=bw[..., :N_FORMANTS] + 4.5, logb_logvar=bw[..., N_FORMANTS:],
                   sonorant_logit=self.sonorant_head(h)[..., 0].float())
        if self.cfg.obs_head:
            pos = torch.stack(means, -1).detach() / FORMANT_AXIS.n_bins
            out["obs_logit"] = self.obs_head(torch.cat([h, pos.to(h.dtype)], -1)).float()  # (B, T, N)
        if self.cfg.vtl_head:
            g = self.vtl_gate
            hv = h.detach() if g == 0 else h if g == 1 else _GradGate.apply(h, float(g))
            v = self.vtl_head(hv).float()
            out["vtl_mu"], out["vtl_logvar"] = v[..., 0], v[..., 1]
            if self.cfg.vtl_inst_head:
                vi = self.vtl_inst_head(hv).float()
                out["vtl_inst_mu"], out["vtl_inst_logvar"] = vi[..., 0], vi[..., 1]
        return out


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
