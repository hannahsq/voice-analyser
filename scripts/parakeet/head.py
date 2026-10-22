"""D-033 part 2: a trained head on Parakeet TDT 0.6B v3's layer 5 (frozen, truncated after it), with teacher v3's own
output heads and loss, so `decode_outputs` reads it like any teacher. Train (run from the main checkout):
  VA_DEVICE=cuda python <worktree>/scripts/parakeet/head.py <out dir> [steps]"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

from vocal_analysis.features.vqt import HOP, N_MAX  # noqa: E402
from vocal_analysis.models import teacher as T  # noqa: E402

TAP = 5  # layer 5: the probe's best for formants and bandwidths on every synthetic set
D = 128


def parakeet_tap(device):
    """The encoder truncated after layer TAP, frozen."""
    enc = load.encoder(device)
    enc.layers = enc.layers[:TAP]
    return enc.requires_grad_(False)


def tap_features(enc, fe, audio_batch, device) -> torch.Tensor:
    """(B, n) audio -> (B, E, 1024) layer-TAP features (80 ms frames, frame e centred on mel frame 8e)."""
    f = fe(list(audio_batch), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        h = enc(f["input_features"].to(device), attention_mask=f["attention_mask"].to(device), output_hidden_states=True)
    return h.hidden_states[TAP].clone()


def interp_time(h: torch.Tensor, mel_pos: torch.Tensor) -> torch.Tensor:
    """(B, E, d) at encoder frames -> (B, T, d) at mel positions (frame e at mel 8e), linear, clamped at the ends."""
    E = h.shape[1]
    p = (mel_pos / load.SUBSAMPLING).clamp(0, E - 1)
    if E == 1:
        return h[:, torch.zeros_like(p, dtype=torch.long)]
    i0 = p.floor().long().clamp(max=E - 2)
    a = (p - i0).to(h.dtype)[None, :, None]
    return h[:, i0] * (1 - a) + h[:, i0 + 1] * a


class ParakeetHead(nn.Module):
    """Layer-5 features -> teacher v3's outputs (same keys, bins and decoding)."""

    def __init__(self, in_dim: int = 1024):
        super().__init__()
        cfg = T.TeacherConfig()
        self.cfg = cfg
        self.norm = nn.LayerNorm(in_dim)  # in_dim: 1024 per tapped layer, concatenated (D-033 part 4)
        self.proj = nn.Linear(in_dim, D)
        self.blocks = nn.Sequential(*[T.ConformerBlock(D, cfg.conformer_heads, cfg.conformer_kernel, cfg.conformer_ff_mult)
                                      for _ in range(2)])
        self.f0_head = nn.Linear(D, T.F0_AXIS.n_bins)
        self.voicing_head = nn.Linear(D, 1)
        self.formant_heads = nn.ModuleList([nn.Linear(D + n, T.FORMANT_AXIS.n_bins) for n in range(T.N_FORMANTS)])
        self.bw_head = nn.Linear(D, 2 * T.N_FORMANTS)
        self.sonorant_head = nn.Linear(D, 1)
        self.register_buffer("fm_bins", torch.arange(T.FORMANT_AXIS.n_bins, dtype=torch.float32), persistent=False)

    def forward(self, feats: torch.Tensor, mel_pos: torch.Tensor) -> dict:
        h = self.blocks(interp_time(self.proj(self.norm(feats)), mel_pos))
        out = {"f0_logits": self.f0_head(h).float(), "voicing_logit": self.voicing_head(h)[..., 0].float()}
        logits, means = [], []
        for n, head in enumerate(self.formant_heads):  # as Teacher.forward: cascaded, soft mask below the previous formant
            inp = torch.cat([h] + [(m / T.FORMANT_AXIS.n_bins)[..., None].to(h.dtype) for m in means], -1) if means else h
            lg = head(inp).float()
            if means:
                below = (means[-1][..., None] - self.cfg.mask_margin_bins - self.fm_bins).clamp_min(0.0)
                lg = lg - self.cfg.mask_strength * torch.tanh(below / self.cfg.mask_margin_bins)
            logits.append(lg)
            means.append((lg.softmax(-1) * self.fm_bins).sum(-1))
        bw = self.bw_head(h).float()
        out.update(formant_logits=torch.stack(logits, 2), logb_mean=bw[..., :T.N_FORMANTS] + 4.5,
                   logb_logvar=bw[..., T.N_FORMANTS:], sonorant_logit=self.sonorant_head(h)[..., 0].float())
        return out


def crop_mel_pos(n_frames: int, device) -> torch.Tensor:
    """Our crop frame j sits at sample N_MAX/2 + j * HOP, i.e. mel frame j + N_MAX/2/HOP of the crop's audio."""
    return torch.arange(n_frames, device=device, dtype=torch.float32) + N_MAX // 2 // HOP


def file_mel_pos(n_frames: int, device) -> torch.Tensor:
    """Whole files: our frame i at t = i * 10 ms = mel frame i."""
    return torch.arange(n_frames, device=device, dtype=torch.float32)


def load_head(path, device):
    ck = torch.load(path, map_location="cpu")
    m = ParakeetHead(ck.get("in_dim", 1024))
    m.load_state_dict(ck["model"])
    return m.eval().to(device)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from vocal_analysis.config import DeviceConfig
    from vocal_analysis.device import setup_device
    from vocal_analysis.synth.channel import ChannelConfig
    from vocal_analysis.train.data import MixedCrops, SynthCrops, collate, consolidate, split_ids
    from vocal_analysis.train.losses import teacher_loss

    sys.stdout.reconfigure(encoding="utf-8")
    OUT = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    BATCH, CROP, WARMUP, LR = 32, 400, 500, 1e-3
    dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
    torch.manual_seed(0)
    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    flat = Path(cfg3["train"]["data"]) / "flat"
    train_ids, _ = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    epoch = STEPS * BATCH
    parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=0, epoch_size=epoch)], []
    for d in cfg3["train"]["resynth"]:
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=1 + len(sizes), epoch_size=epoch,
                                tract_left_weight=tuple(cfg3["train"]["tract_left_weight"])))
        sizes.append(n)
    frac = cfg3["train"]["resynth_frac"]
    ds = MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], epoch, seed=0)
    loader = DataLoader(ds, BATCH, num_workers=8, collate_fn=collate, persistent_workers=True, prefetch_factor=4, drop_last=True)
    fe, enc = load.feature_extractor(), parakeet_tap(dev)
    head = ParakeetHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    lr_at = lambda s: LR * (s + 1) / WARMUP if s < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (STEPS - WARMUP)))  # noqa: E731
    pos = crop_mel_pos(CROP, dev)
    log = open(OUT / "log.jsonl", "a")
    print(f"head: {sum(p.numel() for p in head.parameters()):,} parameters; {STEPS} steps of {BATCH} x 4 s", flush=True)
    t0, step = time.time(), 0
    for audio, lab in loader:
        if step >= STEPS:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        feats = tap_features(enc, fe, audio.numpy(), dev)
        out = head(feats, pos)
        loss, parts_ = teacher_loss(out, {k: v.to(dev) for k, v in lab.items()})
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "lr": lr_at(step), "loss": float(loss), "grad_norm": float(gn),
                   **{k: float(v) for k, v in parts_.items()}, "audio_s_per_s": step * BATCH * CROP * 0.01 / (time.time() - t0)}
            print(json.dumps(rec), file=log, flush=True)
            if step % 250 == 0:
                print(f"step {step}: loss {float(loss):.3f}  ({rec['audio_s_per_s']:.0f} audio s/s, {time.time() - t0:.0f} s)", flush=True)
        if step % 2000 == 0 or step == STEPS:
            torch.save({"model": head.state_dict(), "step": step, "tap": TAP}, OUT / "ckpt_last.pt")
    print("done", flush=True)
