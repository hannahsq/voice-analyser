"""D-034: the student. Our formant encoder's architecture without the comb channel (2 input channels: spectral shape and
level), scaled to channels (32, 64, 96), embed 512, Conformer width 384, 8 layers, 6 heads, feed-forward x4, with a
linear projection to Parakeet's 1024 dimensions. Input: VQT log power (B, F, T) at our 10 ms frames; output (B, T, 1024)."""
import json
from dataclasses import replace

import torch
from torch import nn

from vocal_analysis.models.teacher import Encoder, TeacherConfig, normalise

SIZE = dict(channels=(32, 64, 96), embed=512, conformer_dim=384, conformer_layers=8, conformer_heads=6, conformer_ff_mult=4)


def student_config() -> TeacherConfig:
    base = json.load(open("runs/conformer_v2r2_s1_d16k/config.json"))["model"]
    return replace(TeacherConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in base.items()}), **SIZE)


class Student(nn.Module):
    def __init__(self, cfg: TeacherConfig | None = None):
        super().__init__()
        cfg = cfg or student_config()
        self.enc = Encoder(2, cfg.channels, (cfg.full_res_dilations, (1, 2, 4, 8), (1, 2)), cfg.embed, cfg.gru_hidden,
                           cfg.gru_layers, "conformer", cfg)
        self.proj = nn.Linear(self.enc.out_dim, 1024)

    def forward(self, logpow: torch.Tensor) -> torch.Tensor:
        return self.proj(self.enc(normalise(logpow)))


def load_student(path, device) -> Student:
    m = Student().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return m.eval()
