"""Run a trained teacher on audio and decode per-frame outputs (plan §4 schema, learned part)."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

from ..features.vqt import HOP, TorchVQT
from .bins import F0_AXIS, FORMANT_AXIS, decode
from .teacher import Teacher, TeacherConfig


def load_teacher(path: str | Path, device):
    """path "<ckpt>#ema" loads the checkpoint's moving average of the weights instead of the weights."""
    import torch

    path = str(path)
    use_ema = path.endswith("#ema")
    ck = torch.load(path.removesuffix("#ema"), map_location="cpu", weights_only=False)
    if use_ema and ck.get("ema") is None:
        raise ValueError(f"{path}: the checkpoint has no moving average")
    model = Teacher(TeacherConfig(**ck["teacher_config"]))
    model.load_state_dict(ck["ema"] if use_ema else ck["model"])
    return model.to(device).eval(), ck


def decode_outputs(out: dict) -> dict[str, np.ndarray]:
    """Model outputs for one item (no batch dim, numpy) -> decoded per-frame values."""
    sm = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)  # noqa: E731
    f0, f0_sig = decode(sm(out["f0_logits"]), F0_AXIS)
    fm, fm_sig = decode(sm(out["formant_logits"]), FORMANT_AXIS)
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    return {"f0": f0, "f0_sigma_cents": f0_sig, "voiced_p": sig(out["voicing_logit"]),
            "formants": fm, "formant_sigma_cents": fm_sig, "bandwidths": np.exp(out["logb_mean"]),
            "bandwidth_log_sd": np.exp(0.5 * out["logb_logvar"]), "sonorant_p": sig(out["sonorant_logit"])}


@functools.lru_cache(maxsize=4)
def _vqt(device) -> TorchVQT:
    return TorchVQT(device)


def eval_bf16() -> bool:
    """Measures run in fp32 unless $VA_EVAL_BF16=1. bf16 (autocast, as before 2026-10-11) moved the median closed-phase
    F1 bias by about 1.7 Hz against fp32, and on this GPU it is no faster for these models."""
    return os.environ.get("VA_EVAL_BF16") == "1"


def analyse(model, audio: np.ndarray, device, sr: int = 16000) -> dict[str, np.ndarray]:
    """Whole-file offline analysis; frame i at t = i * 10 ms. fp32 unless $VA_EVAL_BF16=1 (`eval_bf16`)."""
    import torch

    assert sr == 16000
    vq = _vqt(device)
    n_frames = len(audio) // HOP + 1
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and eval_bf16()):
        x = torch.from_numpy(np.ascontiguousarray(audio, np.float32))[None].to(device)
        out = model(vq.log_power(x, n_frames))
    res = decode_outputs({k: v[0].float().cpu().numpy() for k, v in out.items()})
    res["t"] = np.arange(n_frames) * HOP / sr
    return res
