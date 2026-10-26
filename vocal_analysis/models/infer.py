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
    res = {"f0": f0, "f0_sigma_cents": f0_sig, "voiced_p": sig(out["voicing_logit"]),
           "formants": fm, "formant_sigma_cents": fm_sig, "bandwidths": np.exp(out["logb_mean"]),
           "bandwidth_log_sd": np.exp(0.5 * out["logb_logvar"]), "sonorant_p": sig(out["sonorant_logit"])}
    if "obs_logit" in out:  # D-027: per-formant observability (the evidence-only precision label's estimate)
        res["observability"] = sig(out["obs_logit"])
    if "vtl_mu" in out:  # D-036: anatomical tract length, log2 cm, and its sigma (log2 units)
        res["vtl_log2cm"], res["vtl_sigma"] = out["vtl_mu"], np.exp(0.5 * out["vtl_logvar"])
    return res


@functools.lru_cache(maxsize=4)
def _vqt(device) -> TorchVQT:
    return TorchVQT(device)


def eval_bf16() -> bool:
    """Measures run in fp32 unless $VA_EVAL_BF16=1. bf16 (autocast, as before 2026-10-11) moved the median closed-phase
    F1 bias by about 1.7 Hz against fp32, and on this GPU it is no faster for these models."""
    return os.environ.get("VA_EVAL_BF16") == "1"


DEFAULT_WINDOW = 400  # D-028 "windows", the default from 2026-10-27 (roadmap 3)


def analyse(model, audio: np.ndarray, device, sr: int = 16000, window_frames: int | None = DEFAULT_WINDOW) -> dict[str, np.ndarray]:
    """Offline analysis of a file; frame i at t = i * 10 ms. fp32 unless $VA_EVAL_BF16=1 (`eval_bf16`).
    window_frames: files longer than this run in windows of that many frames (hop half a window), each frame taken from
    the window where it is most central (D-028: the models train on clips of at most 400 frames, and longer context
    carries one speaker's tract into another's frames). The default since 2026-10-27; None or 0 = the whole file at once."""
    assert sr == 16000
    n_frames = len(audio) // HOP + 1
    if not window_frames or n_frames <= window_frames:
        return _analyse_once(model, audio, device, n_frames)
    W, H = int(window_frames), int(window_frames) // 2
    starts = list(range(0, n_frames - W + 1, H))
    if starts[-1] + W < n_frames:
        starts.append(n_frames - W)
    res, best = None, np.full(n_frames, -np.inf)
    for st in starts:
        seg = audio[st * HOP: (st + W - 1) * HOP + 1]
        r = _analyse_once(model, seg, device, W)
        if res is None:
            res = {k: np.full((n_frames, *v.shape[1:]), np.nan, v.dtype if v.dtype.kind == "f" else np.float64)
                   for k, v in r.items() if k != "t"}
        centrality = -np.abs(np.arange(W) - (W - 1) / 2)
        take = centrality > best[st: st + W]
        for k in res:
            res[k][st: st + W][take] = r[k][take]
        best[st: st + W][take] = centrality[take]
    res["t"] = np.arange(n_frames) * HOP / sr
    return res


def _analyse_once(model, audio: np.ndarray, device, n_frames: int) -> dict[str, np.ndarray]:
    import torch

    vq = _vqt(device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and eval_bf16()):
        x = torch.from_numpy(np.ascontiguousarray(audio, np.float32))[None].to(device)
        out = model(vq.log_power(x, n_frames))
    res = decode_outputs({k: v[0].float().cpu().numpy() for k, v in out.items()})
    res["t"] = np.arange(n_frames) * HOP / 16000
    return res
