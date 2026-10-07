"""Supervised losses (plan §7.5): heat-map CE with blurred targets, bandwidth NLL, voicing/sonorant BCE."""

from __future__ import annotations

import torch
import torch.nn.functional as Fn

from ..models.bins import F0_AXIS, FORMANT_AXIS, gaussian_targets


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, x, torch.zeros_like(x)).sum() / mask.float().sum().clamp_min(1.0)


def heatmap_ce(logits: torch.Tensor, values_hz: torch.Tensor, mask: torch.Tensor, axis) -> torch.Tensor:
    target, valid = gaussian_targets(torch, values_hz, axis)
    ce = -(target * Fn.log_softmax(logits.float(), -1)).sum(-1)
    return _masked_mean(ce, mask & valid)


WEIGHTS = {"f0": 1.0, "voicing": 0.5, "formants": 1.0, "bandwidth": 0.2, "sonorant": 0.5}


def teacher_loss(out: dict, lab: dict, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    w = {**WEIGHTS, **(weights or {})}
    voiced, son = lab["voiced"], lab["sonorant"]
    parts = {
        "f0": heatmap_ce(out["f0_logits"], lab["f0"], voiced, F0_AXIS),
        "voicing": Fn.binary_cross_entropy_with_logits(out["voicing_logit"].float(), voiced.float()),
        "sonorant": Fn.binary_cross_entropy_with_logits(out["sonorant_logit"].float(), son.float()),
    }
    fl = out["formant_logits"]
    parts["formants"] = sum(heatmap_ce(fl[:, :, n], lab["formants"][..., n], son, FORMANT_AXIS) for n in range(fl.shape[2])) / fl.shape[2]
    logb = torch.log(lab["bandwidths"].clamp_min(1.0))
    bmask = son[..., None] & torch.isfinite(lab["bandwidths"])
    mean, logvar = out["logb_mean"], out["logb_logvar"].clamp(-8, 4)
    nll = 0.5 * (logvar + (torch.nan_to_num(logb) - mean) ** 2 / logvar.exp())
    parts["bandwidth"] = _masked_mean(nll, bmask)
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
