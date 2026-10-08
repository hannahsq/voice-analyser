"""Supervised losses (plan §7.5): heat-map CE with blurred targets, bandwidth NLL, voicing/sonorant BCE.

Optional per-frame weights in the labels (`train.data`). `w_f0` is a mask for the f0, voicing and
sonorant terms (frames with 0 leave the average, so mixing in resynthesised clips doesn't dilute these
terms). `w_formant` scales the formant and bandwidth terms, still averaged over all labelled frames, so
a low-quality clip pulls less. Absent weights count as 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn

from ..models.bins import F0_AXIS, FORMANT_AXIS, gaussian_targets


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean of x over mask, weighted; normalised by the number of masked frames, so weights below 1
    shrink a term rather than being renormalised away."""
    w = mask.float() if weight is None else mask.float() * weight.reshape(weight.shape + (1,) * (mask.dim() - weight.dim()))
    return torch.where(mask, x * w, torch.zeros_like(x)).sum() / mask.float().sum().clamp_min(1.0)


def heatmap_ce(logits: torch.Tensor, values_hz: torch.Tensor, mask: torch.Tensor, axis,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    target, valid = gaussian_targets(torch, values_hz, axis)
    ce = -(target * Fn.log_softmax(logits.float(), -1)).sum(-1)
    return _masked_mean(ce, mask & valid, weight)


WEIGHTS = {"f0": 1.0, "voicing": 0.5, "formants": 1.0, "bandwidth": 0.2, "sonorant": 0.5}


def teacher_loss(out: dict, lab: dict, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    w = {**WEIGHTS, **(weights or {})}
    voiced, son = lab["voiced"], lab["sonorant"]
    w_fm = lab.get("w_formant")
    f0_ok = lab["w_f0"] > 0 if "w_f0" in lab else torch.ones_like(voiced)
    bce = lambda logit, y: Fn.binary_cross_entropy_with_logits(logit.float(), y.float(), reduction="none")  # noqa: E731
    parts = {
        "f0": heatmap_ce(out["f0_logits"], lab["f0"], voiced & f0_ok, F0_AXIS),
        "voicing": _masked_mean(bce(out["voicing_logit"], voiced), f0_ok),
        "sonorant": _masked_mean(bce(out["sonorant_logit"], son), f0_ok),
    }
    fl = out["formant_logits"]
    parts["formants"] = sum(heatmap_ce(fl[:, :, n], lab["formants"][..., n], son, FORMANT_AXIS, w_fm)
                            for n in range(fl.shape[2])) / fl.shape[2]
    logb = torch.log(lab["bandwidths"].clamp_min(1.0))
    bmask = son[..., None] & torch.isfinite(lab["bandwidths"])
    mean, logvar = out["logb_mean"], out["logb_logvar"].clamp(-8, 4)
    nll = 0.5 * (logvar + (torch.nan_to_num(logb) - mean) ** 2 / logvar.exp())
    parts["bandwidth"] = _masked_mean(nll, bmask, w_fm)
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
