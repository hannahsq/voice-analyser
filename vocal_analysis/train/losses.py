"""Supervised losses (plan §7.5): heat-map CE with blurred targets, bandwidth NLL, voicing/sonorant BCE.

Optional per-frame weights in the labels (`train.data`). `w_f0` is a mask for the f0, voicing and
sonorant terms (frames with 0 leave the average, so mixing in resynthesised clips doesn't dilute these
terms). `w_formant` scales the formant and bandwidth terms, still averaged over all labelled frames, so
a low-quality clip pulls less. Absent weights count as 1.

The parts come back as detached tensors: converting them to floats every step would make the host wait for
the GPU (the trainer converts once per log interval).
"""

from __future__ import annotations

import torch
import torch.nn.functional as Fn

from ..models.bins import F0_AXIS, FORMANT_AXIS, UPPER_AXIS, gaussian_targets


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


WEIGHTS = {"f0": 1.0, "voicing": 0.5, "formants": 1.0, "bandwidth": 0.2, "sonorant": 0.5, "obs": 0.2, "vtl": 0.0125, "vtl_inst": 0.0125, "opening": 1.0, "upper": 1.0,
           "formants_acoustic": 1.0, "bandwidth_acoustic": 0.2, "source": 0.1}  # source: D-161's registered weight  # D-085: the acoustic set at the closed set's weights  # vtl: D-036's registered weight


SOURCE_F_MAX = 4000.0  # D-161: the source loss's harmonics, below min(4 kHz, the VQT's top at 7.5 kHz)
SOURCE_UNIT_DB = 6.0


def source_loss(pred: torch.Tensor, y: torch.Tensor, f0: torch.Tensor, voiced: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """D-161: Huber (delta 1) on the level error in units of 6 dB, offset-invariant (each frame's mean error over its
    valid harmonics subtracted: only the shape is defined), over voiced frames with at least 2 finite harmonics below
    4 kHz, weighted by w per frame."""
    k = torch.arange(1, pred.shape[-1] + 1, device=pred.device, dtype=torch.float32)
    ok = torch.isfinite(y) & voiced[..., None] & (f0.float()[..., None] * k < SOURCE_F_MAX) & (w[..., None] > 0)
    e = torch.where(ok, pred.float() - torch.nan_to_num(y.float()), torch.zeros_like(pred, dtype=torch.float32))
    n = ok.sum(-1, keepdim=True)
    r = torch.where(ok, (e - e.sum(-1, keepdim=True) / n.clamp_min(1)) / SOURCE_UNIT_DB, torch.zeros_like(e))
    hub = Fn.huber_loss(r, torch.zeros_like(r), reduction="none", delta=1.0)
    fw = w.float()[..., None] * ok * (n >= 2)
    return (hub * fw).sum() / fw.sum().clamp_min(1e-6)


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
    if "formant_logits_acoustic" in out and "formants_acoustic" in lab:  # D-085: the acoustic set, as the closed one
        fa = out["formant_logits_acoustic"]
        parts["formants_acoustic"] = sum(heatmap_ce(fa[:, :, n], lab["formants_acoustic"][..., n], son, FORMANT_AXIS, w_fm)
                                         for n in range(fa.shape[2])) / fa.shape[2]
        ba = lab["bandwidths_acoustic"]
        amask = son[..., None] & torch.isfinite(ba)
        mean_a, lv_a = out["logb_mean_acoustic"], out["logb_logvar_acoustic"].clamp(-8, 4)
        nll_a = 0.5 * (lv_a + (torch.nan_to_num(torch.log(ba.clamp_min(1.0))) - mean_a) ** 2 / lv_a.exp())
        parts["bandwidth_acoustic"] = _masked_mean(nll_a, amask, w_fm)
    if "obs_logit" in out and "obs" in lab:  # D-027: soft-target BCE on sonorant frames with a label
        o = lab["obs"]
        omask = son[..., None] & torch.isfinite(o)
        parts["obs"] = _masked_mean(bce(out["obs_logit"], torch.nan_to_num(o).clamp(0, 1)), omask)
    if "vtl_mu" in out and "vtl" in lab:  # D-036: Gaussian NLL of log2 length on sonorant frames with a label
        y = lab["vtl"]
        vmask = son & torch.isfinite(y)
        lv = out["vtl_logvar"].clamp(-12, 4)
        parts["vtl"] = _masked_mean(0.5 * (lv + (torch.nan_to_num(y) - out["vtl_mu"]) ** 2 / lv.exp()), vmask)
    if "vtl_inst_mu" in out and "vtl_inst" in lab:  # D-051: the same NLL for the instantaneous length
        y = lab["vtl_inst"]
        vmask = son & torch.isfinite(y)
        lv = out["vtl_inst_logvar"].clamp(-12, 4)
        parts["vtl_inst"] = _masked_mean(0.5 * (lv + (torch.nan_to_num(y) - out["vtl_inst_mu"]) ** 2 / lv.exp()), vmask)
    if "opening" in out and "glottal_opening" in lab:  # D-080: L1 on <a> x 10 on sonorant frames with a label
        y = lab["glottal_opening"]
        omask = son & torch.isfinite(y)
        parts["opening"] = _masked_mean(10.0 * (out["opening"] - torch.nan_to_num(y)).abs(), omask)
    if "upper_logits" in out and "upper_formants" in lab:  # D-090: as the formant term, per formant at F1-F4's
        ul = out["upper_logits"]  # weight (sum / 4); frames with the formant out of band (NaN) carry no loss
        parts["upper"] = sum(heatmap_ce(ul[:, :, k], lab["upper_formants"][..., k], son, UPPER_AXIS, w_fm)
                             for k in range(ul.shape[2])) / 4
    if "source_levels" in out and "source40" in lab:  # D-161
        parts["source"] = source_loss(out["source_levels"], lab["source40"], lab["f0"], voiced, lab["source_w"])
    total = sum(w[k] * v for k, v in parts.items())
    return total, {k: v.detach() for k, v in parts.items()}
