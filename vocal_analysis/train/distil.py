"""D-128 (M5): distil a frozen teacher into a causal copy of itself (`TeacherConfig.causal`). Plain distillation: the
student matches the teacher's outputs on the same audio, no labels; it starts from the teacher's weights (or from
scratch). `va train student` drives this.

Losses, each a per-frame mean over the frames the student may be scored on (the crop's last `lookahead` frames are
left out: there the student read appended silence, not the crop's future):
- f0: KL(teacher || student) of the f0 heat-maps, weighted by the teacher's voicing probability;
- voicing, sonorant: binary KL against the teacher's probabilities (BCE less the teacher's entropy);
- formants: KL per formant heat-map, weighted by the teacher's sonorant probability;
- bandwidths, VTL: KL between the two Gaussians (log B; log2 length), weighted by the sonorant probability.
The part weights are the teacher's supervised ones (`losses.WEIGHTS`).

A run directory holds config.json, log.jsonl and checkpoints in the teacher's format (so `va eval teacher` and
`load_teacher` read a student unchanged); a run resumes from ckpt_last.pt.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..models.bins import F0_AXIS, FORMANT_AXIS, decode
from ..models.teacher import N_FORMANTS, Teacher, TeacherConfig, n_params
from .data import TRACT_LEFT_WEIGHT, LengthBatches, MixedCrops, SynthCrops, collate, consolidate, split_ids
from .losses import WEIGHTS
from .teacher import _git, _saved_channel, _to, cache_batches, evaluate

PARTS = ("f0", "voicing", "sonorant", "formants", "bandwidth", "vtl")


@dataclass
class DistilConfig:
    teacher: str = "runs/d040_5b_s2_b20k/ckpt_last.pt"  # teacher v3.1
    out: str = "runs/student_causal"
    lookahead: int = 0  # frames
    attn_window: int = 400  # frames, the longest training crop
    init: str = "teacher"  # "teacher": the student starts from the teacher's weights; "scratch"
    data: str = "data/synth/train_v2"
    resynth: tuple = ("data/resynth/vocalset_train_d16k", "data/resynth/zeroth_korean_train_d16k")
    resynth_frac: float = 0.3
    resynth_val: tuple = ("data/resynth/vocalset_val_d16k", "data/resynth/zeroth_korean_val_d16k")
    tract_left_weight: tuple = TRACT_LEFT_WEIGHT
    steps: int = 8000
    batch: int = 32
    n_frames: int = 200
    crop_frames: tuple | None = (32, 400)
    lr: float = 3e-4
    warmup: int = 300
    decay_start: int | None = 6000  # WSD: constant, then a square-root decay to 0 at `steps`
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    workers: int = 8
    log_every: int = 50
    eval_every: int = 1000
    ckpt_every: int = 1000
    val_clips: int = 512
    weights: dict = field(default_factory=dict)  # overrides of losses.WEIGHTS for the parts above


def _lr(step: int, cfg: DistilConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if cfg.decay_start is None or step < cfg.decay_start:
        return cfg.lr
    return cfg.lr * (1.0 - math.sqrt(min((step - cfg.decay_start) / max(1, cfg.steps - cfg.decay_start), 1.0)))


def _wmean(x, w):
    return (x * w).sum() / w.sum().clamp_min(1e-6)


def _gauss_kl(mu_t, lv_t, mu_s, lv_s):
    """KL(N(mu_t, e^lv_t) || N(mu_s, e^lv_s))."""
    return 0.5 * (lv_s - lv_t + (lv_t.exp() + (mu_t - mu_s) ** 2) / lv_s.exp() - 1.0)


def distil_loss(s: dict, t: dict, valid, weights: dict | None = None):
    """Student outputs s, teacher outputs t (detached), valid (B, T) frames scored -> (loss, parts)."""
    import torch
    import torch.nn.functional as Fn

    w = {**WEIGHTS, **(weights or {})}
    v = valid.float()
    pv = torch.sigmoid(t["voicing_logit"].float())
    ps = torch.sigmoid(t["sonorant_logit"].float())

    def kl(sl, tl):  # over the last axis
        lt = tl.float().log_softmax(-1)
        return (lt.exp() * (lt - sl.float().log_softmax(-1))).sum(-1)

    def bkl(key, p):  # binary KL(teacher || student): BCE minus the teacher's own entropy (0 when they agree)
        bce = lambda lg: Fn.binary_cross_entropy_with_logits(lg.float(), p, reduction="none")  # noqa: E731
        return bce(s[key]) - bce(t[key])

    parts = {
        "f0": _wmean(kl(s["f0_logits"], t["f0_logits"]), pv * v),
        "voicing": _wmean(bkl("voicing_logit", pv), v),
        "sonorant": _wmean(bkl("sonorant_logit", ps), v),
        "formants": _wmean(kl(s["formant_logits"], t["formant_logits"]).mean(-1), ps * v),
        "bandwidth": _wmean(_gauss_kl(t["logb_mean"].float(), t["logb_logvar"].float().clamp(-8, 4),
                                      s["logb_mean"].float(), s["logb_logvar"].float().clamp(-8, 4)).mean(-1), ps * v),
    }
    if "vtl_mu" in t and "vtl_mu" in s:
        parts["vtl"] = _wmean(_gauss_kl(t["vtl_mu"].float(), t["vtl_logvar"].float().clamp(-12, 4),
                                        s["vtl_mu"].float(), s["vtl_logvar"].float().clamp(-12, 4)), ps * v)
    loss = sum(w[k] * p for k, p in parts.items())
    return loss, {k: p.detach() for k, p in parts.items()}


def agreement(student, teacher, batches, device) -> dict:
    """Student against teacher on fixed batches: median |difference| in cents of decoded f0 (frames the teacher calls
    voiced) and F1-F4 (sonorant), and voicing / sonorant agreement at 0.5."""
    import torch

    student.eval()
    rows = {"f0": [], "fm": [], "voicing": [], "sonorant": []}
    with torch.inference_mode():
        for lp, _ in batches:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                s, t = student(lp), teacher(lp)
            pv = torch.sigmoid(t["voicing_logit"].float()).cpu().numpy() > 0.5
            ps = torch.sigmoid(t["sonorant_logit"].float()).cpu().numpy() > 0.5
            f0s, _ = decode(s["f0_logits"].float().softmax(-1).cpu().numpy(), F0_AXIS)
            f0t, _ = decode(t["f0_logits"].float().softmax(-1).cpu().numpy(), F0_AXIS)
            fs, _ = decode(s["formant_logits"].float().softmax(-1).cpu().numpy(), FORMANT_AXIS)
            ft, _ = decode(t["formant_logits"].float().softmax(-1).cpu().numpy(), FORMANT_AXIS)
            rows["f0"].append(np.abs(1200 * np.log2(f0s / f0t))[pv])
            rows["fm"].append(np.abs(1200 * np.log2(fs / ft))[ps])
            rows["voicing"].append(((torch.sigmoid(s["voicing_logit"].float()).cpu().numpy() > 0.5) == pv).ravel())
            rows["sonorant"].append(((torch.sigmoid(s["sonorant_logit"].float()).cpu().numpy() > 0.5) == ps).ravel())
    student.train()
    f0, fm = np.concatenate(rows["f0"]), np.concatenate(rows["fm"])
    out = {"f0_cents_med": float(np.median(f0)) if len(f0) else float("nan"),
           "voicing_agree": float(np.concatenate(rows["voicing"]).mean()),
           "sonorant_agree": float(np.concatenate(rows["sonorant"]).mean())}
    for n in range(N_FORMANTS):
        out[f"F{n + 1}_cents_med"] = float(np.median(fm[:, n])) if len(fm) else float("nan")
    return out


def distil(cfg: DistilConfig, log=print) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from ..device import setup_device
    from ..features.vqt import TorchVQT
    from ..models.infer import load_teacher

    device = setup_device()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    teacher, tck = load_teacher(cfg.teacher, device)
    teacher.requires_grad_(False)
    channel = _saved_channel(Path(cfg.teacher).parent)
    scfg = replace(teacher.cfg, causal=True, attn_window=cfg.attn_window, lookahead=cfg.lookahead)
    student = Teacher(scfg).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    last = out / "ckpt_last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        torch.manual_seed(cfg.seed + 7919 * step)
        log(f"resumed at step {step}")
    else:
        if cfg.init == "teacher":
            student.load_state_dict(tck["model"])
        elif cfg.init != "scratch":
            raise ValueError(f"unknown init {cfg.init!r}")
    if step == 0:
        from ..resynth import corpus

        (out / "config.json").write_text(json.dumps({
            "distil": asdict(cfg), "model": asdict(scfg), "git": _git(), "n_params": n_params(student),
            "teacher": {"path": Path(cfg.teacher).as_posix(), "sha256": corpus.file_sha256(cfg.teacher), "step": tck.get("step")},
            "channel": asdict(channel)}, indent=1))
    log(f"student: {n_params(student):,} parameters, init {cfg.init}, lookahead {cfg.lookahead}, window {cfg.attn_window}")

    flat = consolidate(cfg.data, log=log)
    n_clips = json.loads((flat / "done").read_text())["n_clips"]
    train_ids, val_ids = split_ids(n_clips)
    val_ids = val_ids[: cfg.val_clips]
    epoch = cfg.batch * 1000
    ds = SynthCrops(flat, train_ids, cfg.n_frames, channel, seed=cfg.seed, epoch_size=epoch)
    if cfg.resynth:
        parts, sizes = [], []
        for d in cfg.resynth:
            rflat = consolidate(d, log=log)
            n = json.loads((rflat / "done").read_text())["n_clips"]
            parts.append(SynthCrops(rflat, np.arange(n), cfg.n_frames, channel, seed=cfg.seed + 1 + len(parts),
                                    epoch_size=epoch, tract_left_weight=cfg.tract_left_weight))
            sizes.append(n)
        weights = [1.0 - cfg.resynth_frac] + [cfg.resynth_frac * n / sum(sizes) for n in sizes]
        ds = MixedCrops([ds, *parts], weights, epoch, seed=cfg.seed)
    common = dict(num_workers=cfg.workers, collate_fn=collate, persistent_workers=True, pin_memory=False, prefetch_factor=4)
    if cfg.crop_frames:
        batches = LengthBatches(1000, epoch, tuple(cfg.crop_frames), cfg.batch * cfg.n_frames, seed=cfg.seed + step,
                                crop_span=cfg.n_frames)
        loader = DataLoader(ds, batch_sampler=batches, **common)
    else:
        loader = DataLoader(ds, cfg.batch, drop_last=True, **common)
    val_channel = replace(channel, gain_db=type(channel)().gain_db)
    vq = TorchVQT(device)
    val_sets = {"eval": cache_batches(DataLoader(SynthCrops(flat, val_ids, cfg.n_frames, val_channel, seed=12345, fixed=True,
                                                            vtl=True), cfg.batch, num_workers=4, collate_fn=collate), vq, device)}
    for d in cfg.resynth_val:
        rflat = consolidate(d, log=log)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.sort(np.random.default_rng(0).permutation(n)[: cfg.val_clips])
        val_sets[f"eval_resynth:{Path(d).name}"] = cache_batches(DataLoader(
            SynthCrops(rflat, ids, cfg.n_frames, val_channel, seed=12345, fixed=True, tract_left_weight=cfg.tract_left_weight),
            cfg.batch, num_workers=4, collate_fn=collate), vq, device)

    logf = (out / "log.jsonl").open("a", encoding="utf-8")
    if step == 0:  # the teacher's own scores on the same batches, for reference
        for vkey, vb in val_sets.items():
            logf.write(json.dumps({"step": 0, vkey + "_teacher": evaluate(teacher, vb, device), "time": time.time()}) + "\n")
        logf.flush()

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": student.state_dict(), "ema": None, "opt": opt.state_dict(), "step": step,
                    "teacher_config": asdict(scfg), "train_config": asdict(cfg), "git": _git()}, tmp)
        tmp.replace(path)

    t0, seen, acc = time.time(), 0, {}
    student.train()
    while step < cfg.steps:
        for audio, lab in loader:
            if step >= cfg.steps:
                break
            for g in opt.param_groups:
                g["lr"] = _lr(step, cfg)
            audio = audio.to(device, non_blocking=True)
            n_t = lab["f0"].shape[1]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                lp = vq.log_power_with_context(audio, n_t)
                with torch.no_grad():
                    tout = teacher(lp)
                sout = student(lp)
            valid = torch.ones(lp.shape[0], n_t, dtype=torch.bool, device=device)
            if cfg.lookahead:
                valid[:, n_t - cfg.lookahead:] = False
            loss, parts = distil_loss(sout, tout, valid, cfg.weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            seen += lab["f0"].numel()
            for k, v in {**parts, "loss": loss.detach(), "grad_norm": gn.detach()}.items():
                acc[k] = acc[k] + v if k in acc else v.float()
            if step % cfg.log_every == 0:
                rec = {"step": step, "lr": _lr(step, cfg), **{k: float(v) / cfg.log_every for k, v in acc.items()},
                       "audio_s_per_s": seen / 100 / (time.time() - t0), "time": time.time()}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log(f"step {step} loss {rec['loss']:.4f} (" + " ".join(f"{k} {rec[k]:.4f}" for k in PARTS if k in rec)
                    + f") {rec['audio_s_per_s']:.0f} s/s")
                acc, t0, seen = {}, time.time(), 0
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                save(last)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                for vkey, vb in val_sets.items():
                    ev, ag = evaluate(student, vb, device), agreement(student, teacher, vb, device)
                    logf.write(json.dumps({"step": step, vkey: ev, vkey + "_agree": ag, "time": time.time()}) + "\n")
                    logf.flush()
                    log(vkey + " agree " + " ".join(f"{k} {v:.3f}" for k, v in ag.items()))
                save(out / f"ckpt_{step:06d}.pt")
    logf.close()
    return out
