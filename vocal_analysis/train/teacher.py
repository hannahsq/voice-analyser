"""Train the teacher on synthetic data (plan M1, M2). `va train teacher` drives this.

Every run directory holds config.json (train + model config, git commit, data index),
log.jsonl (one line per log/eval step) and checkpoints; a run resumes from ckpt_last.pt.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..models.bins import F0_AXIS, FORMANT_AXIS, decode
from ..models.teacher import Teacher, TeacherConfig, n_params
from ..synth.channel import ChannelConfig
from .data import LengthBatches, SynthCrops, collate, consolidate, split_ids
from .losses import teacher_loss


@dataclass
class TrainConfig:
    data: str = "data/synth/train_v2"
    out: str = "runs/teacher_v2"
    steps: int = 40000  # wsd: end of the run (the constant phase, or decay_start + decay_steps for a branch)
    batch: int = 32
    n_frames: int = 200
    # Mixed crop lengths: (lo, hi) frames, log-uniform per batch, batch scaled to batch * n_frames frames (same audio
    # per step). None = every crop n_frames long. Validation also runs on fixed val_short_frames crops ("eval_short").
    crop_frames: tuple[int, int] | None = None
    val_short_frames: int = 50
    lr: float = 1e-3
    warmup: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    workers: int = 12
    # Learning rate: linear warmup, then "wsd" (constant; a branch decays as 1 - sqrt over decay_steps from
    # decay_start, Hagele et al. 2024) or "cosine" to 0 at `steps` (teacher v1).
    schedule: str = "wsd"
    decay_start: int | None = None
    decay_steps: int = 4000
    branch_from: str | None = None  # checkpoint a decay branch starts from (recorded, not read)
    comb_truth_until: float = 0.3  # P(comb from true f0) decays 1 -> 0 over this fraction of `steps` (v1)...
    comb_truth_steps: int | None = 4000  # ...or over this many steps (wsd runs have no fixed length)
    log_every: int = 50
    eval_every: int = 1000
    ckpt_every: int = 1000
    val_clips: int = 512
    seed: int = 0
    loss_weights: dict = field(default_factory=dict)
    # Exponential moving average of the weights, kept alongside (training itself is unchanged) and saved as "ema";
    # 0.998 averages over roughly the last 500 steps. Load with "<ckpt>#ema" (models.infer.load_teacher). 0 = off.
    ema_decay: float = 0.998


def _git() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _saved_channel(out: Path) -> ChannelConfig:
    """The channel a run was started with. Runs from before the channel was recorded predate codecs."""
    saved = json.loads((out / "config.json").read_text()).get("channel")
    if saved is None:
        return replace(ChannelConfig(), p_codec=0.0)
    return ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})


def _lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if cfg.schedule == "wsd":
        if cfg.decay_start is None or step < cfg.decay_start:
            return cfg.lr
        return cfg.lr * (1.0 - math.sqrt(min((step - cfg.decay_start) / cfg.decay_steps, 1.0)))
    if cfg.schedule == "cosine":
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def _p_comb_truth(step: int, cfg: TrainConfig) -> float:
    span = cfg.comb_truth_steps if cfg.comb_truth_steps is not None else cfg.comb_truth_until * cfg.steps
    return max(0.0, 1.0 - step / max(1.0, span))


def branch_config(parent_run: Path, ckpt_step: int, out: str, decay_steps: int | None = None) -> TrainConfig:
    """A decay branch: the parent run's recorded training config, decaying from ckpt_step to the end. A cosine
    parent (teacher v1) decays from its learning rate at ckpt_step, so the rate is continuous at the branch."""
    saved = json.loads((parent_run / "config.json").read_text())["train"]
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**fields)
    if "schedule" not in saved:  # runs from before schedules were recorded (teacher v1) are cosine
        cfg.schedule, cfg.comb_truth_steps = "cosine", None
    if cfg.schedule == "cosine":
        cfg.lr, cfg.schedule = _lr(ckpt_step, cfg), "wsd"
    elif cfg.schedule != "wsd":
        raise ValueError(f"{parent_run} uses the {cfg.schedule!r} schedule; only wsd and cosine runs can branch")
    cfg.decay_steps = decay_steps or cfg.decay_steps
    cfg.decay_start = ckpt_step
    cfg.steps = ckpt_step + cfg.decay_steps
    cfg.out = out
    return cfg


def _to(lab: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in lab.items()}


def evaluate(model, loader, vq, device, max_batches: int | None = None) -> dict:
    """Synthetic validation: formant MAE / DR(20 %, 200 Hz), f0 RPA, voicing accuracy, by f0 band."""
    import torch

    model.eval()
    rows = {"fm_err": [], "fm_ref": [], "f0_c": [], "voiced_ok": [], "f0_ref": [], "son": []}
    with torch.inference_mode():
        for i, (audio, lab) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            audio, lab = audio.to(device), _to(lab, device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(vq.log_power_with_context(audio, lab["f0"].shape[1]))
            fm_p = out["formant_logits"].float().softmax(-1).cpu().numpy()
            f0_p = out["f0_logits"].float().softmax(-1).cpu().numpy()
            fm, _ = decode(fm_p, FORMANT_AXIS)
            f0, _ = decode(f0_p, F0_AXIS)
            ref_f = lab["formants"].cpu().numpy()
            ref_f0 = lab["f0"].cpu().numpy()
            voiced = lab["voiced"].cpu().numpy()
            rows["fm_err"].append(fm - ref_f)
            rows["fm_ref"].append(ref_f)
            rows["son"].append(lab["sonorant"].cpu().numpy())
            rows["f0_c"].append(np.where(voiced, 1200 * np.log2(f0 / np.where(voiced, ref_f0, 1.0)), np.nan))
            rows["f0_ref"].append(np.where(voiced, ref_f0, np.nan))
            rows["voiced_ok"].append((torch.sigmoid(out["voicing_logit"].float()).cpu().numpy() > 0.5) == voiced)
    model.train()
    cat = {k: np.concatenate([r.reshape(-1, *r.shape[2:]) for r in v]) for k, v in rows.items()}
    son = cat["son"].astype(bool)
    err, ref = cat["fm_err"][son], cat["fm_ref"][son]
    res = {}
    for n in range(err.shape[1]):
        e, r = err[:, n], ref[:, n]
        ok = np.isfinite(r)
        res[f"F{n + 1}_mae_hz"] = float(np.mean(np.abs(e[ok])))
        res[f"F{n + 1}_dr"] = float(np.mean((np.abs(e[ok]) <= 0.2 * r[ok]) & (np.abs(e[ok]) <= 200)))
    v = np.isfinite(cat["f0_c"])
    res["f0_rpa"] = float(np.mean(np.abs(cat["f0_c"][v]) <= 50))
    res["voicing_acc"] = float(np.mean(cat["voiced_ok"]))
    f0r = cat["f0_ref"][son]
    for lo, hi in ((0, 150), (150, 300), (300, 500), (500, 2000)):
        m = (f0r >= lo) & (f0r < hi) & np.isfinite(ref[:, 0])
        if m.any():
            res[f"F1_dr_f0_{lo}_{hi}"] = float(np.mean((np.abs(err[m, 0]) <= 0.2 * ref[m, 0]) & (np.abs(err[m, 0]) <= 200)))
    return res


def train(cfg: TrainConfig, model_cfg: TeacherConfig | None = None, log=print) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from ..device import setup_device
    from ..features.vqt import TorchVQT

    device = setup_device()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    flat = consolidate(cfg.data, log=log)
    n_clips = json.loads((flat / "done").read_text())["n_clips"]
    train_ids, val_ids = split_ids(n_clips)
    val_ids = val_ids[: cfg.val_clips]

    last = out / "ckpt_last.pt"
    if last.exists():  # resume this run
        start, channel = last, _saved_channel(out)
    elif cfg.branch_from:  # new decay branch: the parent's weights, optimiser state, step and channel
        start, channel = Path(cfg.branch_from), _saved_channel(Path(cfg.branch_from).parent)
    else:
        start, channel = None, ChannelConfig()
    ck = torch.load(start, map_location="cpu", weights_only=False) if start else None
    if ck is not None:
        model_cfg = TeacherConfig(**ck["teacher_config"])  # the checkpoint's architecture, whatever was passed
    model_cfg = model_cfg or TeacherConfig()
    model = Teacher(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    ema = None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        torch.manual_seed(cfg.seed + 7919 * step)  # a resumed run or branch must not replay the data from step 0
        log(f"{'resumed' if start == last else f'branched from {start}'} at step {step}")
    if cfg.ema_decay > 0:  # from the checkpoint's average if it has one, else from the current weights
        import copy

        ema = copy.deepcopy(model).requires_grad_(False)
        if ck is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        ema_params, model_params = list(ema.parameters()), list(model.parameters())
    if start != last:
        if cfg.branch_from and cfg.decay_start != step:
            raise ValueError(f"branch decay_start {cfg.decay_start} != checkpoint step {step}")
        index = json.loads((Path(cfg.data) / "index.json").read_text())
        (out / "config.json").write_text(json.dumps({"train": asdict(cfg), "model": asdict(model_cfg), "git": _git(),
                                                     "data_index": index, "n_params": n_params(model),
                                                     "channel": asdict(channel)}, indent=1))
    if channel.p_codec > 0:
        from ..synth import codecs

        if not codecs.available():
            raise FileNotFoundError("channel.p_codec > 0 needs ffmpeg (winget install ffmpeg, or set VA_FFMPEG)")
    log(f"teacher: {n_params(model):,} parameters; {len(train_ids):,} train / {len(val_ids)} val clips")

    ds = SynthCrops(flat, train_ids, cfg.n_frames, channel, seed=cfg.seed, epoch_size=cfg.batch * 1000)
    common = dict(num_workers=cfg.workers, collate_fn=collate, persistent_workers=True, pin_memory=False, prefetch_factor=4)
    if cfg.crop_frames:
        batches = LengthBatches(1000, ds.epoch_size, tuple(cfg.crop_frames), cfg.batch * cfg.n_frames, seed=cfg.seed + step)
        loader = DataLoader(ds, batch_sampler=batches, **common)
    else:
        loader = DataLoader(ds, cfg.batch, drop_last=True, **common)
    val_ds = SynthCrops(flat, val_ids, cfg.n_frames, channel, seed=12345, fixed=True)
    val_loader = DataLoader(val_ds, cfg.batch, num_workers=4, collate_fn=collate)
    short_ds = SynthCrops(flat, val_ids, cfg.val_short_frames, channel, seed=12345, fixed=True)
    short_loader = DataLoader(short_ds, cfg.batch * cfg.n_frames // cfg.val_short_frames, num_workers=4, collate_fn=collate)
    vq = TorchVQT(device)
    rng = np.random.default_rng(cfg.seed + step)

    def save(path):
        tmp = path.with_suffix(".tmp")
        if ema is not None:
            for b_ema, b in zip(ema.buffers(), model.buffers()):
                b_ema.copy_(b)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                    "opt": opt.state_dict(), "step": step,
                    "teacher_config": asdict(model_cfg), "train_config": asdict(cfg), "git": _git()}, tmp)
        tmp.replace(path)

    logf = (out / "log.jsonl").open("a", encoding="utf-8")
    t0, seen = time.time(), 0
    acc: dict = {}  # name -> running sum (tensor until the log step)
    model.train()
    while step < cfg.steps:
        for audio, lab in loader:
            if step >= cfg.steps:
                break
            for g in opt.param_groups:
                g["lr"] = _lr(step, cfg)
            audio, lab = audio.to(device, non_blocking=True), _to(lab, device)
            p_truth = _p_comb_truth(step, cfg)
            truth = rng.random() < p_truth
            with torch.autocast(device.type, dtype=torch.bfloat16):
                lp = vq.log_power_with_context(audio, lab["f0"].shape[1])
                out_ = model(lp, comb_f0=lab["f0"] if truth else None, comb_voiced=lab["voiced"].float() if truth else None)
            loss, parts = teacher_loss(out_, lab, cfg.loss_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if ema is not None:
                torch._foreach_lerp_(ema_params, model_params, 1.0 - cfg.ema_decay)
            step += 1
            seen += lab["f0"].numel()  # frames
            # summed on the GPU and read once per log interval: a float() here would make every step wait for it
            for k, v in {**parts, "loss": loss.detach(), "grad_norm": gn.detach()}.items():
                acc[k] = acc[k] + v if k in acc else v.float()
            if step % cfg.log_every == 0:
                acc = {k: float(v) for k, v in acc.items()}
                dt = time.time() - t0
                rec = {"step": step, "lr": _lr(step, cfg), "p_comb_truth": p_truth,
                       **{k: v / cfg.log_every for k, v in acc.items()},
                       "audio_s_per_s": seen / 100 / dt, "time": time.time()}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log(f"step {step} loss {rec['loss']:.3f} (f0 {rec['f0']:.2f} F {rec['formants']:.2f} bw {rec['bandwidth']:.2f}) "
                    f"{rec['audio_s_per_s']:.0f} s/s")
                acc, t0, seen = {}, time.time(), 0
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                save(last)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                for key, net in (("", model), ("_ema", ema)):
                    if net is None:
                        continue
                    for vkey, vl in (("eval", val_loader), ("eval_short", short_loader)):
                        ev = evaluate(net, vl, vq, device)
                        logf.write(json.dumps({"step": step, vkey + key: ev, "time": time.time()}) + "\n")
                        logf.flush()
                        label = vkey + (f" ({cfg.val_short_frames} frames)" if vkey == "eval_short" else "") + (" (ema)" if key else "")
                        log(label + " " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if not k.startswith("F1_dr_f0")))
                save(out / f"ckpt_{step:06d}.pt")
    logf.close()
    return out
