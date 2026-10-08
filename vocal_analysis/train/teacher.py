"""Train the teacher on synthetic data (plan M1, M2), optionally mixed with resynthesised sets (M3).
`va train teacher` drives this.

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
from .data import TRACT_LEFT_WEIGHT, MixedCrops, SynthCrops, collate, consolidate, split_ids
from .losses import teacher_loss


@dataclass
class TrainConfig:
    data: str = "data/synth/train_v2"
    out: str = "runs/teacher_v2"
    steps: int = 40000  # wsd: end of the run (the constant phase, or decay_start + decay_steps for a branch)
    batch: int = 32
    n_frames: int = 200
    lr: float = 1e-3
    warmup: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    workers: int = 12
    # Learning rate: linear warmup, then "wsd" (constant; a branch decays as 1 - sqrt over decay_steps from
    # decay_start, Hagele et al. 2024) or "cosine" to 0 at `steps` (teacher v1).
    # "adamw" with `schedule`, or "schedulefree": AdamW-ScheduleFree (Defazio et al. 2024), which needs no decay
    # (any checkpoint is meant to be final-quality); `schedule` and the decay fields are then unused, and lr
    # warms up over `warmup` steps inside the optimiser.
    optimizer: str = "adamw"
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
    # Resynthesised sets (plan §5.2, D-016): `resynth_frac` of training items come from these, split between
    # them by clip count. Their f0/voicing/sonorant losses are masked and formant losses weighted by QC.
    # During the comb's true-f0 phase their comb comes from the tracker's f0 label.
    resynth: tuple[str, ...] = ()
    resynth_frac: float = 0.3
    resynth_val: tuple[str, ...] = ()  # held-out resynthesised sets, scored on their (exact) formants
    tract_left_weight: tuple[float, float] = TRACT_LEFT_WEIGHT


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
    """A decay branch: the parent run's recorded training config, decaying from ckpt_step to the end."""
    saved = json.loads((parent_run / "config.json").read_text())["train"]
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items() if k in TrainConfig.__dataclass_fields__}
    cfg = TrainConfig(**fields)
    if cfg.schedule != "wsd":
        raise ValueError(f"{parent_run} uses the {cfg.schedule!r} schedule; only wsd runs can branch")
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
    sf = cfg.optimizer == "schedulefree"
    if sf:
        import schedulefree

        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr, warmup_steps=cfg.warmup, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    # Schedule-free keeps the evaluation weights (an average) apart from the training point: checkpoints and
    # evaluations use the former, so they are saved in eval mode, and training resumes in train mode.
    sf_mode = lambda mode: getattr(opt, mode)() if sf else None  # noqa: E731
    step = 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        torch.manual_seed(cfg.seed + 7919 * step)  # a resumed run or branch must not replay the data from step 0
        log(f"{'resumed' if start == last else f'branched from {start}'} at step {step}")
    if start != last:
        if cfg.branch_from and cfg.decay_start != step:
            raise ValueError(f"branch decay_start {cfg.decay_start} != checkpoint step {step}")
        index = json.loads((Path(cfg.data) / "index.json").read_text())
        resynth_idx = {d: json.loads((Path(d) / "index.json").read_text()) for d in (*cfg.resynth, *cfg.resynth_val)}
        (out / "config.json").write_text(json.dumps({"train": asdict(cfg), "model": asdict(model_cfg), "git": _git(),
                                                     "data_index": index, "resynth_indexes": resynth_idx,
                                                     "n_params": n_params(model), "channel": asdict(channel)}, indent=1))
    if channel.p_codec > 0:
        from ..synth import codecs

        if not codecs.available():
            raise FileNotFoundError("channel.p_codec > 0 needs ffmpeg (winget install ffmpeg, or set VA_FFMPEG)")
    log(f"teacher: {n_params(model):,} parameters; {len(train_ids):,} train / {len(val_ids)} val clips")

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
        log(f"resynthesised: {sum(sizes):,} clips in {len(sizes)} sets, {cfg.resynth_frac:.0%} of items")
    loader = DataLoader(ds, cfg.batch, num_workers=cfg.workers, collate_fn=collate, persistent_workers=True,
                        pin_memory=False, prefetch_factor=4, drop_last=True)
    val_ds = SynthCrops(flat, val_ids, cfg.n_frames, channel, seed=12345, fixed=True)
    val_loader = DataLoader(val_ds, cfg.batch, num_workers=4, collate_fn=collate)
    resynth_val = {}
    for d in cfg.resynth_val:
        rflat = consolidate(d, log=log)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.random.default_rng(0).permutation(n)[: cfg.val_clips]  # spread over recordings, fixed
        rv = SynthCrops(rflat, np.sort(ids), cfg.n_frames, channel, seed=12345, fixed=True, tract_left_weight=cfg.tract_left_weight)
        resynth_val[Path(d).name] = DataLoader(rv, cfg.batch, num_workers=4, collate_fn=collate)
    vq = TorchVQT(device)
    rng = np.random.default_rng(cfg.seed + step)

    def save(path):
        sf_mode("eval")
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "teacher_config": asdict(model_cfg), "train_config": asdict(cfg), "git": _git()}, tmp)
        tmp.replace(path)
        sf_mode("train")

    logf = (out / "log.jsonl").open("a", encoding="utf-8")
    t0, seen = time.time(), 0
    acc: dict = {}  # name -> running sum (tensor until the log step)
    model.train()
    sf_mode("train")
    while step < cfg.steps:
        for audio, lab in loader:
            if step >= cfg.steps:
                break
            if not sf:
                for g in opt.param_groups:
                    g["lr"] = _lr(step, cfg)
            audio, lab = audio.to(device, non_blocking=True), _to(lab, device)
            p_truth = _p_comb_truth(step, cfg)
            truth = rng.random() < p_truth
            with torch.autocast(device.type, dtype=torch.bfloat16):
                lp = vq.log_power_with_context(audio, cfg.n_frames)
                comb_f0 = torch.nan_to_num(lab["f0"], nan=100.0)  # padded frames; they are unvoiced, so no comb there
                out_ = model(lp, comb_f0=comb_f0 if truth else None, comb_voiced=lab["voiced"].float() if truth else None)
            loss, parts = teacher_loss(out_, lab, cfg.loss_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            seen += audio.shape[0]
            # summed on the GPU and read once per log interval: a float() here would make every step wait for it
            for k, v in {**parts, "loss": loss.detach(), "grad_norm": gn.detach()}.items():
                acc[k] = acc[k] + v if k in acc else v.float()
            if step % cfg.log_every == 0:
                acc = {k: float(v) for k, v in acc.items()}
                dt = time.time() - t0
                rec = {"step": step, "lr": opt.param_groups[0].get("scheduled_lr", cfg.lr) if sf else _lr(step, cfg), "p_comb_truth": p_truth,
                       **{k: v / cfg.log_every for k, v in acc.items()},
                       "audio_s_per_s": seen * cfg.n_frames / 100 / dt, "time": time.time()}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log(f"step {step} loss {rec['loss']:.3f} (f0 {rec['f0']:.2f} F {rec['formants']:.2f} bw {rec['bandwidth']:.2f}) "
                    f"{rec['audio_s_per_s']:.0f} s/s")
                acc, t0, seen = {}, time.time(), 0
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                save(last)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                sf_mode("eval")
                ev = evaluate(model, val_loader, vq, device)
                logf.write(json.dumps({"step": step, "eval": ev, "time": time.time()}) + "\n")
                logf.flush()
                log("eval " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if not k.startswith("F1_dr_f0")))
                for name, rl in resynth_val.items():  # f0 and voicing here score agreement with the tracker
                    ev = evaluate(model, rl, vq, device)
                    logf.write(json.dumps({"step": step, f"eval_resynth:{name}": ev, "time": time.time()}) + "\n")
                    logf.flush()
                    log(f"eval {name} " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if k[:2] in ("F1", "F2", "F3") and "f0" not in k))
                save(out / f"ckpt_{step:06d}.pt")  # leaves schedule-free in train mode
    logf.close()
    return out
