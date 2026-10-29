"""Train the teacher on synthetic data (plan M1, M2), optionally mixed with resynthesised sets (M3).
`va train teacher` drives this.

Every run directory holds config.json (train + model config, git commit, data index),
log.jsonl (one line per log/eval step) and checkpoints; a run resumes from ckpt_last.pt.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..models.bins import F0_AXIS, FORMANT_AXIS, decode
from ..models.teacher import Teacher, TeacherConfig, n_params
from ..synth.channel import ChannelConfig
from .data import TRACT_LEFT_WEIGHT, LengthBatches, MixedCrops, SynthCrops, collate, consolidate, split_ids
from .losses import teacher_loss


@dataclass
class TrainConfig:
    data: str = "data/synth/train_v2"
    out: str = "runs/teacher_v2"
    steps: int = 40000  # wsd: end of the run (the constant phase, or decay_start + decay_steps for a branch)
    batch: int = 32
    n_frames: int = 200
    # Mixed crop lengths: (lo, hi) frames, 8 lengths spaced geometrically, one per batch, batch scaled to batch * n_frames frames (same audio
    # per step). None = every crop n_frames long. Validation also runs on fixed val_short_frames crops ("eval_short").
    crop_frames: tuple[int, int] | None = None
    crop_long_frac: float | None = None  # with crop_frames: share of frames in crops of >= ~n_frames (None: lengths uniform)
    val_short_frames: int = 50
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
    optimizer: str = "adamw"  # also "adan", "muon" (Muon + AdamW), "muon_adan" (Muon + Adan); D-030
    rest_lr_scale: float = 1.0  # "muon" / "muon_adan": the other optimiser's rate as a multiple of Muon's
    schedule: str = "wsd"
    decay_start: int | None = None
    decay_steps: int = 4000
    decay_hold: int = 0  # D-036: a branch holds the rate constant this many steps before its decay (within decay_steps)
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
    # 0.998 averages over roughly the last 500 steps. Load with "<ckpt>#ema" (models.infer.load_teacher). 0 = off:
    # the default since D-024 (no gain where the rate decays to zero); kept for runs whose rate doesn't.
    ema_decay: float = 0.0
    # Resynthesised sets (plan §5.2, D-016): `resynth_frac` of training items come from these, split between
    # them by clip count. Their f0/voicing/sonorant losses are masked and formant losses weighted by QC.
    # During the comb's true-f0 phase their comb comes from the tracker's f0 label.
    resynth: tuple[str, ...] = ()
    resynth_frac: float = 0.3
    resynth_val: tuple[str, ...] = ()  # held-out resynthesised sets, scored on their (exact) formants
    # D-051: further synthetic sets (e.g. long articulation clips): `synth_extra_frac` of items, split by clip count;
    # the rest keeps the main set and resynthesised sets in their usual proportions. Their last 2 % of clips are held
    # out and scored as "eval_extra:<name>". 0 / () = earlier runs' mix exactly
    synth_extra: tuple[str, ...] = ()
    synth_extra_frac: float = 0.0
    tract_left_weight: tuple[float, float] = TRACT_LEFT_WEIGHT
    # D-036: an anatomical vocal-tract-length head added at a branch. None = none; "new" (initialised with vtl_seed);
    # "frozen:<head.pt>" (D-035's trained head, not updated); "tepid:<head.pt>" (shrink and perturb: 0.5 x D-035's
    # weights + 0.1 x a fresh initialisation). Its own parameter group: linear warmup over vtl_warmup steps from the
    # branch, then cosine to 0 at `steps`. Gradient into the trunk gated: 0 for vtl_detach steps, then a linear ramp to
    # 1 over vtl_gate_ramp steps. Labels: train.data's "vtl"; loss weight loss_weights["vtl"].
    vtl: str | None = None
    # D-051: also the instantaneous-length head (TeacherConfig.vtl_inst_head), initialised with vtl_seed + 1, in the
    # VTL head's parameter group, gate and schedule; labels train.data's "vtl_inst"; loss weight loss_weights["vtl_inst"]
    vtl_inst: bool = False
    vtl_seed: int = 0
    vtl_warmup: int = 300
    vtl_detach: int = 300
    vtl_gate_ramp: int = 700
    # D-040: where the VTL group lives. "rest": in the optimiser of the non-Muon parameters (AdamW or Adan; under "muon" /
    # "muon_adan" at rest_lr_scale x the rate, like the other heads), which for "adamw" runs is D-036's setup exactly.
    # "adamw": AdamW at vtl_lr and vtl_weight_decay (None: lr and weight_decay), a separate optimiser under Muon.
    vtl_optimizer: str = "rest"
    # D-040: the head's rate. "cosine": D-036's (warm-up from the branch, cosine to 0 at `steps`). "model": warm-up over
    # vtl_warmup steps from vtl_detach (the gate's opening, counted from the branch or from step 0), then the model's own
    # schedule's shape (held, then decayed with it), so a from-scratch head starts on a trunk past its warm-up.
    vtl_schedule: str = "cosine"
    # D-040: the step the head's clock counts from. None: the branch point (decay_start, or 0), D-036's case, where the
    # head is added at the branch. A head trained from step 0 sets 0, which a later decay branch keeps (branch_config
    # copies it), so the branch neither restarts the head's warm-up nor closes its gate again.
    vtl_start: int | None = None
    vtl_lr: float | None = None
    vtl_weight_decay: float | None = None


def _git() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    # dirty = uncommitted changes to the code that trains (tracked files under vocal_analysis/ and configs/), listed;
    # untracked reports no longer count (runs before 2026-10-22 were "dirty" whenever any file was)
    changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "vocal_analysis", "configs"],
                             capture_output=True, text=True).stdout.splitlines()  # unstripped: the status columns matter
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(changed), "dirty_files": [c[3:] for c in changed]}


def _saved_channel(out: Path) -> ChannelConfig:
    """The channel a run was started with. Runs from before the channel was recorded predate codecs."""
    saved = json.loads((out / "config.json").read_text()).get("channel")
    if saved is None:
        return replace(ChannelConfig(), p_codec=0.0)
    return ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})


HEAD_MODULES = ("f0_head", "voicing_head", "formant_heads", "bw_head", "sonorant_head", "obs_head")


class _Combined:
    """Two optimisers stepped together (Muon for hidden 2-D weights, another for the rest), exposing their param
    groups as one list so the learning-rate schedule sets both."""

    def __init__(self, *opts):
        self.opts = list(opts)

    def add(self, opt):
        """a further optimiser stepped with the others (D-040: the VTL head's own AdamW); its state comes last"""
        self.opts.append(opt)

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s_ in zip(self.opts, sd["combined"]):
            o.load_state_dict(s_)


def muon_split(model) -> tuple[list, list]:
    """(Muon parameters, the rest): the weights of nn.Linear layers outside the output heads (Conformer attention and
    feed-forward, projections); convolutions, recurrent weights, biases, norms and heads stay with the other optimiser."""
    import torch.nn as nn

    muon_ids = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not name.split(".")[0] in HEAD_MODULES:
            muon_ids.add(id(mod.weight))
    params = list(model.parameters())
    return [p for p in params if id(p) in muon_ids], [p for p in params if id(p) not in muon_ids]


def _optimizer(model, cfg: "TrainConfig"):
    """AdamW, Adan (Xie et al. 2022; adan-pytorch), or Muon (torch.optim.Muon, Jordan et al. 2024) for the hidden 2-D
    weights with AdamW / Adan for the rest ("muon", "muon_adan"). Muon's update is scaled to AdamW's RMS
    ("match_rms_adamw"), so the same learning rate and weight decay apply to every group (D-030)."""
    import torch

    def rest(params):
        if cfg.optimizer in ("adan", "muon_adan"):
            from adan_pytorch import Adan

            return Adan(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.optimizer in ("adamw", "adan"):  # the VTL head (D-036) gets its own group, added by the trainer
        return rest([p for n, p in model.named_parameters() if not n.startswith(VTL_PREFIXES)])
    if cfg.optimizer in ("muon", "muon_adan"):
        vtl_ids = {id(p) for n, p in model.named_parameters() if n.startswith(VTL_PREFIXES)}  # added by the trainer
        mu, other = ([p for p in ps if id(p) not in vtl_ids] for ps in muon_split(model))
        other_opt = rest(other)
        for g in other_opt.param_groups:
            g["lr_scale"] = cfg.rest_lr_scale  # the schedule multiplies it in
        return _Combined(torch.optim.Muon(mu, lr=cfg.lr, weight_decay=cfg.weight_decay, adjust_lr_fn="match_rms_adamw"), other_opt)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if cfg.schedule == "wsd":
        if cfg.decay_start is None or step < cfg.decay_start:
            return cfg.lr
        into = step - cfg.decay_start - cfg.decay_hold
        if into < 0:
            return cfg.lr
        return cfg.lr * (1.0 - math.sqrt(min(into / max(1, cfg.decay_steps - cfg.decay_hold), 1.0)))
    if cfg.schedule == "cosine":
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def _add_vtl_group(opt, params: list, cfg: TrainConfig) -> None:
    """D-036 / D-040: the VTL head's parameters as their own group, scheduled by _vtl_lr (the trainer multiplies in the
    group's lr_scale); where it lives is cfg.vtl_optimizer."""
    import torch

    if not params:
        return
    wd = cfg.weight_decay if cfg.vtl_weight_decay is None else cfg.vtl_weight_decay
    scale = (cfg.vtl_lr if cfg.vtl_lr is not None else cfg.lr) / cfg.lr
    group = {"params": params, "lr": 0.0, "weight_decay": wd, "vtl": True}
    if cfg.vtl_optimizer == "rest":
        if cfg.vtl_lr is not None or cfg.vtl_weight_decay is not None:
            raise ValueError("vtl_lr / vtl_weight_decay apply to vtl_optimizer='adamw' only")
        if isinstance(opt, _Combined):  # the non-Muon optimiser, at its rate
            opt.opts[1].add_param_group({**group, "lr_scale": cfg.rest_lr_scale})
        else:
            opt.add_param_group(group)
    elif cfg.vtl_optimizer == "adamw":
        if isinstance(opt, _Combined):
            own = torch.optim.AdamW(params, lr=0.0, weight_decay=wd)
            own.param_groups[0].update(vtl=True, lr_scale=scale)
            opt.add(own)
        elif isinstance(opt, torch.optim.AdamW):
            opt.add_param_group({**group, "lr_scale": scale})
        else:
            raise NotImplementedError("vtl_optimizer='adamw' beside a single non-AdamW optimiser")
    else:
        raise ValueError(f"unknown vtl_optimizer {cfg.vtl_optimizer!r}")


def _vtl_origin(cfg: TrainConfig) -> int:
    return cfg.vtl_start if cfg.vtl_start is not None else (cfg.decay_start or 0)


def _vtl_lr(step: int, cfg: TrainConfig) -> float:
    """D-036: the VTL head's rate: linear warmup from the branch, then cosine to 0 at the run's end. D-040
    (vtl_schedule "model"): linear warm-up from the gate's opening, times the model's schedule shape."""
    if cfg.vtl_schedule == "model":
        rel = step - _vtl_origin(cfg) - cfg.vtl_detach
        return 0.0 if rel < 0 else _lr(step, cfg) * min(1.0, (rel + 1) / cfg.vtl_warmup)
    if cfg.vtl_schedule != "cosine":
        raise ValueError(f"unknown vtl_schedule {cfg.vtl_schedule!r}")
    rel, span = step - _vtl_origin(cfg), cfg.steps - _vtl_origin(cfg)
    if rel < cfg.vtl_warmup:
        return cfg.lr * (rel + 1) / cfg.vtl_warmup
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min((rel - cfg.vtl_warmup) / max(1, span - cfg.vtl_warmup), 1.0)))


def _vtl_gate(step: int, cfg: TrainConfig) -> float:
    rel = step - _vtl_origin(cfg) - cfg.vtl_detach
    return 0.0 if rel < 0 else min(1.0, rel / max(1, cfg.vtl_gate_ramp))


VTL_PREFIXES = ("vtl_head.", "vtl_inst_head.")  # D-036 / D-051: the heads in the VTL parameter group


def _vtl_params(model) -> list:
    return [p for n, p in model.named_parameters() if n.startswith(VTL_PREFIXES) and p.requires_grad]


def _add_vtl_head(model, cfg: TrainConfig, log) -> None:
    """Initialise the VTL head per cfg.vtl (see TrainConfig)."""
    import torch

    if hasattr(model, "vtl_inst_head"):  # D-051: always fresh (no saved head exists)
        torch.manual_seed(cfg.vtl_seed + 1)
        for m in model.vtl_inst_head:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    torch.manual_seed(cfg.vtl_seed)
    for m in model.vtl_head:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    kind, _, path = cfg.vtl.partition(":")
    if kind == "new":
        return
    old = {k.removeprefix("net."): v for k, v in torch.load(path, map_location="cpu", weights_only=False)["state"].items()}
    if kind == "frozen":
        model.vtl_head.load_state_dict(old)
        model.vtl_head.requires_grad_(False)
    elif kind == "tepid":
        fresh = {k: v.detach().cpu().clone() for k, v in model.vtl_head.state_dict().items()}
        model.vtl_head.load_state_dict({k: 0.5 * old[k] + 0.1 * fresh[k] for k in old})
    else:
        raise ValueError(f"unknown vtl {cfg.vtl!r}")
    log(f"VTL head: {kind} from {path}")


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


def cache_batches(loader, vq, device) -> list:
    """A fixed validation set as (VQT log-power, labels) batches on the device, computed once as evaluate() used to
    each time (same autocast). The VQT has no learned parameters, so the cache stays valid; evals then need no loader
    workers and no codec work."""
    import torch

    out = []
    with torch.inference_mode():
        for audio, lab in loader:
            audio, lab = audio.to(device), _to(lab, device)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out.append((vq.log_power_with_context(audio, lab["f0"].shape[1]), lab))
    return out


def evaluate(model, batches, device) -> dict:
    """Synthetic validation (cache_batches): formant MAE / DR(20 %, 200 Hz), f0 RPA, voicing accuracy, by f0 band; the
    VTL head (D-036) against its label where both exist."""
    import torch

    model.eval()
    rows = {"fm_err": [], "fm_ref": [], "f0_c": [], "voiced_ok": [], "f0_ref": [], "son": [], "obs_p": [], "obs": [],
            "vtl_mu": [], "vtl_lv": [], "vtl": [], "vtl_inst_mu": [], "vtl_inst_lv": [], "vtl_inst": []}
    with torch.inference_mode():
        for lp, lab in batches:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                out = model(lp)
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
            if "obs_logit" in out and "obs" in lab:
                rows["obs_p"].append(torch.sigmoid(out["obs_logit"].float()).cpu().numpy())
                rows["obs"].append(lab["obs"].cpu().numpy())
            if "vtl_mu" in out and "vtl" in lab:
                rows["vtl_mu"].append(out["vtl_mu"].float().cpu().numpy())
                rows["vtl_lv"].append(out["vtl_logvar"].float().clamp(-12, 4).cpu().numpy())  # as the loss clamps it
                rows["vtl"].append(lab["vtl"].cpu().numpy())
            if "vtl_inst_mu" in out and "vtl_inst" in lab:
                rows["vtl_inst_mu"].append(out["vtl_inst_mu"].float().cpu().numpy())
                rows["vtl_inst_lv"].append(out["vtl_inst_logvar"].float().clamp(-12, 4).cpu().numpy())
                rows["vtl_inst"].append(lab["vtl_inst"].cpu().numpy())
    model.train()
    rows = {k: v for k, v in rows.items() if v}
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
    if "obs" in cat:  # D-027: the observability head against its label (sonorant frames with one)
        from scipy.stats import spearmanr

        op, ol = cat["obs_p"][son], cat["obs"][son]
        for n in range(ol.shape[1]):
            ok = np.isfinite(ol[:, n])
            if ok.sum() > 1:
                res[f"F{n + 1}_obs_mae"] = float(np.mean(np.abs(op[ok, n] - ol[ok, n])))
                res[f"F{n + 1}_obs_rho"] = float(spearmanr(op[ok, n], ol[ok, n])[0])
    # D-036 head ("vtl") and D-051's instantaneous head ("vtl_inst"), log2 cm, on sonorant frames with a label (tube
    # clips whose length was not drawn from f0)
    for key in ("vtl", "vtl_inst"):
        if key not in cat:
            continue
        y = cat[key]
        ok = son & np.isfinite(y)
        if ok.any():
            mu, lv, y = cat[f"{key}_mu"][ok], cat[f"{key}_lv"][ok], y[ok]
            ratio = np.exp2(mu - y)
            res[f"{key}_mae_pct"] = float(100 * np.mean(np.abs(ratio - 1)))
            res[f"{key}_bias_pct"] = float(100 * (np.exp2(np.median(mu - y)) - 1))
            res[f"{key}_nll"] = float(np.mean(0.5 * (lv + (y - mu) ** 2 / np.exp(lv))))  # the training loss's form
            res[f"{key}_cover1"] = float(np.mean(np.abs(mu - y) <= np.exp(lv / 2)))  # ideal 0.683
            res[f"{key}_frames"] = int(ok.sum())
    return res


def _raise_priority(log) -> None:
    """$VA_PRIORITY=above_normal: the training process (whose thread queues the GPU work) above normal priority on
    Windows. Loader workers start at normal priority: Windows gives new processes normal priority unless the parent
    is below normal."""
    if os.environ.get("VA_PRIORITY") != "above_normal" or os.name != "nt":
        return
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit pseudo-handle; ctypes' default int truncates it
    k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    if k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000):  # ABOVE_NORMAL_PRIORITY_CLASS
        log("teacher: process priority above normal")
    else:
        log(f"teacher: could not raise the process priority (error {k32.GetLastError()})")


def train(cfg: TrainConfig, model_cfg: TeacherConfig | None = None, log=print) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from ..device import setup_device
    from ..features.vqt import TorchVQT

    _raise_priority(log)
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
    new_vtl = bool(cfg.vtl) and not model_cfg.vtl_head  # D-036: the head is added at this branch
    if cfg.vtl:
        if cfg.optimizer not in ("adamw", "adan", "muon", "muon_adan"):  # D-040: Muon runs keep it out of Muon
            raise ValueError("a VTL head needs adamw, adan, muon or muon_adan (its own parameter group)")
        model_cfg = replace(model_cfg, vtl_head=True, vtl_inst_head=model_cfg.vtl_inst_head or cfg.vtl_inst)
    elif cfg.vtl_inst:
        raise ValueError("vtl_inst needs vtl")
    model = Teacher(model_cfg).to(device)
    sf = cfg.optimizer == "schedulefree"
    if sf:
        import schedulefree

        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=cfg.lr, warmup_steps=cfg.warmup, weight_decay=cfg.weight_decay)
    else:
        opt = _optimizer(model, cfg)
    # Schedule-free keeps the evaluation weights (an average) apart from the training point: checkpoints and
    # evaluations use the former, so they are saved in eval mode, and training resumes in train mode.
    sf_mode = lambda mode: getattr(opt, mode)() if sf else None  # noqa: E731
    step = 0
    ema = None
    vtl_params = _vtl_params(model) if cfg.vtl else []

    def add_vtl_group():
        _add_vtl_group(opt, vtl_params, cfg)

    if cfg.vtl and not new_vtl:
        if cfg.vtl.startswith("frozen"):
            model.vtl_head.requires_grad_(False)
            vtl_params = []
        add_vtl_group()  # resuming: the saved optimiser state already has the group
    if ck is not None:
        if new_vtl:
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            assert not unexpected and all(k.startswith(VTL_PREFIXES) for k in missing), (missing, unexpected)
        else:
            model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        torch.manual_seed(cfg.seed + 7919 * step)  # a resumed run or branch must not replay the data from step 0
        log(f"{'resumed' if start == last else f'branched from {start}'} at step {step}")
    if new_vtl:
        _add_vtl_head(model, cfg, log)
        vtl_params = _vtl_params(model)
        add_vtl_group()
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
    obs = model_cfg.obs_head  # D-027: the loaders add observability labels (NaN where a set has no store)
    if obs and not (flat / "obs_sig.npy").exists():
        raise FileNotFoundError(f"obs_head needs {flat}'s observability store (obs_store.write_store)")
    vtl, vti = bool(cfg.vtl), bool(cfg.vtl_inst)
    ds = SynthCrops(flat, train_ids, cfg.n_frames, channel, seed=cfg.seed, epoch_size=epoch, observability=obs, vtl=vtl,
                    vtl_inst=vti)
    if cfg.resynth:
        parts, sizes = [], []
        for d in cfg.resynth:
            rflat = consolidate(d, log=log)
            n = json.loads((rflat / "done").read_text())["n_clips"]
            parts.append(SynthCrops(rflat, np.arange(n), cfg.n_frames, channel, seed=cfg.seed + 1 + len(parts),
                                    epoch_size=epoch, tract_left_weight=cfg.tract_left_weight, observability=obs, vtl=vtl,
                                    vtl_inst=vti))
            sizes.append(n)
        weights = [1.0 - cfg.resynth_frac] + [cfg.resynth_frac * n / sum(sizes) for n in sizes]
        if not cfg.synth_extra:
            ds = MixedCrops([ds, *parts], weights, epoch, seed=cfg.seed)
        log(f"resynthesised: {sum(sizes):,} clips in {len(sizes)} sets, {cfg.resynth_frac:.0%} of items")
    extra_val = {}
    if cfg.synth_extra:  # D-051: the rest keeps its proportions, scaled by 1 - synth_extra_frac
        base_parts, base_w = ([ds, *parts], weights) if cfg.resynth else ([ds], [1.0])
        xparts, xsizes = [], []
        for d in cfg.synth_extra:
            xflat = consolidate(d, log=log)
            n = json.loads((xflat / "done").read_text())["n_clips"]
            xtr, xva = split_ids(n)
            xparts.append(SynthCrops(xflat, xtr, cfg.n_frames, channel, seed=cfg.seed + 101 + len(xparts), epoch_size=epoch,
                                     observability=obs, vtl=vtl, vtl_inst=vti))
            xsizes.append(len(xtr))
            extra_val[f"eval_extra:{Path(d).name}"] = SynthCrops(xflat, xva[: cfg.val_clips], cfg.n_frames, channel, seed=12345,
                                                                  fixed=True, observability=obs, vtl=vtl, vtl_inst=vti)
        f = cfg.synth_extra_frac
        weights = [(1.0 - f) * w for w in base_w] + [f * n / sum(xsizes) for n in xsizes]
        ds = MixedCrops([*base_parts, *xparts], weights, epoch, seed=cfg.seed)
        log(f"extra synthetic: {sum(xsizes):,} clips in {len(xsizes)} sets, {f:.0%} of items")
    common = dict(num_workers=cfg.workers, collate_fn=collate, persistent_workers=True, pin_memory=False, prefetch_factor=4)
    if cfg.crop_frames:
        batches = LengthBatches(1000, epoch, tuple(cfg.crop_frames), cfg.batch * cfg.n_frames, seed=cfg.seed + step,
                                crop_span=cfg.n_frames, long_frac=cfg.crop_long_frac)
        loader = DataLoader(ds, batch_sampler=batches, **common)
    else:
        loader = DataLoader(ds, cfg.batch, drop_last=True, **common)
    # vtl only adds the label (no random draws), so the crops stay those of runs before it was passed
    val_ds = SynthCrops(flat, val_ids, cfg.n_frames, channel, seed=12345, fixed=True, observability=obs, vtl=vtl, vtl_inst=vti)
    short_ds = SynthCrops(flat, val_ids, cfg.val_short_frames, channel, seed=12345, fixed=True, observability=obs, vtl=vtl,
                          vtl_inst=vti)
    resynth_val = {}
    for d in cfg.resynth_val:
        rflat = consolidate(d, log=log)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.random.default_rng(0).permutation(n)[: cfg.val_clips]  # spread over recordings, fixed
        resynth_val[f"eval_resynth:{Path(d).name}"] = SynthCrops(rflat, np.sort(ids), cfg.n_frames, channel, seed=12345, fixed=True,
                                                                  tract_left_weight=cfg.tract_left_weight)
    vq = TorchVQT(device)
    val_sets = {"eval": cache_batches(DataLoader(val_ds, cfg.batch, num_workers=4, collate_fn=collate), vq, device),
                "eval_short": cache_batches(DataLoader(short_ds, cfg.batch * cfg.n_frames // cfg.val_short_frames,
                                                       num_workers=4, collate_fn=collate), vq, device)}
    for name, rv in (resynth_val | extra_val).items():
        val_sets[name] = cache_batches(DataLoader(rv, cfg.batch, num_workers=4, collate_fn=collate), vq, device)
    rng = np.random.default_rng(cfg.seed + step)

    def save(path):
        sf_mode("eval")
        tmp = path.with_suffix(".tmp")
        if ema is not None:
            for b_ema, b in zip(ema.buffers(), model.buffers()):
                b_ema.copy_(b)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict() if ema is not None else None,
                    "opt": opt.state_dict(), "step": step,
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
                    g["lr"] = (_vtl_lr if g.get("vtl") else _lr)(step, cfg) * g.get("lr_scale", 1.0)
            if cfg.vtl:
                model.vtl_gate = _vtl_gate(step, cfg)
            audio, lab = audio.to(device, non_blocking=True), _to(lab, device)
            p_truth = _p_comb_truth(step, cfg)
            truth = rng.random() < p_truth
            with torch.autocast(device.type, dtype=torch.bfloat16):
                lp = vq.log_power_with_context(audio, lab["f0"].shape[1])
                comb_f0 = torch.nan_to_num(lab["f0"], nan=100.0)  # resynthesised f0 labels can be NaN; unvoiced there, so no comb
                out_ = model(lp, comb_f0=comb_f0 if truth else None, comb_voiced=lab["voiced"].float() if truth else None)
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
                rec = {"step": step, "lr": opt.param_groups[0].get("scheduled_lr", cfg.lr) if sf else _lr(step, cfg), "p_comb_truth": p_truth,
                       **{k: v / cfg.log_every for k, v in acc.items()},
                       "audio_s_per_s": seen / 100 / dt, "time": time.time()}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                vtl_part = (f" vtl {rec['vtl']:.2f}" if "vtl" in rec else "") + (f" inst {rec['vtl_inst']:.2f}" if "vtl_inst" in rec else "")
                log(f"step {step} loss {rec['loss']:.3f} (f0 {rec['f0']:.2f} F {rec['formants']:.2f} bw {rec['bandwidth']:.2f}{vtl_part}) "
                    f"{rec['audio_s_per_s']:.0f} s/s")
                acc, t0, seen = {}, time.time(), 0
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                save(last)
            if step % cfg.eval_every == 0 or step == cfg.steps:
                sf_mode("eval")
                for key, net in (("", model), ("_ema", ema)):
                    if net is None:
                        continue
                    for vkey, batches in val_sets.items():
                        ev = evaluate(net, batches, device)
                        logf.write(json.dumps({"step": step, vkey + key: ev, "time": time.time()}) + "\n")
                        logf.flush()
                        label = vkey + (f" ({cfg.val_short_frames} frames)" if vkey == "eval_short" else "") + (" (ema)" if key else "")
                        if vkey.startswith("eval_resynth:"):  # f0 and voicing there score agreement with the tracker
                            log(label + " " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if k[:2] in ("F1", "F2", "F3") and "f0" not in k))
                        else:
                            log(label + " " + " ".join(f"{k} {v:.3f}" for k, v in ev.items() if not k.startswith("F1_dr_f0")))
                save(out / f"ckpt_{step:06d}.pt")  # leaves schedule-free in train mode
    logf.close()
    return out
