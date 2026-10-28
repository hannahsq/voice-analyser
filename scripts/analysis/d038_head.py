"""D-038 (draft): fine-tune a trunk's own F1-F4 heads (the cascade with its soft masks, `Teacher.forward`) on D-037's frozen
features, with or without the trunk's joint VTL head's output as an extra input.
Arms: F (h alone, the control); Fv (+ log2 length and log sigma); Fi (+ 1/L in cm^-1 and log sigma). The extra inputs are
detached, standardised with the training part's sonorant-frame statistics, and enter each formant head through its own
zero-initialised weights (the same as appending them to the head's input with zero columns), so all three arms start
from the trunk's function.
Training crops: the cache's `mix` part (teacher v3's training mixture, every clip kind and its resynthesised sets; the
first noise-floor check used D-037's tube-only `train` part, which pulled F3 towards tube clips; VA_D038_PART selects).
Recipe (D-037's schedule shape): AdamW 1e-3, wd 0.01, 300 warm-up steps, cosine to 0, 20,000 steps, batch 32 crops, the
teacher's formant loss (blurred heat-map CE averaged over F1-F4, sonorant frames, weighted by w_formant). Validation formant loss every 500
steps on the val part (labels from cache/emb_v1/synth_train_v2, checked equal by d037_cache.py), on all its clips
(every train_v2 kind; the overfitting flag) and on its tube clips (reported): a head is flagged if its final validation
loss on all clips is more than 2 % above its own minimum. Resumable (ckpt.pt every 500 steps).
Usage: python d038_head.py <cache dir> <arm> <seed> <out dir>   (env VA_DEVICE, default cuda; VA_STEPS for code checks)
       python d038_head.py <cache dir> check   (step-0 check: every arm reproduces the cached F1-F4 on val)"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vocal_analysis.models.bins import FORMANT_AXIS, decode
from vocal_analysis.models.infer import load_teacher
from vocal_analysis.train.losses import heatmap_ce

STEPS, WARM, BATCH, LOG_EVERY = int(os.environ.get("VA_STEPS", 20000)), 300, 32, 500
ARMS = ("F", "Fv", "Fi")
EMB = Path("cache/emb_v1/synth_train_v2")
PART = os.environ.get("VA_D038_PART", "mix")


def extra_raw(arm, mu, logvar):
    """-> (..., 2) unstandardised extra inputs, or None for F"""
    if arm == "F":
        return None
    a = mu if arm == "Fv" else np.exp2(-mu)  # log2 cm, or 1/L in cm^-1
    return np.stack([a, 0.5 * logvar], -1).astype(np.float32)


def extra_stats(cache, arm):
    d = cache / PART
    son = np.load(d / "label_sonorant.npy", mmap_mode="r")
    mu, lv = np.load(d / "vtl_mu.npy", mmap_mode="r"), np.load(d / "vtl_logvar.npy", mmap_mode="r")
    xs = []
    for i in range(0, len(son), 4096):
        m = np.asarray(son[i:i + 4096]) > 0.5
        xs.append(extra_raw(arm, np.asarray(mu[i:i + 4096])[m], np.asarray(lv[i:i + 4096])[m]))
    x = np.concatenate(xs).astype(np.float64)
    return x.mean(0).astype(np.float32), x.std(0).astype(np.float32)


class Heads(nn.Module):
    """the trunk's formant heads, as Teacher.forward, plus zero-initialised weights for the extra inputs"""

    def __init__(self, teacher, n_extra):
        super().__init__()
        self.heads = teacher.formant_heads
        self.margin, self.strength = teacher.cfg.mask_margin_bins, teacher.cfg.mask_strength
        self.register_buffer("bins", torch.arange(FORMANT_AXIS.n_bins, dtype=torch.float32), persistent=False)
        self.extra = nn.ModuleList([nn.Linear(n_extra, FORMANT_AXIS.n_bins, bias=False) for _ in self.heads]) if n_extra else None
        if self.extra is not None:
            for e in self.extra:
                nn.init.zeros_(e.weight)

    def forward(self, h, x=None):  # h (..., 128), x (..., n_extra) standardised -> logits (..., 4, bins)
        logits, means = [], []
        for n, head in enumerate(self.heads):
            inp = torch.cat([h] + [(m / FORMANT_AXIS.n_bins)[..., None] for m in means], -1) if means else h
            lg = head(inp)
            if self.extra is not None:
                lg = lg + self.extra[n](x)
            if means:
                below = (means[-1][..., None] - self.margin - self.bins).clamp_min(0.0)
                lg = lg - self.strength * torch.tanh(below / self.margin)
            logits.append(lg)
            means.append((lg.softmax(-1) * self.bins).sum(-1))
        return torch.stack(logits, -2)


def build_model(ckpt, arm, dev):
    return Heads(load_teacher(ckpt, "cpu")[0], 0 if arm == "F" else 2).to(dev)


def build(cache, arm, dev):
    ckpt = json.loads((cache / "train" / "done").read_text())["ckpt"]
    return build_model(ckpt, arm, dev), None if arm == "F" else extra_stats(cache, arm), ckpt


def standardise(arm, stats, mu, lv):
    x = extra_raw(arm, mu, lv)
    return None if x is None else (x - stats[0]) / stats[1]


def formant_loss(logits, fm_hz, son, w=None):
    return sum(heatmap_ce(logits[..., n, :], fm_hz[..., n], son, FORMANT_AXIS, w) for n in range(4)) / 4


def val_data(cache):
    d = cache / "val"
    n = len(np.load(d / "clip.npy"))
    fm = np.load(EMB / "formants.npy").reshape(512, 400, 4)[:n]
    son = np.load(EMB / "sonorant.npy").reshape(512, 400)[:n]
    mu = np.load(d / "vtl_mu.npy") if (d / "vtl_mu.npy").exists() else None
    lv = np.load(d / "vtl_logvar.npy") if mu is not None else None
    ids = np.load(d / "clip.npy")
    cfg = json.load(open("runs/conformer_v2r2_s1/config.json"))  # d037_cache.py's data
    tube = np.load(Path(cfg["train"]["data"]) / "flat" / "clip_tube.npy")[ids]
    return np.load(d / "feats.npy"), fm, son, mu, lv, np.load(d / "log2F.npy"), tube


@torch.no_grad()
def predict(model, h, x, dev, chunk=64):
    """-> (n, T, 4) logits as float32 numpy, in chunks of crops (short GPU bursts)"""
    out = []
    for i in range(0, len(h), chunk):
        xi = None if x is None else torch.from_numpy(x[i:i + chunk]).to(dev)
        out.append(model(torch.from_numpy(np.asarray(h[i:i + chunk], np.float32)).to(dev), xi).float().cpu())
    return torch.cat(out)


def val_loss(model, V, arm, stats, dev):
    """-> (all clips, tube clips)"""
    h, fm, son, mu, lv, _, tube = V
    x = None if arm == "F" else standardise(arm, stats, mu, lv)
    lg = predict(model, h, x, dev)
    fm, son = torch.from_numpy(fm), torch.from_numpy(son)
    return float(formant_loss(lg, fm, son)), float(formant_loss(lg[tube], fm[tube], son[tube]))


def log2_hz(logits):
    return np.log2(decode(torch.softmax(logits, -1).numpy(), FORMANT_AXIS)[0])


def check(cache):
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    V = val_data(cache)
    ref = V[5]
    son = V[2]
    res = {}
    for arm in ARMS:
        model, stats, _ = build(cache, arm, dev)
        x = None if arm == "F" else standardise(arm, stats, V[3], V[4])
        f = log2_hz(predict(model.eval(), V[0], x, dev))
        res[arm] = f
        d = np.abs(f - ref)[son] * 1200
        print(f"{arm}: |step-0 - cached F1-F4| on sonorant val frames, cents: median {np.median(d, 0).round(3)}, "
              f"max {d.max(0).round(2)}; val loss (all, tube) {np.round(val_loss(model, V, arm, stats, dev), 5)}", flush=True)
    for arm in ("Fv", "Fi"):
        print(f"{arm} vs F: max |difference| {np.abs(res[arm] - res['F']).max() * 1200:.2e} cents")


def main():
    cache = Path(sys.argv[1])
    if sys.argv[2] == "check":
        return check(cache)
    arm, seed, out = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    assert arm in ARMS, arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / "head.pt").exists():
        print(f"{out}: done")
        return
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    torch.manual_seed(seed)
    model, stats, ckpt = build(cache, arm, dev)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARM) / max(1, STEPS - WARM)))))
    d = cache / PART
    feats = np.load(d / "feats.npy", mmap_mode="r")
    lab_w = np.load(d / "label_w_formant.npy", mmap_mode="r") if (d / "label_w_formant.npy").exists() else None
    lab_f = np.load(d / "label_formants.npy", mmap_mode="r")
    lab_s = np.load(d / "label_sonorant.npy", mmap_mode="r")
    X = None
    if arm != "F":
        X = np.load(d / "vtl_mu.npy", mmap_mode="r"), np.load(d / "vtl_logvar.npy", mmap_mode="r")
    V = val_data(cache)
    rng = np.random.default_rng(seed)
    curve, step0 = [], 0
    if (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        rng.bit_generator.state, curve, step0 = ck["rng"], ck["curve"], ck["step"]
        print(f"resuming at step {step0}", flush=True)
    run, t0 = [], time.time()
    for step in range(step0, STEPS):
        pick = np.sort(rng.integers(len(lab_s), size=BATCH))
        h = torch.from_numpy(np.asarray(feats[pick], np.float32)).to(dev)
        fm = torch.from_numpy(np.asarray(lab_f[pick])).to(dev)
        son = torch.from_numpy(np.asarray(lab_s[pick]) > 0.5).to(dev)
        w = None if lab_w is None else torch.from_numpy(np.asarray(lab_w[pick])).to(dev)
        x = None if X is None else torch.from_numpy(standardise(arm, stats, np.asarray(X[0][pick]), np.asarray(X[1][pick]))).to(dev)
        loss = formant_loss(model(h, x), fm, son, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if (step + 1) % LOG_EVERY == 0:
            model.eval()
            vl, vt = val_loss(model, V, arm, stats, dev)
            model.train()
            curve.append({"step": step + 1, "train": float(np.mean(run[-LOG_EVERY:])), "val": vl, "val_tube": vt})
            print(f"{time.strftime('%H:%M:%S')} {arm} s{seed} step {step + 1}: train {curve[-1]['train']:.4f} val {vl:.4f} (tube {vt:.4f}) "
                  f"({time.time() - t0:.0f} s)", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "curve": curve, "step": step + 1}, out / "ckpt.pt")
    vals = [c["val"] for c in curve]
    flag = bool(vals and vals[-1] > 1.02 * min(vals))
    (out / "curve.json").write_text(json.dumps({"curve": curve, "overfit_flag": flag, "final_val": vals[-1] if vals else None,
                                                "min_val": min(vals) if vals else None}, indent=1))
    torch.save({"state": model.state_dict(), "arm": arm, "seed": seed, "trunk": ckpt, "steps": STEPS,
                "stats": None if stats is None else [s.tolist() for s in stats], "part": PART}, out / "head.pt")
    print(f"done; overfit flag {flag}", flush=True)


if __name__ == "__main__":
    main()
