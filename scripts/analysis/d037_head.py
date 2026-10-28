"""D-037 (draft): a VTL head refitted on one trunk's frozen features (d037_cache.py). D-035's head (LayerNorm -> 256 ->
GELU -> 256 -> GELU -> mean, log variance of log2 cm) and recipe (AdamW 1e-3, wd 0.01, 300 warm-up steps, cosine, batch 32
crops with all their sonorant frames, Gaussian NLL), for 20,000 steps.
Inputs: "h" = the formant encoder's output; "hF" = h plus the trunk's own F1-F4 (log2 Hz, detached by construction,
standardised with training-set statistics: the mean and SD over sonorant frames of the first 2,000 training crops).
Every 500 steps: training NLL (mean of the last 500 steps) and validation NLL / SNR on train_v2's held-out tube clips
with length not drawn from f0 (sonorant frames; per-clip mean NLL, averaged over clips; SNR = 20 log10(label SD / sigma)
with sigma = exp(NLL - 1/2)). Overfitting flag: final validation NLL more than 0.05 nats above the run's minimum.
Resumable: a checkpoint every 500 steps (a cut-off run continues from it). Args: cache dir, inputs (h | hF), seed,
out dir. Env: VA_STEPS (code checks only), VA_DEVICE (default cpu), VA_OPT (adamw, or adan for D-040's check)."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parakeet"))
from vtl_head import Head, nll  # noqa: E402

STEPS, WARM, BATCH, EVERY = int(os.environ.get("VA_STEPS", 20000)), 300, 32, 500
OPT = os.environ.get("VA_OPT", "adamw")  # "adan": D-040's optimiser check (same schedule)
FLAG_NATS = 0.05
LABEL_SD = 0.2852  # log2 SD of the labelled tube training crops (d037_noise_floor.py), so SNRs compare across entries


def inputs(feats, log2F, stats, kind):
    x = np.asarray(feats, np.float32)
    if kind == "hF":
        x = np.concatenate([x, (np.asarray(log2F, np.float32) - stats[0]) / stats[1]], -1)
    return x


def main():
    cache, kind, seed, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    assert kind in ("h", "hF"), kind
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done").exists():
        print(f"{out}: done")
        return
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    tr = cache / "train"
    feats = np.load(tr / "feats.npy", mmap_mode="r")
    f_tr = np.load(tr / "log2F.npy", mmap_mode="r")
    son = np.load(tr / "label_sonorant.npy", mmap_mode="r")
    y_tr = np.load(tr / "log2_length.npy")
    ok = np.array([(np.asarray(son[i]) > 0.5).sum() > 0 for i in range(len(y_tr))])
    usable = np.flatnonzero(ok)
    first = usable[:2000]
    fr = np.concatenate([np.asarray(f_tr[i])[np.asarray(son[i]) > 0.5] for i in first])
    stats = (fr.mean(0), fr.std(0))

    flat = Path("data/synth/train_v2/flat")
    clip = np.load(cache / "val" / "clip.npy")
    L = np.log2(np.load(flat / "clip_length_cm.npy")[clip])
    keep = np.load(flat / "clip_tube.npy")[clip] & ~np.load(flat / "clip_length_from_f0.npy")[clip]
    vson = np.load(Path("cache/emb_v1/synth_train_v2/sonorant.npy")).reshape(512, 400)[:len(clip)] > 0.5
    vf, vF = np.load(cache / "val" / "feats.npy"), np.load(cache / "val" / "log2F.npy")
    val = [(torch.from_numpy(inputs(vf[c][vson[c]], vF[c][vson[c]], stats, kind)).to(dev), float(L[c]))
           for c in np.flatnonzero(keep) if vson[c].any()]

    in_dim = 128 + (4 if kind == "hF" else 0)
    torch.manual_seed(seed)
    head = Head(in_dim).to(dev)
    if OPT == "adan":  # D-040's check: the rate and decay a head would get in rA's Adan group (2e-3 x rest_lr_scale 5, wd 0.007)
        from adan_pytorch import Adan

        opt = Adan(head.parameters(), lr=1e-2, weight_decay=0.007)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / WARM if s < WARM else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - WARM) / max(1, STEPS - WARM)))))
    rng = np.random.default_rng(seed)
    curve, run, start = [], [], 0
    ck = out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cpu", weights_only=False)
        head.load_state_dict(s["head"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        rng.bit_generator.state = s["rng"]
        torch.set_rng_state(s["torch_rng"])
        curve, start = s["curve"], s["step"]
        print(f"resumed at step {start}", flush=True)

    def validate():
        head.eval()
        with torch.inference_mode():
            v = [float(nll(*head(x), torch.full((len(x),), y, device=dev))) for x, y in val]
        head.train()
        n = float(np.mean(v))
        return n, 20 * math.log10(LABEL_SD / math.exp(n - 0.5))

    t0 = time.time()
    for step in range(start, STEPS):
        pick = rng.choice(usable, size=BATCH)
        xs, ys = [], []
        for i in pick:
            s = np.asarray(son[i]) > 0.5
            xs.append(inputs(feats[i][s], f_tr[i][s], stats, kind))
            ys.append(np.full(s.sum(), y_tr[i], np.float32))
        x, y = torch.from_numpy(np.concatenate(xs)).to(dev), torch.from_numpy(np.concatenate(ys)).to(dev)
        loss = nll(*head(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run.append(loss.item())
        if (step + 1) % EVERY == 0 or step + 1 == STEPS:
            vn, vs = validate()
            curve.append({"step": step + 1, "train_nll": float(np.mean(run[-EVERY:])), "val_nll": vn, "val_snr_db": vs})
            run = []
            print(f"{time.strftime('%H:%M:%S')} {kind} s{seed} step {step + 1}: train {curve[-1]['train_nll']:+.4f}  "
                  f"val {vn:+.4f} ({vs:.2f} dB)  {time.time() - t0:.0f} s", flush=True)
            torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "curve": curve, "step": step + 1}, ck)
    vmin = min(c["val_nll"] for c in curve)
    flag = curve[-1]["val_nll"] - vmin > FLAG_NATS
    torch.save({"state": head.state_dict(), "in_dim": in_dim, "kind": kind, "seed": seed, "stats": stats,
                "n_crops": int(len(usable))}, out / "head.pt")
    (out / "curve.json").write_text(json.dumps(curve, indent=1))
    (out / "done").write_text(json.dumps({"kind": kind, "seed": seed, "steps": STEPS, "n_crops": int(len(usable)),
                                          "final_val_nll": curve[-1]["val_nll"], "final_val_snr_db": curve[-1]["val_snr_db"],
                                          "min_val_nll": vmin, "overfit_flag": bool(flag)}))
    print(f"{out}: final val {curve[-1]['val_nll']:+.4f} ({curve[-1]['val_snr_db']:.2f} dB), min {vmin:+.4f}, overfit flag {flag}")


if __name__ == "__main__":
    main()
