"""D-084: what do the other formants tell a formant head (M, frozen v3.1 features + true others), and does v3.1's
formant chain pass F1's errors on (S, the chain re-run with true fed means)?

    python scripts/analysis/d084_formant_links.py M v31_s1 v31_s2                      (VA_DEVICE=cuda)
    python scripts/analysis/d084_formant_links.py S v31_s1=<ckpt> v31_s2=<ckpt>
    python scripts/analysis/d084_formant_links.py decide v31_s1 v31_s2
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

O = Path("reports/analysis/d084")
SEEDS = (1, 2)
VAL = Path("data/synth/val_v2ko/flat")
SR, HOP = 16000, 160


def frames(trunk, split):
    from d081_oracle_heads import aligned

    h, lab = aligned(trunk, split)
    f = lab["formants"][:, :4]
    m = lab["sonorant"].astype(bool) & np.all(np.isfinite(f) & (f > 0), 1)
    return h[m].astype(np.float32), np.log(f[m])


def matrix(trunk):
    import torch

    from d081_label_oracle import gb
    from d081_oracle_heads import fit

    out = O / f"M_{trunk}.json"
    if out.exists():
        return
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    (ht, yt), (hv, yv) = frames(trunk, "train"), frames(trunk, "val")
    res = {}
    for n in range(4):
        others = [m for m in range(4) if m != n]
        r = {}
        for name, keys in [("h", [])] + [(f"h+F{m + 1}", [m]) for m in others] + [("h+others", others)]:
            Xt = np.column_stack([ht] + [yt[:, k] for k in keys])
            Xv = np.column_stack([hv] + [yv[:, k] for k in keys])
            per = {s: float(np.mean(np.abs(np.exp(fit(Xt, yt[:, n:n + 1], Xv, s, dev)[:, 0]) - np.exp(yv[:, n])))) for s in SEEDS}
            r[name] = {"by_seed": per, "mean": float(np.mean(list(per.values()))), "spread": float(abs(per[1] - per[2]))}
            print(f"{trunk} F{n + 1} {name}: {r[name]['mean']:.2f} Hz", flush=True)
        p = gb(yt[:, others], yt[:, n]).predict(yv[:, others])
        r["label only (others)"] = float(np.mean(np.abs(np.exp(p) - np.exp(yv[:, n]))))
        res[f"F{n + 1}"] = r
    O.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def chain(model, h, fed):
    """v3.1's flat read-out with the means fed to later heads (and the mask) overridden: fed[n] is a (B, T) bin index
    tensor or None (the chain's own). Returns the expected bin of each formant."""
    import torch

    from vocal_analysis.models.teacher import N_FORMANTS

    cfg, means, used = model.cfg, [], []
    for n in range(N_FORMANTS):
        inp = torch.cat([h] + [(m / model.fm_bins.numel())[..., None].to(h.dtype) for m in used], -1) if used else h
        lg = model.formant_heads[n](inp).float()
        if used:
            below = (used[-1][..., None] - cfg.mask_margin_bins - model.fm_bins).clamp_min(0.0)
            lg = lg - cfg.mask_strength * torch.tanh(below / cfg.mask_margin_bins)
        mu = (lg.softmax(-1) * model.fm_bins).sum(-1)
        means.append(mu)
        used.append(fed[n] if fed[n] is not None else mu)
    return means


def swap(name, ck):
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.bins import FORMANT_AXIS
    from vocal_analysis.models.infer import load_teacher

    out = O / f"S_{name}.json"
    if out.exists():
        return
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ck, dev)
    assert model.cfg.readout == "flat"
    vq = TorchVQT(dev)
    got = {}
    model.enc.register_forward_hook(lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    a_off, f_off = np.load(VAL / "audio_offsets.npy"), np.load(VAL / "frame_offsets.npy")
    audio = np.load(VAL / "audio.npy", mmap_mode="r")
    lab_f = np.load(VAL / "label_formants.npy", mmap_mode="r")
    son = np.load(VAL / "label_sonorant.npy", mmap_mode="r")
    arms = ("chain", "S1 true F1", "S2 true F1..Fn-1", "F1 fed x0.97", "F1 fed x1.03")
    P, Y = {a: [] for a in arms}, []
    with torch.inference_mode():
        for c in range(len(a_off) - 1):
            x = audio[a_off[c]:a_off[c + 1]].astype(np.float32) / 32768.0
            pad = (-len(x)) % SR
            xx = torch.from_numpy(np.pad(x, (0, pad)))[None].to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                model(vq.log_power_with_context(xx, (len(x) + pad) // HOP))
                h = got["h"]
                t = min(int(f_off[c + 1] - f_off[c]), h.shape[1])
                y = np.asarray(lab_f[f_off[c]:f_off[c] + t, :4], np.float64)
                s = np.asarray(son[f_off[c]:f_off[c] + t]).astype(bool) & np.all(np.isfinite(y) & (y > 0), 1)
                if not s.any():
                    continue
                tb = np.nan_to_num(FORMANT_AXIS.to_bin(np.where(y > 0, y, 1.0)))
                tb = torch.from_numpy(np.pad(tb, ((0, h.shape[1] - t), (0, 0)), mode="edge")).float().to(dev)[None]  # (1, T, 4)
                own = chain(model, h, [None] * 4)
                if c == 0:  # the re-run chain must reproduce the model's own read-out
                    ref = (model(vq.log_power_with_context(xx, (len(x) + pad) // HOP))["formant_logits"].float().softmax(-1)
                           * model.fm_bins).sum(-1)
                    assert torch.allclose(torch.stack(own, -1).float(), ref, atol=0.5), "chain re-run differs from the model"
                    h = got["h"]
                feds = {"chain": [None] * 4, "S1 true F1": [tb[..., 0], None, None, None],
                        "S2 true F1..Fn-1": [tb[..., 0], tb[..., 1], tb[..., 2], None]}
                k = 1200 / FORMANT_AXIS.cents_per_bin
                feds["F1 fed x0.97"] = [own[0] + k * np.log2(0.97), None, None, None]
                feds["F1 fed x1.03"] = [own[0] + k * np.log2(1.03), None, None, None]
                for a, fed in feds.items():
                    mb = torch.stack(chain(model, h, fed), -1)[0, :t].float().cpu().numpy()
                    P[a].append(FORMANT_AXIS.to_hz(mb)[s])
            Y.append(y[s])
    Y = np.concatenate(Y)
    P = {a: np.concatenate(v) for a, v in P.items()}
    res = {a: [float(np.mean(np.abs(P[a][:, n] - Y[:, n]))) for n in range(4)] for a in arms}
    d1 = (P["F1 fed x1.03"] - P["F1 fed x0.97"])
    dF1 = P["chain"][:, 0] * 0.06
    res["sensitivity |dFn| / |dF1| (n = 2-4), median"] = [float(np.median(np.abs(d1[:, n]) / dF1)) for n in (1, 2, 3)]
    res["frames"] = int(len(Y))
    O.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(name, json.dumps(res), flush=True)


def decide(names):
    L = ["D-084: formant links (v3.1 frozen features; MAE Hz on val_v2ko sonorant frames, seed means)", "", "M, the oracle matrix:"]
    inf = {}
    for t in names:
        M = json.loads((O / f"M_{t}.json").read_text())
        for n in range(1, 5):
            r = M[f"F{n}"]
            h = r["h"]
            cells = []
            for k, v in r.items():
                if k in ("h", "label only (others)"):
                    continue
                rel = v["mean"] / h["mean"] - 1
                good = rel <= -0.05 and h["mean"] - v["mean"] > 3 * max(h["spread"], v["spread"])
                if k != "h+others":
                    inf.setdefault((n, k), []).append(good)
                cells.append(f"{k} {v['mean']:.1f} ({100 * rel:+.1f} %{'*' if good else ''})")
            lo = f"; label only {r['label only (others)']:.1f}" if "label only (others)" in r else ""
            L.append(f"  {t} F{n}: h {h['mean']:.1f}; " + ", ".join(cells) + lo)
    yes = [f"F{n} <- {k[2:]}" for (n, k), v in sorted(inf.items()) if all(v)]
    L += ["", "S, v3.1's chain re-run (F1-F4 MAE):"]
    s1 = []
    for t in names:
        S = json.loads((O / f"S_{t}.json").read_text())
        for a in ("chain", "S1 true F1", "S2 true F1..Fn-1", "F1 fed x0.97", "F1 fed x1.03"):
            L.append(f"  {t} {a:18s} " + " / ".join(f"{x:.1f}" for x in S[a]))
        L.append(f"  {t} sensitivity |dFn| / |dF1| (F2-F4): " + " / ".join(f"{x:.3f}" for x in S["sensitivity |dFn| / |dF1| (n = 2-4), median"]))
        s1.append(S["S1 true F1"][1] <= 0.95 * S["chain"][1])
    L += ["", f"DECISION: informative cells: {', '.join(yes) if yes else 'none'}; "
          f"S1 {'the chain passes F1 errors on' if all(s1) else 'no (F2 not cut by 5 % on both trunks)'}"]
    (O / "decide_d084.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "M":
        for t in sys.argv[2:]:
            matrix(t)
    elif step == "S":
        for a in sys.argv[2:]:
            swap(*a.split("=", 1))
    elif step == "decide":
        decide(sys.argv[2:])
