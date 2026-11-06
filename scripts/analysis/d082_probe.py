"""D-082: is the cascade's closed-F1 shift learnable from aspiration-free audio? Steps (each skips if its output exists):

    python scripts/analysis/d082_probe.py render [workers]                 (CPU)
    python scripts/analysis/d082_probe.py features <name>=<ckpt> ...       (GPU; VA_DEVICE=cuda)
    python scripts/analysis/d082_probe.py heads <name> ...                 (GPU)
    python scripts/analysis/d082_probe.py decide <name> ...
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

C, O = Path("cache/d082"), Path("reports/analysis/d082")
SETS = {"train": (3000, 78), "val": (1000, 7801)}
SR, HOP = 16000, 160
KEYS = ("formants", "formants_acoustic", "bandwidths_acoustic", "glottal_opening", "coupling_df", "gap_offset",
        "open_quotient", "rd", "f0", "sonorant")
LOG = lambda x: np.log(np.maximum(x, 1e-6))  # noqa: E731


def config():
    import d078_render

    return replace(d078_render.config(), hnr_db=None)


def one(job):
    from vocal_analysis.synth.sampler import generate

    i, seed = job
    c = generate(i, config(), seed)
    n = len(c.labels["f0"])
    lab = {}
    for k in KEYS:
        v = c.labels.get(k)
        if v is None:
            v = np.full((n, 8) if k in ("coupling_df", "bandwidths_acoustic") else n, np.nan)
        lab[k] = np.asarray(v, np.float32)
    return i, np.asarray(c.audio, np.float32), lab


def render(workers):
    C.mkdir(parents=True, exist_ok=True)
    for split, (n, seed) in SETS.items():
        out = C / f"{split}_render.npz"
        if out.exists():
            continue
        res = {}
        with ProcessPoolExecutor(workers) as ex:
            for i, a, lab in ex.map(one, [(i, seed) for i in range(n)], chunksize=8):
                res[i] = (a, lab)
        a_off = np.cumsum([0] + [len(res[i][0]) for i in range(n)])
        f_off = np.cumsum([0] + [len(res[i][1]["f0"]) for i in range(n)])
        np.savez(out, audio=np.concatenate([res[i][0] for i in range(n)]), a_off=a_off, f_off=f_off,
                 **{k: np.concatenate([res[i][1][k] for i in range(n)]) for k in KEYS})
        print(f"render {split}: {n} clips", flush=True)


def features(name, ck):
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ck, dev)
    vq = TorchVQT(dev)
    got = {}
    model.enc.register_forward_hook(lambda m, i, o: got.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    for split in SETS:
        out = C / f"{name}_{split}.npz"
        if out.exists():
            continue
        z = np.load(C / f"{split}_render.npz")
        audio, a_off, f_off = z["audio"], z["a_off"], z["f_off"]
        H, rows, clip = [], [], []
        with torch.inference_mode():
            for c in range(len(a_off) - 1):
                x = audio[a_off[c]:a_off[c + 1]]
                pad = (-len(x)) % SR
                xx = torch.from_numpy(np.pad(x, (0, pad)))[None].to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    model(vq.log_power_with_context(xx, (len(x) + pad) // HOP))
                t = min(int(f_off[c + 1] - f_off[c]), got["h"].shape[1])
                H.append(got["h"][0, :t].float().cpu().numpy().astype(np.float16))
                rows.append(np.arange(f_off[c], f_off[c] + t))
                clip.append(np.full(t, c))
        np.savez(out, h=np.concatenate(H), rows=np.concatenate(rows), clip=np.concatenate(clip))
        print(f"features {name} {split}", flush=True)


def load(name, split):
    z = np.load(C / f"{name}_{split}.npz")
    r = np.load(C / f"{split}_render.npz")
    lab = {k: r[k][z["rows"]] for k in KEYS}
    lab["clip"] = z["clip"]
    lab["df1"] = lab["coupling_df"][:, 0]
    lab["shift"] = LOG(lab["formants_acoustic"][:, 0]) - LOG(lab["formants"][:, 0])
    m = (lab["sonorant"].astype(bool) & np.isfinite(lab["glottal_opening"]) & np.isfinite(lab["df1"])
         & np.all(np.isfinite(lab["formants_acoustic"][:, :4]), 1) & np.isfinite(lab["formants"][:, 0]))
    return z["h"][m].astype(np.float32), {k: v[m] for k, v in lab.items()}


def full_labels(z):
    return np.column_stack([z["formants_acoustic"][:, :4], LOG(z["bandwidths_acoustic"][:, :4]), z["f0"], z["rd"],
                            z["open_quotient"]])


def r2(y, p):
    return float(1 - np.var(y - p) / np.var(y))


def heads(name):
    import torch

    from d081_label_oracle import Slope, gb
    from d081_oracle_heads import fit

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    O.mkdir(parents=True, exist_ok=True)
    out = O / f"heads_{name}.json"
    if out.exists():
        return
    (ht, lt), (hv, lv) = load(name, "train"), load(name, "val")
    S = Slope(lv)
    res = {"frames": int(len(hv)), "closed label slope": S(lv["formants"][:, 0])}
    # C1: label ceiling and closed-F1 heads
    xt, xv = full_labels(lt), full_labels(lv)
    okt, okv = np.all(np.isfinite(xt), 1), np.all(np.isfinite(xv), 1)
    pc = np.full(len(xv), np.nan)
    pc[okv] = np.exp(gb(xt[okt], LOG(lt["formants"][okt, 0])).predict(xv[okv]))
    res["label ceiling slope"] = S(pc)
    yt = LOG(lt["formants"][:, :1])
    for v, keys in (("h", []), ("h+<a>", ["glottal_opening"]), ("h+<a>+df1", ["glottal_opening", "df1"])):
        Xt = np.column_stack([ht] + [lt[k] for k in keys])
        Xv = np.column_stack([hv] + [lv[k] for k in keys])
        per = {s: S(np.exp(fit(Xt, yt, Xv, s, dev)[:, 0])) for s in (1, 2)}
        res[f"C1 {v}"] = {"by_seed": per, "mean": float(np.mean(list(per.values())))}
        print(f"{name} C1 {v}: {res[f'C1 {v}']['mean']:.3f}", flush=True)
    # C2: shift and df1 beyond the vowel part
    vt, vv = lt["formants_acoustic"][:, :4], lv["formants_acoustic"][:, :4]
    for tgt in ("shift", "df1"):
        g = gb(vt, lt[tgt])
        gv = g.predict(vv)
        resid = lv[tgt] - gv
        lab_ref = np.full(len(xv), np.nan)
        lab_ref[okv] = gb(xt[okt], lt[tgt][okt]).predict(xv[okv])
        r = {"label reference": r2(resid[okv], lab_ref[okv] - gv[okv]), "vowel part R2": r2(lv[tgt], gv)}
        per, slopes = {}, {}
        for s in (1, 2):
            p = fit(ht, lt[tgt][:, None], hv, s, dev)[:, 0]
            per[s] = r2(resid, p - gv)
            if tgt == "shift":
                slopes[s] = S(lv["formants_acoustic"][:, 0] * np.exp(-p))
        r.update(by_seed=per, mean=float(np.mean(list(per.values()))), spread=float(abs(per[1] - per[2])))
        if slopes:
            r["slope"] = float(np.mean(list(slopes.values())))
        res[f"C2 {tgt}"] = r
        print(f"{name} C2 {tgt}: beyond vowel {r['mean']:.3f} (label ref {r['label reference']:.3f})", flush=True)
    out.write_text(json.dumps(res, indent=1, default=float))


def decide(names):
    R = {n: json.loads((O / f"heads_{n}.json").read_text()) for n in names}
    L = ["D-082: the closed-F1 shift on aspiration-free cascade audio (v3.1 frozen features)", ""]
    for n, r in R.items():
        L.append(f"{n} ({r['frames']} val frames): closed label slope {r['closed label slope']:+.3f}; label ceiling {r['label ceiling slope']:+.3f}; "
                 f"h {r['C1 h']['mean']:+.3f}, h+<a> {r['C1 h+<a>']['mean']:+.3f}, h+<a>+df1 {r['C1 h+<a>+df1']['mean']:+.3f}")
        for t in ("shift", "df1"):
            c = r[f"C2 {t}"]
            L.append(f"  {t}: R² beyond the vowel part {c['mean']:+.3f} (seeds {', '.join(f'{x:+.3f}' for x in c['by_seed'].values())}); "
                     f"label reference {c['label reference']:+.3f}; vowel part alone R² {c['vowel part R2']:.3f}"
                     + (f"; slope via the shift head {c['slope']:+.3f}" if "slope" in c else ""))
    gap = [r["C1 h"]["mean"] - r["label ceiling slope"] for r in R.values()]
    c1 = "confirmed" if all(x <= 0.05 for x in gap) else "refuted" if all(x >= 0.15 for x in gap) else "unclear"
    c2 = {}
    for t in ("shift", "df1"):
        v = [r[f"C2 {t}"]["mean"] for r in R.values()]
        c2[t] = "learnable" if all(x >= 0.3 for x in v) else "not learnable" if all(x < 0.1 for x in v) else "partly"
    L += ["", f"DECISION: C1 artefact explanation {c1} (h - ceiling: {', '.join(f'{x:+.3f}' for x in gap)}); "
          f"C2 shift {c2['shift']}; df1 {c2['df1']}"]
    (O / "decide_d082.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "render":
        render(int(sys.argv[2]) if len(sys.argv) > 2 else 24)
    elif step == "features":
        for a in sys.argv[2:]:
            features(*a.split("=", 1))
    elif step == "heads":
        for n in sys.argv[2:]:
            heads(n)
    elif step == "decide":
        decide(sys.argv[2:])
