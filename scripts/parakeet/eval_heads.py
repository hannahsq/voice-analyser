"""D-033 part 4: heads trained on cached layers (train_cached.py), scored from the scoring cache (cache.py: layers 0-6,
teacher v3's outputs on the same frames) with part 2's bias analysis: synthetic F1-F3 by f0 band, Hillenbrand dev by
group, PTDB-TUG dev closed phase by sex; ensembles with teacher v3 (a) mean of all formants, (b) precision-weighted,
(c) F1 only; the part-2 rule per head; then part 4's comparison rule (layers 3-6 vs layer 5).
Args: scoring cache dir, out txt, name=ckpt ... (the first two heads are compared: H5 first, H3456 second)."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head import load_head  # noqa: E402

from vocal_analysis.models.infer import decode_outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
C, OUTF = Path(sys.argv[1]), sys.argv[2]
heads = [a.split("=", 1) for a in sys.argv[3:] if not a.startswith("col:")]
COLS = [a[4:] for a in sys.argv[3:] if a.startswith("col:")]  # other models' cached columns (s2_score.py), no ensembles
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
torch.backends.cudnn.enabled = False
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
models = {n: load_head(ck, dev) for n, ck in heads}
layers = {n: torch.load(ck, map_location="cpu", weights_only=False)["layers"] for n, ck in heads}


def load(name):
    return {f.stem: np.load(f, mmap_mode="r" if f.stem == "emb" else None) for f in (C / name).glob("*.npy")}


def head_outputs(s, name):
    """per reference frame: log2 F1-F4 and sigma (cents) from head `name`, items in order."""
    eo, ro, mel = s["emb_off"], s["ref_off"], s["mel"]
    F, S = [], []
    for i in range(len(eo) - 1):
        e = torch.from_numpy(np.asarray(s["emb"][eo[i]:eo[i + 1]][:, list(layers[name])], np.float32)).to(dev).flatten(1)[None]
        m = torch.from_numpy(np.asarray(mel[ro[i]:ro[i + 1]], np.float32)).to(dev)
        with torch.inference_mode():
            out = models[name](e, m)
        d = decode_outputs({k: v[0].float().cpu().numpy() for k, v in out.items()})
        F.append(np.log2(d["formants"][:, :4]))
        S.append(d["formant_sigma_cents"][:, :4])
    return np.concatenate(F), np.concatenate(S)


def ensembles(s):
    t, ts = np.asarray(s["t_log2F"]), np.asarray(s["t_sigma"])
    e = {"teacher_v3": t, **{c: np.asarray(s[f"m_{c}_log2F"]) for c in COLS if f"m_{c}_log2F" in s}}
    for name in models:
        h, hs = head_outputs(s, name)
        e[name] = h
        e[f"a:{name}"] = (t + h) / 2
        wt, wh = 1 / np.maximum(ts, 1.0) ** 2, 1 / np.maximum(hs, 1.0) ** 2
        e[f"b:{name}"] = (wt * t + wh * h) / (wt + wh)
        c = t.copy()
        c[:, 0] = (t[:, 0] + h[:, 0]) / 2
        e[f"c:{name}"] = c
    return e


def table(title, est, ref, groups, nf):
    p(f"\n## {title}  (median signed error / MAE, Hz)")
    p(f"{'':12s}" + "".join(f"{g + ' F' + str(k + 1):>16s}" for g in groups for k in range(nf)))
    res = {}
    for name, x in est.items():
        cells = []
        for g, gm in groups.items():
            for k in range(nf):
                m = gm & np.isfinite(ref[:, k]) & (ref[:, k] > 0) & np.isfinite(x[:, k])
                err = 2 ** x[m, k] - ref[m, k]
                res[(name, g, k)] = (float(np.median(err)), float(np.mean(np.abs(err))))
                cells.append(f"{np.median(err):+7.1f} /{np.mean(np.abs(err)):6.1f}")
        p(f"{name:12s}" + "".join(f"{c:>16s}" for c in cells))
    return res


for sname in ("synth_train_v2", "synth_train_v2s"):
    s = load(sname)
    son = np.asarray(s["sonorant"]) > 0.5
    f0 = np.where(np.asarray(s["voiced"]) > 0.5, np.asarray(s["f0"]), np.nan)
    bands = {"<150": son & (f0 < 150), "150-250": son & (f0 >= 150) & (f0 < 250), "250-350": son & (f0 >= 250) & (f0 < 350),
             ">=350": son & (f0 >= 350), "unvoiced": son & ~np.isfinite(f0)}
    table(f"synthetic {sname} by f0 band", ensembles(s), np.asarray(s["formants"]), bands, 3)
s = load("gold_hillenbrand")
g = np.asarray(s["group"])
hb = table("Hillenbrand dev by group", ensembles(s), np.asarray(s["ref_formants"]),
           {"all": np.ones(len(g), bool), "men": g == "m", "women": g == "w", "children": np.isin(g, ["b", "g"])}, 4)
s = load("gold_ptdb_cp")
sex = np.asarray(s["sex"])
cp = table("PTDB-TUG dev closed phase by sex", ensembles(s), np.column_stack([s["ref_cp"], np.full(len(sex), np.nan)]),
           {"women": sex == "f", "men": sex == "m"}, 3)

p("\nPart-2 rule per head, ensemble (c), F1 only")
gain = {}
for name in models:
    w0, w1 = abs(cp[("teacher_v3", "women", 0)][0]), abs(cp[(f"c:{name}", "women", 0)][0])
    m0, m1 = abs(cp[("teacher_v3", "men", 0)][0]), abs(cp[(f"c:{name}", "men", 0)][0])
    h0, h1 = hb[("teacher_v3", "all", 0)][1], hb[(f"c:{name}", "all", 0)][1]
    ok = (w0 - w1 >= 5) and (m1 - m0 <= 5) and (h1 - h0 <= 4)
    gain[name] = (w0 - w1, h0 - h1)
    p(f"  {name}: women |F1 bias| {w0:.1f} -> {w1:.1f}; men {m0:.1f} -> {m1:.1f}; Hillenbrand F1 MAE {h0:.1f} -> {h1:.1f}"
      f"  DECISION: {'worth a full arm' if ok else 'no'}")
if len(models) >= 2:
    a, b = list(models)[:2]
    db, dh = gain[b][0] - gain[a][0], gain[b][1] - gain[a][1]
    better = (db >= 2 and dh > -2) or (dh >= 2 and db > -2)
    p(f"\nPart-4 rule, {b} against {a}: extra cut in women's closed-phase |F1 bias| {db:+.1f} Hz, in Hillenbrand F1 MAE {dh:+.1f} Hz"
      f"  DECISION: {b + ' better' if better else 'layer 5 stays'}")
