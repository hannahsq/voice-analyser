"""D-033 full arm (pre-registered 2026-10-23): labeller v3 + H5x6, scored from a scoring cache (cache.py) with the teacher
seeds' columns (s2_score.py: tv3_s0, tv3_s1, tv3_s2).
  labeller v3 = per-frame mean in log frequency of teacher seeds 1 and 2 (D-025); H5x6 = mean of the six H5 heads;
  arm = labeller v3 with F1 the 1:1 mean of labeller v3's F1 and H5x6; arm_t = the same on teacher v3 alone;
  med3 = per-frame median of the three teacher seeds (reported only).
Tables as eval_heads.py (F1-F4 by set and group), the dev rule (part 2's thresholds, the arm against labeller v3), and a
paired speaker bootstrap (2,000 resamples within sex / group, seed 20261023).
Args: cache dir, out txt, tier label, H5 checkpoints (six) [--semi E1 E2: the semi-private replication rule]."""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head import load_head  # noqa: E402

from vocal_analysis.models.infer import decode_outputs  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
argv = sys.argv[1:]
SEMI = None
if "--semi" in argv:
    i = argv.index("--semi")
    SEMI = (float(argv[i + 1]), float(argv[i + 2]))
    argv = argv[:i] + argv[i + 3:]
C, OUTF, TIER, CKS = Path(argv[0]), argv[1], argv[2], argv[3:]
assert len(CKS) == 6, "six H5 checkpoints"
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
torch.backends.cudnn.enabled = False
dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
heads = [load_head(ck, dev) for ck in CKS]
LAYER = [torch.load(ck, map_location="cpu", weights_only=False)["layers"] for ck in CKS]
assert all(list(x) == [5] for x in LAYER), "H5 heads only"


def load(name):
    return {f.stem: np.load(f, mmap_mode="r" if f.stem == "emb" else None) for f in (C / name).glob("*.npy")}


def head_f(s, m):
    eo, ro, mel = s["emb_off"], s["ref_off"], s["mel"]
    F = []
    for i in range(len(eo) - 1):
        e = torch.from_numpy(np.asarray(s["emb"][eo[i]:eo[i + 1]][:, [5]], np.float32)).to(dev).flatten(1)[None]
        mm = torch.from_numpy(np.asarray(mel[ro[i]:ro[i + 1]], np.float32)).to(dev)
        with torch.inference_mode():
            out = m(e, mm)
        F.append(np.log2(decode_outputs({k: v[0].float().cpu().numpy() for k, v in out.items()})["formants"][:, :4]))
    return np.concatenate(F)


CHECK = []


def estimates(s, sname):
    t = np.asarray(s["t_log2F"])
    T = {k: np.asarray(s[f"m_tv3_{k}_log2F"]) for k in ("s0", "s1", "s2")}
    d = np.abs(T["s1"] - t)
    CHECK.append((sname, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0))
    L3 = (T["s1"] + T["s2"]) / 2
    hs = [head_f(s, m) for m in heads]
    H = np.mean(hs, 0)
    e = {"teacher_v3": t, "tv3_s0": T["s0"], "tv3_s2": T["s2"], "labeller_v3": L3,
         "med3": np.median(np.stack([T["s0"], T["s1"], T["s2"]]), 0), "H5x6": H}
    a = L3.copy()
    a[:, 0] = (L3[:, 0] + H[:, 0]) / 2
    e["arm"] = a
    at = t.copy()
    at[:, 0] = (t[:, 0] + H[:, 0]) / 2
    e["arm_t"] = at
    for k, h in enumerate(hs):
        x = L3.copy()
        x[:, 0] = (L3[:, 0] + h[:, 0]) / 2
        e[f"arm#{k}"] = x
    return e


def table(title, est, ref, groups, nf, show):
    p(f"\n## {title}  (median signed error / MAE, Hz)")
    p(f"{'':13s}" + "".join(f"{g + ' F' + str(k + 1):>16s}" for g in groups for k in range(nf)))
    res = {}
    for name, x in est.items():
        cells = []
        for g, gm in groups.items():
            for k in range(nf):
                m = gm & np.isfinite(ref[:, k]) & (ref[:, k] > 0) & np.isfinite(x[:, k])
                err = 2 ** x[m, k] - ref[m, k]
                res[(name, g, k)] = (float(np.median(err)), float(np.mean(np.abs(err))))
                cells.append(f"{np.median(err):+7.1f} /{np.mean(np.abs(err)):6.1f}")
        if name in show:
            p(f"{name:13s}" + "".join(f"{c:>16s}" for c in cells))
    return res


SHOW = ["teacher_v3", "tv3_s0", "tv3_s2", "labeller_v3", "med3", "H5x6", "arm", "arm_t"]
p(f"D-033 full arm, {TIER} speakers; H5 heads: {', '.join(Path(c).parent.name for c in CKS)}")
if TIER == "dev":
    for sname in ("synth_train_v2", "synth_train_v2s"):
        s = load(sname)
        son = np.asarray(s["sonorant"]) > 0.5
        f0 = np.where(np.asarray(s["voiced"]) > 0.5, np.asarray(s["f0"]), np.nan)
        bands = {"<150": son & (f0 < 150), "150-250": son & (f0 >= 150) & (f0 < 250), "250-350": son & (f0 >= 250) & (f0 < 350),
                 ">=350": son & (f0 >= 350), "unvoiced": son & ~np.isfinite(f0)}
        table(f"synthetic {sname} by f0 band", estimates(s, sname), np.asarray(s["formants"]), bands, 4, SHOW)
    for sname in ("resynth_vocalset_val_d16k", "resynth_zeroth_korean_val_d16k"):
        s = load(sname)
        table(f"{sname}, sonorant frames", estimates(s, sname), np.asarray(s["formants"]),
              {"all": np.asarray(s["sonorant"]) > 0.5}, 4, SHOW)
s = load("gold_hillenbrand")
g, spk_h = np.asarray(s["group"]), np.asarray(s["speaker"])
EH, RH = estimates(s, "gold_hillenbrand"), np.asarray(s["ref_formants"])
hb = table(f"Hillenbrand {TIER} by group", EH, RH,
           {"all": np.ones(len(g), bool), "men": g == "m", "women": g == "w", "children": np.isin(g, ["b", "g"])}, 4, SHOW)
s = load("gold_ptdb_cp")
sex, spk_c = np.asarray(s["sex"]), np.asarray(s["speaker"])
EC, RC = estimates(s, "gold_ptdb_cp"), np.asarray(s["ref_cp"])
cp = table(f"PTDB-TUG {TIER} closed phase by sex", EC, np.column_stack([RC, np.full(len(sex), np.nan)]),
           {"women": sex == "f", "men": sex == "m"}, 3, SHOW)
p("\nseed-1 rescoring against the cached teacher v3 columns, max |difference| in log2 F: "
  + ", ".join(f"{n} {v:.2g}" for n, v in CHECK))


def rule(k, base):
    w0, w1 = abs(cp[(base, "women", 0)][0]), abs(cp[(k, "women", 0)][0])
    m0, m1 = abs(cp[(base, "men", 0)][0]), abs(cp[(k, "men", 0)][0])
    h0, h1 = hb[(base, "all", 0)][1], hb[(k, "all", 0)][1]
    return w0 - w1, m1 - m0, h0 - h1, f"women |F1 bias| {w0:.1f} -> {w1:.1f}; men {m0:.1f} -> {m1:.1f}; Hillenbrand F1 MAE {h0:.1f} -> {h1:.1f}"


p("")
for k, base in (("arm", "labeller_v3"), ("arm_t", "teacher_v3"), ("labeller_v3", "teacher_v3"), ("med3", "labeller_v3")):
    cw, rm, ch, txt = rule(k, base)
    if SEMI is None and (k, base) == ("arm", "labeller_v3"):
        ok = cw >= 5 and rm <= 5 and ch >= -4
        p(f"  {k} vs {base}: {txt}  (E1 {cw:+.1f}, E2 {ch:+.1f})  DECISION (dev): {'passes: semi-private confirmation' if ok else 'no'}")
    elif SEMI is not None and (k, base) == ("arm", "labeller_v3"):
        ok = cw >= SEMI[0] / 2 and ch >= SEMI[1] / 2 and rm <= 5
        p(f"  {k} vs {base}: {txt}  (E1 {cw:+.1f} against dev {SEMI[0]:+.1f}, E2 {ch:+.1f} against dev {SEMI[1]:+.1f})"
          f"  DECISION (semi-private): {'replicates' if ok else 'does not replicate'}")
    else:
        p(f"  {k} vs {base} (reported): {txt}")

# ---- paired speaker bootstrap
B, rng = 2000, np.random.default_rng(20261023)


def strata(spk, lab):
    out = []
    for st in np.unique(lab):
        sp = sorted(set(spk[lab == st]))
        out.append([np.flatnonzero(spk == x) for x in sp])
    return out


SC, SH = strata(spk_c, sex), strata(spk_h, g)
EST = ["teacher_v3", "labeller_v3", "arm", "arm_t"] + [f"arm#{k}" for k in range(6)]
errc = {k: 2 ** EC[k][:, 0] - RC[:, 0] for k in EST}
errh = {k: 2 ** EH[k][:, 0] - RH[:, 0] for k in EST}
okc, okh = np.isfinite(RC[:, 0]) & (RC[:, 0] > 0), np.isfinite(RH[:, 0]) & (RH[:, 0] > 0)
stats = {k: [] for k in ("w", "m", "h")}
for _ in range(B):
    ic = np.concatenate([np.concatenate([st[j] for j in rng.integers(0, len(st), len(st))]) for st in SC])
    ih = np.concatenate([np.concatenate([st[j] for j in rng.integers(0, len(st), len(st))]) for st in SH])
    w, m, h = {}, {}, {}
    for k in EST:
        ec, eh = errc[k][ic], errh[k][ih]
        fc, fh = okc[ic] & np.isfinite(ec), okh[ih] & np.isfinite(eh)
        w[k] = abs(np.median(ec[fc & (sex[ic] == "f")]))
        m[k] = abs(np.median(ec[fc & (sex[ic] == "m")]))
        h[k] = np.mean(np.abs(eh[fh]))
    for d, key in ((w, "w"), (m, "m"), (h, "h")):
        d["single-seed mean"] = np.mean([d[f"arm#{k}"] for k in range(6)])
        stats[key].append(d)


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return f"{np.mean(vals):6.1f} [{lo:6.1f}, {hi:6.1f}]"


p(f"\n## Paired speaker bootstrap ({B} resamples within sex / group; mean [95 % interval], Hz)")
for key, title in (("w", "women's closed-phase |F1 median signed error|"), ("m", "men's closed-phase |F1 median signed error|"),
                   ("h", "Hillenbrand F1 MAE (all)")):
    p(f"  {title}")
    for k in ["teacher_v3", "labeller_v3", "arm", "arm_t", "single-seed mean"] + [f"arm#{k}" for k in range(6)]:
        p(f"    {k:18s} {ci([d[k] for d in stats[key]])}")
    for a, b in (("arm", "labeller_v3"), ("arm", "teacher_v3"), ("arm", "single-seed mean"), ("arm_t", "teacher_v3")):
        p(f"    {a + ' - ' + b:30s} {ci([d[a] - d[b] for d in stats[key]])}")
p(f"  share of resamples with the arm's women's |bias| below labeller v3's: {np.mean([d['arm'] < d['labeller_v3'] for d in stats['w']]):.3f};"
  f" below the single-seed mean's: {np.mean([d['arm'] < d['single-seed mean'] for d in stats['w']]):.3f}")
