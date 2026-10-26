"""D-033 part 2: bias analysis and ensembles, with teacher v3, the layer-5 linear probe and the layer-5 head.
- Synthetic validation (train_v2, train_v2s; 512 fixed 4 s crops, seed 12345): F1-F3 median signed error and MAE (Hz)
  by f0 band of the frame (unvoiced sonorant frames separately).
- Hillenbrand dev: F1-F4 by group. PTDB-TUG dev closed-phase frames (ptdb_egg_sub.npz): F1-F3 by sex.
- Per-frame correlation of signed log errors, teacher v3 against each Parakeet member (Hillenbrand, closed phase).
- Ensembles, nothing fitted: (a) mean in log frequency, F1-F4; (b) precision-weighted by each model's 1 / sigma^2
  (head only; the probe has no sigma); (c) F1 only, teacher v3's F2-F4.
Rule (amended before any result): per member, (c) cuts women's closed-phase F1 |median signed| by >= 5 Hz, raises
men's by <= 5 Hz, and keeps Hillenbrand dev F1 MAE (all) within +4 Hz of teacher v3: "worth a full arm", else "no".
Run from the main checkout: VA_DEVICE=cuda python <worktree>/scripts/parakeet/bias.py <scratch dir> <head ckpt> <probe coef> <out>"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402
from head import crop_mel_pos, interp_time, load_head, parakeet_tap, tap_features  # noqa: E402

from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.data.manifest import read_manifest  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402
from vocal_analysis.eval import tiers as TI  # noqa: E402
from vocal_analysis.eval.datasets import load_audio, registry  # noqa: E402
from vocal_analysis.features.vqt import TorchVQT  # noqa: E402
from vocal_analysis.models.infer import analyse, decode_outputs, load_teacher  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import SynthCrops, split_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
S, HEAD, COEF, OUTF = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))
teacher = load_teacher("runs/conformer_v2r2_s1_d16k/ckpt_last.pt", dev)[0]
head = load_head(HEAD, dev)
fe, enc = load.feature_extractor(), parakeet_tap(dev)
pc = torch.load(COEF, weights_only=False)  # our own probe.py output (holds a numpy array)
P_W, P_MU, P_SD = pc["coef"][5, :, :4].float().to(dev), pc["mu"][5].to(dev), pc["sd"][5].to(dev)  # layer 5, F1-F4
vq = TorchVQT(dev)
cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})


def members(feats, mel_pos):
    """layer-5 features (B, E, 1024), mel positions (T,) -> probe log2 F1-F4 (B, T, 4); head log2 F1-F4 and sigma (cents)."""
    h = interp_time(feats, mel_pos)
    X = (h - P_MU) / P_SD
    probe = (torch.cat([X, torch.ones(*X.shape[:2], 1, device=dev)], -1) @ P_W).clamp(4, 13).cpu().numpy()
    with torch.inference_mode():
        out = head(feats, mel_pos)
    hd = [decode_outputs({k: v[i].float().cpu().numpy() for k, v in out.items()}) for i in range(feats.shape[0])]
    return probe, np.stack([np.log2(d["formants"][:, :4]) for d in hd]), np.stack([d["formant_sigma_cents"][:, :4] for d in hd])


def ensembles(t, ts, probe, hd, hs):
    """all log2 (n, 4) (+ sigma cents) -> dict of estimates, log2."""
    e = {"teacher_v3": t, "probe_L5": probe, "head_L5": hd,
         "a:probe": (t + probe) / 2, "a:head": (t + hd) / 2}
    wt, wh = 1 / np.maximum(ts, 1.0) ** 2, 1 / np.maximum(hs, 1.0) ** 2
    e["b:head"] = (wt * t + wh * hd) / (wt + wh)
    for name, x in (("probe", probe), ("head", hd)):
        c = t.copy()
        c[:, 0] = (t[:, 0] + x[:, 0]) / 2
        e[f"c:{name}"] = c
    return e


ORDER = ["teacher_v3", "probe_L5", "head_L5", "a:probe", "a:head", "b:head", "c:probe", "c:head"]


def table(title, est, ref, groups, nf):
    """est: name -> (n, 4) log2; ref (n, >=nf) Hz; groups: name -> mask. Median signed / MAE."""
    p(f"\n## {title}  (median signed error / MAE, Hz)")
    p(f"{'':12s}" + "".join(f"{g + ' F' + str(k + 1):>16s}" for g in groups for k in range(nf)))
    res = {}
    for name in ORDER:
        cells = []
        for g, gm in groups.items():
            for k in range(nf):
                m = gm & np.isfinite(ref[:, k]) & (ref[:, k] > 0) & np.isfinite(est[name][:, k])
                err = 2 ** est[name][m, k] - ref[m, k]
                res[(name, g, k)] = (float(np.median(err)), float(np.mean(np.abs(err))), int(m.sum()))
                cells.append(f"{np.median(err):+7.1f} /{np.mean(np.abs(err)):6.1f}")
        p(f"{name:12s}" + "".join(f"{c:>16s}" for c in cells))
    return res


def corr(title, est, ref, nf):
    rows = []
    for x in ("probe_L5", "head_L5"):
        r = []
        for k in range(nf):
            m = np.isfinite(ref[:, k]) & (ref[:, k] > 0)
            a, b = est["teacher_v3"][m, k] - np.log2(ref[m, k]), est[x][m, k] - np.log2(ref[m, k])
            r.append(np.corrcoef(a, b)[0, 1])
        rows.append(f"  teacher_v3 vs {x}: " + "  ".join(f"F{k + 1} {v:+.2f}" for k, v in enumerate(r)))
    p(f"Correlation of signed log errors ({title}):\n" + "\n".join(rows))


# ---------------------------------------------------------------- synthetic validation by f0 band
v2s = Path("data/synth/train_v2s")
sets = {"train_v2": SynthCrops(Path(cfg3["train"]["data"]) / "flat",
                               split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])[1][:512],
                               400, channel, seed=12345, fixed=True),
        "train_v2s": SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                400, channel, seed=12345, fixed=True)}
pos = crop_mel_pos(400, dev)
QUICK = os.environ.get("VA_QUICK") == "1"  # code check only: a few items per set
for sname, ds in sets.items():
    T_, PR, HD, HS, TS, REF, F0, SON = [], [], [], [], [], [], [], []
    for s in range(0, 32 if QUICK else len(ds), 16):
        items = [ds[i] for i in range(s, min(s + 16, len(ds)))]
        audio = np.stack([a for a, _ in items])
        feats = tap_features(enc, fe, audio, dev)
        pr, hd, hs = members(feats, pos)
        with torch.inference_mode():
            out = teacher(vq.log_power_with_context(torch.from_numpy(audio).to(dev), 400))
        for i, (_, lab) in enumerate(items):
            d = decode_outputs({k: v[i].float().cpu().numpy() for k, v in out.items()})
            T_.append(np.log2(d["formants"][:, :4]))
            TS.append(d["formant_sigma_cents"][:, :4])
            REF.append(lab["formants"][:, :4])
            F0.append(np.where(lab["voiced"] > 0.5, lab["f0"], np.nan))
            SON.append(lab["sonorant"] > 0.5)
        PR.append(pr.reshape(-1, 4))
        HD.append(hd.reshape(-1, 4))
        HS.append(hs.reshape(-1, 4))
    t, ts, ref, f0, son = (np.concatenate(x) for x in (T_, TS, REF, F0, SON))
    est = ensembles(t, ts, np.concatenate(PR), np.concatenate(HD), np.concatenate(HS))
    bands = {"<150": son & (f0 < 150), "150-250": son & (f0 >= 150) & (f0 < 250), "250-350": son & (f0 >= 250) & (f0 < 350),
             ">=350": son & (f0 >= 350), "unvoiced": son & ~np.isfinite(f0)}
    table(f"synthetic {sname} by f0 band", est, ref, bands, 3)

# ---------------------------------------------------------------- Hillenbrand dev
mods = registry()
rows = TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",))[: 60 if QUICK else None]
T_, TS, PR, HD, HS, REF, GRP = [], [], [], [], [], [], []
for row in rows:
    item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
    ok = np.isfinite(item.times)
    if not ok.any():
        continue
    x = load_audio(item, 16000)
    tt = item.times[ok]
    r = analyse(teacher, x, dev)
    idx = np.clip(np.round(tt / 0.01).astype(int), 0, len(r["f0"]) - 1)
    T_.append(np.log2(r["formants"][idx, :4]))
    TS.append(r["formant_sigma_cents"][idx, :4])
    pr, hd, hs = members(tap_features(enc, fe, x[None], dev), torch.from_numpy(tt / 0.01).float().to(dev))
    PR.append(pr[0])
    HD.append(hd[0])
    HS.append(hs[0])
    REF.append(item.formants[ok][:, :4])
    GRP += [item.meta.get("group", "?")] * int(ok.sum())
est = ensembles(*(np.concatenate(x) for x in (T_, TS, PR, HD, HS)))
ref, GRP = np.concatenate(REF), np.array(GRP)
hb = table("Hillenbrand dev by group", est, ref, {"all": np.ones(len(GRP), bool), "men": GRP == "m", "women": GRP == "w",
                                                   "children": np.isin(GRP, ["b", "g"])}, 4)
corr("Hillenbrand dev", est, ref, 4)

# ---------------------------------------------------------------- PTDB-TUG dev closed phase
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
tmap = TI.load()
devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == "dev" for s in d["speaker"]])
n = len(d["t"])
T_, TS, PR, HD, HS = (np.full((n, 4), np.nan) for _ in range(5))
for s, u in sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])})[: 12 if QUICK else None]:
    x, sr = sf.read(Path("data/ptdb_tug") / ("FEMALE" if s[0] == "F" else "MALE") / "MIC" / s / f"{u}.wav")
    x = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
    i = np.flatnonzero((d["speaker"] == s) & (d["utt"] == u))
    tt = d["t"][i].astype(float)
    r = analyse(teacher, x, dev)
    idx = np.clip(np.round(tt / 0.01).astype(int), 0, len(r["f0"]) - 1)
    T_[i], TS[i] = np.log2(r["formants"][idx, :4]), r["formant_sigma_cents"][idx, :4]
    pr, hd, hs = members(tap_features(enc, fe, x[None], dev), torch.from_numpy(tt / 0.01).float().to(dev))
    PR[i], HD[i], HS[i] = pr[0], hd[0], hs[0]
if QUICK:
    devm &= np.isfinite(T_[:, 0])
chk = devm & np.isfinite(d["conformer_v2r2_s1_d16k"][:, 0])
p(f"\nclosed phase: {devm.sum()} dev frames; teacher v3 recomputed vs ptdb_egg_sub.npz, F1 max |diff| "
  f"{np.nanmax(np.abs(2 ** T_[chk, 0] - d['conformer_v2r2_s1_d16k'][chk, 0])):.2f} Hz")
est = {k: v[devm] for k, v in ensembles(T_, TS, PR, HD, HS).items()}
ref = np.column_stack([d["cp"][devm], np.full(devm.sum(), np.nan)])
sex = d["sex"][devm]
cp = table("PTDB-TUG dev closed phase by sex", est, ref, {"women": sex == "f", "men": sex == "m"}, 3)
corr("closed phase", est, ref, 3)

# ---------------------------------------------------------------- rule
p("\nRule (amended before results): ensemble (c), F1 only")
for x in ("probe", "head"):
    w0, w1 = abs(cp[("teacher_v3", "women", 0)][0]), abs(cp[(f"c:{x}", "women", 0)][0])
    m0, m1 = abs(cp[("teacher_v3", "men", 0)][0]), abs(cp[(f"c:{x}", "men", 0)][0])
    h0, h1 = hb[("teacher_v3", "all", 0)][1], hb[(f"c:{x}", "all", 0)][1]
    ok = (w0 - w1 >= 5) and (m1 - m0 <= 5) and (h1 - h0 <= 4)
    p(f"  {x}: women |F1 bias| {w0:.1f} -> {w1:.1f}; men {m0:.1f} -> {m1:.1f}; Hillenbrand F1 MAE {h0:.1f} -> {h1:.1f}"
      f"  DECISION: {'worth a full arm' if ok else 'no'}")
