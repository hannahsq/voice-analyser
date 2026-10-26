"""D-033: linear probe of Parakeet TDT 0.6B v3's encoder, layer by layer, for every target teacher v3 trains on.

Fitted only on teacher v3's training mixture (runs/conformer_v2r2_s1/config.json: train_v2 training clips + 30 %
resynthesised VocalSet and Zeroth-Korean, its saved channel model): 3000 fixed 4 s crops (seed 777), every 2nd 10 ms
frame. Features: each hidden state (subsampling output = layer 0, layers 1-24), linearly interpolated from the 80 ms
encoder frames (frame e centred on mel frame 8e, click-checked) to our 10 ms frames, standardised. Ridge per layer and
target, strength chosen on the last 20 % of the crops, refitted on all. Targets and masks as the trainer's: log2 F1-F4
on sonorant frames (resynthesised clips weighted by their QC weight), log2 B1-B4, log2 f0 (voiced), voicing and
sonorant (0/1) on synthetic clips only.
Scored on synthetic validation (512 fixed 4 s crops each, seed 12345: train_v2, train_v2s, resynthesised VocalSet /
Zeroth-Korean val) and the baseline gold sets on dev speakers (Hillenbrand formants; PTDB-TUG, MIR-1K, MDB-stem-synth
f0 and voicing; whole files). Teacher v3 on the same frames. Run from the main checkout (data/, runs/):
  VA_DEVICE=cuda python scripts/parakeet/probe.py <out dir>"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load  # noqa: E402

from vocal_analysis.data.manifest import read_manifest  # noqa: E402
from vocal_analysis.eval.datasets import load_audio, registry  # noqa: E402
from vocal_analysis.eval.tiers import filter_rows  # noqa: E402
from vocal_analysis.features.vqt import HOP, N_MAX, TorchVQT  # noqa: E402
from vocal_analysis.models.infer import analyse, decode_outputs, load_teacher  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.train.data import MixedCrops, SynthCrops, consolidate, split_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
from vocal_analysis.config import DeviceConfig  # noqa: E402
from vocal_analysis.device import setup_device  # noqa: E402

dev = setup_device(DeviceConfig(device=os.environ.get("VA_DEVICE", "cuda")))  # MIOpen policy as training
N_TRAIN, N_VAL, CROP, BATCH, SEL_FRAC = int(os.environ.get("VA_N_TRAIN", 3000)), int(os.environ.get("VA_N_VAL", 512)), 400, 16, 0.2
L = load.N_LAYERS + 1
TEACHER = "runs/conformer_v2r2_s1_d16k/ckpt_last.pt"
TARGETS = ["F1", "F2", "F3", "F4", "B1", "B2", "B3", "B4", "f0", "voiced", "sonorant"]
LOG = open(OUT / "probe_log.txt", "a", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    print(time.strftime("%H:%M:%S"), msg, file=LOG, flush=True)


cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
tlw = tuple(cfg3["train"]["tract_left_weight"])
fe = load.feature_extractor()
enc = load.encoder(dev)


def hidden(audio_batch: np.ndarray) -> torch.Tensor:
    """(B, n) audio -> (L, B, E, 1024) hidden states."""
    f = fe(list(audio_batch), sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        h = enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states
    return torch.stack(h)


def interp(h: torch.Tensor, mel_pos: torch.Tensor) -> torch.Tensor:
    """h (L, B, E, D), mel_pos (T,) in mel frames -> (L, B, T, D), linear between encoder frame centres 8e."""
    p = (mel_pos / load.SUBSAMPLING).clamp(0, h.shape[2] - 1)
    i0 = p.floor().long().clamp(max=h.shape[2] - 2) if h.shape[2] > 1 else p.floor().long()
    if h.shape[2] == 1:
        return h[:, :, i0]
    a = (p - i0).to(h.dtype)[None, None, :, None]
    return h[:, :, i0] * (1 - a) + h[:, :, i0 + 1] * a


def crop_targets(lab: dict) -> tuple[np.ndarray, np.ndarray]:
    """labels of one crop -> targets (T, 11) and weights (T, 11) (0 = not used)."""
    T = len(lab["f0"])
    y, w = np.zeros((T, 11)), np.zeros((T, 11))
    son = lab["sonorant"] > 0.5
    syn = float(lab["w_f0"][0]) > 0
    for k in range(4):
        F, B = lab["formants"][:, k], lab["bandwidths"][:, k]
        ok = son & np.isfinite(F) & (F > 0)
        y[ok, k], w[ok, k] = np.log2(F[ok]), lab["w_formant"][ok]
        okb = son & np.isfinite(B) & (B > 0) & syn
        y[okb, 4 + k], w[okb, 4 + k] = np.log2(B[okb]), 1.0
    v = (lab["voiced"] > 0.5) & np.isfinite(lab["f0"]) & (lab["f0"] > 0) & syn
    y[v, 8], w[v, 8] = np.log2(lab["f0"][v]), 1.0
    if syn:
        y[:, 9], w[:, 9] = (lab["voiced"] > 0.5), 1.0
        y[:, 10], w[:, 10] = son, 1.0
    return y, w


def batches(ds, n):
    for s in range(0, n, BATCH):
        items = [ds[i] for i in range(s, min(s + BATCH, n))]
        yield np.stack([a for a, _ in items]), [lab for _, lab in items]


# ---------------------------------------------------------------- 1. fit on teacher v3's training mixture
flat = Path(cfg3["train"]["data"]) / "flat"
n_clips = json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"]
train_ids, val_ids = split_ids(n_clips)
val_ids = val_ids[: cfg3["train"]["val_clips"]]
parts, sizes = [SynthCrops(flat, train_ids, CROP, channel, seed=777, epoch_size=N_TRAIN, fixed=True)], []
for d in cfg3["train"]["resynth"]:
    rflat = consolidate(d, log=log)
    n = json.loads((rflat / "done").read_text())["n_clips"]
    parts.append(SynthCrops(rflat, np.arange(n), CROP, channel, seed=778 + len(sizes), epoch_size=N_TRAIN, fixed=True,
                            tract_left_weight=tlw))
    sizes.append(n)
frac = cfg3["train"]["resynth_frac"]
train_ds = MixedCrops(parts, [1 - frac] + [frac * n / sum(sizes) for n in sizes], N_TRAIN, seed=777, fixed=True)
lab_frames = torch.arange(0, CROP, 2, device=dev)  # every 2nd frame
mel_of_frame = lambda j: j + N_MAX // 2 // HOP  # noqa: E731  crop frame j sits at sample N_MAX/2 + j * HOP

stats_path = OUT / "ridge_stats.pt"
if stats_path.exists():
    S = torch.load(stats_path)
    log(f"loaded {stats_path}")
else:
    t0 = time.time()
    mu = sd = None
    D = 1025
    S = {sp: {"xx": torch.zeros(11, L, D, D, device=dev), "xy": torch.zeros(11, L, D, device=dev),
              "yy": torch.zeros(11, device=dev, dtype=torch.float64), "w": torch.zeros(11, device=dev, dtype=torch.float64)}
         for sp in ("fit", "sel")}
    n_done = 0
    for audio, labs in batches(train_ds, N_TRAIN):
        H = interp(hidden(audio), mel_of_frame(lab_frames).float())  # (L, B, T, 1024)
        H = H.reshape(L, -1, H.shape[-1])
        if mu is None:  # standardisation from the first batches (fixed thereafter)
            mu, sd = H.mean(1, keepdim=True), H.std(1, keepdim=True) + 1e-4
        X = torch.cat([(H - mu) / sd, torch.ones(L, H.shape[1], 1, device=dev)], -1)
        yw = [crop_targets({k: v[::2] for k, v in lab.items()}) for lab in labs]
        Y = torch.from_numpy(np.concatenate([a for a, _ in yw])).to(dev, torch.float32)
        W = torch.from_numpy(np.concatenate([b for _, b in yw])).to(dev, torch.float32)
        sp = "fit" if n_done < (1 - SEL_FRAC) * N_TRAIN else "sel"
        for t in range(11):
            m = W[:, t] > 0
            if not m.any():
                continue
            Xm, wm, ym = X[:, m], W[m, t], Y[m, t]
            S[sp]["xx"][t] += torch.einsum("lnd,n,lne->lde", Xm, wm, Xm)
            S[sp]["xy"][t] += torch.einsum("lnd,n->ld", Xm, wm * ym)
            S[sp]["yy"][t] += (wm * ym * ym).double().sum()
            S[sp]["w"][t] += wm.double().sum()
        n_done += len(labs)
        if n_done % 320 == 0:
            log(f"  fit crops {n_done}/{N_TRAIN} ({time.time() - t0:.0f} s)")
    S = {sp: {k: v.cpu() for k, v in d.items()} for sp, d in S.items()} | {"mu": mu.cpu(), "sd": sd.cpu()}
    torch.save(S, stats_path)

LAMBDAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def solve(xx, xy, n, lam):
    reg = torch.eye(xx.shape[-1], dtype=torch.float64, device=xx.device) * lam * n
    reg[-1, -1] = 0  # no penalty on the bias
    return torch.linalg.solve(xx + reg, xy)


coef = torch.zeros(L, 1025, 11, dtype=torch.float64)
chosen = np.zeros((L, 11))
for t in range(11):
    for li in range(L):
        f, s = S["fit"], S["sel"]
        # on the GPU: this torch build's CPU LAPACK (OpenBLAS) takes 0.8 s per 1025^2 solve, the GPU 0.013 s
        xx, xy, n = f["xx"][t, li].to(dev, torch.float64), f["xy"][t, li].to(dev, torch.float64), float(f["w"][t])
        sxx, sxy, syy, sn = (s["xx"][t, li].to(dev, torch.float64), s["xy"][t, li].to(dev, torch.float64), float(s["yy"][t]),
                             float(s["w"][t]))
        best = None
        for lam in LAMBDAS:
            b = solve(xx, xy, n, lam)
            mse = (syy - 2 * b @ sxy + b @ sxx @ b).item() / max(sn, 1e-9)
            if best is None or mse < best[0]:
                best = (float(mse), lam)
        chosen[li, t] = best[1]
        coef[li, :, t] = solve(xx + sxx, xy + sxy, n + sn, best[1]).cpu()
torch.save({"coef": coef, "lambda": chosen, "mu": S["mu"], "sd": S["sd"]}, OUT / "probe_coef.pt")
log("ridge strengths chosen (log10 lambda, layers x targets):")
log("\n".join(f"  L{li:2d} " + " ".join(f"{np.log10(chosen[li, t]):+.0f}" for t in range(11)) for li in range(L)))
mu, sd = S["mu"].to(dev), S["sd"].to(dev)
coef_d = coef.float().to(dev)


def predict(H: torch.Tensor) -> np.ndarray:
    """(L, N, 1024) -> (L, N, 11) probe outputs (log2 Hz / probabilities)."""
    X = torch.cat([(H - mu) / sd, torch.ones(L, H.shape[1], 1, device=dev)], -1)
    out = torch.einsum("lnd,ldt->lnt", X, coef_d)
    out[..., :9] = out[..., :9].clamp(4.0, 13.0)  # log2 Hz: 16 Hz to 8 kHz, so a wild extrapolation stays finite
    return out.cpu().numpy()


# ---------------------------------------------------------------- 2. score
teacher = load_teacher(TEACHER, dev)[0]
vq = TorchVQT(dev)
results = {}


def synth_set(name, ds, n, formants_only=False):
    P, T_, Wt, TV = [], [], [], []
    frames = torch.arange(CROP, device=dev)
    for audio, labs in batches(ds, n):
        H = interp(hidden(audio), mel_of_frame(frames).float()).reshape(L, -1, 1024)
        P.append(predict(H))
        with torch.inference_mode():
            out = teacher(vq.log_power_with_context(torch.from_numpy(audio).to(dev), CROP))
        for i, lab in enumerate(labs):
            d = decode_outputs({k: v[i].float().cpu().numpy() for k, v in out.items()})
            TV.append(np.column_stack([np.log2(d["formants"][:, :4]), np.log2(d["bandwidths"][:, :4]), np.log2(d["f0"]),
                                       d["voiced_p"], d["sonorant_p"]]))
            y, w = crop_targets(lab)
            T_.append(y)
            Wt.append(w)
    P, Y, Wt, TV = np.concatenate(P, 1), np.concatenate(T_), np.concatenate(Wt), np.concatenate(TV)
    rows = {}
    for who, pred in [*((f"L{li}", P[li]) for li in range(L)), ("teacher_v3", TV)]:
        r = {}
        for t in range(4 if formants_only else 11):
            m = Wt[:, t] > 0
            if not m.any():
                continue
            if t < 8:
                r[TARGETS[t] + "_mae_hz"] = float(np.mean(np.abs(2 ** pred[m, t] - 2 ** Y[m, t])))
            elif t == 8:
                r["f0_rpa"] = float(np.mean(np.abs(1200 * (pred[m, t] - Y[m, t])) <= 50))
            else:
                r[TARGETS[t] + "_acc"] = float(np.mean((pred[m, t] > 0.5) == (Y[m, t] > 0.5)))
        rows[who] = r
    results[name] = rows
    log(f"[{name}] done: teacher {rows['teacher_v3']}")


val_sets = [("synth:train_v2", SynthCrops(flat, val_ids, CROP, channel, seed=12345, fixed=True), False)]
v2s = Path("data/synth/train_v2s")
n_s = json.loads((v2s / "index.json").read_text())["n_clips"]
val_sets.append(("synth:train_v2s", SynthCrops(v2s / "flat", split_ids(n_s)[1][:N_VAL], CROP, channel, seed=12345, fixed=True), False))
for d in cfg3["train"]["resynth_val"]:
    rflat = consolidate(d, log=log)
    n = json.loads((rflat / "done").read_text())["n_clips"]
    ids = np.sort(np.random.default_rng(0).permutation(n)[:N_VAL])
    val_sets.append((f"resynth:{Path(d).name}", SynthCrops(rflat, ids, CROP, channel, seed=12345, fixed=True, tract_left_weight=tlw), True))
for name, ds, fo in val_sets:
    synth_set(name, ds, min(N_VAL, len(ds)), fo)
    json.dump(results, open(OUT / "probe_results.json", "w"), indent=1)


def gold_file(x, times):
    """one file -> probe outputs (L, n_times, 11) and teacher v3's (n_times, 11) at the reference times."""
    H = hidden(x[None])
    mel = torch.from_numpy(np.asarray(times) / 0.01).float().to(dev)
    P = predict(interp(H, mel).reshape(L, -1, 1024))
    r = analyse(teacher, x, dev)
    idx = np.clip(np.round(np.asarray(times) / 0.01).astype(int), 0, len(r["f0"]) - 1)
    TV = np.column_stack([np.log2(r["formants"][idx, :4]), np.log2(r["bandwidths"][idx, :4]), np.log2(r["f0"][idx]),
                          r["voiced_p"][idx], r["sonorant_p"][idx]])
    return P, TV


mods = registry()
# Hillenbrand: formants by group
rows = filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",))
P, TV, REF, GRP = [], [], [], []
for i, row in enumerate(rows):
    item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
    ok = np.isfinite(item.times)
    if not ok.any():
        continue
    p, tv = gold_file(load_audio(item, 16000), item.times[ok])
    P.append(p)
    TV.append(tv)
    REF.append(item.formants[ok][:, :4])
    GRP += [item.meta.get("group", "?")] * int(ok.sum())
P, TV, REF, GRP = np.concatenate(P, 1), np.concatenate(TV), np.concatenate(REF), np.array(GRP)
grp = {"all": np.ones(len(GRP), bool), "men": GRP == "m", "women": GRP == "w", "children": np.isin(GRP, ["b", "g"])}
hb = {}
for who, pred in [*((f"L{li}", P[li]) for li in range(L)), ("teacher_v3", TV)]:
    r = {}
    for gname, gm in grp.items():
        for k in range(4):
            m = gm & np.isfinite(REF[:, k]) & (REF[:, k] > 0)
            r[f"F{k + 1}_mae_{gname}"] = float(np.mean(np.abs(2 ** pred[m, k] - REF[m, k])))
    hb[who] = r
results["gold:hillenbrand_dev"] = hb
log(f"[hillenbrand dev] {len(rows)} tokens; teacher {hb['teacher_v3']}")
json.dump(results, open(OUT / "probe_results.json", "w"), indent=1)

# f0 corpora
for corpus in ("ptdb_tug", "mir1k", "mdb_stem_synth"):
    rows = filter_rows(read_manifest(f"data/manifests/{corpus}.csv"), ("dev",))
    P, TV, REF = [], [], []
    for i, row in enumerate(rows):
        item = mods[corpus].load(Path("data") / corpus, row)
        p, tv = gold_file(load_audio(item, 16000), item.times)
        P.append(p)
        TV.append(tv)
        REF.append(item.f0.astype(float))
        if (i + 1) % 500 == 0:
            log(f"  [{corpus}] {i + 1}/{len(rows)}")
    P, TV, REF = np.concatenate(P, 1), np.concatenate(TV), np.concatenate(REF)
    ok = np.isfinite(REF)
    v = ok & (REF > 0)
    res = {}
    for who, pred in [*((f"L{li}", P[li]) for li in range(L)), ("teacher_v3", TV)]:
        res[who] = {"f0_rpa": float(np.mean(np.abs(1200 * (pred[v, 8] - np.log2(REF[v]))) <= 50)),
                    "voiced_acc": float(np.mean((pred[ok, 9] > 0.5) == (REF[ok] > 0)))}
    results[f"gold:{corpus}_dev"] = res
    log(f"[{corpus} dev] {len(rows)} files; teacher {res['teacher_v3']}")
    json.dump(results, open(OUT / "probe_results.json", "w"), indent=1)
log("done")
