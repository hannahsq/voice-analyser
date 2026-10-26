"""D-034, reported only: Parakeet and the students at 2 ms offsets. For the first 256 crops of each held-out crop set
(distil_cache_v4.py's crops, same draws), Parakeet layer 5 on the audio shifted by 2k ms (k = 0..39: forty offsets over
one 80 ms frame), and each student on the audio shifted by 2s ms (s = 0..4: five offsets inside its 10 ms hop).
Alignment: Parakeet frame e at offset 2k ms against student frame 8e + q - 4 at sub-offset s, with q = k // 5, s = k % 5
(k = 5q reproduces v4's targets, checked against its cache). Writes, for v2 and v4,
  <prefix>_r2.csv       model,set,offset_ms,r2: distil_train_v4.py's R^2 at each offset (set variance per dimension);
  <prefix>_crops.csv    model,set,crop,offset_ms,r2: each crop's R^2 at each offset (mean over crops = the line above);
  <prefix>_spectrum.csv model,set,hz,power_db: the 2 ms trajectory's power spectrum (Parakeet: 40 offsets interleaved;
                        students: 5 sub-offsets interleaved), 1960 points, 0-250 Hz, dB relative to Parakeet's peak.
Students trained against another cache's statistics are restandardised to v4's (as fid_v4_on.py).
Args: v4 cache dir, out prefix, then LABEL=CKPT=STATS_DIR per student. Run from the main checkout (VA_DEVICE=cuda).
VA_CROPS=<csv with set,crop columns>: only those crops of those sets, in place of the first N (e.g. one f0 range).
VA_ARM=base|edge|clock (D-034's ripple check): the synthetic sets only, drawn with three extra frames (30 ms) of clip audio
after the crop. base: the crop alone, zero-padded (as without VA_ARM); edge: the real audio after it in place of the
padding; clock: as edge, the crop's start moved by 0-9 ms (seeded per crop). Parakeet sees the same span in every arm
(no context outside the crop). Also writes <prefix>_f0.csv (set,crop,f0_hz: median voiced f0).
D-034's follow-ups (as base unless stated): regen / jit0 / jit2 / hnr0 / whisper: each clip regenerated from its index
and seed (as drawn; jitter fixed at 1e-6 or 0.02; aspiration at 0 dB HNR; whispered whole), then cropped and channelled
exactly as SynthCrops; end0 / end40 / endR: crops drawn with
80 ms of extra clip audio, Parakeet's input ending at the crop's end, 40 ms later, or 0-78 ms later (seeded per crop)."""
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_cache as D  # noqa: E402
import load  # noqa: E402
from student import load_student  # noqa: E402
from distil_cache_v4 import ALL, Phased  # noqa: E402

from vocal_analysis.features.vqt import TorchVQT  # noqa: E402
from vocal_analysis.synth.channel import ChannelConfig  # noqa: E402
from vocal_analysis.synth.sampler import SynthConfig, generate  # noqa: E402
from vocal_analysis.synth.shards import INT16_SCALE  # noqa: E402
from vocal_analysis.train.data import SynthCrops, split_ids  # noqa: E402

ARM = os.environ.get("VA_ARM")
N, K, S, E4, OFF = int(os.environ.get("VA_N", 256)), 40, 5, 49, 32  # crops, Parakeet offsets, student sub-offsets, encoder frames, samples per 2 ms
T2 = E4 * K  # 2 ms trajectory length
# SynthConfig overrides per arm; jitter 1e-6, not 0, keeps the generator's random stream (whisper changes it: unpaired)
REGEN = {"regen": {}, "jit0": {"jitter": (1e-6, 1e-6)}, "jit2": {"jitter": (0.02, 0.02)}, "hnr0": {"hnr_db": (0.0, 0.0)},
         "whisper": {"p_whisper_clip": 1.0}}


class Regen:
    """stand-in for a flat store's audio array: slicing one clip returns it regenerated from its index and seed, stored
    as the shards store it (int16), with the config overrides given"""

    def __init__(self, flat, a_off, over):
        ix = json.loads((Path(flat).parent / "index.json").read_text())
        cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ix["synth_config"].items()})
        self.cfg = dataclasses.replace(cfg, **over)
        self.seed, self.cid = ix["seed"], {int(a): c for c, a in enumerate(a_off[:-1])}

    def __getitem__(self, sl):
        a = generate(self.cid[sl.start], self.cfg, self.seed).audio
        assert len(a) == sl.stop - sl.start, "regenerated clip differs in length"
        return np.clip(np.round(a * INT16_SCALE), -32768, 32767).astype(np.int16)


class RegenCrops(SynthCrops):
    def __init__(self, *a, over=None, **k):
        super().__init__(*a, **k)
        self.over = over or {}

    def _open(self):
        arr = super()._open()
        if not isinstance(arr["audio"], Regen):
            arr["audio"] = Regen(self.dir, arr["a_off"], self.over)
        return arr


class SubHop(torch.utils.data.Dataset):
    """per crop: five student inputs (shifted 0..8 ms) and forty Parakeet contexts (shifted 0..78 ms, leading frames)"""

    def __init__(self, base, idx):
        self.p, self.idx = Phased(base, ALL), idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if ARM:
            x, lab = self.p.ds.ds[self.idx[i]]
            x, v = np.asarray(x, np.float32), lab["voiced"][:D.CROP].astype(bool)
            f0 = float(np.median(lab["f0"][:D.CROP][v])) if v.any() else float("nan")
            if ARM in ("end0", "end40", "endR"):
                e = {"end0": 0, "end40": 640, "endR": OFF * int(np.random.default_rng([4343, self.idx[i]]).integers(0, 40))}[ARM]
                return [x[OFF * j:OFF * j + D.NS] for j in range(S)], [(x[OFF * k:D.NS + e], 0) for k in range(K)], f0
            if ARM in ("base",) + tuple(REGEN):
                x = x[:D.NS]
                pad = np.concatenate([x, np.zeros(S * OFF, np.float32)])
                return [pad[OFF * j:OFF * j + D.NS] for j in range(S)], [(x[OFF * k:], 0) for k in range(K)], f0
            u = 16 * int(np.random.default_rng([4242, self.idx[i]]).integers(0, 10)) if ARM == "clock" else 0
            return [x[u + OFF * j:u + OFF * j + D.NS] for j in range(S)], [(x[u + OFF * k:u + D.NS], 0) for k in range(K)], f0
        x, s = self.p.source(self.idx[i])
        if s is None:  # synthetic: the crop alone
            pad = np.concatenate([x, np.zeros(S * OFF, np.float32)])
            return [pad[OFF * j:OFF * j + D.NS] for j in range(S)], [(x[OFF * k:], 0) for k in range(K)]
        xp = np.concatenate([x, np.zeros(S * OFF, np.float32)])
        stud = [xp[s + OFF * j:s + OFF * j + D.NS] for j in range(S)]
        ctx = []
        for k in range(K):
            s2 = s + OFF * k
            pre = (min(s2, D.CTX) // 1280) * 1280
            ctx.append((x[s2 - pre:s + D.NS + D.CTX], pre // 1280))
        return stud, ctx


def main():
    C4, PRE, SPECS = Path(sys.argv[1]), sys.argv[2], [a.split("=", 2) for a in sys.argv[3:]]
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    vq = TorchVQT(dev)
    st = np.load(C4 / "stats.npz")
    MU4, SD4 = (torch.tensor(st[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    fe, enc = load.feature_extractor(), load.encoder(dev)
    enc.layers = enc.layers[:D.LAYER]
    students = []
    for label, ck, sdir in SPECS:
        s = np.load(Path(sdir) / "stats.npz")
        students.append((label, load_student(ck, dev).requires_grad_(False),
                         *(torch.tensor(s[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))))

    def parakeet(ctxs):
        f = fe(list(ctxs), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            return enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[D.LAYER].float()

    def student(m, MU, SD, audio):
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16):
            y = m(vq.log_power_with_context(audio, D.CROP)).float()
        return (y * SD + MU - MU4) / SD4

    cfg3 = json.load(open("runs/conformer_v2r2_s1/config.json"))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    flat = Path(cfg3["train"]["data"]) / "flat"
    _, val_ids = split_ids(json.loads((Path(cfg3["train"]["data"]) / "index.json").read_text())["n_clips"])
    v2s = Path("data/synth/train_v2s")
    pools = D.real_files() if not ARM else None
    SETS = {
        "held_synth_v2": D.SynthWrap(SynthCrops(flat, val_ids[:512], D.CROP, channel, seed=12345, fixed=True), 512),
        "held_synth_v2s": D.SynthWrap(SynthCrops(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                                 D.CROP, channel, seed=12345, fixed=True), 512),
        "held_vocalset": D.RealCrops(pools["vocalset_val"], 512, 12345),
        "held_zeroth": D.RealCrops(pools["zeroth_korean_val"], 512, 12345),
    } if not ARM else {}
    if ARM:  # the synthetic sets, with 30 ms (80 ms for the end arms) of clip audio after each crop
        nf = D.CROP + (8 if ARM.startswith("end") else 3)
        mk = (lambda *a, **k: RegenCrops(*a, over=REGEN[ARM], **k)) if ARM in REGEN else SynthCrops
        SETS = {
            "held_synth_v2": D.SynthWrap(mk(flat, val_ids[:512], nf, channel, seed=12345, fixed=True), 512),
            "held_synth_v2s": D.SynthWrap(mk(v2s / "flat", split_ids(json.loads((v2s / "index.json").read_text())["n_clips"])[1][:512],
                                             nf, channel, seed=12345, fixed=True), 512),
        }
        ff = open(f"{PRE}_f0.csv", "w", newline="")
        wf = csv.writer(ff)
        wf.writerow(["set", "crop", "f0_hz"])
    IDX = {}  # set -> crop indices (the first N when VA_CROPS is unset)
    if os.environ.get("VA_CROPS"):
        for r in csv.DictReader(open(os.environ["VA_CROPS"], encoding="utf-8")):
            IDX.setdefault(r["set"], []).append(int(r["crop"]))
        SETS = {k: v for k, v in SETS.items() if k in IDX}
    j0 = torch.arange(1, E4 + 1, device=dev) * 8 - 4
    HANN = torch.hann_window(T2, periodic=False, device=dev)[None, :, None]
    FREQ = np.fft.rfftfreq(T2, 0.002)
    fr, fc, fs = (open(f"{PRE}_{n}.csv", "w", newline="") for n in ("r2", "crops", "spectrum"))
    wr, wc, ws = csv.writer(fr), csv.writer(fc), csv.writer(fs)
    wr.writerow(["model", "set", "offset_ms", "r2"])
    wc.writerow(["model", "set", "crop", "offset_ms", "r2"])
    ws.writerow(["model", "set", "hz", "power_db"])

    for name, base in SETS.items():
        idx = IDX.get(name) or list(range(N))
        NC = len(idx)
        cached = np.load(C4 / name / "target.npy", mmap_mode="r")
        se = {lab: torch.zeros(NC, K, 1024, dtype=torch.float64) for lab, *_ in students}
        s1 = torch.zeros(K, 1024, dtype=torch.float64, device=dev)
        s2 = torch.zeros(K, 1024, dtype=torch.float64, device=dev)
        P = {lab: torch.zeros(T2 // 2 + 1, dtype=torch.float64, device=dev) for lab in ["parakeet"] + [s[0] for s in students]}
        worst = 0.0
        dl = torch.utils.data.DataLoader(SubHop(base, idx), batch_size=16, num_workers=6, collate_fn=D.as_list)
        i0 = 0
        for b in dl:
            B = len(b)
            tg = torch.zeros(B, K, E4, 1024, device=dev)
            for k in range(K):
                h = parakeet([it[1][k][0] for it in b])
                for r, it in enumerate(b):
                    pre = it[1][k][1]
                    tg[r, k] = h[r, pre + 1:pre + 1 + E4]
            if ARM:
                for r, it in enumerate(b):
                    wf.writerow([name, idx[i0 + r], round(it[2], 2)])
            for q in range(8 if not ARM else 0):  # k = 5q is v4's phase q (the arms' draws are not the cached ones)
                worst = max(worst, float((tg[:, 5 * q] - torch.from_numpy(np.stack([np.asarray(cached[j, q], np.float32) for j in idx[i0:i0 + B]])).to(dev)).abs().max()))
            tg = (tg - MU4) / SD4
            s1 += tg.sum((0, 2)).double()
            s2 += (tg ** 2).sum((0, 2)).double()
            z = tg.permute(0, 2, 1, 3).reshape(B, T2, 1024)  # time index 40(e - 1) + k
            P["parakeet"] += (torch.fft.rfft((z - z.mean(1, keepdim=True)) * HANN, dim=1).abs() ** 2).sum((0, 2)).double()
            for lab, m, MU, SD in students:
                ys = torch.stack([student(m, MU, SD, torch.from_numpy(np.stack([it[0][j] for it in b])).to(dev)) for j in range(S)], 1)  # (B, 5, 400, 1024)
                for k in range(K):
                    q, s = divmod(k, S)
                    se[lab][i0:i0 + B, k] = ((ys[:, s, j0 + q] - tg[:, k]) ** 2).sum(1).double().cpu()
                z = ys[:, :, 4:4 + 8 * E4].permute(0, 2, 1, 3).reshape(B, T2, 1024)  # time index 5(j - 4) + s
                P[lab] += (torch.fft.rfft((z - z.mean(1, keepdim=True)) * HANN, dim=1).abs() ** 2).sum((0, 2)).double()
            i0 += B
            D.log(f"[{name}] {i0}/{NC}")
        cnt = NC * E4
        var = (s2 / cnt - (s1 / cnt) ** 2).cpu()
        for lab in se:
            r2c = (1 - (se[lab] / (E4 * var[None])).mean(-1)).numpy()  # (NC, K)
            for k in range(K):
                wr.writerow([lab, name, 2 * k, round(float(r2c[:, k].mean()), 4)])
                for i in range(NC):
                    wc.writerow([lab, name, idx[i], 2 * k, round(float(r2c[i, k]), 4)])
            D.log(f"[{name}] {lab} R2 by 2 ms offset: {np.round(r2c.mean(0), 3).tolist()}")
        ref = P["parakeet"][1:].max()
        for lab, p in P.items():
            for f, v in zip(FREQ, (10 * torch.log10(p / ref)).cpu().numpy()):
                ws.writerow([lab, name, f"{f:.3f}", round(float(v), 2)])
        D.log(f"[{name}] Parakeet at k = 5q against v4's cached targets: max |difference| {worst:.4f}")
        for f in (fr, fc, fs):
            f.flush()
    for f in (fr, fc, fs) + ((ff,) if ARM else ()):
        f.close()


if __name__ == "__main__":
    main()
