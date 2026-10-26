"""D-034 follow-up 3 (pre-registered 2026-10-26; reported only): how much does Parakeet's layer 5 on a 4 s segment depend
on how much audio surrounds it? Long streams: each speaker's (singer's) files joined in order, for Zeroth-Korean val,
VocalSet val and LibriTTS-R dev-clean (speakers by numeric id only). 64 segments of 4 s per corpus (seeded), each at least
14 s inside a stream, so every window fits. Windows of 4, 8, 16 and 30 s centred on the segment, starting on its 80 ms
grid, so the segment's encoder frames are the same frames in every window (frames 1..48 after its start, as the distil
caches index them). R^2 of each window's frames against the 30 s window's, with the per-dimension variance over the 30 s
frames: middle 2 s (frames 13..37), first and last 0.5 s (frames 1..6, 43..48); 95 % bootstrap intervals over segments.
Writes <out>.csv (corpus,window_s,region,r2,lo,hi) and prints the table. Run from the main checkout (VA_DEVICE=cuda).
VA_STATS30=1 (pre-registered 2026-10-26): also each 4, 8 and 16 s window with its own features converted to the 30 s
window's normalisation statistics (per mel band, scale and shift fitted on the interior frames against the 30 s
features at the same frames, applied to every frame), encoder on the short window alone; rows arm=own / stats30. The
segments are the same as without it (the extra bootstrap draws use their own generator)."""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_cache as D  # noqa: E402
import load  # noqa: E402

OUT = sys.argv[1]
SEG, ENC, NSEG, MARGIN = 64000, 1280, 64, 14 * 16000
WINDOWS = (4, 8, 16, 30)
REGIONS = {"middle 2 s": np.arange(13, 38), "first 0.5 s": np.arange(1, 7), "last 0.5 s": np.arange(43, 49)}
CAP = 180 * 16000  # read at most 3 minutes per stream


def streams():
    out = {}
    for corpus, root in (("zeroth_korean", "data/zeroth_korean"), ("vocalset", "data/vocalset")):
        by = defaultdict(list)
        for r in csv.DictReader(open(f"data/manifests/{corpus}.csv", encoding="utf-8")):
            if r["split"] == "val":
                by[r["speaker_id"]].append(str(Path(root) / r["path"]))
        out[corpus] = {k: sorted(v) for k, v in sorted(by.items())}
    by = defaultdict(list)
    for p in sorted((D.DS / "libritts_r" / "LibriTTS_R" / "dev-clean").rglob("*.wav")):
        by[p.parts[-3]].append(str(p))
    out["libritts_r_dev"] = dict(sorted(by.items()))
    return out


def join(paths):
    xs, n = [], 0
    for p in paths:
        x = D.read16k(p)
        xs.append(x)
        n += len(x)
        if n >= CAP:
            break
    return np.concatenate(xs)[:CAP]


def main():
    dev = torch.device(os.environ.get("VA_DEVICE", "cuda"))
    fe, enc = load.feature_extractor(), load.encoder(dev)
    enc.layers = enc.layers[:D.LAYER]

    STATS30 = os.environ.get("VA_STATS30") == "1"

    def feats(x):  # features and the extractor's mask (it marks the last frame as padding)
        f = fe([x], sampling_rate=16000, return_tensors="pt")
        return f["input_features"][0].numpy(), f["attention_mask"]

    def from_feats(f, mask):
        t = torch.from_numpy(np.ascontiguousarray(f, np.float32))[None].to(dev)
        with torch.inference_mode():
            return enc(t, attention_mask=mask.to(dev), output_hidden_states=True).hidden_states[D.LAYER][0].float().cpu().numpy()

    def layer5(x):
        f = fe([x], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            return enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[D.LAYER][0].float().cpu().numpy()

    def to_stats(fw, f30, off):
        """fw's features with f30's normalisation: per band, fw = a * f30 + b on the interior frames"""
        ref = f30[off:off + len(fw)]
        i = slice(3, len(fw) - 3)
        out = np.empty_like(fw)
        for k in range(fw.shape[1]):
            a, b = np.polyfit(ref[i, k], fw[i, k], 1)
            out[:, k] = (fw[:, k] - b) / a
        return out

    rng2 = np.random.default_rng(2611)
    rng = np.random.default_rng(2610)
    rows = []
    for corpus, spk in streams().items():
        audio = {k: join(v) for k, v in spk.items()}
        ok = [k for k, x in audio.items() if len(x) >= 2 * MARGIN + SEG]
        D.log(f"[{corpus}] {len(ok)} of {len(audio)} streams of at least 30 s")
        H = {w: [] for w in WINDOWS}
        H2 = {w: [] for w in WINDOWS[:-1]}  # stats30
        for n in range(NSEG):
            x = audio[ok[rng.integers(len(ok))]]
            s = int(rng.integers(MARGIN, len(x) - MARGIN - SEG + 1))
            m30 = int(round((30 * 16000 - SEG) / 2 / ENC))
            f30 = feats(x[s - m30 * ENC:s - m30 * ENC + 30 * 16000])[0] if STATS30 else None
            for w in WINDOWS:
                m = int(round((w * 16000 - SEG) / 2 / ENC))  # encoder frames of context before the segment
                a = s - m * ENC
                h = layer5(x[a:a + w * 16000])
                H[w].append(h[m:m + 49])  # the segment's frames 0..48 (1..48 used)
                if STATS30 and w < 30:
                    fw, mask = feats(x[a:a + w * 16000])
                    h2 = from_feats(to_stats(fw, f30, 8 * (m30 - m)), mask)
                    H2[w].append(h2[m:m + 49])
            if (n + 1) % 16 == 0:
                D.log(f"[{corpus}] {n + 1}/{NSEG}")
        ref = np.stack(H[30])  # (seg, 49, 1024)
        var = ref[:, 1:].reshape(-1, 1024).var(0)
        for w in WINDOWS:
            y = np.stack(H[w])
            for reg, fr in REGIONS.items():
                err = ((y[:, fr] - ref[:, fr]) ** 2 / var).mean((1, 2))  # per segment
                r2 = 1 - err.mean()
                bs = [1 - err[rng.integers(0, NSEG, NSEG)].mean() for _ in range(1000)]
                rows.append([corpus, w, reg, round(float(r2), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
                print(f"{corpus:15s} {w:2d} s  {reg:12s} R2 {r2:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]", flush=True)
        for w in H2 if STATS30 else ():
            y = np.stack(H2[w])
            for reg, fr in REGIONS.items():
                err = ((y[:, fr] - ref[:, fr]) ** 2 / var).mean((1, 2))
                err0 = ((np.stack(H[w])[:, fr] - ref[:, fr]) ** 2 / var).mean((1, 2))
                r2 = 1 - err.mean()
                bi = rng2.integers(0, NSEG, (1000, NSEG))
                bs = [1 - err[i].mean() for i in bi]
                gr = [err[i].mean() / err0[i].mean() for i in bi]  # stats30's gap to 1 over own's
                rows.append([corpus, w, reg + " (stats30)", round(float(r2), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
                print(f"{corpus:15s} {w:2d} s  {reg:12s} stats30 R2 {r2:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]"
                      f"  gap ratio {err.mean() / err0.mean():.2f} [{np.percentile(gr, 2.5):.2f}, {np.percentile(gr, 97.5):.2f}]", flush=True)
    with open(f"{OUT}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["corpus", "window_s", "region", "r2", "lo", "hi"], *rows])


if __name__ == "__main__":
    main()
