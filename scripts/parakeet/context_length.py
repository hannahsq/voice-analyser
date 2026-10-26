"""D-034 follow-up 3 (pre-registered 2026-10-26; reported only): how much does Parakeet's layer 5 on a 4 s segment depend
on how much audio surrounds it? Long streams: each speaker's (singer's) files joined in order, for Zeroth-Korean val,
VocalSet val and LibriTTS-R dev-clean (speakers by numeric id only). 64 segments of 4 s per corpus (seeded), each at least
14 s inside a stream, so every window fits. Windows of 4, 8, 16 and 30 s centred on the segment, starting on its 80 ms
grid, so the segment's encoder frames are the same frames in every window (frames 1..48 after its start, as the distil
caches index them). R^2 of each window's frames against the 30 s window's, with the per-dimension variance over the 30 s
frames: middle 2 s (frames 13..37), first and last 0.5 s (frames 1..6, 43..48); 95 % bootstrap intervals over segments.
Writes <out>.csv (corpus,window_s,region,r2,lo,hi) and prints the table. Run from the main checkout (VA_DEVICE=cuda)."""
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

    def layer5(x):
        f = fe([x], sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            return enc(f["input_features"].to(dev), attention_mask=f["attention_mask"].to(dev), output_hidden_states=True).hidden_states[D.LAYER][0].float().cpu().numpy()

    rng = np.random.default_rng(2610)
    rows = []
    for corpus, spk in streams().items():
        audio = {k: join(v) for k, v in spk.items()}
        ok = [k for k, x in audio.items() if len(x) >= 2 * MARGIN + SEG]
        D.log(f"[{corpus}] {len(ok)} of {len(audio)} streams of at least 30 s")
        H = {w: [] for w in WINDOWS}
        for n in range(NSEG):
            x = audio[ok[rng.integers(len(ok))]]
            s = int(rng.integers(MARGIN, len(x) - MARGIN - SEG + 1))
            for w in WINDOWS:
                m = int(round((w * 16000 - SEG) / 2 / ENC))  # encoder frames of context before the segment
                a = s - m * ENC
                h = layer5(x[a:a + w * 16000])
                H[w].append(h[m:m + 49])  # the segment's frames 0..48 (1..48 used)
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
    with open(f"{OUT}.csv", "w", newline="") as f:
        csv.writer(f).writerows([["corpus", "window_s", "region", "r2", "lo", "hi"], *rows])


if __name__ == "__main__":
    main()
