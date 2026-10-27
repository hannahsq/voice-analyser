"""D-037, before registering (AGENTS.md: a measure's noise floor): the head-seed spread of synthetic validation NLL and SNR
for D-035's three frozen-feature T heads (teacher v3's trunk, 4,000 steps), from the cached features; CPU only.
Validation: train_v2's first 512 held-out clips (vtl_cache.py's val part), sonorant frames of tube clips whose length was
not drawn from f0. SNR (dB) = 20 log10(label SD / sigma), sigma from the NLL as a calibrated Gaussian (NLL = ln sigma + 1/2,
the training loss's form without its constant); label SD = log2 SD of the labelled training crops (cache_heads.py's).
Also each seed's per-frame NLL split by clip halves (even / odd clip index) to show the measure's sampling noise.
Args: vtl cache dir [cache/vtl], head cache dir [cache/heads_l3_6]."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parakeet"))
from vtl_head import Head  # noqa: E402


def main():
    vtl = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/vtl")
    hc = Path(sys.argv[2] if len(sys.argv) > 2 else "cache/heads_l3_6")
    v = np.load(hc / "vtl.npz")
    m = v["tube"].astype(bool) & ~v["from_f0"].astype(bool) & np.isfinite(v["length_cm"])
    sd = np.log2(v["length_cm"][m]).std()
    flat = Path("data/synth/train_v2/flat")
    clip = np.load(vtl / "val" / "clip.npy")
    L = np.log2(np.load(flat / "clip_length_cm.npy")[clip])
    keep = np.load(flat / "clip_tube.npy")[clip] & ~np.load(flat / "clip_length_from_f0.npy")[clip]
    son = np.load(Path("cache/emb_v1/synth_train_v2/sonorant.npy")).reshape(len(clip), 400) > 0.5
    feats = np.load(vtl / "val" / "feats.npy")
    print(f"label log2 SD {sd:.4f} ({m.sum()} labelled training crops); validation clips {keep.sum()} of {len(clip)}")
    rows = []
    for s in (0, 1, 2):
        ck = torch.load(vtl / "heads" / f"T_s{s}" / "head.pt", map_location="cpu", weights_only=False)
        h = Head(ck["in_dim"])
        h.load_state_dict(ck["state"])
        h.eval()
        per_clip = []
        for c in np.flatnonzero(keep):
            x = feats[c][son[c]]
            if len(x) == 0:
                continue
            with torch.inference_mode():
                mu, lv = (t.numpy() for t in h(torch.from_numpy(np.asarray(x, np.float32))))
            lv = np.clip(lv, -12, 4)
            per_clip.append((c, np.mean(0.5 * (lv + (L[c] - mu) ** 2 / np.exp(lv)))))
        cid, nll = np.array([a for a, _ in per_clip]), np.array([b for _, b in per_clip])
        snr = lambda n: 20 * np.log10(sd / np.exp(n - 0.5))  # noqa: E731
        halves = [nll[cid % 2 == k].mean() for k in (0, 1)]
        rows.append((nll.mean(), snr(nll.mean())))
        print(f"T seed {s}: validation NLL {nll.mean():+.3f}  SNR {snr(nll.mean()):5.2f} dB  (clip halves {snr(halves[0]):5.2f} / {snr(halves[1]):5.2f} dB)")
    r = np.array(rows)
    print(f"\nhead-seed spread: NLL SD {r[:, 0].std(ddof=1):.3f} nats; SNR SD {r[:, 1].std(ddof=1):.2f} dB; "
          f"Q1 bound max(1 dB, 2 SD) = {max(1.0, 2 * r[:, 1].std(ddof=1)):.2f} dB")


if __name__ == "__main__":
    main()
