"""D-049: teacher v3's F1 bias on resynthesised real speech (Zeroth-Korean val), by sex.

    python scripts/analysis/d049_resynth_speech.py pilot <out dir>   (men only: spreads for the noise floor)
    python scripts/analysis/d049_resynth_speech.py run <out dir>     (both sexes, both models; writes d049.txt)

Clips: data/resynth/zeroth_korean_val_d16k (10 held-out speakers: real source and dynamics, a perturbed synthetic tract
with exact labels), as stored (no channel). The store keeps no clip-to-file map, so only shards whose 8 files share one
speaker are used (52 of 58). Sex: Zeroth's recorded SEX column (numeric ids only). Per clip, the median signed F1 error
(model - label) on label-voiced frames with finite F1, 5 frames trimmed at each end.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

STORE = Path("data/resynth/zeroth_korean_val_d16k")
MODELS = {"teacher_v3": Path("runs/conformer_v2r2_s1_d16k/ckpt_020000.pt"),  # trained with Zeroth train resynthesis
          "no_resynth": Path("runs/conformer_v2_s1_d16k/ckpt_020000.pt")}    # same seed and recipe, train_v2 only
SEX = {"104": "m", "105": "f", "112": "m", "118": "m", "121": "f", "126": "f", "132": "f", "137": "f", "147": "f", "149": "m"}
TRIM = 5
N_BOOT = 2000


def clip_speakers():
    """{flat clip index: speaker} for clips from single-speaker shards (flat order = sorted shards with clips)"""
    out, k = {}, 0
    for p in sorted(glob.glob(str(STORE / "shard_*.json"))):
        d = json.load(open(p))
        n = d["n_clips"]
        if not n:
            continue
        spk = {f.split("/")[2] for f in d["files"]}
        if len(spk) == 1:
            s = spk.pop()
            out.update({k + j: s for j in range(n)})
        k += n
    assert k == json.loads((STORE / "index.json").read_text())["n_clips"]
    return out


def errors(model_path, clips, dev):
    from vocal_analysis.models.infer import analyse, load_teacher
    from vocal_analysis.synth.shards import INT16_SCALE

    m = load_teacher(model_path, dev)[0]
    flat = STORE / "flat"
    audio = np.load(flat / "audio.npy", mmap_mode="r")
    ao, fo = np.load(flat / "audio_offsets.npy"), np.load(flat / "frame_offsets.npy")
    lv, lf = np.load(flat / "label_voiced.npy", mmap_mode="r"), np.load(flat / "label_formants.npy", mmap_mode="r")
    out = {}
    for c in clips:
        x = audio[ao[c]:ao[c + 1]].astype(np.float32) / INT16_SCALE
        r = analyse(m, x, dev)
        v, F = np.asarray(lv[fo[c]:fo[c + 1]]), np.asarray(lf[fo[c]:fo[c + 1], 0])
        n = min(len(v), len(r["formants"]))
        keep = v[:n].astype(bool) & np.isfinite(F[:n])
        keep[:TRIM] = False
        keep[n - TRIM:] = False
        e = r["formants"][:n, 0] - F[:n]
        e = e[keep & np.isfinite(e)]
        out[c] = (float(np.median(e)), float(np.median(np.abs(e))), int(len(e))) if len(e) >= 20 else (np.nan, np.nan, int(len(e)))
    return out


def summarise(E, cs, sx, rng):
    """speaker-weighted mean of per-clip median errors, two-level bootstrap (speakers, then clips within them)"""
    spk = sorted({cs[c] for c in E if SEX[cs[c]] == sx})
    by = {s: np.array([E[c][0] for c in E if cs[c] == s and np.isfinite(E[c][0])]) for s in spk}
    point = np.mean([by[s].mean() for s in spk])
    bs = []
    for _ in range(N_BOOT):
        ss = rng.choice(spk, len(spk))
        bs.append(np.mean([rng.choice(by[s], len(by[s])).mean() for s in ss]))
    return point, np.percentile(bs, 2.5), np.percentile(bs, 97.5), {s: (by[s].mean(), len(by[s])) for s in spk}


def main():
    import torch

    stage, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    cs = clip_speakers()
    rng = np.random.default_rng(49)
    lines = [f"D-049 ({time.strftime('%Y-%m-%d %H:%M')}): {len(cs)} clips from single-speaker shards; "
             + ", ".join(f"{s} ({SEX[s]}) {sum(v == s for v in cs.values())}" for s in sorted(SEX))]
    sexes = ("m",) if stage == "pilot" else ("f", "m")
    models = {"teacher_v3": MODELS["teacher_v3"]} if stage == "pilot" else MODELS
    reading = None
    for name, path in models.items():
        clips = [c for c, s in cs.items() if SEX[s] in sexes]
        cache = out / f"{name}.json"
        E = {int(k): tuple(v) for k, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
        todo = [c for c in clips if c not in E]
        if todo:
            E.update(errors(path, todo, dev))
            cache.write_text(json.dumps({str(k): v for k, v in E.items()}))
        E = {c: E[c] for c in clips}
        for sx in sexes:
            p, lo, hi, per = summarise(E, cs, sx, rng)
            within = [np.std([E[c][0] for c in E if cs[c] == s and np.isfinite(E[c][0])]) for s in per]
            lines.append(f"{name} {'women' if sx == 'f' else 'men'}: mean signed F1 error {p:+.1f} [{lo:+.1f}, {hi:+.1f}] Hz; "
                         f"per speaker " + ", ".join(f"{s} {m:+.1f} (n {n})" for s, (m, n) in per.items())
                         + f"; within-speaker clip SD median {np.median(within):.1f} Hz; mean |error| "
                         f"{np.nanmean([E[c][1] for c in E if SEX[cs[c]] == sx]):.1f}")
            if name == "teacher_v3" and sx == "f":
                reading = ("bias appears on resynthesised speech" if lo >= 4 else
                           "no bias on resynthesised speech" if lo > -4 and hi < 4 else
                           "underread on resynthesised speech" if hi <= -4 else "inconclusive")
    if reading:
        lines.append(f"READING (teacher v3, women, +-4 Hz): {reading}")
    (out / f"{stage}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
