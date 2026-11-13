"""D-112: D-109's F1 shift across speaking styles and excitation in one recording session (one voice, one chain).
Inference only. The recordings are private: per-take tables go to <out dir> (outside the repo); stdout carries summaries.

Inputs: the session folder (block1_round{1,2,3}.flac, block2.flac, block3.flac) and the take list (markdown, the order
each block was recorded in; a row ticked "redo" was recorded twice and the later take is kept).
Segmentation by energy (10 ms frames, threshold 10 dB above the file's 10th-percentile level): sustained takes close
gaps < 0.3 s and keep runs >= 2 s; words close gaps < 0.15 s and keep voiced runs (pyworld) within 15 dB of the
block's median level; sentences close gaps < 0.35 s and keep runs >= 0.8 s. The counts must match the take list.
Each unit is cut with 0.3 s of its own recording on each side and analysed alone (16 kHz, as `load_audio`).
Frames: sustained voiced takes and words, every model voiced and sonorant (p > 0.5); whispered takes, every model
sonorant; sustained takes drop their first and last 0.3 s; sentences, every model voiced and sonorant.
Per unit: the median over its frames of each model's F1-F3. Sentences are also split by 5b's own F1 per frame into
the lower and upper thirds of the block's frames (the split cancels in d109 - D-065, which carries no 5b term).
Measures, per style (sustained voiced at A2, A3; whispered; words; sentences): S = d109 - 5b, S0 = D-065 - 5b (branch
null), and D = S - S0 = d109 - D-065 (seed means each); the open - close contrast (open: æ, ɐ; close: i, ʉ; sentences:
upper - lower third) as the difference of the groups' medians over units, with a 95 % bootstrap over units (2000).

    python scripts/analysis/d112_session.py <session dir> <take list> <out dir> [--null-only]
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

B = {"b1": "runs/d040_5b_s1_b20k/ckpt_last.pt", "b2": "runs/d040_5b_s2_b20k/ckpt_last.pt"}
N = {"n1": "runs/d065_gain_s1_b20k/ckpt_last.pt", "n2": "runs/d065_gain_s2_b20k/ckpt_last.pt"}
R = {"r1": "runs/d109_s1/ckpt_last.pt", "r2": "runs/d109_s2/ckpt_last.pt"}
OPEN, CLOSE = ("æ", "ɐ"), ("i", "ʉ")
PAD, SEED = 0.3, 5112


def parse_takes(md):
    """Block -> ordered list of unit dicts, from the take list's tables."""
    rows, block, rnd = {}, None, None
    for line in Path(md).read_text(encoding="utf-8").splitlines():
        if line.startswith("## Block"):
            block, rnd = int(line.split()[2].rstrip(":")), None
        elif line.startswith("### Round"):
            rnd = int(line.split()[2])
        elif line.startswith("| ") and re.match(r"\| \d+ \|", line):
            c = [x.strip() for x in line.strip().split("|")[1:]]
            redo = "[x]" in c[4] if block == 1 else ("[x]" in c[3] if block == 3 else "[x]" in c[4])
            if block == 1:
                v = c[1].strip("/").replace("̥", "")
                how = "wh" if "whisper" in c[2] else ("A2" if "A2" in c[2] else "A3")
                u = {"vowel": v, "style": how, "id": int(c[0])}
            elif block == 2:
                u = {"vowel": c[2].strip("/"), "style": "word", "word": c[1], "id": int(c[0])}
            else:
                u = {"vowel": None, "style": "sentence", "text": c[1], "id": int(c[0])}
            key = f"b{block}r{rnd}" if block == 1 else f"b{block}"
            rows.setdefault(key, []).append((u, redo))
    out = {}
    for key, lst in rows.items():
        units = []
        for u, redo in lst:  # a redo row follows the take it repeats: both were recorded, the later one is kept
            units.append(dict(u, keep=True))
            if redo:
                units[-2]["keep"] = False
        out[key] = units
    return out


def segment(x, sr, gap, min_len, words=False):
    import pyworld as pw

    hop = int(0.01 * sr)
    db = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "valid")[::hop] + 1e-12))
    on = db > np.percentile(db, 10) + 10
    g, segs, i, n = int(gap * 100), [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or on[j:j + g].any()):
                j += 1
            if j - i >= int(min_len * 100):
                segs.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    if words:
        lv = np.array([np.median(db[a // hop:b // hop]) for a, b in segs])
        keep = []
        for (a, b), l in zip(segs, lv):
            f0, _ = pw.harvest(x[a:b].astype(np.float64), sr, f0_floor=60, f0_ceil=600, frame_period=10)
            if (f0 > 0).mean() >= 0.5 and l >= np.median(lv) - 15:
                keep.append((a, b))
        segs = keep
    return segs


def to16(x, sr):
    from scipy.signal import resample_poly

    g = np.gcd(int(sr), 16000)
    return resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)


def run(session, md, models, dev):
    import soundfile as sf

    from vocal_analysis.models.infer import analyse, load_teacher

    takes = parse_takes(md)
    plan = [("block1_round1", "b1r1", 0.3, 2.0, False), ("block1_round2", "b1r2", 0.3, 2.0, False),
            ("block1_round3", "b1r3", 0.3, 2.0, False), ("block2", "b2", 0.15, 0.2, True), ("block3", "b3", 0.35, 0.8, False)]
    cuts = []
    for fname, key, gap, ml, words in plan:
        x, sr = sf.read(str(Path(session) / f"{fname}.flac"), dtype="float32", always_2d=True)
        x = x.mean(1)
        segs = segment(x, sr, gap, ml, words)
        units = takes[key]
        if len(segs) != len(units):
            raise SystemExit(f"{fname}: {len(segs)} segments, take list has {len(units)}")
        p = int(PAD * sr)
        for (a, b), u in zip(segs, units):
            if u["keep"]:
                lo, hi = max(0, a - p), min(len(x), b + p)
                cuts.append((dict(u, block=key, t0=a / sr, dur=(b - a) / sr), to16(x[lo:hi], sr), (a - lo) / sr, (hi - b) / sr))
    res = {m: [] for m in models}
    for m, ck in models.items():
        model, _ = load_teacher(ck, dev)
        res[m] = [analyse(model, y, dev) for _, y, _, _ in cuts]
        del model
    units = []
    for i, (u, y, pre, post) in enumerate(cuts):
        o = [res[m][i] for m in models]
        T = len(o[0]["f0"])
        t = np.arange(T) * 0.01
        inside = (t >= pre) & (t <= T * 0.01 - post)
        son = np.all([r["sonorant_p"] > 0.5 for r in o], 0)
        voi = np.all([r["voiced_p"] > 0.5 for r in o], 0)
        if u["style"] in ("A2", "A3", "wh"):
            inside &= (t >= pre + PAD) & (t <= T * 0.01 - post - PAD)
        ok = inside & son & (voi if u["style"] != "wh" else True)
        u = dict(u, n=int(ok.sum()))
        if ok.sum() >= 5:
            for m in models:
                u[m] = np.median(res[m][i]["formants"][ok, :3], 0).tolist()
            if u["style"] == "sentence":
                u["frames"] = {m: res[m][i]["formants"][ok, 0].tolist() for m in models}
        units.append(u)
    return units


def summarise(units, have_r):
    rng = np.random.default_rng(SEED)
    M = lambda u, a, b: (np.array(u[a]) + np.array(u[b])) / 2  # noqa: E731
    lines = []

    def boot_contrast(vo, vc):
        if len(vo) == 0 or len(vc) == 0:
            return np.nan, np.nan, np.nan
        bs = [np.median(rng.choice(vo, len(vo))) - np.median(rng.choice(vc, len(vc))) for _ in range(2000)]
        return np.median(vo) - np.median(vc), *np.percentile(bs, [2.5, 97.5])

    for style in ("A2", "A3", "wh", "word"):
        us = [u for u in units if u["style"] == style and "b1" in u]
        if not us:
            continue
        lines.append(f"{style}: {len(us)} units, {sum(u['n'] for u in us)} frames (dropped for too few frames: "
                     f"{sum(1 for u in units if u['style'] == style and 'b1' not in u)})")
        meas = {"S0 = D-065 - 5b": lambda u: (M(u, "n1", "n2") - M(u, "b1", "b2"))[0],
                "seed null 5b s1 - s2": lambda u: u["b1"][0] - u["b2"][0]}
        if have_r:
            meas = {"S = d109 - 5b": lambda u: (M(u, "r1", "r2") - M(u, "b1", "b2"))[0],
                    "D = d109 - D-065 (primary)": lambda u: (M(u, "r1", "r2") - M(u, "n1", "n2"))[0],
                    "  seed 1 (d109 - D-065)": lambda u: u["r1"][0] - u["n1"][0],
                    "  seed 2 (d109 - D-065)": lambda u: u["r2"][0] - u["n2"][0], **meas}
        for name, fn in meas.items():
            v = np.array([fn(u) for u in us])
            vo = np.array([fn(u) for u in us if u["vowel"] in OPEN])
            vc = np.array([fn(u) for u in us if u["vowel"] in CLOSE])
            c, lo, hi = boot_contrast(vo, vc)
            byv = ", ".join(f"{vw} {np.median([fn(u) for u in us if u['vowel'] == vw]):+.1f}"
                            for vw in ("i", "ʉ", "e", "ə", "ɔ", "ɐ", "æ") if any(u["vowel"] == vw for u in us))
            lines.append(f"  {name:28s} F1 median {np.median(v):+6.1f}; open - close {c:+6.1f} [{lo:+.1f}, {hi:+.1f}]   by vowel: {byv}")
        if have_r:
            for k, lab in ((1, "F2"), (2, "F3")):
                v = np.array([(M(u, "r1", "r2") - M(u, "n1", "n2"))[k] for u in us])
                lines.append(f"  D {lab} median {np.median(v):+.1f}")
    ss = [u for u in units if u["style"] == "sentence" and "frames" in u]
    if ss:
        f1b = np.concatenate([(np.array(u["frames"]["b1"]) + np.array(u["frames"]["b2"])) / 2 for u in ss])
        lo_t, hi_t = np.percentile(f1b, [100 / 3, 200 / 3])
        lines.append(f"sentence: {len(ss)} sentences, {len(f1b)} frames; 5b F1 thirds split at {lo_t:.0f} / {hi_t:.0f} Hz")
        pairs = {"S0 = D-065 - 5b": ("n", "b"), "seed null 5b s1 - s2": None}
        if have_r:
            pairs = {"S = d109 - 5b": ("r", "b"), "D = d109 - D-065 (primary)": ("r", "n"), **pairs}
        for name, pr in pairs.items():
            up, lw, al = [], [], []
            for u in ss:
                fr = {m: np.array(v) for m, v in u["frames"].items()}
                fb = (fr["b1"] + fr["b2"]) / 2
                d = fr["b1"] - fr["b2"] if pr is None else (fr[pr[0] + "1"] + fr[pr[0] + "2"]) / 2 - (fr[pr[1] + "1"] + fr[pr[1] + "2"]) / 2
                al.append(np.median(d))
                if (fb >= hi_t).sum() >= 3:
                    up.append(np.median(d[fb >= hi_t]))
                if (fb < lo_t).sum() >= 3:
                    lw.append(np.median(d[fb < lo_t]))
            c, lo, hi = boot_contrast(np.array(up), np.array(lw))
            lines.append(f"  {name:28s} F1 median {np.median(al):+6.1f}; upper - lower third {c:+6.1f} [{lo:+.1f}, {hi:+.1f}] "
                         f"(upper {np.median(up):+.1f}, lower {np.median(lw):+.1f})")
    return lines


if __name__ == "__main__":
    import torch

    session, md, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    null_only = "--null-only" in sys.argv
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    units = run(session, md, {**B, **N} if null_only else {**B, **N, **R}, dev)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("d112_units_null.json" if null_only else "d112_units.json")).write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")
    print("\n".join(summarise(units, not null_only)))
