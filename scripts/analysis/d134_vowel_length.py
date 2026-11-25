"""D-134: does the tract-length head depend on the vowel, and does KL change it? Arm A re-reads D-132's cascade clips
by vowel; arm B renders the same shapes through the KL waveguide.

    python scripts/analysis/d134_vowel_length.py render             -> data/probes/vowel_length_kl_v1/
    python scripts/analysis/d134_vowel_length.py score NAME CKPT    -> reports/probes/vowel_length_kl/NAME.npz ($VA_DEVICE)
    python scripts/analysis/d134_vowel_length.py analyse            -> reports/analysis/d134/d134.txt
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/probes/vowel_length_kl_v1"
RAW = ROOT / "reports/probes/vowel_length_kl"
D132_CACHE = ROOT / "data/probes/length_f0_v1"
D132_RAW = ROOT / "reports/probes/length_f0"
OUT = ROOT / "reports/analysis/d134"
LENGTHS = (13.0, 14.5, 15.6, 17.0, 18.5)
F0S = (110.0, 180.0, 280.0)
SEEDS = (0, 1)
DUR = 1.5
EDGE = 20
GAP = 0.1
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")
BACK = ("o", "u", "U", "O")
B = 1000


def _peak(L: float) -> float:
    return float(np.clip(0.07 + (L - 15.6) / (17.46 - 15.6) * 0.05, 0.05, 0.14))


def cmd_render(only: int | None = None) -> None:
    from vocal_analysis.synth import branches as br
    from vocal_analysis.synth import kl, sampler, tube

    CACHE.mkdir(parents=True, exist_ok=True)
    items = []
    k = 0
    for L in LENGTHS:
        rel = L / tube.STORY_LENGTH_CM
        for v, (q1, q2) in tube.STORY_VOWELS.items():
            s = tube.build_shape(q1, q2, L)
            (F, _, _), = kl.frame_labels([s])
            for f0 in F0S:
                for seed in SEEDS:
                    tag = "".join(c + "^" if c.isupper() else c for c in v)  # Windows names ignore case
                    name = f"L{L:g}_{tag}_f{f0:g}_s{seed}"
                    items.append({"name": name, "length": L, "vowel": v, "f0": f0, "seed": seed})
                    k += 1
                    if only is not None and k > only:
                        continue
                    path = CACHE / f"{name}.npz"
                    if path.exists():
                        continue
                    n = int(DUR * kl.FS)
                    frames = n // kl.HOP + 1
                    rng = np.random.default_rng(zlib.crc32(name.encode()))
                    f0_t = f0 * 2 ** sampler._smooth_noise(rng, frames, 15, 2.0 / 12.0)
                    ag = kl.glottal_area(f0_t, n, kl.FS, _peak(L), oq=0.6, sq=1.3, gap=GAP)
                    a = kl.render([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel,
                                  subglottal=br.subglottal(L, kl.section_length()))
                    x = a["audio"].astype(np.float64)
                    fade = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / 320.0, 0, 1)
                    x = 0.5 * x * fade / (np.abs(x * fade).max() + 1e-12)
                    np.savez_compressed(path, audio=np.round(x * 32000).astype(np.int16), formants=np.asarray(F[:3]))
        print(f"rendered length {L}", flush=True)
    if only is None:
        (CACHE / "meta.json").write_text(json.dumps({"items": items}, indent=1))
        print(f"{len(items)} items")


def cmd_score(name: str, ckpt: str) -> None:
    import torch

    from vocal_analysis.models.infer import DEFAULT_WINDOW, analyse, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model, _ = load_teacher(ckpt, dev)
    causal = bool(model.cfg.causal)
    items = json.loads((CACHE / "meta.json").read_text())["items"]
    out = np.full((len(items), 4), np.nan)
    for i, it in enumerate(items):
        d = np.load(CACHE / f"{it['name']}.npz")
        r = analyse(model, d["audio"].astype(np.float32) / 32000, dev, window_frames=None if causal else DEFAULT_WINDOW)
        s = slice(EDGE, len(r["sonorant_p"]) - EDGE)
        out[i] = [np.nanmedian(r["vtl_log2cm"][s])] + [np.nanmedian(np.log2(r["formants"][s, k])) for k in range(3)]
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / f"{name}.npz", est=out)
    print(f"{name}: {len(items)} clips")


def _arm(which):
    """items, per model (n, 4) [log2 cm, log2 F1-F3], true (n, 3) Hz"""
    if which == "cascade":
        items = json.loads((D132_CACHE / "meta.json").read_text())["items"]
        est = {n: np.load(D132_RAW / f"{n}.npz")["est"][:, [0, 2, 3, 4]] for n in TEACHERS + STUDENTS}
        true = []
        for it in items:
            lab = np.load(D132_CACHE / f"{it['name']}.npz")["formants"]
            true.append(np.median(lab[EDGE:-EDGE, :3], 0))
    else:
        items = json.loads((CACHE / "meta.json").read_text())["items"]
        est = {n: np.load(RAW / f"{n}.npz")["est"] for n in TEACHERS + STUDENTS}
        true = [np.load(CACHE / f"{it['name']}.npz")["formants"] for it in items]
    return items, est, np.array(true)


def cmd_analyse() -> None:
    from vocal_analysis.synth import tube

    vowels = list(tube.STORY_VOWELS)
    rng = np.random.default_rng(134)
    lines = ["D-134: tract-length bias (median estimate - truth, cm) by vowel, averaged over lengths and f0; back rounded "
             "(o, u, U, O) minus the other six [95 % bootstrap over (length, f0, seed) cells]; F1-F3 median error (cents)", ""]
    arms = {}
    for which in ("cascade", "kl"):
        items, est, true = _arm(which)
        arms[which] = (items, est, true)
        V = np.array([it["vowel"] for it in items])
        L = np.array([it["length"] for it in items])
        cellkey = np.array([f"{it['length']}_{it['f0']}_{it['seed']}" for it in items])
        cells = np.unique(cellkey)
        lines.append(f"{'arm A, cascade (D-132 clips)' if which == 'cascade' else 'arm B, KL'}: {len(items)} clips")
        groups = [(n, [n]) for n in TEACHERS] + [("teacher (seed mean)", list(TEACHERS)), ("students (mean of four)", list(STUDENTS))]
        for gname, names in groups:
            bias = np.mean([2 ** est[n][:, 0] - L for n in names], 0)
            per_v = {v: np.mean(bias[V == v]) for v in vowels}
            back = np.isin(V, BACK)

            def contrast(rows):
                return np.mean(bias[rows & back]) - np.mean(bias[rows & ~back])

            boots = []
            for _ in range(B):
                pick = rng.choice(cells, len(cells))
                rows = np.zeros(len(items), bool)
                w = np.zeros(len(items))
                for c in pick:
                    w[cellkey == c] += 1
                rows = w > 0
                bb = (np.sum(w * bias * back) / np.sum(w * back)) - (np.sum(w * bias * ~back) / np.sum(w * ~back))
                boots.append(bb)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rng_v = max(per_v.values()) - min(per_v.values())
            lines.append(f"  {gname:24s} range {rng_v:.2f} cm; back - others {contrast(np.ones(len(items), bool)):+.2f} "
                         f"[{lo:+.2f}, {hi:+.2f}]; by vowel " + ", ".join(f"{v} {per_v[v]:+.2f}" for v in vowels))
        lines.append("  F1-F3 median error (cents), teacher seed mean / students, by vowel:")
        for v in vowels:
            m = V == v
            te = [np.median(np.mean([1200 * (est[n][m, k + 1] - np.log2(true[m, k])) for n in TEACHERS], 0)) for k in range(3)]
            st = [np.median(np.mean([1200 * (est[n][m, k + 1] - np.log2(true[m, k])) for n in STUDENTS], 0)) for k in range(3)]
            lines.append(f"    {v:3s} " + " ".join(f"F{k + 1} {te[k]:+5.0f} / {st[k]:+5.0f}" for k in range(3)))
        lines.append("")
    # KL minus cascade, paired by (length, vowel), f0 110-280 Hz, seed-averaged
    lines.append("KL - cascade, paired by (length, vowel), f0 110-280 Hz: length bias (cm) and F1-F3 (cents), teacher seed mean / students")
    (ic, ec, tc), (ik, ek, tk) = arms["cascade"], arms["kl"]

    def table(items, est, true, names):
        out = {}
        for i, it in enumerate(items):
            if not 110 <= it["f0"] <= 280:
                continue
            key = (it["length"], it["vowel"])
            row = [np.mean([2 ** est[n][i, 0] for n in names])] + [np.mean([1200 * est[n][i, k + 1] for n in names]) for k in range(3)]
            out.setdefault(key, []).append(row)
        return {k: np.mean(v, 0) for k, v in out.items()}

    for v in vowels:
        parts = []
        for names in (TEACHERS, STUDENTS):
            c, k = table(ic, ec, tc, names), table(ik, ek, tk, names)
            keys = [key for key in c if key[1] == v and key in k]
            dif = np.mean([k[key] - c[key] for key in keys], 0)
            parts.append(f"length {dif[0]:+.2f}, F1 {dif[1]:+4.0f}, F2 {dif[2]:+4.0f}, F3 {dif[3]:+4.0f}")
        lines.append(f"  {v:3s} " + " / ".join(parts))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d134.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "render":
        cmd_render(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == "score":
        cmd_score(sys.argv[2], sys.argv[3])
    elif cmd == "analyse":
        cmd_analyse()
