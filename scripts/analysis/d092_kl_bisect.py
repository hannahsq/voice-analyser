"""D-092 (with amendment 1): does v3.1's F1 level on KL renders need the coupled flow?

D-060's grid (ten Story vowels, 8 f0 per sex, 1-semitone 5 Hz vibrato, 1 s) at gaps 0 and 0.3. Per file:
  C   coupled render (D-060's);  I   C's own flow `ug` through a rigid glottis (`flow=`): the identity check;
  R   the uncoupled render's flow through a rigid glottis;  RT  R x (f / 1 kHz)^-0.76 above 50 Hz, zero phase;
  R6  R at -6 dB (level control). All peak-normalised to 0.5 as D-060 (R6 to 0.25).
Error = D-060's file_error (median F1 over 0.3-0.8 of the file, voiced frames, minus the closed-glottis label), seed
mean of d040_5b_s{1,2}_b20k. L_C = median(C - dE) on D-077's fitted files; L_R = median R error (fitted files decide;
all reported). Bootstrap over (vowel, f0) pairs, 2000.

    python scripts/analysis/d092_kl_bisect.py [--serial]      (renders on CPU, then scores on VA_DEVICE)
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("reports/analysis/d092")
CACHE = Path("cache/d092")
MODELS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
GAPS = (0.0, 0.3)
VARIANTS = ("C", "I", "R", "RT", "R6")
SEED = 5092


def tag(sex, v, f0, gap):
    return f"{sex}_{''.join(c + '^' if c.isupper() else c for c in v)}_{f0}_{gap:.1f}"


def tilt_filter(x, fs=16000):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    g = np.where(f > 50, (np.maximum(f, 50) / 1000.0) ** -0.76, (50 / 1000.0) ** -0.76)
    return np.fft.irfft(X * g, len(x))


def norm(x, peak=0.5):
    return (peak * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


def one(job):
    sex, v, f0, gap = job
    path = CACHE / f"{tag(*job)}.npz"
    if path.exists():
        return job
    import d060_kl_gap as D
    from vocal_analysis.synth import branches as br
    from vocal_analysis.synth import kl, tube

    L, peak = D.SIZE[sex]
    rel = L / tube.STORY_LENGTH_CM
    s = tube.build_shape(*tube.STORY_VOWELS[v], L, area_scale=rel**2)
    (F, _, _), = kl.frame_labels([s])
    n = kl.FS
    frames = n // kl.HOP + 1
    t = np.arange(frames) * kl.HOP / kl.FS
    f0_t = f0 * 2.0 ** (D.VIB_ST * np.sin(2 * np.pi * D.VIB_HZ * t) / 12)
    ag = kl.glottal_area(f0_t, n, kl.FS, peak, oq=0.6, sq=1.3, gap=gap)
    kw = dict(subglottal=br.subglottal(L, kl.section_length()))
    args = ([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel)
    c = kl.render(*args, **kw)
    u = kl.render(*args, coupled=False, **kw)
    i = kl.render(*args, flow=c["ug"].astype(float), **kw)
    r = kl.render(*args, flow=u["ug"].astype(float), **kw)
    ra = r["audio"].astype(float)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, C=norm(c["audio"]), I=norm(i["audio"]), R=norm(ra), RT=norm(tilt_filter(ra)), R6=norm(ra, 0.25),
             f1=F[0])
    return job


def tilt_db_oct(x, fs=16000, nwin=1024):
    """D-045's tilt (slope of the dB spectrum on log2 f, 100 Hz-4 kHz), median over windows in 0.3-0.8 of the file"""
    w = np.hanning(nwin)
    f = np.fft.rfftfreq(nwin, 1 / fs)
    band = (f >= 100) & (f <= 4000)
    out = []
    for a in range(int(0.3 * len(x)), int(0.8 * len(x)) - nwin, nwin // 2):
        db = 20 * np.log10(np.abs(np.fft.rfft(x[a:a + nwin] * w)) + 1e-9)
        out.append(np.polyfit(np.log2(f[band]), db[band], 1)[0])
    return float(np.median(out))


def score(jobs, dev):
    """per job and variant: F1 error per model, and C's aVTL (median log2 cm over the same frames) per model"""
    path = CACHE / "scores.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["err"].item(), z["vtl"].item()
    from vocal_analysis.models.infer import analyse, load_teacher

    err, vtl = {}, {}
    for m in MODELS:
        model = load_teacher(f"runs/{m}/ckpt_last.pt", dev)[0]
        for j in jobs:
            z = np.load(CACHE / f"{tag(*j)}.npz")
            for var in VARIANTS:
                r = analyse(model, z[var], dev)
                n = len(r["formants"])
                mid = slice(int(0.3 * n), int(0.8 * n))
                vo = r["voiced_p"][mid] > 0.5
                err[m, j, var] = float(np.nanmedian(r["formants"][mid, 0][vo]) - float(z["f1"]))
                if var == "C":
                    vtl[m, j] = float(np.nanmedian(r["vtl_log2cm"][mid][vo]))
        del model
    np.savez(path, err=np.array(err, dtype=object), vtl=np.array(vtl, dtype=object))
    return err, vtl


def boot(x, rng, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


if __name__ == "__main__":
    import d060_kl_gap as D

    jobs = [(sx, v, f0, g) for sx in D.SIZE for v in D.MAIN["vowels"] for f0 in D.MAIN["f0"][sx] for g in GAPS]
    todo = [j for j in jobs if not (CACHE / f"{tag(*j)}.npz").exists()]
    if todo:
        if "--serial" in sys.argv:
            for j in todo:
                one(j)
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(6) as ex:
                list(ex.map(one, todo, chunksize=4))
        print(f"rendered {len(todo)} files", file=sys.stderr, flush=True)
    import torch

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    err, vtl = score(jobs, dev)
    E = lambda j, var: float(np.mean([err[m, j, var] for m in MODELS]))  # noqa: E731
    dE = {}
    for j in jobs:
        p = Path("cache/d077") / f"{tag(*j)}.npy"
        dE[j] = float(np.load(p)[1] - np.load(p)[4]) if p.exists() else np.nan
    rng = np.random.default_rng(SEED)
    L = [f"D-092: v3.1's F1 level on KL renders, coupled (C) against the uncoupled flow through a rigid glottis (R); "
         f"seed mean of {' + '.join(MODELS)}; Hz; median over files [95 % bootstrap over (vowel, f0)]", ""]

    # identity check first
    idd = np.array([E(j, "I") - E(j, "C") for j in jobs])
    L.append(f"identity check (I - C, v3.1's reading on C's own flow through a rigid glottis minus on C): median "
             f"{np.median(idd):+.2f} Hz, median |.| {np.median(np.abs(idd)):.2f}, largest |.| {np.max(np.abs(idd)):.1f} "
             f"-> {'passes' if abs(np.median(idd)) <= 1 else 'FAILS (nothing else is read)'}")
    ident_ok = abs(np.median(idd)) <= 1
    L.append("")
    res = {}
    for g in GAPS:
        L.append(f"gap {g:.1f}:")
        for sx in D.SIZE:
            fj = [j for j in jobs if j[0] == sx and j[3] == g and np.isfinite(dE[j])]
            aj = [j for j in jobs if j[0] == sx and j[3] == g]
            lc = boot([E(j, "C") - dE[j] for j in fj], rng)
            lr = boot([E(j, "R") for j in fj], rng)
            lra = boot([E(j, "R") for j in aj], rng)
            c_raw = boot([E(j, "C") for j in aj], rng)
            rt = boot([E(j, "R") - E(j, "RT") for j in aj], rng)
            r6 = boot([E(j, "R") - E(j, "R6") for j in aj], rng)
            res[g, sx] = dict(lc=lc, lr=lr, rt=rt)
            f = lambda b: f"{b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}]"  # noqa: E731
            L.append(f"  {sx:6s} L_C {f(lc)} ({len(fj)} fitted files); L_R {f(lr)} (fitted), {f(lra)} (all {len(aj)}); "
                     f"C raw {f(c_raw)}; R - RT {f(rt)}; R - R6 {f(r6)}")
        L.append("")

    # rule (women, gap 0)
    w, m = res[0.0, "Female"], res[0.0, "Male"]

    def cat(b):
        return "linear" if b[0] >= 11 else "coupled" if b[0] <= 6 else "mixed"

    verdict = {"linear": "THE LEVEL IS IN THE LINEAR PATH", "coupled": "THE LEVEL COMES WITH THE COUPLED FLOW'S RIPPLE AND SKEW",
               "mixed": "MIXED"}[cat(w["lr"])]
    opp = {"linear": "coupled", "coupled": "linear"}.get(cat(w["lr"]))
    disagree = opp is not None and cat(m["lr"]) == opp and (m["lr"][2] < w["lr"][1] or m["lr"][1] > w["lr"][2])
    tilt_share = w["rt"][0] >= 5 and w["rt"][1] > 0
    L.append(f"rule (women, gap 0): L_R {w['lr'][0]:+.1f} -> {cat(w['lr'])}; men {m['lr'][0]:+.1f} -> {cat(m['lr'])}"
             + ("; the sexes disagree (no reading)" if disagree else ""))
    L.append(f"tilt: women R - RT {w['rt'][0]:+.1f} [{w['rt'][1]:+.1f}, {w['rt'][2]:+.1f}] -> "
             f"{'tilt carries a share' if tilt_share else 'no tilt share'}")
    L.append("")

    # reported: by vowel, against log2(F1 / f0), tilt per variant, aVTL by gap
    L.append("reported: by vowel, gap 0, median over f0 (Hz): C - dE / R / R - RT")
    for sx in D.SIZE:
        L.append(f"  {sx}: " + "  ".join(
            f"{v} {np.nanmedian([E(j, 'C') - dE[j] for j in jobs if j[:2] == (sx, v) and j[3] == 0.0]):+.0f}/"
            f"{np.median([E(j, 'R') for j in jobs if j[:2] == (sx, v) and j[3] == 0.0]):+.0f}/"
            f"{np.median([E(j, 'R') - E(j, 'RT') for j in jobs if j[:2] == (sx, v) and j[3] == 0.0]):+.0f}"
            for v in D.MAIN["vowels"]))
    L.append("reported: R's error (gap 0) by log2(F1 / f0) band, both sexes pooled (median Hz, files)")
    z0 = [j for j in jobs if j[3] == 0.0]
    ratio = np.array([np.log2(float(np.load(CACHE / f"{tag(*j)}.npz")["f1"]) / j[2]) for j in z0])
    for lo_, hi_ in ((0, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 9)):
        k = [j for j, r_ in zip(z0, ratio) if lo_ <= r_ < hi_]
        if k:
            L.append(f"  F1/f0 2^[{lo_}, {hi_}): R {np.median([E(j, 'R') for j in k]):+.1f}, C - dE "
                     f"{np.nanmedian([E(j, 'C') - dE[j] for j in k]):+.1f} ({len(k)} files; women {sum(j[0] == 'Female' for j in k)})")
    L.append("reported: spectral tilt (dB/oct, D-045's measure), median over files, gap 0")
    for sx in D.SIZE:
        k = [j for j in jobs if j[0] == sx and j[3] == 0.0]
        L.append(f"  {sx}: " + ", ".join(f"{var} {np.median([tilt_db_oct(np.load(CACHE / f'{tag(*j)}.npz')[var]) for j in k]):+.1f}"
                                         for var in ("C", "R", "RT")))
    L.append("reported: aVTL on C, paired gap 0.3 - 0 per (vowel, f0), median [95 %]; ideal 0")
    for sx in D.SIZE:
        d_ = [np.mean([vtl[mm, (sx, v, f0, 0.3)] - vtl[mm, (sx, v, f0, 0.0)] for mm in MODELS])
              for v in D.MAIN["vowels"] for f0 in D.MAIN["f0"][sx]]
        b = boot(100 * (2 ** np.array(d_) - 1), rng)
        L.append(f"  {sx}: {b[0]:+.2f} % [{b[1]:+.2f}, {b[2]:+.2f}] ({b[0] / 100 * D.SIZE[sx][0]:+.2f} cm at {D.SIZE[sx][0]} cm)")
    L.append("")
    L.append(f"DECISION: {verdict if ident_ok and not disagree else ('IDENTITY CHECK FAILED' if not ident_ok else 'SEXES DISAGREE')}"
             f" (women L_R {w['lr'][0]:+.1f}, L_C {w['lc'][0]:+.1f}); tilt {'carries a share' if tilt_share else 'no share'}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d092.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
