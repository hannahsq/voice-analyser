"""D-046: QCP-FB as a formant reference without an EGG. Stage 1 on D-043's renders.

Per render (D-043's interactive tube renderer, render D), the same five frame times (0.3-0.7 s), median over frames,
minus the closed-glottis poles (Hz), for:
  cp       the closed-phase reference as D-043 (10 kHz, order 12, ideal instants from the area);
  A@8      arm A: QCP-FB as published (duration quotient 0.7), 8 kHz, GCIs from SEDREAMS (f0_mean = the render's f0);
  B@R      arm B: QCP-FB, duration quotient 0.3, at rate R (8, 10, 16 kHz), SEDREAMS GCIs;
  BI@R     arm B with ideal GCIs;
  qB@8     forward-only QCP, duration quotient 0.3, 8 kHz, SEDREAMS GCIs.
Also per render: SEDREAMS' identification rate (true closures, shifted by the glottis-to-lips lag that `speech_lag`
measures, with an estimate within +-0.5 ms) and the share of frames each method returns. "Ideal GCIs" are those
lag-shifted closures.

    python scripts/analysis/d046_qcp.py stage1 <out dir> [--smoke | --high]
--smoke: development renders off the grid (women, f0 160 / 240, open quotient 0.50 / 0.65, gaps 0.03 / 0.08, every
third vowel); --high: the 400-800 Hz grid (women's size).
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

FRAME_TIMES = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
RATES = (8000.0, 10000.0, 16000.0)
METHODS = ("cp", "A@8", *[f"B@{int(r / 1000)}" for r in RATES], *[f"BI@{int(r / 1000)}" for r in RATES], "qB@8")
DQ_B = 0.3


def one(job):
    from scipy.signal import resample_poly

    from cp_ref_grid import FS, render_one
    from vocal_analysis.eval import closed_phase as cp
    from vocal_analysis.eval import qcp

    size, f0, oq, gap, vowel = job
    y, area, F, _ = render_one(size, f0, oq, gap, vowel, "D")
    n = len(y)
    out = {}
    x10, e10 = resample_poly(y, 10000, FS), resample_poly(-area, 10000, FS)
    # true closures: the area pulse's open phase spans cycle fraction [0, oq); it closes at oq. Shifted by the
    # acoustic lag from glottis to lips, measured as the reference measures it (`speech_lag`)
    lag = cp.speech_lag(x10, cp.egg_instants(e10, 10000.0)["gci"], 10000.0) / 10000.0
    k = np.arange(int(n * f0 / FS) + 2)
    gci_t = (k + oq) / f0 + lag
    gci_t = gci_t[gci_t * FS < n]
    r = cp.analyse(x10, e10, 10000.0, FRAME_TIMES, order=12)["cp"]
    out["cp"] = r[:, :3]
    sed_hit = np.nan
    for rate in RATES:
        tag = int(rate / 1000)
        a = qcp.analyse(y, FS, FRAME_TIMES, f0_mean=f0, fs=rate, dq=DQ_B)
        out[f"B@{tag}"] = a["F"][:, :3]
        out[f"BI@{tag}"] = qcp.analyse(y, FS, FRAME_TIMES, fs=rate, gci=np.round(gci_t * rate).astype(int), dq=DQ_B)["F"][:, :3]
        if tag == 8:
            est = a["gci"] / rate
            sel = (gci_t > 0.05) & (gci_t < n / FS - 0.05)
            sed_hit = float(np.mean([np.min(np.abs(est - g)) <= 0.0005 for g in gci_t[sel]])) if len(est) else 0.0
            out["A@8"] = qcp.analyse(y, FS, FRAME_TIMES, fs=rate, gci=a["gci"])["F"][:, :3]
            out["qB@8"] = qcp.analyse(y, FS, FRAME_TIMES, fs=rate, gci=a["gci"], forward_only=True, dq=DQ_B)["F"][:, :3]
    row = [*job, float(F[0]), float(F[1]), float(F[2]), sed_hit]
    for m in METHODS:
        e = out[m] - F[:3]
        ok = np.isfinite(e[:, 0])
        row += [float(ok.mean()), *(np.nanmedian(e, 0) if ok.any() else [np.nan] * 3)]
    return tuple(row)


def columns():
    cols = "size f0 oq gap vowel F1 F2 F3 sedreams_hit".split()
    for m in METHODS:
        cols += [f"{m}_cov", f"{m}_dF1", f"{m}_dF2", f"{m}_dF3"]
    return cols


def stage1(out: Path, mode: str):
    from cp_ref_grid import F0S, GAPS, HIGH_F0S, OQS, SIZES
    from vocal_analysis.synth import tube

    vowels = list(tube.STORY_VOWELS)
    if mode == "smoke":
        jobs = list(itertools.product(["women"], (160.0, 240.0), (0.50, 0.65), (0.03, 0.08), vowels[::3]))
    elif mode == "high":
        jobs = list(itertools.product(["women"], HIGH_F0S, OQS, GAPS, vowels))
    else:
        jobs = list(itertools.product(SIZES, F0S, OQS, GAPS, vowels))
    from multiprocessing import Pool

    t0 = time.time()
    with Pool(12) as pool:
        rows = pool.map(one, jobs, chunksize=2)
    print(f"{len(rows)} renders in {time.time() - t0:.0f} s", flush=True)
    name = {"smoke": "d046_stage1_smoke", "high": "d046_stage1_high"}.get(mode, "d046_stage1")
    (out / f"{name}.tsv").write_text("\t".join(columns()) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    summary(rows, out / f"{name}.txt")
    if mode == "grid":
        gate(rows, out / "d046_gate.txt")


def summary(rows, path):
    cols = columns()
    c = {k: i for i, k in enumerate(cols)}
    lines = [f"D-046 stage 1 ({len(rows)} renders): median over renders of the per-render median error (Hz); coverage = mean share of frames",
             f"SEDREAMS identification rate (+-0.5 ms): median {np.nanmedian([r[c['sedreams_hit']] for r in rows]):.3f}, "
             f"min {np.nanmin([r[c['sedreams_hit']] for r in rows]):.3f}", ""]
    for m in METHODS:
        e = np.array([[r[c[f"{m}_dF{k}"]] for k in (1, 2, 3)] for r in rows], float)
        cov = np.mean([r[c[f"{m}_cov"]] for r in rows])
        med = np.nanmedian(e, 0)
        mad = np.nanmedian(np.abs(e), 0)
        lines.append(f"{m:7s} cov {cov:.2f}  F1 {med[0]:+6.1f} (|.| {mad[0]:5.1f})  F2 {med[1]:+6.1f} (|.| {mad[1]:5.1f})  F3 {med[2]:+6.1f} (|.| {mad[2]:5.1f})")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def gate(rows, path):
    """D-046's registered stage-1 criteria for arm B at each rate; arm A and the variants reported the same way."""
    cols = columns()
    c = {k: i for i, k in enumerate(cols)}
    rng = np.random.default_rng(0)
    women = [r for r in rows if r[0] == "women" and r[1] in (180.0, 220.0, 270.0)]
    men = [r for r in rows if r[0] == "men" and r[1] in (100.0, 140.0)]

    def col(sel, key):
        return np.array([r[c[key]] for r in sel], float)

    def boot(sel, f):
        """f over sel and its vowel-bootstrap 95 % interval (vowels resampled with all their renders)."""
        vow = sorted({r[4] for r in sel})
        by = {v: [r for r in sel if r[4] == v] for v in vow}
        bs = [f([r for v in rng.choice(vow, len(vow)) for r in by[v]]) for _ in range(2000)]
        return f(sel), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    def med(key):
        return lambda sel: np.nanmedian(col(sel, key))

    def mad(key):
        return lambda sel: np.nanmedian(np.abs(col(sel, key)))

    def half(sel, f):
        _, lo, hi = boot(sel, f)
        return (hi - lo) / 2

    lines = ["D-046 stage 1 gate (registered criteria; arm B candidates, arm A and variants reported)", ""]
    hit = col(rows, "sedreams_hit")
    by_f0 = " ".join(f"f0 {f:.0f}: {np.nanmean(col([r for r in rows if r[1] == f], 'sedreams_hit')):.3f}"
                     for f in sorted({r[1] for r in rows}))
    lines.append(f"SEDREAMS identification rate: mean {np.nanmean(hit):.3f}; {by_f0}")
    sed_ok = np.nanmean(hit) >= 0.95
    lines.append(f"GCI gate (mean >= 0.95): {'passes' if sed_ok else 'FAILS: stop'}")
    leaky = {"gap 0.10": [r for r in women if r[3] == 0.10], "OQ 0.70": [r for r in women if r[2] == 0.70]}
    verdict = {}
    for m in ("A@8", "B@8", "B@10", "B@16", "BI@8", "qB@8"):
        lines.append(f"\n[{m}]")
        ok = {}
        bw, bm = med(f"{m}_dF1")(women), med(f"{m}_dF1")(men)
        ok["bias"] = abs(bw) <= 4 and abs(bm) <= 4
        lines.append(f"  1 bias: women's region F1 {bw:+.1f}, men's {bm:+.1f} (within +-4): {ok['bias']}")
        r2 = []
        for lab, sel in leaky.items():
            q, rf, hw = med(f"{m}_dF1")(sel), med("cp_dF1")(sel), half(sel, med("cp_dF1"))
            r2.append(abs(q) <= abs(rf) + hw)
            lines.append(f"  2 {lab}: F1 {q:+.1f} vs reference {rf:+.1f} + half-width {hw:.1f}: {r2[-1]}")
        ok["leaky"] = all(r2)
        r3 = []
        for k in (2, 3):
            q, rf, hw = med(f"{m}_dF{k}")(women), med(f"cp_dF{k}")(women), half(women, med(f"cp_dF{k}"))
            r3.append(abs(q) <= abs(rf) + hw)
            lines.append(f"  3 F{k}: {q:+.1f} vs reference {rf:+.1f} + half-width {hw:.1f}: {r3[-1]}")
        ok["F2F3"] = all(r3)

        def pdiff(sel, m=m):
            return np.nanmedian(np.abs(col(sel, f"{m}_dF1")) - np.abs(col(sel, "cp_dF1")))

        p, _, hi = boot(women, pdiff)
        hw = half(women, mad("cp_dF1"))
        ok["precision"] = hi <= hw
        lines.append(f"  4 precision: paired |{m}| - |reference| F1 median {p:+.1f}, upper bound {hi:+.1f} "
                     f"<= reference half-width {hw:.1f}: {ok['precision']}")
        cov = np.nanmean(col(rows, f"{m}_cov"))
        ok["coverage"] = cov >= 0.9
        lines.append(f"  5 coverage {cov:.2f} (>= 0.90): {ok['coverage']}")
        verdict[m] = bool(sed_ok and all(ok.values()))
        lines.append(f"  women's-region median |F1 error| {mad(f'{m}_dF1')(women):.1f}; PASSES: {verdict[m]}")
    passing = [m for m in ("B@8", "B@10", "B@16") if verdict[m]]
    choice = None
    if verdict["B@8"]:
        choice = "B@8"
        for m in ("B@10", "B@16"):
            if verdict[m] and mad(f"{m}_dF1")(women) <= mad(f"{choice}_dF1")(women):
                choice = m
    lines.append(f"\nArm B passing rates: {passing or 'none'}; chosen: {choice or 'none (the 8 kHz candidate did not pass)'}")
    if choice:
        wsz = [r for r in rows if r[0] == "women"]
        valid = [f for f in sorted({r[1] for r in wsz})
                 if np.nanmedian(np.abs(col([r for r in wsz if r[1] == f], f"{choice}_dF1"))) <= 8]
        lines.append(f"Valid f0 range (women's size, median |F1 error| <= 8 Hz): {valid}")
        for lab, sel in (("women", women), ("men", men)):
            d, lo, hi = boot(sel, lambda s_: np.nanmedian(col(s_, f"{choice}_dF1") - col(s_, "cp_dF1")))
            lines.append(f"Stage-2 prediction, {lab}: ({choice} - reference) F1 median {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "stage1":
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if "--smoke" in sys.argv else "high" if "--high" in sys.argv else "grid"
    stage1(out, mode)


if __name__ == "__main__":
    main()
