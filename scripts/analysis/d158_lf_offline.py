"""D-158: LF sources off Fant's Rd line (Ra, Rg or Rk moved one at a time from anchors Rd 0.6 / 1.0 / 2.0), crossed with
B1 and B2 on D-157's grid; 5b's readings. The free-parameter source replaces the cascade's Rd lookup inside this script
only (synth/ is unchanged). CPU, inference only.

    python scripts/analysis/d158_lf_offline.py run [MODEL]   -> cache/d158/<model>.json (one process per model)
    python scripts/analysis/d158_lf_offline.py report        -> reports/analysis/d158/d158.{txt,json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d158"
CACHE = ROOT / "cache/d158"
FS = 16000
FIVEB = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
ANCHORS = (0.6, 1.0, 2.0)
STEPS = {"Ra": (0.5, 2.0, 3.0), "Rg": (0.8, 1.25), "Rk": (0.7, 1.4)}
B1S = (150.0, 360.0)
B2S = (90.0, 250.0)
SEEDS = (20, 21, 22)


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _line(rd):
    """Fant's (1995) regression, as `glottal.lf_params`."""
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    rg = rk * (0.5 + 1.2 * rk) / (0.44 * rd - 4.0 * ra * (0.5 + 1.2 * rk))
    return ra, rk, rg


def _valid(ra, rk, rg):
    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    return 0 < tp < te < 1 and 0 < ra < 1.0 - te


def sources():
    """[(anchor, parameter, factor, (ra, rk, rg))], the anchor itself as parameter '-' factor 1."""
    out = []
    for rd in ANCHORS:
        ra, rk, rg = _line(rd)
        out.append((rd, "-", 1.0, (ra, rk, rg)))
        for p, fs in STEPS.items():
            for f in fs:
                v = (ra * (f if p == "Ra" else 1), rk * (f if p == "Rk" else 1), rg * (f if p == "Rg" else 1))
                if _valid(*v):
                    out.append((rd, p, f, v))
    return out


def _waveform(ra, rk, rg, n):
    """One period of the LF flow derivative for free (Ra, Rk, Rg): `glottal.lf_params` / `lf_waveform` after the
    regression, unchanged."""
    from scipy.optimize import brentq

    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    ta = ra
    tb = 1.0 - te
    eps = brentq(lambda e: e * ta - 1.0 + np.exp(-e * tb), 1e-6 / ta, 1e3 / ta)
    ret_area = -(1.0 / (eps * ta)) * ((1.0 - np.exp(-eps * tb)) / eps - tb * np.exp(-eps * tb))
    wg = np.pi / tp

    def net_area(a):
        e0 = -1.0 / (np.exp(a * te) * np.sin(wg * te))
        return e0 * (np.exp(a * te) * (a * np.sin(wg * te) - wg * np.cos(wg * te)) + wg) / (a * a + wg * wg) + ret_area

    alpha = brentq(net_area, -50.0, 200.0)
    e0 = -1.0 / (np.exp(alpha * te) * np.sin(wg * te))
    t = np.arange(n) / n
    out = np.empty(n)
    op = t <= te
    out[op] = e0 * np.exp(alpha * t[op]) * np.sin(np.pi / tp * t[op])
    tr = t[~op]
    out[~op] = -(1.0 / (eps * ta)) * (np.exp(-eps * (tr - te)) - np.exp(-eps * (1.0 - te)))
    return out


def _coefs(params):
    from vocal_analysis.synth.glottal import N_HARMONICS

    n = 8192
    return np.fft.rfft(_waveform(*params, n))[1:N_HARMONICS + 1] / n


def _audio(Fs, f0, b1, b2, seed, coefs):
    """D-157's clip (D-142's seed formula) with the cascade's source coefficients replaced by `coefs`."""
    from vocal_analysis.synth import sampler, tract, voice

    orig = voice.source_coefficients
    voice.source_coefficients = lambda rd, k_max=len(coefs): np.broadcast_to(coefs[:k_max], np.shape(rd) + (k_max,)).copy()
    try:
        C = int(3.0 * FS / voice.CTRL) + 1
        tc = np.arange(C) * voice.CTRL / FS
        pf = np.full((C, tract.N_POLES), np.nan)
        pb = pf.copy()
        pf[:, :4], pb[:, :4] = Fs, (b1, b2, 120, 150)
        rng = np.random.default_rng(int(1420 + 1000 * seed + f0 + b1 + Fs[0]))
        tr = voice.Trajectories(f0=f0 * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12), rd=np.full(C, 1.0),
                                voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1), whisper_amp=np.zeros(C),
                                fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
        y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
    finally:
        voice.source_coefficients = orig
    return 0.5 * y / np.abs(y).max()


def _keys():
    d142 = _mod("d142_flags")
    return [(src, s, f0, b1, b2, sd, Fs) for src in sources() for s, Fs in d142.SETS.items() for f0 in d142.F0S
            for b1 in B1S for b2 in B2S for sd in SEEDS]


def _key(k):
    src = k[0]
    return f"{src[0]}|{src[1]}{src[2]:g}|{k[1]}|f{k[2]:.0f}|b{k[3]:.0f}|B{k[4]:.0f}|s{k[5]}"


def _score(r, y, Fs, b1, b2):
    n = len(r["voiced_p"])
    i = np.arange(int(0.2 * n), int(0.8 * n))
    i = i[(r["voiced_p"][i] > 0.5) & np.isfinite(r["f0"][i]) & (i * 160 >= 512) & (i * 160 + 512 <= len(y))]
    i = i[np.all(np.isfinite(r["formants"][i, :4]), 1) & np.all(np.isfinite(r["bandwidths"][i, :4]), 1)]
    if not len(i):
        return None
    F, B = r["formants"][i], r["bandwidths"][i]
    return {"fail": float(np.mean(np.abs(np.log(F[:, 1] / Fs[2])) < np.abs(np.log(F[:, 1] / Fs[1])))),
            "e1": float(np.median(np.abs(np.log2(F[:, 0] / Fs[0])) * 1200)),
            "e2": float(np.median(np.abs(np.log2(F[:, 1] / Fs[1])) * 1200)),
            "rb1": float(np.median(np.log2(B[:, 0] / b1))), "rb2": float(np.median(np.log2(B[:, 1] / b2)))}


def cmd_run(models: list[str]) -> None:
    import torch

    from vocal_analysis.models.infer import analyse, load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    coefs = {s[:3]: _coefs(s[3]) for s in sources()}
    for m in models:
        out = CACHE / f"{m}.json"
        if out.exists():
            continue
        md = load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]
        res = {}
        for k in _keys():
            y = _audio(k[6], k[2], k[3], k[4], k[5], coefs[k[0][:3]]).astype(np.float32)
            s = _score(analyse(md, y, dev), y, k[6], k[3], k[4])
            if s is not None:
                res[_key(k)] = s
        out.write_text(json.dumps(res), encoding="utf-8")
        print(m, len(res), flush=True)


def cmd_report() -> None:
    sc = {m: json.loads((CACHE / f"{m}.json").read_text(encoding="utf-8")) for m in FIVEB}
    ref = {m: json.loads((ROOT / f"cache/d157/{m}.json").read_text(encoding="utf-8")) for m in FIVEB}
    ks = _keys()
    chk = []
    for m in FIVEB:
        for k in ks:
            src = k[0]
            if src[1] == "-" and src[0] in (0.6, 1.0) and _key(k) in sc[m]:
                rk = f"{k[1]}|f{k[2]:.0f}|b{k[3]:.0f}|B{k[4]:.0f}|r{src[0]:.1f}|s{k[5]}"
                if rk in ref[m]:
                    chk.append(abs(sc[m][_key(k)]["fail"] - ref[m][rk]))
    mism = sum(d > 1e-9 for d in chk)

    def mean(m, src, sel, q="fail"):
        v = [sc[m][_key(k)][q] for k in ks if k[0][:3] == src and sel(k) and _key(k) in sc[m]]
        return float(np.mean(v)) if v else float("nan")

    lines = ["D-158: LF off Fant's Rd line (one parameter at a time from Rd 0.6 / 1.0 / 2.0) x B1 / B2 on D-157's grid; 5b",
             f"  render check (on-line Rd 0.6 and 1.0 against D-157, per clip): {mism} of {len(chk)} differ", "",
             "  low f0 (110, 140); per source: failure at B2 90 (both B1); B2 effect (250 - 90); B1 effect (360 - 150); "
             "F1 / F2 error (c); model B1 / B2 vs truth (log2, narrow cell B1 150 B2 90)"]
    srcs = [s[:3] for s in sources()]
    eff = {}
    for who in FIVEB + ("mean",):
        ms = FIVEB if who == "mean" else (who,)
        eff[who] = {}
        for src in srcs:
            for lo in (True, False):
                f = lambda sel, q="fail": float(np.mean([mean(m, src, sel, q) for m in ms]))  # noqa: E731
                L = (lambda k: k[2] <= 140) if lo else (lambda k: k[2] > 140)
                base = f(lambda k: L(k) and k[4] == 90)
                e2 = f(lambda k: L(k) and k[4] == 250) - base
                e1 = f(lambda k: L(k) and k[3] == 360) - f(lambda k: L(k) and k[3] == 150)
                eff[who][(src, lo)] = {"base": base, "b2": e2, "b1": e1,
                                       "e1": f(lambda k: L(k), "e1"), "e2": f(lambda k: L(k), "e2"),
                                       "rb1": f(lambda k: L(k) and k[3] == 150 and k[4] == 90, "rb1"),
                                       "rb2": f(lambda k: L(k) and k[3] == 150 and k[4] == 90, "rb2")}
    for lo in (True, False):
        if not lo:
            lines += ["", "  f0 180-220, the same (reported)"]
        for src in srcs:
            e = eff["mean"][(src, lo)]
            lines.append(f"    Rd {src[0]} {src[1]:>2s} x{src[2]:<4g} fail {e['base']:.2f}  B2 eff {e['b2']:+.2f}  B1 eff {e['b1']:+.2f}  "
                         f"F1 {e['e1']:4.0f} c  F2 {e['e2']:4.0f} c  B1 {e['rb1']:+.2f}  B2 {e['rb2']:+.2f}")
    # ranges per anchor and parameter (low f0), the anchor included
    lines += ["", "  ranges over each parameter's steps (anchor included), low f0: source effect / B2 effect / B1 effect"]
    rng, worst = {}, {"src": 0.0, "eff": 0.0}
    for rd in ANCHORS:
        for p in STEPS:
            grp = [s for s in srcs if s[0] == rd and s[1] in ("-", p)]
            row = {}
            for who in FIVEB + ("mean",):
                g = [eff[who][(s, True)] for s in grp]
                row[who] = {q: max(x[q] for x in g) - min(x[q] for x in g) for q in ("base", "b2", "b1")}
            rng[f"{rd}|{p}"] = row
            mn = row["mean"]
            lines.append(f"    Rd {rd} {p}: {mn['base']:.2f} / {mn['b2']:.2f} / {mn['b1']:.2f}   (seeds: "
                         + ", ".join(f"{row[m]['b2']:.2f}, {row[m]['b1']:.2f}" for m in FIVEB) + ")")
            worst["src"] = max(worst["src"], mn["base"])
            worst["eff"] = max(worst["eff"], mn["b2"], mn["b1"])
    joint = [(c, q) for c, row in rng.items() for q in ("b2", "b1")
             if row["mean"][q] >= 0.15 and all(row[m][q] >= 0.10 for m in FIVEB)]
    if joint:
        out = "JOINT: leaving the line changes how a wide bandwidth reads (" + ", ".join(f"Rd {c} {q.upper()} effect" for c, q in joint) + ")"
    elif worst["eff"] < 0.05 and worst["src"] < 0.10:
        out = "NONE: off-line sources neither tip the failure nor change the wide-bandwidth effects"
    else:
        out = "PARTLY: reported, Hannah decides"
    res = {"render_mismatches": mism, "effects": {f"{w}|{s[0]}|{s[1]}{s[2]:g}|{'low' if lo else 'high'}": v
                                                   for w, d in eff.items() for (s, lo), v in d.items()},
           "ranges": rng, "reading": out}
    lines += ["", f"Reading (registered; largest effect range {worst['eff']:.2f}, largest source effect {worst['src']:.2f}): {out}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d158.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d158.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2].split(",") if len(sys.argv) > 2 else list(FIVEB))
    else:
        cmd_report()
