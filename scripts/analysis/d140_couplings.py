"""D-140: what widens F1 on a real voice? Harmonic residuals against an all-pole envelope (a zero shows as a local dip:
subglottal at a fixed frequency in every vowel, nasal in open vowels only) and H1*-H2* against F1 bandwidth (glottal).
Validation on synthetic clips is committed; the recording's per-take numbers go to OUT (outside the repo).

    python scripts/analysis/d140_couplings.py validate               -> reports/analysis/d140/d140_validate.txt
    python scripts/analysis/d140_couplings.py run REC OUT FILES      -> OUT/d140.json, OUT/d140.txt (needs D-138's OUT/levels.json)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d140"
FS = 16000
BINS = np.arange(200, 3001, 25.0)
ORDER = 14
CLOSE, OPEN = ("/i/", "/ʉ/", "/e/"), ("/ɔ/", "/ɐ/", "/ə/", "/æ/")
SG1, SG2 = (450, 750), (1200, 1600)
B = 2000


def _d138():
    spec = importlib.util.spec_from_file_location("d138", ROOT / "scripts/analysis/d138_real_balance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lpc(z):
    from scipy.linalg import solve_toeplitz

    ac = np.zeros(ORDER + 1)
    for i in range(0, len(z) - 512, 160):
        w = z[i:i + 512] * np.hanning(512)
        ac += np.array([w[:512 - k] @ w[k:] for k in range(ORDER + 1)])
    return np.r_[1.0, -solve_toeplitz(ac[:ORDER], ac[1:ORDER + 1])]


def _env_db(a, f):
    w = np.exp(-2j * np.pi * np.outer(f, np.arange(len(a))) / FS)
    return -20 * np.log10(np.abs(w @ a))


def _b1(a, F1):
    r = np.roots(a)
    r = r[np.imag(r) > 0]
    f, b = np.angle(r) * FS / (2 * np.pi), -np.log(np.abs(r)) * FS / np.pi
    k = np.argmin(np.abs(np.log(f / F1)))
    return float(b[k])


def _harmonics(x, i, f0, kmax_hz):
    """dB amplitudes and frequencies of harmonics 1..K in the 64 ms frame centred on frame i."""
    fr = x[i * 160 - 512:i * 160 + 512] * np.hanning(1024)
    s = 20 * np.log10(np.abs(np.fft.rfft(fr, 4096)) + 1e-9)
    f = np.fft.rfftfreq(4096, 1 / FS)
    ks = np.arange(1, int(kmax_hz / f0) + 1)
    h = np.array([s[(f >= k * f0 - f0 / 4) & (f <= k * f0 + f0 / 4)].max() for k in ks])
    return ks * f0, h


def probe(y, f0, voiced, F):
    """Per take: binned median harmonic residual (dB), H1*-H2* (dB) and the LPC F1 bandwidth (Hz)."""
    from scipy.signal import lfilter

    from vocal_analysis.synth import filters

    n = len(y)
    z = lfilter([1, -0.97], [1], y)
    a = _lpc(z[int(0.2 * n):int(0.8 * n)])
    nf = len(f0)
    fs_, rs, h12 = [], [], []
    for i in range(int(0.2 * nf), int(0.8 * nf)):
        if not voiced[i] or i * 160 < 512 or i * 160 + 512 > n or not np.isfinite(f0[i]):
            continue
        hf, h = _harmonics(z, i, f0[i], 3000)
        r = h - _env_db(a, hf)
        r -= np.median(r)
        fs_.append(hf)
        rs.append(r)
        _, g = _harmonics(y, i, f0[i], 2.5 * f0[i])
        T = 20 * np.log10(np.abs(filters.response(np.array([f0[i], 2 * f0[i]]), np.array([F[0], F[1]])[None, :],
                                                  np.array([80.0, 100.0])[None, :], FS)))
        h12.append((g[0] - T[0]) - (g[1] - T[1]))
    fs_, rs = np.concatenate(fs_), np.concatenate(rs)
    idx = np.digitize(fs_, BINS - 12.5) - 1
    resid = np.array([np.median(rs[idx == j]) if np.sum(idx == j) >= 5 else np.nan for j in range(len(BINS))])
    return resid, float(np.median(h12)), _b1(a, F[0])


def _group(resids, rng):
    R = np.array(resids)
    m = np.nanmean(R, 0)
    bs = np.array([np.nanmean(R[rng.integers(0, len(R), len(R))], 0) for _ in range(B)])
    return m, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def _deepest(m, lo, hi, rng_=None):
    w = (BINS >= rng_[0]) & (BINS <= rng_[1])
    j = np.flatnonzero(w)[np.nanargmin(m[w])]
    return j


def cmd_validate() -> None:
    from vocal_analysis.synth import sampler, tract, voice

    lines = ["D-140 validation: cascade at f0 110 Hz (D-131 jitter, 0.3 st f0 wander), three seeds per case; with or without a "
             "pole-zero pair (zero 580 Hz / 60, pole 640 Hz / 80); deepest harmonic-residual bin in 450-750 Hz (mean over seeds)", ""]
    ok = True
    for vname, Fs in (("i-like", (320, 2200, 2800, 3500)), ("O-like", (450, 800, 2800, 3500))):
        for b1 in (60, 250):
            for pair in (False, True):
                res = []
                for seed in range(3):
                    C = int(3.0 * FS / voice.CTRL) + 1
                    tc = np.arange(C) * voice.CTRL / FS
                    pf = np.full((C, tract.N_POLES), np.nan)
                    pb = pf.copy()
                    pf[:, :4], pb[:, :4] = Fs, (b1, 90, 120, 150)
                    zf, zb = np.zeros((C, 0)), np.zeros((C, 0))
                    if pair:
                        pf[:, 4], pb[:, 4] = 640.0, 80.0
                        zf, zb = np.full((C, 1), 580.0), np.full((C, 1), 60.0)
                    rng = np.random.default_rng(1400 + seed + 10 * b1 + pair)
                    f0c = 110 * 2 ** sampler._smooth_noise(rng, C, 150, 0.3 / 12)
                    tr = voice.Trajectories(f0=f0c, rd=np.full(C, 1.0), voice_amp=np.clip(np.minimum(tc, tc[-1] - tc) / .02, 0, 1),
                                            whisper_amp=np.zeros(C), fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=zf, zeros_b=zb)
                    y = voice.synthesize(tr, rng, jitter=0.01, shimmer_db=0.5)["total"]
                    y = 0.5 * y / np.abs(y).max()
                    nfr = len(y) // 160
                    f0f = np.interp(np.arange(nfr) * 0.01, tc, f0c)
                    res.append(probe(y, f0f, np.ones(nfr, bool), Fs)[0])
                m = np.nanmean(res, 0)
                j = _deepest(m, None, None, SG1)
                found = m[j] <= -3 and abs(BINS[j] - 580) <= 50
                passed = found if pair else m[j] > -3
                ok &= passed
                lines.append(f"  {vname} B1 {b1:3d} pair {'yes' if pair else 'no '}: deepest {BINS[j]:4.0f} Hz {m[j]:+5.1f} dB -> "
                             f"{'pass' if passed else 'FAIL'}; residual 400-800 Hz: "
                             + " ".join(f"{v:+.0f}" for v in m[(BINS >= 400) & (BINS <= 800)][::2]))
    lines.append("")
    lines.append(f"Validation {'passed' if ok else 'FAILED: subglottal and nasal readings withheld'}")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d140_validate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_run(rec: Path, out: Path, files: list[str]) -> None:
    d138 = _d138()
    from vocal_analysis.models.infer import analyse

    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    dev, models = d138._models()
    teacher = models[d138.TEACHERS[0]]
    rows = []
    for name, r, y in d138._load(rec, files):
        if name not in lv or lv[name]["f0"] >= 160 or r["item"] not in CLOSE + OPEN:
            continue
        res = analyse(teacher, y.astype(np.float32), dev)
        resid, h12, b1 = probe(y, res["f0"], res["voiced_p"] > 0.5, lv[name]["F"])
        rows.append({"take": name, "vowel": r["item"], "resid": resid.tolist(), "h12": h12, "b1": b1})
        print(name, flush=True)
    (out / "d140.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    rng = np.random.default_rng(140)
    lines = ["D-140 (private: per-take numbers; not for the repo)", ""]
    g = {}
    for gname, vs in (("close", CLOSE), ("open", OPEN)):
        rs = [np.array(x["resid"]) for x in rows if x["vowel"] in vs]
        g[gname] = _group(rs, rng)
        m, lo, hi = g[gname]
        lines.append(f"{gname} (n = {len(rs)}): mean residual (dB) [95 %] in 25 Hz bins, 300-1700 Hz:")
        for j in np.flatnonzero((BINS >= 300) & (BINS <= 1700)):
            flag = " *" if (abs(m[j]) >= 3 and (hi[j] < 0 or lo[j] > 0)) else ""
            lines.append(f"    {BINS[j]:5.0f} {m[j]:+5.1f} [{lo[j]:+5.1f}, {hi[j]:+5.1f}]{flag}")
    lines.append("")
    for lab, band in (("Sg1", SG1), ("Sg2", SG2)):
        js = {k: _deepest(g[k][0], None, None, band) for k in g}
        sup = all(g[k][0][js[k]] <= -3 and g[k][2][js[k]] < 0 for k in g) and abs(BINS[js["close"]] - BINS[js["open"]]) <= 75
        lines.append(f"Subglottal {lab} band {band}: deepest close {BINS[js['close']]:.0f} Hz {g['close'][0][js['close']]:+.1f}, open "
                     f"{BINS[js['open']]:.0f} Hz {g['open'][0][js['open']]:+.1f} -> {'supported' if sup else 'not supported'}")
    mo, lo_, hi_ = g["open"]
    mc = g["close"][0]
    w = (BINS >= 200) & (BINS <= 1200) & (np.abs(mo) >= 3) & ((hi_ < 0) | (lo_ > 0)) & (np.abs(mc) <= 1.5)
    lines.append("Nasal (open-only features >= 3 dB, close within 1.5 dB): "
                 + (", ".join(f"{BINS[j]:.0f} Hz {mo[j]:+.1f}" for j in np.flatnonzero(w)) + " -> supported" if w.any() else "none -> not supported"))
    V = np.array([x["vowel"] for x in rows])
    h = np.array([x["h12"] for x in rows])
    b = np.array([x["b1"] for x in rows])
    hc, bc = h.copy(), b.copy()
    for v in np.unique(V):
        hc[V == v] -= h[V == v].mean()
        bc[V == v] -= b[V == v].mean()
    r = np.corrcoef(hc, bc)[0, 1]
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(h), len(h))
        bs.append(np.corrcoef(hc[i], bc[i])[0, 1])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    lines.append(f"Glottal: within-vowel r(H1*-H2*, B1) = {r:+.2f} [{lo:+.2f}, {hi:+.2f}], n {len(h)} -> "
                 f"{'supported' if r >= 0.5 and lo > 0 else 'not supported'}")
    lines.append("  per vowel: " + "; ".join(f"{v} H1*-H2* {h[V == v].mean():+.1f} dB, B1 {b[V == v].mean():.0f} Hz" for v in np.unique(V)))
    lines.append("  per take: " + "; ".join(f"{x['take']} {x['vowel']} {x['h12']:+.1f}/{x['b1']:.0f}" for x in rows))
    (out / "d140.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "validate":
        cmd_validate()
    elif cmd == "run":
        cmd_run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
