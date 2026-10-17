"""Source structure the observability label's analyst can't explain (D-027 "fix", 2026-10-18). On voiced frames of
regenerated training-distribution clips (train_v2: LF through the cascade or the tube; train_v2m: interactive
Rosenberg with feedback; train_v2s: plus the closure corner; train_v2s with the return phase), each harmonic's level
is read from the clean voiced component (the label's analysis: 25 ms Hann, n_fft 1024, peak within one bin of k f0),
the tract's response at k f0 (all labelled poles, closed glottis) is subtracted, and the label's smooth envelope (six
cosines in log2 f, 50-7600 Hz) is fitted to the rest by least squares. The remainder is source (and source-tract
interaction) structure a smooth envelope can't absorb. Reported: its SD by frequency band and f0 band, and its
correlation between harmonics as a function of their distance in octaves (pairs within a frame).
Seed 5151 (used by no set). Args: out file, clips per config (default 60)."""
import json
import sys
from dataclasses import replace

import numpy as np

from vocal_analysis.synth import observability as ob
from vocal_analysis.synth.obs_store import config_from_index
from vocal_analysis.synth.sampler import generate

sys.stdout.reconfigure(encoding="utf-8")
OUT = open(sys.argv[1], "w", encoding="utf-8")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
FS, W, NFFT, FMAX = 16000, 400, 1024, 7600.0
win = np.hanning(W)
cfg_ob = ob.ObservabilityConfig()
lo, hi = np.log2(50.0), np.log2(FMAX)

configs = {}
for name in ("train_v2", "train_v2m", "train_v2s"):
    configs[name] = config_from_index(json.load(open(f"data/synth/{name}/index.json")))
configs["train_v2s+return"] = replace(configs["train_v2s"], interactive_return=True)


def harmonic_db(x, t, f0):
    """(K,) dB of the voiced component's peaks within one bin of k f0, frame centred at t (s)."""
    c = int(round(t * FS))
    seg = np.zeros(W)
    a, b = max(c - W // 2, 0), min(c + W - W // 2, len(x))
    seg[a - (c - W // 2): b - (c - W // 2)] = x[a:b]
    P = np.abs(np.fft.rfft(seg * win, NFFT)) ** 2
    k = np.arange(1, int(FMAX // f0) + 1)
    b0 = np.round(k * f0 / FS * NFFT).astype(int)
    pk = np.max(np.stack([P[np.clip(b0 + d, 0, len(P) - 1)] for d in (-1, 0, 1)]), 0)
    return k * f0, 10 * np.log10(np.maximum(pk, 1e-30))


rows = []  # (config, f0, log2 f, deviation dB, frame id)
fid = 0
for cname, cfg in configs.items():
    for i in range(N):
        c = generate(i, cfg, 5151, keep_components=True)
        L = c.labels
        v = c.components.get("voiced")
        if v is None:
            continue
        for j in range(5, len(L["t"]) - 5, 3):  # every third voiced frame
            if not L["voiced"][j] or not np.isfinite(L["f0"][j]) or L["f0"][j] < 60:
                continue
            f, a = harmonic_db(v, L["t"][j], L["f0"][j])
            if len(f) < 8:
                continue
            F, B = L["formants"][j], L["bandwidths"][j]
            ok = np.isfinite(F) & np.isfinite(B) & (F > 0)
            tract_db = ob.resonance_db(f[:, None], F[ok][None, :], B[ok][None, :]).sum(1)
            r = a - tract_db
            x = (np.log2(f) - lo) / (hi - lo)
            G = np.cos(np.pi * np.arange(cfg_ob.n_env)[None, :] * x[:, None])
            coef, *_ = np.linalg.lstsq(G, r, rcond=None)
            dev = r - G @ coef
            for ff, dd in zip(f, dev):
                rows.append((cname, L["f0"][j], np.log2(ff), dd, fid))
            fid += 1
    print(f"  {cname} done ({fid} frames so far)", flush=True)

cn = np.array([r[0] for r in rows])
f0 = np.array([r[1] for r in rows])
lf = np.array([r[2] for r in rows])
dv = np.array([r[3] for r in rows])
fr = np.array([r[4] for r in rows])
p(f"Source structure beyond a smooth envelope (D-027): {fid} voiced frames, {len(rows)} harmonics, {N} clips per config, seed 5151")
p("SD (dB) of the deviation, by config and frequency band:")
bands = ((50, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4500), (4500, 7600))
p(f"{'':18s}" + "".join(f"{f'{a}-{b}':>11s}" for a, b in bands) + f"{'all':>8s}")
for name in configs:
    m = cn == name
    row = [np.std(dv[m & (lf >= np.log2(a)) & (lf < np.log2(b))]) for a, b in bands]
    p(f"{name:18s}" + "".join(f"{x:11.2f}" for x in row) + f"{np.std(dv[m]):8.2f}")
p("SD (dB), all configs, by f0 band (rows) and frequency band (columns):")
for a0, b0 in ((60, 150), (150, 250), (250, 350), (350, 600), (600, 1200)):
    m = (f0 >= a0) & (f0 < b0)
    if m.sum() < 200:
        continue
    row = [np.std(dv[m & (lf >= np.log2(a)) & (lf < np.log2(b))]) if (m & (lf >= np.log2(a)) & (lf < np.log2(b))).sum() > 50 else np.nan for a, b in bands]
    p(f"  f0 {a0:4d}-{b0:4d}  " + "".join(f"{x:11.2f}" for x in row))
p("Correlation of deviations between two harmonics of one frame, by their distance (octaves), all configs:")
order = np.argsort(fr, kind="stable")
fr_s, lf_s, dv_s = fr[order], lf[order], dv[order]
starts = np.r_[0, np.flatnonzero(np.diff(fr_s)) + 1, len(fr_s)]
edges = np.array([0.0, 0.05, 0.1, 0.2, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0])
sxy, sxx, syy, cnt = (np.zeros(len(edges) - 1) for _ in range(4))
for a, b in zip(starts[:-1], starts[1:]):
    l, d = lf_s[a:b], dv_s[a:b]
    iu = np.triu_indices(len(l), 1)
    bins = np.digitize(np.abs(l[:, None] - l[None, :])[iu], edges) - 1
    di, dj = d[iu[0]], d[iu[1]]
    for q in range(len(edges) - 1):
        sel = bins == q
        sxy[q] += (di[sel] * dj[sel]).sum()
        sxx[q] += (di[sel] ** 2).sum()
        syy[q] += (dj[sel] ** 2).sum()
        cnt[q] += sel.sum()
for q in range(len(edges) - 1):
    r = sxy[q] / np.sqrt(max(sxx[q] * syy[q], 1e-30))
    p(f"  {edges[q]:.2f}-{edges[q + 1]:.2f} oct: {r:+.3f}  (pairs {int(cnt[q])})")
p("The same, harmonics below 4.5 kHz only:")
sxy, sxx, syy, cnt = (np.zeros(len(edges) - 1) for _ in range(4))
for a, b in zip(starts[:-1], starts[1:]):
    keep = lf_s[a:b] < np.log2(4500)
    l, d = lf_s[a:b][keep], dv_s[a:b][keep]
    iu = np.triu_indices(len(l), 1)
    bins = np.digitize(np.abs(l[:, None] - l[None, :])[iu], edges) - 1
    di, dj = d[iu[0]], d[iu[1]]
    for q in range(len(edges) - 1):
        sel = bins == q
        sxy[q] += (di[sel] * dj[sel]).sum()
        sxx[q] += (di[sel] ** 2).sum()
        syy[q] += (dj[sel] ** 2).sum()
        cnt[q] += sel.sum()
for q in range(len(edges) - 1):
    r = sxy[q] / np.sqrt(max(sxx[q] * syy[q], 1e-30))
    p(f"  {edges[q]:.2f}-{edges[q + 1]:.2f} oct: {r:+.3f}  (pairs {int(cnt[q])})")
