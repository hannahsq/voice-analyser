"""D-061: H1-H2 with and without formant correction (Iseli & Alwan 2004), by source parameter, on D-059's clips; and
the real recordings' low-frequency roll-off from pause spectra. Descriptive, readings in docs/decisions.md.

    python scripts/analysis/d061_source.py labels <out>    (CPU, parallel: D-059's clips regenerated with labels)
    python scripts/analysis/d061_source.py measure <out>   (GPU in short bursts for teacher v3's F / B, then CPU)
    python scripts/analysis/d061_source.py rolloff <out>   (CPU: pause and speech spectra, real and synthetic channel)
    python scripts/analysis/d061_source.py report <out>
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

D059_CLIPS = Path("cache/d059/clips")
FS = 16000
PER_CLIP = 60
SEED = 5061
F0_BINS = ((80, 130), (130, 180), (180, 250), (250, 350))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tract_db(f, F, B):
    """Level (dB) of the all-pole tract with resonances F, B (n, k) at frequencies f (n,), each resonance 0 dB at DC;
    missing formants contribute nothing."""
    r = np.exp(-np.pi * B / FS)
    th, w = 2 * np.pi * F / FS, (2 * np.pi * f / FS)[:, None]
    num = 1 - 2 * r * np.cos(th) + r**2
    den = np.sqrt((1 - 2 * r * np.cos(th + w) + r**2) * (1 - 2 * r * np.cos(th - w) + r**2))
    return np.nansum(20 * np.log10(num / den), axis=1)


def corrected(h1h2, f0, F, B):
    return h1h2 - (tract_db(f0, F[:, :3], B[:, :3]) - tract_db(2 * f0, F[:, :3], B[:, :3]))


def _clip(args):
    name, i = args
    from dataclasses import replace

    from vocal_analysis.synth import kl_set, sampler

    d059 = _load("d059_realness")
    base = d059.synth_config()
    if name == "kl":
        c = kl_set.generate(i, replace(kl_set.KLSetConfig(), synth=replace(base, p_tube=1.0)), d059.SEED)
        src = c.meta["kl"]
    elif name == "tube":
        c = sampler.generate(i, replace(base, p_tube=1.0, p_interactive=1.0), d059.SEED)
        src = c.meta["tube"]
    else:
        c = sampler.generate(i, replace(base, p_tube=0.0), d059.SEED)
        src = {"pulse": "lf", "area_sq": None}
    lab = c.labels
    keep = {k: lab[k] for k in ("t", "f0", "voiced", "whisper", "formants", "bandwidths", "rd", "source_h_db")}
    keep["mixed"] = lab.get("formants_mixed", np.zeros_like(lab["formants"], bool))
    saved = np.load(D059_CLIPS / name / f"{i:04d}.npz")["audio"]
    same = saved.shape == c.audio.shape and np.array_equal(saved, c.audio)
    return name, i, keep, {"pulse": src.get("pulse") or "lf", "area_sq": src.get("area_sq"),
                           "n_extra": c.meta.get("n_extra_poles", 0) + c.meta.get("n_extra_zeros", 0), "same": same}


def labels(out: Path):
    d059 = _load("d059_realness")
    jobs = [(s, i) for s in d059.SETS for i in range(d059.N_CLIPS) if not (out / "labels" / s / f"{i:04d}.npz").exists()]
    for s in d059.SETS:
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(10) as ex:
        for k, (s, i, lab, meta) in enumerate(ex.map(_clip, jobs, chunksize=2), 1):
            np.savez(out / "labels" / s / f"{i:04d}.npz", meta=json.dumps(meta), **lab)
            if k % 40 == 0:
                print(f"labels {k}/{len(jobs)}", flush=True)


def measure(out: Path):
    import torch

    from vocal_analysis.features.vqt import HOP
    from vocal_analysis.synth.channel import ChannelConfig, augment

    D, d059 = _load("d045_discriminator"), _load("d059_realness")
    T = D.Trunk(torch.device("cuda"))
    cfg3 = json.load(open(D.TEACHER_V3_CFG))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    rng = np.random.default_rng(SEED)
    for s in d059.SETS:
        path = out / f"{s}.npz"
        if path.exists():
            continue
        rows = []
        for i in range(d059.N_CLIPS):
            z = np.load(D059_CLIPS / s / f"{i:04d}.npz")
            lab = np.load(out / "labels" / s / f"{i:04d}.npz")
            meta = json.loads(str(lab["meta"]))
            x0 = z["audio"].astype(np.float64)
            centres = np.round(lab["t"] * FS).astype(int)
            ok = (lab["voiced"] & ~lab["whisper"] & (lab["f0"] > 0) & np.isfinite(lab["formants"][:, :3]).all(1)
                  & np.isfinite(lab["bandwidths"][:, :3]).all(1) & ~lab["mixed"][:, :3].any(1)
                  & (centres > 0.02 * FS) & (centres < len(x0) - 0.02 * FS))
            idx = np.flatnonzero(ok)
            if not len(idx):
                continue
            idx = np.sort(rng.choice(idx, min(PER_CLIP, len(idx)), replace=False))
            for cond in ("clean", "channel"):
                x = x0
                if cond == "channel":  # D-059's draw for this clip
                    mask = np.pad(np.repeat(z["voiced"], 160), (0, max(0, len(x0) - 160 * len(z["voiced"]))))[: len(x0)]
                    x, _ = augment(x0, np.random.default_rng([d059.SEED, i]), channel, mask, codec_stream=False)
                r = T.file(x.astype(np.float32))
                ti = np.clip(np.round(centres[idx] / HOP).astype(int), 0, len(r["f0"]) - 1)
                m = D.measures(x, centres[idx], lab["f0"][idx], lab["formants"][idx, 0])
                f0, F, B = lab["f0"][idx], lab["formants"][idx], lab["bandwidths"][idx]
                sh = lab["source_h_db"][idx]
                rows.append({"cond": np.full(len(idx), cond), "clip": np.full(len(idx), i), "f0": f0,
                             "h1h2": m[:, 0], "tilt": m[:, 4], "exact": corrected(m[:, 0], f0, F, B),
                             "teacher": corrected(m[:, 0], f0, r["F"][ti], r["B"][ti]),
                             "truth": sh[:, 0] - sh[:, 1], "rd": lab["rd"][idx],
                             "pulse": np.full(len(idx), meta["pulse"]),
                             "sq": np.full(len(idx), np.nan if meta["area_sq"] is None else meta["area_sq"]),
                             "n_extra": np.full(len(idx), meta["n_extra"]), "same": np.full(len(idx), meta["same"])})
        np.savez(path, **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
        D.log(f"{s}: {sum(len(r['f0']) for r in rows)} frames")


def _band_levels(x, quiet: bool):
    """Mean power spectrum (dB) of the quietest 10 % (quiet) or loudest 50 % of 32 ms frames, as band levels re
    300-600 Hz: 30-60, 60-100, 100-150 Hz. None if the frames are digitally silent."""
    n = 512
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 2] * np.hanning(n)
    if len(fr) < 20:
        return None
    e = (fr**2).sum(1)
    o = np.argsort(e)
    sel = o[: max(2, len(o) // 10)] if quiet else o[len(o) // 2:]
    if e[sel].max() <= 1e-14:
        return None
    p = (np.abs(np.fft.rfft(fr[sel], 4096)) ** 2).mean(0)
    f = np.fft.rfftfreq(4096, 1 / FS)
    lv = lambda a, b: 10 * np.log10(p[(f >= a) & (f < b)].mean() + 1e-30)  # noqa: E731
    ref = lv(300, 600)
    return [lv(30, 60) - ref, lv(60, 100) - ref, lv(100, 150) - ref]


def rolloff(out: Path):
    import soundfile as sf
    from scipy.signal import resample_poly

    from vocal_analysis.synth.channel import ChannelConfig, augment

    D, d059 = _load("d045_discriminator"), _load("d059_realness")
    rng = np.random.default_rng(SEED)

    def read16(path):
        x, sr = sf.read(path)
        x = x if x.ndim == 1 else x.mean(1)
        return resample_poly(x, FS, sr) if sr != FS else x

    sources = {}
    zer = D.zeroth_speakers()
    sources["zeroth"] = {s: (sx, sorted(map(str, (D.ZEROTH / "test_data_01").glob(f"*/{s}/*.flac")))) for s, sx in zer.items()}
    so = D.so_speakers()
    sources["so_adult"] = {s: (sx, sorted(map(str, (D.SO / "WAVE" / s).glob("*.WAV")))) for s, (sx, a) in so.items() if a >= 18}
    rows = []
    for cname, spk in sources.items():
        for s, (sx, files) in spk.items():
            for f in (rng.choice(files, min(3, len(files)), replace=False) if files else []):
                x = read16(f)
                for quiet in (True, False):
                    lv = _band_levels(x, quiet)
                    rows.append((cname, sx, "pause" if quiet else "speech", lv))
    cfg3 = json.load(open(D.TEACHER_V3_CFG))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    for s in d059.SETS:
        for i in range(d059.N_CLIPS):
            z = np.load(D059_CLIPS / s / f"{i:04d}.npz")
            x0 = z["audio"].astype(np.float64)
            mask = np.pad(np.repeat(z["voiced"], 160), (0, max(0, len(x0) - 160 * len(z["voiced"]))))[: len(x0)]
            x, _ = augment(x0, np.random.default_rng([d059.SEED, i]), channel, mask, codec_stream=False)
            for quiet in (True, False):
                rows.append((f"{s}_channel", "?", "pause" if quiet else "speech", _band_levels(x, quiet)))
    with open(out / "rolloff.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh)


def report(out: Path):
    D, d059 = _load("d045_discriminator"), _load("d059_realness")
    syn = {s: {k: v for k, v in np.load(out / f"{s}.npz").items()} for s in d059.SETS}
    load = lambda p: {k: v for k, v in np.load(p, allow_pickle=True).items()}  # noqa: E731
    zer, so = load(Path("cache/d045/zeroth.npz")), load(Path("cache/d045/so.npz"))
    real = {"zeroth": zer, "so_adult": {k: v[so["group"] == "adult"] for k, v in so.items()}}
    rng = np.random.default_rng(61)

    def med(v, g=None, n=1000):
        v = np.asarray(v, float)
        ok = np.isfinite(v)
        v = v[ok]
        if len(v) < 10:
            return f"{'n/a':>22s} ({len(v)})"
        if g is None:
            b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
        else:
            g = np.asarray(g)[ok]
            us = np.unique(g)
            by = {u: np.flatnonzero(g == u) for u in us}
            b = [np.median(v[np.concatenate([by[u] for u in rng.choice(us, len(us))])]) for _ in range(n)]
        return f"{np.median(v):+6.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}] ({len(v)})"

    L = ["D-061: H1-H2 and H1*-H2* (Iseli & Alwan, F1-F3), dB; median [95 % bootstrap over clips / speakers] (frames)", ""]
    same = {s: bool(d["same"].all()) for s, d in syn.items()}
    L.append("regenerated audio equal to D-059's clips: " + ", ".join(f"{s} {'yes' if v else 'NO'}" for s, v in same.items()))
    L.append("")
    c = syn["cascade"]
    m = (c["cond"] == "clean") & np.isfinite(c["truth"])
    L.append("1. Correction valid: clean cascade frames, exact H1*-H2* minus source truth")
    L.append(f"   all            {med(c['exact'][m] - c['truth'][m], c['clip'][m])}")
    m0 = m & (c["n_extra"] == 0)
    L.append(f"   no extra p/z   {med(c['exact'][m0] - c['truth'][m0], c['clip'][m0])}")
    L.append(f"   |diff| median  {np.median(np.abs(c['exact'][m] - c['truth'][m])):.2f} (rule <= 1.5)")
    for lo, hi in F0_BINS:
        mb = m & (c["f0"] >= lo) & (c["f0"] < hi)
        L.append(f"   f0 {lo}-{hi:<4}    {med(c['exact'][mb] - c['truth'][mb], c['clip'][mb])}")
    L += ["", "2. Teacher correction: synthetic channel frames, teacher minus exact H1*-H2*, per f0 bin (readable if |median| <= 2)"]
    readable = {}
    for lo, hi in F0_BINS:
        parts, allv = [], []
        for s, d in syn.items():
            mb = (d["cond"] == "channel") & (d["f0"] >= lo) & (d["f0"] < hi)
            parts.append(f"{s} {med(d['teacher'][mb] - d['exact'][mb], d['clip'][mb])}")
            allv.append(d["teacher"][mb] - d["exact"][mb])
        a = np.concatenate(allv)
        readable[lo, hi] = bool(np.isfinite(a).sum() >= 10 and abs(np.nanmedian(a)) <= 2)
        L.append(f"   f0 {lo}-{hi:<4} all {np.nanmedian(a):+.2f} -> {'readable' if readable[lo, hi] else 'not readable'};  " + ";  ".join(parts))
    L += ["", "3. Teacher-corrected H1*-H2* by f0 bin: real (per sex) and synthetic channel (and its exact, uncorrected)"]
    gap = {}
    for lo, hi in F0_BINS:
        L.append(f"   f0 {lo}-{hi} ({'readable' if readable[lo, hi] else 'not readable'})")
        rmed = {}
        for cname, d in real.items():
            for sx in "fm":
                mb = (d["sex"] == sx) & (d["f0"] >= lo) & (d["f0"] < hi)
                v = corrected(d["meas"][mb, 0], d["f0"][mb], d["F"][mb], d["B"][mb])
                L.append(f"     real {cname:9s} {sx}  teacher {med(v, d['speaker'][mb])}   H1-H2 {med(d['meas'][mb, 0], d['speaker'][mb])}")
                if np.isfinite(v).sum() >= 10:
                    rmed.setdefault(cname, []).append(v[np.isfinite(v)])
        smed = []
        for s, d in syn.items():
            mb = (d["cond"] == "channel") & (d["f0"] >= lo) & (d["f0"] < hi)
            L.append(f"     {s + '_channel':19s}  teacher {med(d['teacher'][mb], d['clip'][mb])}   exact {med(d['exact'][mb], d['clip'][mb])}"
                     f"   H1-H2 {med(d['h1h2'][mb], d['clip'][mb])}")
            if np.isfinite(d["teacher"][mb]).sum() >= 10:
                smed.append(np.nanmedian(d["teacher"][mb]))
        if readable[lo, hi] and smed and len(rmed) == 2:
            gap[lo, hi] = {cn: min(smed) - np.median(np.concatenate(v)) for cn, v in rmed.items()}
    L.append("")
    for k, g in gap.items():
        both = all(v >= 3 for v in g.values())
        L.append(f"   reading 3, f0 {k[0]}-{k[1]}: lowest synthetic set minus real " + ", ".join(f"{c} {v:+.2f}" for c, v in g.items())
                 + f" -> {'source H1 too strong after correction' if both else 'no'}")
    L += ["", "4. Low-frequency levels re 300-600 Hz (dB), median over files [IQR] (files; silent)"]
    rows = json.load(open(out / "rolloff.json", encoding="utf-8"))
    groups = {}
    for cname, sx, kind, lv in rows:
        groups.setdefault((cname, sx, kind), []).append(lv)
    p60 = {}
    for key in sorted(groups):
        vals = [v for v in groups[key] if v is not None]
        nsil = len(groups[key]) - len(vals)
        if not vals:
            L.append(f"   {key[0]:16s} {key[1]} {key[2]:6s}  no frames ({nsil} silent)")
            continue
        a = np.array(vals)
        L.append(f"   {key[0]:16s} {key[1]} {key[2]:6s}  " + "  ".join(
            f"{b} {np.median(a[:, j]):+6.1f} [{np.percentile(a[:, j], 25):+.1f}, {np.percentile(a[:, j], 75):+.1f}]"
            for j, b in enumerate(("30-60", "60-100", "100-150"))) + f"  ({len(a)}; {nsil})")
        if key[2] == "pause":
            p60.setdefault(key[0], []).append(a[:, 1])
    syn60 = np.median(np.concatenate([v for k, vs in p60.items() if k.endswith("_channel") for v in vs]))
    res = {c: np.median(np.concatenate(p60[c])) - syn60 for c in real if c in p60}
    L.append(f"   reading 4: real pause 60-100 Hz minus synthetic channel ({syn60:+.1f}): "
             + ", ".join(f"{c} {v:+.1f}" for c, v in res.items())
             + f" -> {'real recordings roll off more (channel candidate)' if len(res) == 2 and all(v <= -10 for v in res.values()) else 'no'}")
    L += ["", "5. Parameter split (clean frames): H1-H2 / exact H1*-H2* / tilt, median (frames)"]
    for s, d in syn.items():
        cl = d["cond"] == "clean"
        q = np.nanpercentile(d["rd"][cl], [33.3, 66.7])
        for j, (lo, hi) in enumerate(((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf))):
            mb = cl & (d["rd"] >= lo) & (d["rd"] < hi)
            L.append(f"   {s:8s} Rd tertile {j + 1} ({np.nanmedian(d['rd'][mb]):.2f}): {np.nanmedian(d['h1h2'][mb]):+6.2f} / "
                     f"{np.nanmedian(d['exact'][mb]):+6.2f} / {np.nanmedian(d['tilt'][mb]):+6.2f}  ({mb.sum()})")
        if s != "cascade":
            for p in np.unique(d["pulse"][cl]):
                mb = cl & (d["pulse"] == p)
                L.append(f"   {s:8s} pulse {p:9s}: {np.nanmedian(d['h1h2'][mb]):+6.2f} / {np.nanmedian(d['exact'][mb]):+6.2f} / "
                         f"{np.nanmedian(d['tilt'][mb]):+6.2f}  ({mb.sum()})")
            sq = d["sq"][cl]
            if np.isfinite(sq).sum() >= 30:
                q = np.nanpercentile(sq, [33.3, 66.7])
                for j, (lo, hi) in enumerate(((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf))):
                    mb = cl & (d["sq"] >= lo) & (d["sq"] < hi)
                    L.append(f"   {s:8s} sq tertile {j + 1} ({np.nanmedian(d['sq'][mb]):.2f}): {np.nanmedian(d['h1h2'][mb]):+6.2f} / "
                             f"{np.nanmedian(d['exact'][mb]):+6.2f} / {np.nanmedian(d['tilt'][mb]):+6.2f}  ({mb.sum()})")
    text = "\n".join(L)
    print(text)
    (out / "d061.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"labels": labels, "measure": measure, "rolloff": rolloff, "report": report}[sys.argv[1]](Path(sys.argv[2]))
