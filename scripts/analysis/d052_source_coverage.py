"""D-052 (descriptive): does the synthetic glottal source cover real voices' source measures?

Part 1, recovery accuracy: OPENGLOT Repository II (evaluation only; channel 1 speech, channel 2 the true glottal
flow), through the resynthesis pipeline (teacher v2's analysis, `source_from_audio` with the Zeroth store's
settings), against the true flow's derivative at the same frames.
Part 2, coverage: recovered sources of the non-gold resynthesis pools (Zeroth-Korean val: numeric ids and recorded
sex only; VocalSet val) against 300 new synthetic clips (train_v2's settings, seed 5055) through the same pipeline.
Measures per voiced frame: H1-H2, H1-H4, source slope (dB / octave over H1-H10), NAQ, QOQ, HNR, CPP.

    python scripts/analysis/d052_source_coverage.py [part1|part2|report]   (CPU)
"""

import json
import sys
from pathlib import Path

import numpy as np

FS, HOP = 16000, 160
OUT = Path("reports/analysis/source_coverage")
CACHE = Path("cache/d052")
TEACHER_V2 = "runs/teacher_v2_d16k/ckpt_last.pt"
ZEROTH = Path("C:/github/datasets/zeroth_korean")
STORES = {"zeroth": ("data/resynth/zeroth_korean_val_d16k", "data/zeroth_korean"),
          "vocalset": ("data/resynth/vocalset_val_d16k", "data/vocalset")}
FILES_PER_SPEAKER = {"zeroth": 10, "vocalset": 8}
FRAMES_PER_FILE = 200
SYNTH_SEED, SYNTH_CLIPS = 5055, 300  # seed used by no other set
MEASURES = ("h1h2", "h1h4", "slope", "naq", "qoq", "hnr", "cpp")
F0_BANDS = ((0, 150), (150, 250), (250, 2000))


# ---------------------------------------------------------------- measures

def _polarity(d: np.ndarray) -> np.ndarray:
    """Closure is the sharp event of a flow derivative: flip so its largest peaks are negative."""
    return -d if np.percentile(d, 99.5) > -np.percentile(d, 0.5) else d


def _cycle(d: np.ndarray, c: int, p: int) -> tuple[int, int] | None:
    """Two consecutive closures (derivative minima) around sample c for period p."""
    a = c - p // 2
    if a < 0 or c + 2 * p >= len(d) or p < 8:
        return None
    g1 = a + int(np.argmin(d[a : a + p]))
    lo, hi = g1 + p // 2, g1 + (3 * p) // 2
    if hi >= len(d):
        return None
    return g1, lo + int(np.argmin(d[lo:hi]))


def flow_measures(d: np.ndarray, c: int, f0: float) -> tuple[float, float]:
    """NAQ and QOQ (50 %) of the cycle at sample c, from the flow derivative d."""
    cyc = _cycle(d, c, int(round(FS / f0)))
    if cyc is None:
        return np.nan, np.nan
    g1, g2 = cyc
    seg = d[g1:g2]
    flow = np.cumsum(seg)
    flow = flow - np.linspace(flow[0], flow[-1], len(flow))  # closed phase at both ends: remove the drift
    f_ac, d_peak = flow.max() - flow.min(), -d[g2]
    if d_peak <= 0 or f_ac <= 0:
        return np.nan, np.nan
    qoq = float(np.mean(flow > flow.min() + 0.5 * f_ac))
    return float(f_ac / (d_peak * (g2 - g1))), qoq


def noise_measures(x: np.ndarray, c: int, f0: float) -> tuple[float, float]:
    """HNR (normalised autocorrelation at the period, best within +-10 %) and CPP (dB) over 40 ms at sample c."""
    n = 640
    a = c - n // 2
    if a < 0 or a + n > len(x):
        return np.nan, np.nan
    w = np.hanning(n)
    seg = x[a : a + n] * w
    p = FS / f0
    ac = np.correlate(seg, seg, "full")[n - 1 :]
    if ac[0] <= 0:
        return np.nan, np.nan
    lags = np.arange(int(p * 0.9), min(int(p * 1.1) + 1, n // 2))
    if len(lags) == 0:
        return np.nan, np.nan
    wac = np.correlate(w, w, "full")[n - 1 :]  # Boersma (1993): divide out the window's own autocorrelation
    r = float(np.clip(np.max(ac[lags] / ac[0] / (wac[lags] / wac[0])), 1e-4, 0.9999))
    spec = 20 * np.log10(np.abs(np.fft.rfft(seg, 2048)) + 1e-12)
    ceps = np.fft.irfft(spec)
    q = np.arange(len(ceps)) / FS
    rng = (q >= 1 / 500) & (q <= 1 / 60)
    k = np.flatnonzero(rng)[np.argmax(ceps[rng])]
    line = np.polyfit(q[rng], ceps[rng], 1)
    return 10 * np.log10(r / (1 - r)), float(ceps[k] - np.polyval(line, q[k]))


def frame_measures(d: np.ndarray, f0: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """(len(frames), len(MEASURES)) for a flow derivative d and per-frame f0."""
    from vocal_analysis.backend.harmonics import harmonic_amplitudes

    times = frames * HOP / FS
    h = harmonic_amplitudes(d, FS, times, f0[frames], k_max=10)
    k = np.log2(np.arange(1, 11))
    out = np.full((len(frames), len(MEASURES)), np.nan)
    for j, i in enumerate(frames):
        hj = h[j]
        ok = np.isfinite(hj)
        if ok[:4].all():
            out[j, 0], out[j, 1] = hj[0] - hj[1], hj[0] - hj[3]
        if ok.sum() >= 5:
            out[j, 2] = np.polyfit(k[ok], hj[ok], 1)[0]
        out[j, 3], out[j, 4] = flow_measures(d, i * HOP, f0[i])
        out[j, 5], out[j, 6] = noise_measures(d, i * HOP, f0[i])
    return out


# ---------------------------------------------------------------- pipeline

def _cfg(store: str):
    from vocal_analysis.eval.probes.counterfactual import _cfg_from_store

    return _cfg_from_store(json.loads((Path(store) / "index.json").read_text(encoding="utf-8")))


def recover(audio: np.ndarray, analysis: dict, cfg) -> tuple[np.ndarray, np.ndarray]:
    """(source, accepted voiced frames) through the resynthesis pipeline; frames as the stores accept them."""
    from vocal_analysis.resynth.render import source_from_audio

    src = source_from_audio(audio, analysis, cfg)
    tl = src["tract_left"]
    ok = src["voiced"] & (~np.isfinite(tl) | (np.abs(tl) <= cfg.max_tract_left))
    return _polarity(np.asarray(src["source"], float)), np.flatnonzero(ok[5:-5]) + 5


def _pick(frames: np.ndarray, rng) -> np.ndarray:
    return np.sort(rng.choice(frames, FRAMES_PER_FILE, replace=False)) if len(frames) > FRAMES_PER_FILE else frames


def part1(log=print) -> None:
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import analyse, load_teacher

    dev = torch.device("cpu")
    model = load_teacher(TEACHER_V2, dev)[0]
    cfg = _cfg(STORES["zeroth"][0])
    rows = []
    rng = np.random.default_rng(0)
    for sex in ("female", "male"):
        for f in sorted(Path(f"data/openglot/RepositoryII_{sex}").glob("*.wav")):
            x, sr = sf.read(str(f), always_2d=True)
            speech = resample_poly(x[:, 0], FS, sr)
            flow = resample_poly(x[:, 1], FS, sr)
            a = analyse(model, speech.astype(np.float32), dev)
            src, frames = recover(speech, {k: a[k] for k in ("f0", "voiced_p", "sonorant_p", "formants", "bandwidths")}, cfg)
            true = np.r_[0.0, np.diff(flow)]  # lip radiation is in the speech, so the source is the flow derivative
            n = min(len(src), len(true))
            frames = _pick(frames[frames * HOP + 3 * 320 < n], rng)
            if len(frames) == 0:
                continue
            got, ref = frame_measures(src[:n], a["f0"], frames), frame_measures(true[:n], a["f0"], frames)
            _, vowel, f0, add = f.stem.split("_")
            rows.append({"file": f.name, "sex": sex, "vowel": vowel, "f0": f0, "adduction": add,
                         "recovered": np.nanmedian(got, 0).tolist(), "true": np.nanmedian(ref, 0).tolist()})
            log(f"  {f.name}: {len(frames)} frames")
    OUT.mkdir(parents=True, exist_ok=True)
    (CACHE).mkdir(parents=True, exist_ok=True)
    (CACHE / "part1.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def zeroth_sex() -> dict[str, str]:
    """{speaker id: sex} for test_data_01, from AUDIO_INFO's SPEAKERID, SEX and DATASET columns only (names never read
    into anything kept)."""
    out = {}
    with open(ZEROTH / "AUDIO_INFO", encoding="utf-8", errors="replace") as f:
        cols = [c.strip() for c in f.readline().rstrip("\n").split("|")]
        si, xi, di = cols.index("SPEAKERID"), cols.index("SEX"), cols.index("DATASET")
        for line in f:
            p = [c.strip() for c in line.rstrip("\n").split("|")]
            if len(p) > max(si, xi, di) and p[di] == "test_data_01":
                out[p[si]] = p[xi].lower()
    return out


def part2(log=print) -> None:
    import torch

    from vocal_analysis.resynth.corpus import _analysis_name, load_audio

    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    zsex = zeroth_sex()
    for name, (store, root) in STORES.items():
        if (CACHE / f"real_{name}.npz").exists():
            continue
        index = json.loads((Path(store) / "index.json").read_text(encoding="utf-8"))
        cfg = _cfg(store)
        adir = Path("data/resynth/analysis") / index["corpus"] / index["teacher"]["sha256"]
        by: dict[str, list[str]] = {}
        for f in index["files"]:
            spk = f.split("/")[2] if name == "zeroth" else f.split("/")[1]
            by.setdefault(spk, []).append(f)
        M, F0, SEX = [], [], []
        for spk, files in sorted(by.items()):
            sex = zsex.get(spk, "?") if name == "zeroth" else ("female" if spk.startswith("female") else "male")
            sex = {"f": "female", "m": "male"}.get(sex, sex)
            for f in [files[int(i)] for i in np.linspace(0, len(files) - 1, min(FILES_PER_SPEAKER[name], len(files)))]:
                audio = load_audio(Path(root) / f)
                a = dict(np.load(adir / _analysis_name(f)))
                src, frames = recover(audio, a, cfg)
                frames = _pick(frames, rng)
                if len(frames):
                    M.append(frame_measures(src, a["f0"], frames))
                    F0.append(a["f0"][frames])
                    SEX += [sex] * len(frames)
            log(f"  {name} {spk}: {len(SEX)} frames so far")
        np.savez_compressed(CACHE / f"real_{name}.npz", m=np.concatenate(M), f0=np.concatenate(F0), sex=np.array(SEX))
    if not (CACHE / "synth.npz").exists():
        from vocal_analysis.models.infer import analyse, load_teacher
        from vocal_analysis.synth.sampler import SynthConfig, generate

        dev = torch.device("cpu")
        model = load_teacher(TEACHER_V2, dev)[0]
        cfg = _cfg(STORES["zeroth"][0])
        M, F0, SEX = [], [], []
        for i in range(SYNTH_CLIPS):
            clip = generate(i, SynthConfig(), SYNTH_SEED)
            a = analyse(model, clip.audio, dev)
            src, frames = recover(clip.audio, {k: a[k] for k in ("f0", "voiced_p", "sonorant_p", "formants", "bandwidths")}, cfg)
            frames = _pick(frames, rng)
            if len(frames):
                M.append(frame_measures(src, a["f0"], frames))
                F0.append(a["f0"][frames])
                g = clip.meta.get("length_group")
                SEX += [{"women": "female", "men": "male"}.get(g, "other")] * len(frames)
            if (i + 1) % 25 == 0:
                log(f"  synth {i + 1}/{SYNTH_CLIPS}")
        np.savez_compressed(CACHE / "synth.npz", m=np.concatenate(M), f0=np.concatenate(F0), sex=np.array(SEX))


# ---------------------------------------------------------------- report

def report() -> str:
    lines = ["D-052 source coverage (descriptive)", "", "Part 1: recovery accuracy on OPENGLOT Repository II "
             "(recovered minus true, median over files [IQR]); a coverage gap smaller than this is not read"]
    p1 = json.loads((CACHE / "part1.json").read_text(encoding="utf-8"))
    err = np.array([np.array(r["recovered"]) - np.array(r["true"]) for r in p1])
    res = {"part1": {}}
    for j, m in enumerate(MEASURES):
        e = err[:, j][np.isfinite(err[:, j])]
        if len(e):
            q = np.percentile(e, [25, 50, 75])
            res["part1"][m] = {"median": q[1], "iqr": [q[0], q[2]], "n": len(e)}
            lines.append(f"  {m:6s} {q[1]:+8.3f} [{q[0]:+.3f}, {q[2]:+.3f}]  ({len(e)} files)")
    syn = np.load(CACHE / "synth.npz")
    res["part2"] = {}
    for name in STORES:
        real = np.load(CACHE / f"real_{name}.npz")
        lines += ["", f"Part 2: {name} val (real) against synthetic; median [5-95 %] real | synthetic; share of real "
                  "frames outside the synthetic 5-95 %"]
        for sex in ("female", "male"):
            for lo, hi in F0_BANDS:
                rm = (real["sex"] == sex) & (real["f0"] >= lo) & (real["f0"] < hi)
                sm = (syn["sex"] == sex) & (syn["f0"] >= lo) & (syn["f0"] < hi)
                if rm.sum() < 50 or sm.sum() < 50:
                    continue
                lines.append(f"  {sex} f0 {lo}-{hi} Hz ({rm.sum()} real, {sm.sum()} synthetic frames)")
                for j, m in enumerate(MEASURES):
                    r, s = real["m"][rm, j], syn["m"][sm, j]
                    r, s = r[np.isfinite(r)], s[np.isfinite(s)]
                    if len(r) < 30 or len(s) < 30:
                        continue
                    rq, sq = np.percentile(r, [5, 50, 95]), np.percentile(s, [5, 50, 95])
                    out = float(np.mean((r < sq[0]) | (r > sq[2])))
                    res["part2"][f"{name}/{sex}/{lo}-{hi}/{m}"] = {"real": rq.tolist(), "synth": sq.tolist(), "outside": out}
                    lines.append(f"    {m:6s} {rq[1]:+8.2f} [{rq[0]:+.2f}, {rq[2]:+.2f}] | {sq[1]:+8.2f} "
                                 f"[{sq[0]:+.2f}, {sq[2]:+.2f}]  outside {out:.0%}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source_coverage.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    text = "\n".join(lines)
    (OUT / "source_coverage.txt").write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("part1", "all"):
        part1(lambda m: print(m, flush=True))
    if what in ("part2", "all"):
        part2(lambda m: print(m, flush=True))
    if what in ("report", "all"):
        print(report())
