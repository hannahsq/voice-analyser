"""D-063: the spectral slope of /h/ noise, tract-corrected, in real speech by stratum and in each synthesis engine's
/h/ onsets at a grid of noise tilts. Readings in docs/decisions.md.

    python scripts/analysis/d063_h_noise.py audit <out>       (CPU: label audit, segment lists)
    python scripts/analysis/d063_h_noise.py real <out>        (GPU in short bursts for teacher v3, then CPU)
    python scripts/analysis/d063_h_noise.py synth <out>       (CPU parallel render, then GPU / CPU as real)
    python scripts/analysis/d063_h_noise.py report <out>

Real sources (fit nothing that D-045 / D-059 score against): Zeroth-Korean train speakers (the minimal-pairs MFA cache,
korean_mfa; recorded sex only, numeric ids, never names), Common Voice 27.0 Korean validated (MFA with minimal-pairs'
patched dictionary; age band and gender), speechocean762 children (< 18, train and test; MFA english_mfa).
Segments: the label `h` on a word whose spelling starts with ㅎ / h (not wh), word-initial after >= 100 ms of no word
("strict"; without the pause condition: "relaxed", reported), >= 60 ms, before a non-high vowel.
"""

from __future__ import annotations

import collections
import csv
import glob
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FS = 16000
NWIN, NFFT = 512, 1024
BAND = (1000.0, 5000.0)
SEED = 5063
DS = Path("C:/github/datasets")
MP_CACHE = Path("C:/github/minimal-pairs/data/cache/align")
ZEROTH = DS / "zeroth_korean"
CV = DS / "mfa_ko" / "cv27_ko"
SO = {"train": DS / "mfa_en" / "so762_train", "test": DS / "mfa_en" / "so762_test"}
ZEROTH_TEST = {"104", "105", "112", "118", "121", "126", "132", "137", "147", "149"}  # D-045's real side
KO_NONHIGH = {"ɐ", "a", "ʌ", "o", "e", "ɛ", "ɐː", "ʌː", "oː", "eː", "ɛː"}
EN_NONHIGH = {"æ", "ɑ", "ɑː", "ɒ", "ʌ", "ə", "ɐ", "ɛ", "ej", "aj", "aw", "ow", "ɔ", "ɔː", "ɔj", "ɜː", "ɝ"}
TILTS = (None, 0.0, -6.0, -12.0)
ENGINES = ("cascade", "tube", "kl")
N_SYNTH = int(os.environ.get("D063_N", 48))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hangul_initial(ch):
    o = ord(ch) - 0xAC00
    return o // 588 if 0 <= o < 11172 else None


def textgrid(path):
    """{tier: [(start, end, text)]} from a long-format TextGrid."""
    t = Path(path).read_text(encoding="utf-8")
    out = {}
    for name, body in re.findall(r'name = "([^"]+)"(.*?)(?=item \[\d+\]:|\Z)', t, re.S):
        out[name] = [(float(a), float(b), c) for a, b, c in
                     re.findall(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', body)]
    return out


def candidates(words, lang):
    """words: [(spelling, start, end, [(phone, start, end)])] in order. Yields (h start, h end, strict, label) for
    word-initial h-spelled words; label is the first phone's label (the audit counts every label)."""
    for k, (w, ws, we, ph) in enumerate(words):
        if not w or not ph:
            continue
        spelled = hangul_initial(w[0]) == 18 if lang == "ko" else (w.lower().startswith("h") and not w.lower().startswith("wh"))
        if not spelled:
            continue
        lab, a, b = ph[0]
        nxt = ph[1][0] if len(ph) > 1 else ""
        gap = ws - words[k - 1][2] if k else 9.0
        ok = lab == "h" and b - a >= 0.06 and nxt in (KO_NONHIGH if lang == "ko" else EN_NONHIGH)
        yield a, b, ok and gap >= 0.1, ok, lab


def zeroth_sex():
    out = {}
    with open(ZEROTH / "AUDIO_INFO", encoding="utf-8", errors="replace") as f:
        cols = [c.strip() for c in f.readline().rstrip("\n").split("|")]
        si, xi, di = cols.index("SPEAKERID"), cols.index("SEX"), cols.index("DATASET")
        for line in f:
            p = [c.strip() for c in line.rstrip("\n").split("|")]
            if len(p) > max(si, xi, di) and p[di] == "train_data_01":
                out[p[si]] = p[xi]
    return out


def audit(out: Path):
    """Segment lists (one per source) and label counts."""
    out.mkdir(parents=True, exist_ok=True)
    sex = zeroth_sex()
    segs, labels = [], collections.Counter()
    for f in sorted(MP_CACHE.glob("*.json")):
        a = json.load(open(f, encoding="utf-8")).get("alignment")
        if not a:
            continue
        uid = a["utterance_id"]
        spk = uid.split("_")[0]
        if spk in ZEROTH_TEST or spk not in sex:
            continue
        words = [(w, ws, we, [tuple(p) for p in ph]) for w, ws, we, ph in a["words"]]
        for s, e, strict, relaxed, lab in candidates(words, "ko"):
            labels["zeroth", lab] += 1
            if relaxed:
                segs.append({"src": "zeroth", "spk": spk, "stratum": f"zeroth_{sex[spk].lower()}", "file":
                             str(ZEROTH / "train_data_01" / "003" / spk / f"{uid}.flac"), "s": s, "e": e, "strict": strict})
    cvmeta = {r["speaker"]: r for r in csv.DictReader(open(CV.parent / f"{CV.name}_speakers.tsv", encoding="utf-8"), delimiter="\t")}
    for f in sorted(glob.glob(str(CV.parent / f"{CV.name}_aligned" / "*" / "*.TextGrid"))):
        spk = Path(f).parent.name
        m = cvmeta.get(spk, {})
        g = {"male_masculine": "m", "female_feminine": "f"}.get(m.get("gender", ""), None)
        tg = textgrid(f)
        words = _words(tg)
        for s, e, strict, relaxed, lab in candidates(words, "ko"):
            labels["cv", lab] += 1
            if relaxed and g and m.get("age"):
                segs.append({"src": "cv", "spk": f"cv{spk}", "stratum": f"cv_{g}_{m['age']}",
                             "file": str(CV / spk / f"{Path(f).stem}.wav"), "s": s, "e": e, "strict": strict})
    D = _load("d045_discriminator")
    so = D.so_speakers()
    for split, root in SO.items():
        for f in sorted(glob.glob(str(root.parent / f"{root.name}_aligned" / "*" / "*.TextGrid"))):
            spk = "SPEAKER" + Path(f).parent.name
            if spk not in so or so[spk][1] >= 18:
                continue
            sx, age = so[spk]
            words = _words(textgrid(f))
            for s, e, strict, relaxed, lab in candidates(words, "en"):
                labels["so_child", lab] += 1
                if relaxed:
                    band = "5-10" if age <= 10 else "11-17"
                    segs.append({"src": "so_child", "spk": spk, "stratum": f"so_child_{sx}_{band}",
                                 "file": str(root / Path(f).parent.name / f"{Path(f).stem}.wav"), "s": s, "e": e,
                                 "strict": strict})
    json.dump(segs, open(out / "segments.json", "w", encoding="utf-8"))
    lines = ["D-063 label audit: first-phone labels of word-initial h-spelled words", ""]
    for src in ("zeroth", "cv", "so_child"):
        c = {lab: n for (s, lab), n in labels.items() if s == src}
        tot = sum(c.values())
        lines.append(f"{src}: {tot} words; " + ", ".join(f"{k} {v} ({100 * v / tot:.0f} %)"
                                                          for k, v in sorted(c.items(), key=lambda x: -x[1])[:12]))
    cnt = collections.Counter((s["stratum"], s["strict"]) for s in segs)
    spk = collections.defaultdict(set)
    for s in segs:
        if s["strict"]:
            spk[s["stratum"]].add(s["spk"])
    lines += ["", "segments by stratum: strict / relaxed-only (strict speakers)"]
    for st in sorted({s["stratum"] for s in segs}):
        lines.append(f"  {st:22s} {cnt[st, True]:5d} / {cnt[st, False]:5d}  ({len(spk[st])})")
    (out / "audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _words(tg):
    words = [w for w in tg.get("words", []) if w[2]]
    phones = [p for p in tg.get("phones", []) if p[2]]
    out = []
    for ws, we, w in words:
        out.append((w, ws, we, [(p[2], p[0], p[1]) for p in phones if p[0] >= ws - 1e-6 and p[1] <= we + 1e-6]))
    return out


# ---------------------------------------------------------------- the measurement (shared by real and synthetic)

def segment_slopes(x, r, segs):
    """For each (start s, end e) in seconds: the tract-corrected and raw slopes (dB/oct over BAND) of the mean power
    spectrum of the segment's middle 60 % (32 ms Hann frames, 8 ms hop; frames teacher v3 calls voiced (p >= 0.2) or
    within 15 dB of the file's pause floor dropped), corrected by the all-pole tract of the following vowel (teacher
    F1-F3 / B1-B3, median over voiced frames in the 60 ms after e); and the residual peak (1-3 kHz) above the line."""
    from vocal_analysis.features.vqt import HOP

    d061 = _load("d061_source")
    win = np.hanning(NWIN)
    fr = np.lib.stride_tricks.sliding_window_view(x, NWIN)[::NWIN // 2]
    e_all = 10 * np.log10((fr**2).sum(1) + 1e-20)
    floor = np.percentile(e_all, 10) if len(e_all) else -200.0
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    band = (f >= BAND[0]) & (f <= BAND[1])
    lf = np.log2(f[band])
    out = []
    for s, e in segs:
        d = e - s
        lo, hi = int((s + 0.2 * d) * FS), int((e - 0.2 * d) * FS)
        cs = np.arange(lo + NWIN // 2, hi - NWIN // 2 + 1, 128)
        keep = []
        for c in cs:
            seg = x[c - NWIN // 2: c + NWIN // 2]
            if len(seg) < NWIN:
                continue
            ti = min(int(round(c / HOP)), len(r["voiced_p"]) - 1)
            if r["voiced_p"][ti] >= 0.2 or 10 * np.log10((seg**2).sum() + 1e-20) < floor + 15:
                continue
            keep.append(np.abs(np.fft.rfft(seg * win, NFFT)) ** 2)
        a, b = int(round(e * FS / HOP)), int(round((e + 0.06) * FS / HOP))
        vi = np.arange(a, min(b + 1, len(r["voiced_p"])))
        vi = vi[r["voiced_p"][vi] > 0.5] if len(vi) else vi
        if not keep or not len(vi):
            out.append(None)
            continue
        F = np.nanmedian(r["F"][vi], 0)
        B = np.nanmedian(r["B"][vi], 0)
        db = 10 * np.log10(np.mean(keep, 0) + 1e-30)
        corr = db - d061.tract_db(f, np.tile(F, (len(f), 1)), np.tile(B, (len(f), 1)))
        p = np.polyfit(lf, corr[band], 1)
        sm = np.convolve(corr, np.ones(9) / 9, mode="same")
        res = sm[band] - np.polyval(p, lf)
        pk = float(res[(f[band] <= 3000)].max())
        out.append((float(p[0]), float(np.polyfit(lf, db[band], 1)[0]), pk, len(keep), float(F[0]), float(F[1])))
    return out


def _read16(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path)
    x = x if x.ndim == 1 else x.mean(1)
    return (resample_poly(x, FS, sr) if sr != FS else x).astype(np.float64)


def real(out: Path):
    import torch

    D = _load("d045_discriminator")
    T = D.Trunk(torch.device("cuda"))
    segs = json.load(open(out / "segments.json", encoding="utf-8"))
    by = collections.defaultdict(list)
    for s in segs:
        by[s["file"]].append(s)
    rows = []
    items = list(by.items())[: int(os.environ.get("D063_MAXFILES", 10**9))]  # a smaller run, to check the code path
    for k, (fpath, ss) in enumerate(items):
        x = _read16(fpath)
        x = 0.5 * x / (np.abs(x).max() + 1e-12)
        r = T.file(x.astype(np.float32))
        for s, m in zip(ss, segment_slopes(x, r, [(s["s"], s["e"]) for s in ss])):
            if m is not None:
                rows.append({**{k2: s[k2] for k2 in ("src", "spk", "stratum", "strict")}, "slope": m[0], "raw": m[1],
                             "peak": m[2], "frames": m[3], "F1": m[4], "F2": m[5]})
        if k % 500 == 0:
            D.log(f"real: {k}/{len(items)} files, {len(rows)} segments")
    json.dump(rows, open(out / "real.json", "w", encoding="utf-8"))
    D.log(f"real: {len(rows)} segments measured of {len(segs)}")


def _synth_clip(args):
    engine, t, i = args
    from dataclasses import replace

    from vocal_analysis.synth import kl_set, sampler

    base = replace(_load("d059_realness").synth_config(), p_h_onset=1.0, asp_tilt_db_oct=t, whisper_tilt_db_oct=t)
    if engine == "kl":
        c = kl_set.generate(i, replace(kl_set.KLSetConfig(), synth=replace(base, p_tube=1.0), p_lf_pulse=0.0), SEED)
    elif engine == "tube":
        c = sampler.generate(i, replace(base, p_tube=1.0, p_interactive=1.0), SEED)
    else:
        c = sampler.generate(i, replace(base, p_tube=0.0), SEED)
    hs = [(g["start_s"], g["end_s"]) for g in c.meta["segments"] if g.get("kind") == "h_onset"]
    return engine, t, i, c.audio, c.labels["voiced"], hs


def synth(out: Path):
    import torch

    from vocal_analysis.synth.channel import ChannelConfig, augment

    D = _load("d045_discriminator")
    tag = lambda t: "asis" if t is None else f"{t:+.0f}"  # noqa: E731
    cdir = out / "synth_clips"
    cdir.mkdir(parents=True, exist_ok=True)
    jobs = [(e, t, i) for e in ENGINES for t in TILTS for i in range(N_SYNTH)
            if not (cdir / f"{e}_{tag(t)}_{i:03d}.npz").exists()]
    with ProcessPoolExecutor(10) as ex:
        for e, t, i, audio, voiced, hs in ex.map(_synth_clip, jobs, chunksize=2):
            np.savez(cdir / f"{e}_{tag(t)}_{i:03d}.npz", audio=audio, voiced=voiced, hs=np.array(hs, float).reshape(-1, 2))
    T = D.Trunk(torch.device("cuda"))
    cfg3 = json.load(open(D.TEACHER_V3_CFG))
    channel = ChannelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg3["channel"].items()})
    rows = []
    for e in ENGINES:
        for t in TILTS:
            for i in range(N_SYNTH):
                z = np.load(cdir / f"{e}_{tag(t)}_{i:03d}.npz")
                hs = [tuple(h) for h in z["hs"] if h[1] - h[0] >= 0.06]
                if not hs:
                    continue
                x0 = z["audio"].astype(np.float64)
                mask = np.pad(np.repeat(z["voiced"], 160), (0, max(0, len(x0) - 160 * len(z["voiced"]))))[: len(x0)]
                xc = augment(x0, np.random.default_rng([SEED, i]), channel, mask, codec_stream=False)[0]
                for cond, x in (("clean", x0), ("channel", xc)):
                    r = T.file(x.astype(np.float32))
                    for m in segment_slopes(x, r, hs):
                        if m is not None:
                            rows.append({"engine": e, "tilt": tag(t), "clip": i, "cond": cond, "slope": m[0], "raw": m[1],
                                         "peak": m[2], "frames": m[3]})
            D.log(f"synth {e}: {sum(r['engine'] == e for r in rows)} segments")
    json.dump(rows, open(out / "synth.json", "w", encoding="utf-8"))


def report(out: Path):
    real_rows = json.load(open(out / "real.json", encoding="utf-8"))
    syn = json.load(open(out / "synth.json", encoding="utf-8"))
    rng = np.random.default_rng(63)
    NB = 1000

    def spk_stats(rows, key="slope"):
        """median over speakers of speaker medians, bootstrap over speakers; SD of speaker medians"""
        by = collections.defaultdict(list)
        for r in rows:
            by[r["spk"]].append(r[key])
        sm = np.array([np.median(v) for v in by.values()])
        if len(sm) < 3:
            return np.nan, np.nan, np.nan, np.nan, len(sm), len(rows)
        b = [np.median(sm[rng.integers(0, len(sm), len(sm))]) for _ in range(NB)]
        return float(np.median(sm)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float(sm.std(ddof=1)), len(sm), len(rows)

    L = ["D-063: /h/ noise slope, tract-corrected (dB/oct, 1-5 kHz); real: median over speakers of speaker medians "
         "[95 % bootstrap over speakers], SD across speakers (speakers; segments)", ""]
    L += (out / "audit.txt").read_text(encoding="utf-8").splitlines() + [""]
    targets = {}
    for which in ("strict", "relaxed"):
        L.append(f"Real, {which}:")
        sel = [r for r in real_rows if r["strict"] or which == "relaxed"]
        groups = sorted({r["stratum"] for r in sel})
        pooled = {"zeroth+cv f": [r for r in sel if r["src"] in ("zeroth", "cv") and r["stratum"].split("_")[1] == "f"],
                  "zeroth+cv m": [r for r in sel if r["src"] in ("zeroth", "cv") and r["stratum"].split("_")[1] == "m"],
                  "so_child all": [r for r in sel if r["src"] == "so_child"]}
        for g in groups + list(pooled):
            rows = pooled.get(g) or [r for r in sel if r["stratum"] == g]
            m, lo, hi, sd, ns, nr = spk_stats(rows)
            pm = spk_stats(rows, "peak")
            raw = spk_stats(rows, "raw")
            L.append(f"  {g:22s} {m:+6.2f} [{lo:+.2f}, {hi:+.2f}] SD {sd:.2f}  raw {raw[0]:+6.2f}  residual peak "
                     f"{pm[0]:+5.2f} dB  ({ns}; {nr})")
            if which == "strict":
                targets[g] = (m, lo, hi, sd, pm[0])
        L.append("")
    # 4. children's /h/: glottal if their residual peak is within 3 dB of Korean adults'
    ko_pk = np.nanmean([targets["zeroth+cv f"][4], targets["zeroth+cv m"][4]])
    ch_pk = targets["so_child all"][4]
    child_ok = bool(np.isfinite(ch_pk) and abs(ch_pk - ko_pk) <= 3)
    L.append(f"4. children's residual peak {ch_pk:+.2f} against Korean adults' {ko_pk:+.2f} dB -> "
             f"{'glottal-like: used' if child_ok else 'not shown glottal: reported, not used'}")
    L.append("")
    # 2. the synthetic mapping per engine and condition
    L.append("Synthetic /h/ onsets: segment slope median [bootstrap over clips] (segments), per engine x tilt")
    fits = {}
    for cond in ("channel", "clean"):
        L.append(f" {cond}:")
        for e in ENGINES:
            pts = []
            for t in ("asis", "+0", "-6", "-12"):
                rows = [r for r in syn if r["engine"] == e and r["tilt"] == t and r["cond"] == cond]
                if not rows:
                    continue
                clips = np.array([r["clip"] for r in rows])
                v = np.array([r["slope"] for r in rows])
                uc = np.unique(clips)
                byc = {c: np.flatnonzero(clips == c) for c in uc}
                b = [np.median(v[np.concatenate([byc[c] for c in rng.choice(uc, len(uc))])]) for _ in range(NB)]
                L.append(f"   {e:7s} {t:>4s}: {np.median(v):+6.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]"
                         f"  raw {np.median([r['raw'] for r in rows]):+6.2f}  peak {np.median([r['peak'] for r in rows]):+5.2f}  ({len(v)})")
                if t != "asis":
                    pts.append((float(t), float(np.median(v))))
            if len(pts) == 3:
                tt, ss = np.array(pts).T
                a, b0 = np.polyfit(tt, ss, 1)
                r2 = 1 - np.sum((ss - np.polyval([a, b0], tt)) ** 2) / np.sum((ss - ss.mean()) ** 2)
                fits[e, cond] = (a, b0, r2)
                ok = 0.7 <= a <= 1.3 and r2 >= 0.9
                L.append(f"   {e:7s} map: measured = {a:.2f} x tilt {b0:+.2f} (R^2 {r2:.3f}) -> {'valid' if ok else 'NOT 1:1'}")
    L.append("")
    L.append("3. Tilt that matches each real stratum (channel mapping; from the strict medians and their intervals)")
    for g, (m, lo, hi, sd, _) in targets.items():
        if g.startswith("so_child") and not child_ok:
            continue
        if not np.isfinite(m):
            continue
        parts = []
        for e in ENGINES:
            if (e, "channel") in fits:
                a, b0, _ = fits[e, "channel"]
                parts.append(f"{e} {(m - b0) / a:+.1f} [{(lo - b0) / a:+.1f}, {(hi - b0) / a:+.1f}]")
        L.append(f"  {g:22s} " + "  ".join(parts) + f"   (speaker SD {sd:.2f} dB/oct)")
    text = "\n".join(L)
    print(text)
    (out / "d063.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"audit": audit, "real": real, "synth": synth, "report": report}[sys.argv[1]](Path(sys.argv[2]))
