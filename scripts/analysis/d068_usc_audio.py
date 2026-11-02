"""D-068: is the USC 75-speaker MRI audio usable for checking vocal-tract length? Label-free, dev speakers only, never
compared with any MRI length, and no per-speaker length is printed or committed (docs/decisions.md).

    python scripts/analysis/d068_usc_audio.py extract    (CPU: ffmpeg, the passages' audio to 16 kHz wav in cache/d068)
    python scripts/analysis/d068_usc_audio.py analyse    (GPU: teacher v3.1 seeds 2 and 1, per-file summaries)
    python scripts/analysis/d068_usc_audio.py report     (CPU: aggregates against PTDB-TUG dev; reports/analysis/d068/)

Per file (sonorant voiced frames: voiced_p and sonorant_p > 0.5): the fraction of such frames, the median anatomical
length (VTL head) and its sigma, the median formant sigma F1-F4; seed mean of the two models. Per corpus and sex:
within-speaker SD of the per-file lengths (%), between-speaker SD, reliability (between / (between + within) variance),
with a speaker bootstrap; for USC, the paired repeat difference |r1 - r2| of the same passage.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

USC = Path("C:/github/datasets/usc75_2drt_video/full")
SEX = Path("C:/github/datasets/usc75_subjects/sex.tsv")
CACHE = Path("cache/d068")
OUT = Path("reports/analysis/d068")
PASSAGES = ("06_rainbow", "07_grandfather1", "08_grandfather2", "09_northwind1", "10_northwind2")
MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")
PTDB_PER_SPEAKER = 10
SEED = 5068


def usc_dev() -> list[str]:
    from vocal_analysis.eval.tiers import tier_of

    return [f"sub{n:03d}" for n in range(1, 76) if tier_of(f"usc75:sub{n:03d}") == "dev" and (USC / f"sub{n:03d}").exists()]


def extract():
    (CACHE / "usc").mkdir(parents=True, exist_ok=True)
    n = 0
    for s in usc_dev():
        for p in PASSAGES:
            for r in ("r1", "r2"):
                src = USC / s / "2drt" / "video" / f"{s}_2drt_{p}_{r}_video.mp4"
                dst = CACHE / "usc" / f"{s}_{p}_{r}.wav"
                if not src.exists() or dst.exists():
                    continue
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
                               check=True, stdin=subprocess.DEVNULL)
                n += 1
    print(f"extracted {n} files for {len(usc_dev())} dev speakers", flush=True)


def _files():
    """(corpus, speaker, sex, key, loader) for every file analysed."""
    import soundfile as sf

    from vocal_analysis.eval.datasets import load_audio, registry
    from vocal_analysis.eval.probes.channel import _rows

    sex = dict(line.strip().lstrip("\ufeff").split("\t")[:2] for line in SEX.read_text(encoding="utf-8").splitlines() if line.strip())
    out = []
    for f in sorted((CACHE / "usc").glob("*.wav")):
        s = f.stem.split("_")[0]
        out.append(("usc", s, sex[s].lower(), f.stem, lambda f=f: sf.read(f)[0]))
    reg = registry()
    by_spk: dict[str, list] = {}
    for r in _rows("ptdb_tug"):
        by_spk.setdefault(r.speaker_id, []).append(r)
    for spk, rs in sorted(by_spk.items()):
        for i in np.linspace(0, len(rs) - 1, min(PTDB_PER_SPEAKER, len(rs))).astype(int):
            r = rs[i]
            sx = "f" if spk.split(":")[-1].startswith("F") else "m"
            out.append(("ptdb", spk, sx, Path(r.path).stem, lambda r=r: load_audio(reg["ptdb_tug"].load(Path("data") / "ptdb_tug", r))))
    return out


def analyse():
    import torch

    from vocal_analysis.models.infer import analyse as run, load_teacher

    dev = torch.device("cuda")
    path = CACHE / "files.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    models = [load_teacher(f"runs/{m}/ckpt_last.pt", dev)[0] for m in MODELS]
    for corpus, spk, sx, key, load in _files():
        k = f"{corpus}:{key}"
        if k in done:
            continue
        x = np.asarray(load(), np.float32)
        x = 0.5 * x / (np.abs(x).max() + 1e-12)
        per = []
        for m in models:
            r = run(m, x, dev)
            on = (r["voiced_p"] > 0.5) & (r["sonorant_p"] > 0.5)
            per.append({"frac": float(on.mean()), "n": int(on.sum()),
                        "vtl": float(np.median(2 ** r["vtl_log2cm"][on])) if on.any() else None,
                        "vtl_sigma": float(np.median(r["vtl_sigma"][on])) if on.any() else None,
                        "Fsig": [float(np.median(r["formant_sigma_cents"][on, j])) if on.any() else None for j in range(4)]})
        mean = lambda key: None if any(p[key] is None for p in per) else float(np.mean([p[key] for p in per]))  # noqa: E731
        done[k] = {"corpus": corpus, "speaker": spk, "sex": sx, "frac": mean("frac"), "n": per[0]["n"], "vtl": mean("vtl"),
                   "vtl_sigma": mean("vtl_sigma"),
                   "Fsig": [None if any(p["Fsig"][j] is None for p in per) else float(np.mean([p["Fsig"][j] for p in per])) for j in range(4)]}
        if len(done) % 25 == 0:
            path.write_text(json.dumps(done))
            print(f"{len(done)} files", flush=True)
    path.write_text(json.dumps(done))
    print(f"analysed {len(done)} files", flush=True)


def _rel(groups: list[np.ndarray]):
    """within-speaker SD (% of the grand mean), between-speaker SD (%), reliability, from per-file log lengths"""
    g = [np.log(v) for v in groups if len(v) >= 2]
    within = np.sqrt(np.mean([np.var(v, ddof=1) for v in g]))
    means = np.array([v.mean() for v in g])
    k = np.mean([len(v) for v in g])
    between_var = max(np.var(means, ddof=1) - within ** 2 / k, 0.0)
    return 100 * within, 100 * np.sqrt(between_var), between_var / (between_var + within ** 2)


def report():
    rng = np.random.default_rng(SEED)
    d = json.loads((CACHE / "files.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["D-068: USC 75-speaker MRI audio (dev speakers, passages r1 and r2) against PTDB-TUG dev (studio), label-free.",
             "Teacher v3.1 seeds 2 and 1, seed mean per file; sonorant voiced frames. No per-speaker length is shown.", ""]
    res = {}
    for corpus in ("usc", "ptdb"):
        for sx in ("f", "m"):
            rows = [v for v in d.values() if v["corpus"] == corpus and v["sex"] == sx and v["vtl"] is not None and v["n"] >= 50]
            by: dict[str, list] = {}
            for v in rows:
                by.setdefault(v["speaker"], []).append(v["vtl"])
            groups = {s: np.array(v) for s, v in by.items()}
            w, b, rel = _rel(list(groups.values()))
            spk = list(groups)
            bs = [_rel([groups[s] for s in rng.choice(spk, len(spk))]) for _ in range(1000)]
            bs = np.array([x for x in bs if np.all(np.isfinite(x))])
            q = lambda j: f"[{np.percentile(bs[:, j], 2.5):.2f}, {np.percentile(bs[:, j], 97.5):.2f}]"  # noqa: E731
            all_rows = [v for v in d.values() if v["corpus"] == corpus and v["sex"] == sx]
            frac = np.median([v["frac"] for v in all_rows])
            sig = np.median([v["vtl_sigma"] for v in rows])
            fs = [np.median([v["Fsig"][j] for v in rows if v["Fsig"][j] is not None]) for j in range(4)]
            res[corpus, sx] = (w, rel, sig)
            lines.append(f"{corpus} {'women' if sx == 'f' else 'men'}: {len(groups)} speakers, {len(rows)} files "
                         f"(of {len(all_rows)}; files with < 50 sonorant voiced frames left out)")
            lines.append(f"  sonorant voiced fraction (median file) {frac:.2f}; VTL sigma (median, log2) {sig:.3f}; "
                         f"formant sigma F1-F4 (cents) " + " / ".join(f"{x:.0f}" for x in fs))
            lines.append(f"  within-speaker SD {w:.2f} % {q(0)}; between-speaker SD {b:.2f} % {q(1)}; reliability {rel:.2f} {q(2)}")
            if corpus == "usc":
                diffs = []
                for s in groups:
                    for p in PASSAGES:
                        a, c = d.get(f"usc:{s}_{p}_r1"), d.get(f"usc:{s}_{p}_r2")
                        if a and c and a["vtl"] and c["vtl"] and a["n"] >= 50 and c["n"] >= 50:
                            diffs.append(100 * abs(np.log(a["vtl"] / c["vtl"])))
                lines.append(f"  same passage, r1 against r2: median |difference| {np.median(diffs):.2f} % ({len(diffs)} pairs)")
            lines.append("")
    verdict = []
    for sx in ("f", "m"):
        (wu, ru, su), (wp, rp, sp) = res["usc", sx], res["ptdb", sx]
        ok = wu <= 1.5 * wp and ru >= 0.5 and su <= 1.5 * sp
        bad = wu > 2 * wp or ru < 0.3
        verdict.append("usable" if ok else ("not usable" if bad else "doubtful"))
        lines.append(f"{'women' if sx == 'f' else 'men'}: within-SD ratio USC / PTDB {wu / wp:.2f}, USC reliability {ru:.2f}, "
                     f"sigma ratio {su / sp:.2f} -> {verdict[-1]}")
    overall = "usable" if all(v == "usable" for v in verdict) else ("not usable" if "not usable" in verdict else "doubtful")
    lines.append(f"\nVERDICT: the MRI audio is {overall} for checking vocal-tract length")
    text = "\n".join(lines)
    print(text)
    (OUT / "d068.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"extract": extract, "analyse": analyse, "report": report}[sys.argv[1]]()
