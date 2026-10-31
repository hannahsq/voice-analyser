"""D-058: does the KL waveguide reproduce OPENGLOT Repository II's growth of the F1 overread with the glottal gap?

KL arms: Story's vowels (tube.STORY_VOWELS a / i / u / ae), male at Story's size and female at 15.6 cm (areas by
size^2), OPENGLOT II's f0 grids, closed-phase gap 0 / 0.1 / 0.2 / 0.3, coupled with the subglottal system, no nose, no
jitter or noise; labels kl.frame_labels. OPENGLOT II (evaluation only; dev; fits nothing): its 96 files against Table 2.
Per file the median F1 over the middle 30-80 % minus the label; files where a model is over 150 Hz off left out.
Args: out_dir name=ckpt ... (VA_DEVICE for the models). Renders cached in runs/d058/."""
import os
import re
import sys
from pathlib import Path

import numpy as np

FR1 = {"Male": {"a": 752, "i": 340, "u": 367, "ae": 693}, "Female": {"a": 848, "i": 379, "u": 420, "ae": 795}}  # Table 2
OG_ARMS = {"Male": {"small": 0.03, "medium": 0.06, "large": 0.09}, "Female": {"small": 0.05, "medium": 0.07, "large": 0.09}}
VOWELS = {"a": "A", "i": "i", "u": "u", "ae": "ae"}
GAPS = (0.0, 0.1, 0.2, 0.3)
CACHE = Path("runs/d058")


def kl_sex():
    from vocal_analysis.synth import tube

    return {"Male": dict(L=tube.STORY_LENGTH_CM, f0=(82, 110, 156, 220), peak=0.12),
            "Female": dict(L=15.6, f0=(175, 196, 220, 294), peak=0.07)}


def render(sex: str, vowel: str, f0: float, gap: float) -> tuple[np.ndarray, float]:
    from vocal_analysis.synth import branches as br
    from vocal_analysis.synth import kl, tube

    path = CACHE / f"{sex}_{vowel}_{f0}_{gap:.1f}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], float(z["f1"])
    p = kl_sex()[sex]
    rel = p["L"] / tube.STORY_LENGTH_CM
    s = tube.build_shape(*tube.STORY_VOWELS[VOWELS[vowel]], p["L"], area_scale=rel**2)
    (F, _, _), = kl.frame_labels([s])
    n = kl.FS
    frames = n // kl.HOP + 1
    ag = kl.glottal_area(np.full(frames, float(f0)), n, kl.FS, p["peak"], oq=0.6, sq=1.3, gap=gap)
    a = kl.render([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel, subglottal=br.subglottal(p["L"], kl.section_length()))
    x = (0.5 * a["audio"] / (np.abs(a["audio"]).max() + 1e-12)).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, f1=F[0])
    return x, float(F[0])


def mid_f1(model, x: np.ndarray, dev) -> float:
    from vocal_analysis.models.infer import analyse

    n = len(x) // 160 + 1
    return float(np.nanmedian(analyse(model, x, dev)["formants"][int(0.3 * n): int(0.8 * n), 0]))


def boot(v: np.ndarray, rng, n: int = 2000) -> tuple[float, float, float]:
    b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import load_teacher

    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
    rows = []  # (set, sex, arm, vowel, f0, {model: error})
    for sex, p in kl_sex().items():
        for vowel in VOWELS:
            for f0 in p["f0"]:
                for gap in GAPS:
                    x, f1 = render(sex, vowel, f0, gap)
                    rows.append(("KL", sex, gap, vowel, f0, {k: mid_f1(m, x, dev) - f1 for k, m in models.items()}))
    for f in sorted(Path("data/openglot").glob("RepositoryII_*/**/*.wav")):
        m = re.match(r"(Male|Female)_([A-Za-z]+)_(\d+)Hz_(small|medium|large)$", f.stem)
        if not m:
            continue
        sex, vowel, f0 = m.group(1), m.group(2).lower(), int(m.group(3))
        x, sr = sf.read(f)
        x = x[:, 0] if x.ndim == 2 else x
        x16 = resample_poly(x, 16000, sr).astype(np.float32)
        x16 = 0.5 * x16 / (np.abs(x16).max() + 1e-12)
        rows.append(("OPENGLOT", sex, OG_ARMS[sex][m.group(4)], vowel, f0,
                     {k: mid_f1(mod, x16, dev) - FR1[sex][vowel] for k, mod in models.items()}))
    names = list(models)
    with open(out / "per_file.tsv", "w", encoding="utf-8") as fh:
        fh.write("set\tsex\tarm\tvowel\tf0\t" + "\t".join(names) + "\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t" + "\t".join(f"{r[5][k]:+.1f}" for k in names) + "\n")

    rng = np.random.default_rng(58)
    kept = [r for r in rows if all(abs(r[5][k]) <= 150 for k in names)]
    mean = lambda r: float(np.mean([r[5][k] for k in names]))  # noqa: E731
    lines = [f"D-058: F1 error (Hz, model minus label; seed mean of {', '.join(names)}), median over files [95 % "
             f"bootstrap]. {len(rows) - len(kept)} of {len(rows)} files left out (a model over 150 Hz off).", ""]
    summary = {}
    for st in ("KL", "OPENGLOT"):
        for sex in ("Male", "Female"):
            for arm in sorted({r[2] for r in kept if r[0] == st and r[1] == sex}):
                sel = [r for r in kept if r[0] == st and r[1] == sex and r[2] == arm]
                v = np.array([mean(r) for r in sel])
                per = "  ".join(f"{k} {np.median([r[5][k] for r in sel]):+.0f}" for k in names)
                med, lo, hi = boot(v, rng)
                summary[(st, sex, arm)] = med
                lines.append(f"{st:8s} {sex:6s} arm {arm:<4}: {med:+6.1f} [{lo:+.1f}, {hi:+.1f}]  ({per}; {len(sel)} files)")
        lines.append("")
    verdict = {}
    for sex in ("Male", "Female"):
        a = {(r[3], r[4]): mean(r) for r in kept if r[0] == "KL" and r[1] == sex and r[2] == 0.0}
        b = {(r[3], r[4]): mean(r) for r in kept if r[0] == "KL" and r[1] == sex and r[2] == 0.3}
        d = np.array([b[k] - a[k] for k in a if k in b])
        med, lo, hi = boot(d, rng)
        verdict[sex] = med >= 10 and lo > 0
        lines.append(f"KL {sex}: gap 0.3 minus gap 0, paired over {len(d)} files: {med:+.1f} [{lo:+.1f}, {hi:+.1f}] Hz"
                     f" -> {'meets' if verdict[sex] else 'misses'} (1)")
    og = summary.get(("OPENGLOT", "Female", 0.09))
    klw = summary.get(("KL", "Female", 0.3))
    ratio = klw / og if og else float("nan")
    two = og is not None and 0.5 <= ratio <= 2.0
    lines.append(f"Women's widest gap: KL {klw:+.1f} against OPENGLOT II {og:+.1f} Hz, ratio {ratio:.2f} -> "
                 f"{'meets' if two else 'misses'} (2)")
    one = verdict["Male"] and verdict["Female"]
    lines.append(f"D-058: {'REPRODUCES' if one and two else 'PARTLY (1 only)' if one else 'DOES NOT REPRODUCE'}")
    text = "\n".join(lines)
    print(text)
    (out / "d058.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
