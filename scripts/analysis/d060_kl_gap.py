"""D-060: the glottal-gap growth of the F1 overread on KL renders, with harmonic capture broken (D-058's women's paired
test was diluted by files whose F1 sat on a harmonic throughout a steady vowel).

Renders as D-058 (Story's vowels, male at Story's size, female at 15.6 cm; Rosenberg oq 0.6 / sq 1.3; coupled with the
subglottal system; no nose, no noise) but with vibrato: f0(t) = f0 2^(VIB_ST sin(2 pi VIB_HZ t) / 12), so harmonics sweep
across F1. Per file the median over the middle 30-80 % of frames of the per-frame F1 error (frames the models call
voiced); paired per (vowel, f0) between gap 0.3 and gap 0; seed mean of the models.

    python scripts/analysis/d060_kl_gap.py pilot <out> name=ckpt ...   (the noise pilot: its own vowels and f0s)
    python scripts/analysis/d060_kl_gap.py main <out> name=ckpt ...
"""

import os
import sys
from pathlib import Path

import numpy as np

VIB_ST, VIB_HZ = 1.0, 5.0
GAPS = (0.0, 0.1, 0.2, 0.3)
MAIN = {"vowels": ("i", "I", "E", "ae", "V", "A", "O", "U", "o", "u"),
        "f0": {"Male": (85, 98, 110, 124, 139, 156, 175, 196), "Female": (175, 185, 196, 208, 220, 247, 262, 294)}}
PILOT = {"vowels": ("A", "i", "u"), "f0": {"Male": (92, 131), "Female": (180, 233)}, "gaps": (0.0, 0.3)}
SIZE = {"Male": (17.46, 0.12), "Female": (15.6, 0.07)}  # tract length (cm), peak glottal area (cm^2)


def render(cache: Path, sex: str, vowel: str, f0: float, gap: float) -> tuple[np.ndarray, float]:
    from vocal_analysis.synth import branches as br
    from vocal_analysis.synth import kl, tube

    path = cache / f"{sex}_{vowel}_{f0}_{gap:.1f}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], float(z["f1"])
    L, peak = SIZE[sex]
    rel = L / tube.STORY_LENGTH_CM
    s = tube.build_shape(*tube.STORY_VOWELS[vowel], L, area_scale=rel**2)
    (F, _, _), = kl.frame_labels([s])
    n = kl.FS
    frames = n // kl.HOP + 1
    t = np.arange(frames) * kl.HOP / kl.FS
    f0_t = f0 * 2.0 ** (VIB_ST * np.sin(2 * np.pi * VIB_HZ * t) / 12)
    ag = kl.glottal_area(f0_t, n, kl.FS, peak, oq=0.6, sq=1.3, gap=gap)
    a = kl.render([s] * frames, ag, 8000.0, 0.3 * rel, 1.4 * rel, subglottal=br.subglottal(L, kl.section_length()))
    x = (0.5 * a["audio"] / (np.abs(a["audio"]).max() + 1e-12)).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, f1=F[0])
    return x, float(F[0])


def file_error(model, x: np.ndarray, f1: float, dev) -> float:
    from vocal_analysis.models.infer import analyse

    r = analyse(model, x, dev)
    n = len(r["formants"])
    mid = slice(int(0.3 * n), int(0.8 * n))
    v = r["voiced_p"][mid] > 0.5 if "voiced_p" in r else np.ones(mid.stop - mid.start, bool)
    return float(np.nanmedian(r["formants"][mid, 0][v]) - f1)


def boot(v: np.ndarray, rng, n: int = 2000) -> tuple[float, float, float]:
    b = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    import torch

    from vocal_analysis.models.infer import load_teacher

    sys.stdout.reconfigure(encoding="utf-8")
    mode, out = sys.argv[1], Path(sys.argv[2])
    spec = PILOT if mode == "pilot" else MAIN
    gaps = spec.get("gaps", GAPS)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs") / f"d060_{mode}"
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[3:]}
    names = list(models)
    rows = []
    for sex in ("Male", "Female"):
        for vowel in spec["vowels"]:
            for f0 in spec["f0"][sex]:
                for gap in gaps:
                    x, f1 = render(cache, sex, vowel, f0, gap)
                    rows.append((sex, gap, vowel, f0, {k: file_error(m, x, f1, dev) for k, m in models.items()}))
    with open(out / f"{mode}_per_file.tsv", "w", encoding="utf-8") as fh:
        fh.write("sex\tgap\tvowel\tf0\t" + "\t".join(names) + "\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t" + "\t".join(f"{r[4][k]:+.1f}" for k in names) + "\n")
    rng = np.random.default_rng(60)
    mean = lambda r: float(np.mean([r[4][k] for k in names]))  # noqa: E731
    lines = [f"D-060 {mode}: F1 error (Hz, seed mean of {', '.join(names)}), median over files [95 % bootstrap]; "
             f"vibrato {VIB_ST} semitone at {VIB_HZ} Hz", ""]
    verdict = {}
    for sex in ("Male", "Female"):
        for gap in gaps:
            v = np.array([mean(r) for r in rows if r[0] == sex and r[1] == gap and np.isfinite(mean(r))])
            med, lo, hi = boot(v, rng)
            lines.append(f"{sex:6s} gap {gap:.1f}: {med:+6.1f} [{lo:+.1f}, {hi:+.1f}]  ({len(v)} files)")
        a = {(r[2], r[3]): mean(r) for r in rows if r[0] == sex and r[1] == 0.0}
        b = {(r[2], r[3]): mean(r) for r in rows if r[0] == sex and r[1] == 0.3}
        d = np.array([b[k] - a[k] for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])])
        med, lo, hi = boot(d, rng)
        verdict[sex] = (med, lo, hi)
        lines.append(f"{sex} paired gap 0.3 - 0 over {len(d)} files: {med:+.1f} [{lo:+.1f}, {hi:+.1f}] Hz; "
                     f"files with |diff| < 5 Hz: {np.mean(np.abs(d) < 5):.0%}")
        lines.append("")
    if mode == "main":
        ok = all(m >= 10 and lo > 0 for m, lo, _ in verdict.values())
        null = all(hi < 10 for _, _, hi in verdict.values())
        lines.append(f"D-060: {'GROWS' if ok else 'DOES NOT GROW' if null else 'UNDECIDED'}")
    text = "\n".join(lines)
    print(text)
    (out / f"{mode}.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
