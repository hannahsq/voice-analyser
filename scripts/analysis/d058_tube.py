"""D-058 amendment 1 (reported only): D-058's KL arms rendered by the interactive source instead (the training
synthesiser's source-tract interaction: `interactive.render` through `tract_responses` of the same Story shape, its own
subglottal load), labelled tube.poles, read by the same models. Args: out_dir name=ckpt ... (VA_DEVICE)."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

CACHE = Path("runs/d058_tube")


def _d058():
    spec = importlib.util.spec_from_file_location("d058", Path(__file__).with_name("d058_kl_openglot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render(d058, sex: str, vowel: str, f0: float, gap: float) -> tuple[np.ndarray, float]:
    from vocal_analysis.synth import interactive, kl, tube
    from vocal_analysis.synth.coupling import Glottis
    from vocal_analysis.synth.voice import CTRL

    path = CACHE / f"{sex}_{vowel}_{f0}_{gap:.1f}.npz"
    if path.exists():
        z = np.load(path)
        return z["x"], float(z["f1"])
    p = d058.kl_sex()[sex]
    rel = p["L"] / tube.STORY_LENGTH_CM
    s = tube.build_shape(*tube.STORY_VOWELS[d058.VOWELS[vowel]], p["L"], area_scale=rel**2)
    F, _ = tube.poles(s, 5000.0)
    n = kl.FS
    ag = kl.glottal_area(np.full(n // kl.HOP + 1, float(f0)), n, kl.FS, p["peak"], oq=0.6, sq=1.3, gap=gap)
    g = Glottis(peak_area_cm2=p["peak"], gap_offset=0.0, length_cm=1.4 * rel, depth_cm=0.3 * rel, pressure=8000.0,
                sub_scale=1.0 / rel)
    resp = interactive.tract_responses(s, kl.FS, sub_scale=g.sub_scale)
    nb = n // CTRL + 1
    y = interactive.render(ag, [resp], np.zeros(nb, np.int64), np.zeros(nb), g, CTRL, float(kl.FS), True)["pressure"]
    x = (0.5 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, f1=F[0])
    return x, float(F[0])


def main():
    import torch

    from vocal_analysis.models.infer import load_teacher

    sys.stdout.reconfigure(encoding="utf-8")
    d058 = _d058()
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], dev)[0] for a in sys.argv[2:]}
    names = list(models)
    rows = []
    for sex, p in d058.kl_sex().items():
        for vowel in d058.VOWELS:
            for f0 in p["f0"]:
                for gap in d058.GAPS:
                    x, f1 = render(d058, sex, vowel, f0, gap)
                    rows.append((sex, gap, vowel, f0, {k: d058.mid_f1(m, x, dev) - f1 for k, m in models.items()}))
    kept = [r for r in rows if all(abs(r[4][k]) <= 150 for k in names)]
    mean = lambda r: float(np.mean([r[4][k] for k in names]))  # noqa: E731
    rng = np.random.default_rng(58)
    lines = [f"D-058 amendment 1 (reported only): interactive-source tube clips on D-058's arms. F1 error (Hz, seed mean of "
             f"{', '.join(names)}), median over files [95 % bootstrap]; {len(rows) - len(kept)} of {len(rows)} files left out.", ""]
    for sex in ("Male", "Female"):
        for gap in d058.GAPS:
            sel = [r for r in kept if r[0] == sex and r[1] == gap]
            med, lo, hi = d058.boot(np.array([mean(r) for r in sel]), rng)
            per = "  ".join(f"{k} {np.median([r[4][k] for r in sel]):+.0f}" for k in names)
            lines.append(f"tube     {sex:6s} arm {gap:<4}: {med:+6.1f} [{lo:+.1f}, {hi:+.1f}]  ({per}; {len(sel)} files)")
        a = {(r[2], r[3]): mean(r) for r in kept if r[0] == sex and r[1] == 0.0}
        b = {(r[2], r[3]): mean(r) for r in kept if r[0] == sex and r[1] == 0.3}
        med, lo, hi = d058.boot(np.array([b[k] - a[k] for k in a if k in b]), rng)
        lines.append(f"tube {sex}: gap 0.3 minus gap 0, paired: {med:+.1f} [{lo:+.1f}, {hi:+.1f}] Hz")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (out / "tube.txt").write_text(text + "\n", encoding="utf-8")
    with open(out / "tube_per_file.tsv", "w", encoding="utf-8") as fh:
        fh.write("sex\tarm\tvowel\tf0\t" + "\t".join(names) + "\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t" + "\t".join(f"{r[4][k]:+.1f}" for k in names) + "\n")


if __name__ == "__main__":
    main()
