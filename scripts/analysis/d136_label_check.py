"""D-136: are the cascade's labels for Story's shapes (`tube.poles`) what physical engines produce, and do narrow
passages explain the disagreement? Labels only (no rendering for the models), plus D-134's and D-135's stored scores
read against each engine's own resonances.

    python scripts/analysis/d136_label_check.py      -> reports/analysis/d136/d136.txt
Needs the local CHIMERA label driver ($CHIMERA_DRIVER, as D-135).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/analysis/d136"
WORK = ROOT / "data/probes/d136_labels"
DRIVER = Path(os.environ.get("CHIMERA_DRIVER", ROOT / "scratchpad/chimera_driver/target/release/chimera-driver.exe"))
LENGTHS = (13.0, 14.5, 15.6, 17.0, 18.5)
CLAMP = 0.5  # cm^2
N_PH, N_OR = 16, 17  # CHIMERA's sections at 48 kHz (D-135)
TEACHERS = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
STUDENTS = ("d128_L0_s0", "d128_L0_s1", "d128_L2_s0", "d128_L2_s1")


def _tag(v):
    return "".join(c + "^" if c.isupper() else c for c in v)


def _resample(a, n):
    x = (np.arange(n) + 0.5) / n * len(a) - 0.5
    return np.interp(x, np.arange(len(a)), a)


def _shape(areas, L):
    from vocal_analysis.synth import tube

    base = tube.build_shape(0.0, 0.0, L)
    return tube.TubeShape(areas=np.asarray(areas, float), lengths=np.full(len(areas), L / len(areas)),
                          perimeter_factor=base.perimeter_factor)


def _tube(areas, L):
    from vocal_analysis.synth import tube

    return tube.poles(_shape(areas, L), f_max_hz=5000.0)[0][:3]


def _chimera(rows):
    """{name: F1-F3} from CHIMERA's closed-glottis transfer function at 48 kHz."""
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = WORK / "jobs.csv"
    jobs.write_text("name,kind,female,f0,secs,vals\n" + "\n".join(
        f"{n},story,{float(np.clip((17.5 - L) / 3.0, 0, 1)):.4f},110,0.2,{L:g}," + ",".join(f"{a:.6f}" for a in A)
        for n, L, A in rows) + "\n", encoding="utf-8")
    subprocess.run([str(DRIVER), "jobs", str(jobs), str(WORK), "48000"], check=True, stderr=subprocess.DEVNULL)
    out = {r["name"]: np.array([float(r[k]) for k in ("F1", "F2", "F3")]) for r in csv.DictReader(open(WORK / "labels.csv"))}
    for f in WORK.glob("*.f32"):
        f.unlink()
    return out


def main() -> None:
    from vocal_analysis.synth import kl, tube

    vowels = list(tube.STORY_VOWELS)
    rows, res = [], {}
    for L in LENGTHS:
        for v in vowels:
            A = tube.vowel_areas(*tube.STORY_VOWELS[v])
            Ac = np.maximum(A, CLAMP)
            Ae = A.copy()
            Ae[:4] = np.maximum(Ae[:4], CLAMP)  # epilarynx (Story's sections 1-4) only
            A33 = np.r_[_resample(A[:22], N_PH), _resample(A[22:], N_OR)]
            k = (L, v)
            res[k] = {"tube": _tube(A, L), "tube33": _tube(A33, L), "tube_clamp": _tube(Ac, L), "tube_epi": _tube(Ae, L),
                      "min_area": A.min(), "epi_area": A[:4].mean(), "lip_area": A[-2:].mean()}
            s = tube.build_shape(*tube.STORY_VOWELS[v], L)
            (fk, _), = kl.frame_poles([s])
            (fk0, _), = kl.frame_poles([s], walls=False)
            res[k]["kl"], res[k]["kl_nowall"] = fk[:3], fk0[:3]
            sc = _shape(Ac, L)
            (fkc, _), = kl.frame_poles([sc])
            res[k]["kl_clamp"] = fkc[:3]
            for tag_, AA in (("", A), ("_clamp", Ac), ("_epi", Ae)):
                rows.append((f"L{L:g}_{_tag(v)}{tag_}", L, AA))
    ch = _chimera(rows)
    for (L, v), r in res.items():
        r["chimera"] = ch[f"L{L:g}_{_tag(v)}"]
        r["chimera_clamp"] = ch[f"L{L:g}_{_tag(v)}_clamp"]
        r["chimera_epi"] = ch[f"L{L:g}_{_tag(v)}_epi"]

    c = lambda a, b: 1200 * np.log2(a / b)  # noqa: E731
    lines = ["D-136: Story's ten shapes at 13.0-18.5 cm (five lengths); differences in cents, median over lengths, F1 / F2 / F3",
             "tube = tube.poles (the training labels); kl = the waveguide's measured closed-glottis resonances (walls on); "
             "chimera = CHIMERA's linearised network (48 kHz, 16 + 17 sections); tube33 = tube.poles of the shape resampled to "
             f"those sections; _clamp = every area raised to at least {CLAMP} cm^2; _epi = only sections 1-4", ""]

    def row(name, f):
        vals = np.array([[f(res[(L, v)]) for L in LENGTHS] for v in vowels])  # (vowel, length, 3)
        med = np.median(vals, 1)
        lines.append(f"{name}:")
        for v, m in zip(vowels, med):
            lines.append(f"  {v:3s} " + " / ".join(f"{x:+5.0f}" for x in m))
        lines.append(f"  all " + " / ".join(f"{x:+5.0f}" for x in np.median(vals.reshape(-1, 3), 0))
                     + "   (median |x|: " + " / ".join(f"{x:.0f}" for x in np.median(np.abs(vals.reshape(-1, 3)), 0)) + ")")
        return med

    d_kl = row("1. kl - tube", lambda r: c(r["kl"], r["tube"]))
    row("   kl without walls - tube", lambda r: c(r["kl_nowall"], r["tube"]))
    d_ch = row("2. chimera - tube", lambda r: c(r["chimera"], r["tube"]))
    row("3. tube33 - tube (resolution alone)", lambda r: c(r["tube33"], r["tube"]))
    row("4. chimera - tube33 (beyond resolution)", lambda r: c(r["chimera"], r["tube33"]))
    row("5. chimera_clamp - tube_clamp (narrow passages removed)", lambda r: c(r["chimera_clamp"], r["tube_clamp"]))
    row("   chimera_epi - tube_epi (epilarynx widened only)", lambda r: c(r["chimera_epi"], r["tube_epi"]))
    row("   kl_clamp - tube_clamp", lambda r: c(r["kl_clamp"], r["tube_clamp"]))
    lines.append("")
    feats = {f: np.array([np.median([res[(L, v)][f] for L in LENGTHS]) for v in vowels]) for f in ("min_area", "epi_area", "lip_area")}
    lines.append("Per vowel areas (cm^2): " + "; ".join(f"{v} min {feats['min_area'][i]:.2f} epi {feats['epi_area'][i]:.2f} lip {feats['lip_area'][i]:.2f}" for i, v in enumerate(vowels)))
    for name, d in (("chimera - tube", d_ch), ("kl - tube", d_kl)):
        for k in range(3):
            rs = {f: np.corrcoef(np.log(feats[f]), d[:, k])[0, 1] for f in feats}
            lines.append(f"  {name} F{k + 1}: correlation over vowels with log area: " + ", ".join(f"{f} {r:+.2f}" for f, r in rs.items()))
    lines.append("")

    # the models' stored readings against each truth
    lines.append("6. The models' readings (D-134 KL clips; D-135 arm 1 CHIMERA clips) against each engine's resonances and "
                 "against tube.poles: median signed error (cents) F1 / F2 / F3, and median |error|, teacher seed mean / students")
    for arm, meta_p, raw_dir, key in (("KL", ROOT / "data/probes/vowel_length_kl_v1/meta.json", ROOT / "reports/probes/vowel_length_kl", None),
                                      ("CHIMERA", ROOT / "data/probes/chimera_v1/story/meta.json", ROOT / "reports/probes/chimera", "story")):
        items = json.loads(meta_p.read_text())["items"]
        for truth in (("tube", "tube.poles"), ("kl" if arm == "KL" else "chimera", "the engine's own")):
            parts = []
            for names in (TEACHERS, STUDENTS):
                est = []
                for n in names:
                    z = np.load(raw_dir / f"{n}.npz")
                    e = z[key] if key else z["est"]
                    est.append(e[:, 1:4])
                est = np.mean(est, 0)
                T = np.array([res[(it["length"], it["vowel"])][truth[0]] for it in items])
                err = 1200 * (est - np.log2(T))
                parts.append(" / ".join(f"{np.median(err[:, k]):+4.0f}" for k in range(3)) + " (|e| "
                             + " / ".join(f"{np.median(np.abs(err[:, k])):.0f}" for k in range(3)) + ")")
            lines.append(f"  {arm} clips against {truth[1]:16s} teacher {parts[0]}; students {parts[1]}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d136.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
