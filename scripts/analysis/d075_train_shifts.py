"""D-075: the F1 coupling shift as the training sampler draws it, under each definition. Tube targets with a glottis
drawn as `coupling_dist.py` (D-023) draws them (the sampler's own speaker, glottis and target functions; train_v2c's
glottal settings); per clip (1) the cascade's leak shift F1 df sc g, (2) its cycle mean F1 df sc <a>, (3) the
quasi-static leak shift (`open_poles` at g x peak) x sc, (4) the quasi-static cycle mean x sc; also at scale 1.

    python scripts/analysis/d075_train_shifts.py [n] [--serial]
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path("reports/analysis/d075")
AREAS = np.array([0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0])  # fractions of the peak
SEED = 5075


def clip(i):
    from vocal_analysis.synth import coupling, tract, tube
    from vocal_analysis.synth.glottal import open_phase

    rng = np.random.default_rng([SEED, i])
    length = rng.uniform(13.0, 18.5)
    spk = tube.sample_speaker(rng, length)
    g = coupling.sample_glottis(rng, length)
    sc = rng.uniform(0.0, 1.0)
    rd = rng.uniform(0.3, 2.7)
    f, _, info = tube.sample_target(rng, spk, tract.N_POLES, tract.F_MAX_HZ, glottis=g)
    shape = tube.build_shape(info["q1"], info["q2"], spk.length_cm, pharynx_ratio=spk.pharynx_ratio,
                             area_scale=spk.area_scale, omega_warp=spk.omega_warp, constrict=info.get("constrict"),
                             pg=info["pg_cm"], pm=info["pm_cm"], perimeter_factor=spk.perimeter_factor)
    F, B = tube.poles(shape, tract.F_MAX_HZ)
    te = open_phase(round(rd, 2))
    gap = float(coupling.gap_from_open_quotient(te, g.gap_offset))
    frac = np.linspace(0, 1, 2000, endpoint=False)
    a = gap + (1 - gap) * np.where(frac < te, np.sin(np.pi * frac / te) ** 2, 0.0)
    f1 = np.array([F[0]] + [coupling.open_poles(shape, g.peak_area_cm2 * x, g, F, B, n=1)[0][0] for x in AREAS[1:]])
    ok = bool(np.all(np.isfinite(f1)))
    leak_q = float(np.interp(gap, AREAS, f1) - F[0]) if ok else np.nan
    mean_q = float(np.mean(np.interp(a, AREAS, f1)) - F[0]) if ok else np.nan
    df = info["open_df"][0]
    return length, F[0], sc, gap, a.mean(), df, F[0] * df * gap, F[0] * df * a.mean(), leak_q, mean_q


def q(x):
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:+6.1f}" for v in np.percentile(x, [10, 25, 50, 75, 90]))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 2000
    if "--serial" in sys.argv:
        rows = [clip(i) for i in range(n)]
    else:
        with ProcessPoolExecutor(6) as ex:
            rows = list(ex.map(clip, range(n), chunksize=20))
    r = np.array(rows, float)
    L, F1, sc, gap, am, df, c_leak, c_mean, q_leak, q_mean = r.T
    lines = [f"D-075: F1 coupling shifts as the sampler draws them ({n} tube targets, train_v2c's glottal settings, "
             "draws as coupling_dist.py). Hz; percentiles 10 / 25 / 50 / 75 / 90", ""]
    for name, k in (("short tract <15.5 cm", L < 15.5), ("long tract >=15.5 cm", L >= 15.5)):
        lines.append(f"{name} (n={k.sum()}): median gap {np.median(gap[k]):.2f}, <a> {np.median(am[k]):.2f}, "
                     f"F1 {np.median(F1[k]):.0f} Hz; quasi-static continuation failed on {int(np.sum(~np.isfinite(q_leak[k])))}")
        for lab, x in (("(1) cascade leak, scale 1   ", c_leak), ("(2) cascade mean, scale 1   ", c_mean),
                       ("(3) quasi-static leak, sc. 1", q_leak), ("(4) quasi-static mean, sc. 1", q_mean)):
            lines.append(f"  {lab} {q(x[k])}   as v2c (x sc): {q(x[k] * sc[k])}")
        m = k & np.isfinite(q_leak) & (np.abs(q_leak) > 0.5)
        ratio = float(np.median(c_leak[m] / q_leak[m]))
        lines.append(f"  cascade / quasi-static leak, median ratio per clip (|quasi-static| > 0.5 Hz, n={m.sum()}): {ratio:.2f}"
                     f" -> {'misstates (> 30 %)' if abs(ratio - 1) > 0.3 else 'within 30 %'}")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "clips.npy", r)
    (OUT / "d075.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
