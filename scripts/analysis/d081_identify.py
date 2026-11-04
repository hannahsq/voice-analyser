"""D-081 I: is the cascade's gap offset audible once the cycle-mean tract is matched? Per coupled cascade clip
(train_v2ko's settings, seed 8101), variants with the clip's gap offset forced (the same draws otherwise):
R (offset -0.3), U (+0.1), MF (+0.1, closed poles rescaled per control point so the cycle-mean F1-F4 equal R's),
MFB (MF plus matched bandwidths), F1x1.01 and F1x1.03 (R with F1 scaled). The rescaled tract is the clip's tract, so
aspiration goes through it too. Measure: RMS of the VQT log-power difference (dB) against R on R's voiced frames,
the clip's median difference removed; all bins and below 1.5 kHz. With the clip's aspiration and without (hnr_db None).

    python scripts/analysis/d081_identify.py [n_clips] [workers]
"""

import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

O = Path("reports/analysis/d081")
SEED, LO, HI = 8101, -0.3, 0.1
VARIANTS = ("U", "MF", "MFB", "F1x1.01", "F1x1.03")
REPORTED = ("MFB*",)
STATE = {"mode": "R"}


def _wrapped(orig):
    from scipy.ndimage import uniform_filter1d

    from vocal_analysis.synth import voice
    from vocal_analysis.synth.coupling import N_COUPLED, gap_from_open_quotient

    def synth(traj, rng, coupling=None, **kw):
        mode = STATE["mode"]
        if coupling is None or mode in ("R", "U"):
            return orig(traj, rng, coupling=coupling, **kw)
        pf, pb = traj.poles_f.copy(), traj.poles_b.copy()
        if mode.startswith("F1x"):
            pf[:, 0] *= float(mode[3:])
        else:
            st = copy.deepcopy(rng.bit_generator.state)
            cycles = orig(traj, rng, coupling=coupling, **kw)["cycles"]
            rng.bit_generator.state = st
            te_s = voice._open_phase_te(traj)
            idx = np.minimum(np.arange(len(pf)) * voice.CTRL, len(te_s) - 1)

            def mean_a(off):
                a = voice.glottal_opening(cycles, te_s, gap_from_open_quotient(te_s, off))
                return uniform_filter1d(a, voice.CTRL * 10)[idx][:, None]  # the 10 ms cycle mean, at control points

            a_lo, a_hi = mean_a(LO), mean_a(coupling["gap_offset"])
            m = min(N_COUPLED, coupling["df"].shape[1])
            df, bm = coupling["df"][:, :m], coupling["bm"][:, :m]
            pf[:, :m] *= (1 + df * a_lo) / (1 + df * a_hi)
            if mode == "MFB":
                pb[:, :m] *= (1 + (bm - 1) * a_lo) / (1 + (bm - 1) * a_hi)
                pf[:, 0] *= STATE.get("s", 1.0)  # the best-fit search (reported): F1 and B1 scaled on top of the match
                pb[:, 0] *= STATE.get("b", 1.0)
        return orig(replace(traj, poles_f=pf, poles_b=pb), rng, coupling=coupling, **kw)

    return synth


def _vqt(x, n):
    import torch

    from vocal_analysis.features.vqt import TorchVQT

    lp = TorchVQT(torch.device("cpu")).log_power(torch.from_numpy(x.astype(np.float32))[None], n)[0].numpy()
    return lp if lp.shape[0] == n else lp.T  # (frames, bins)


def one(job):
    import torch

    torch.set_num_threads(1)
    import d078_render

    from vocal_analysis.features.vqt import bin_frequencies
    from vocal_analysis.synth import sampler

    if not hasattr(sampler, "_d081_orig"):
        sampler._d081_orig = sampler.synthesize
        sampler.synthesize = _wrapped(sampler._d081_orig)
    i, asp = job
    base = d078_render.config()
    if not asp:
        base = replace(base, hnr_db=None)
    low = bin_frequencies() < 1500
    out = {}

    def render(mode, off):
        STATE["mode"] = mode
        return sampler.generate(i, replace(base, glottal_gap_offset=(off, off)), SEED)

    r = render("R", LO)
    if "gap_offset" not in r.labels:  # interactive or uncoupled: not this check's clip
        return i, asp, None
    n = len(r.labels["f0"])
    v = np.asarray(r.labels["voiced"], bool)
    lr = _vqt(r.audio, n)
    for mode in VARIANTS:
        c = render(mode, HI if mode in ("U", "MF", "MFB") else LO)
        d = (_vqt(c.audio, n) - lr)[v]
        d -= np.median(d)
        out[mode] = (float(np.sqrt(np.mean(d ** 2))), float(np.sqrt(np.mean(d[:, low] ** 2))))

    # Reported (added after a 4-clip smoke test, rule unchanged): the closed F1 and B1 that bring the +0.1 render
    # closest to R (all bins), searched on top of MFB: F1 x s in [0.9, 1.1], B1 x b in [0.5, 2], s again.
    from scipy.optimize import minimize_scalar

    def dist(s_, b_):
        STATE["s"], STATE["b"] = s_, b_
        d = (_vqt(render("MFB", HI).audio, n) - lr)[v]
        d -= np.median(d)
        return float(np.sqrt(np.mean(d ** 2))), float(np.sqrt(np.mean(d[:, low] ** 2)))

    s_ = minimize_scalar(lambda x: dist(x, 1.0)[0], bounds=(0.9, 1.1), method="bounded", options={"xatol": 2e-3, "maxiter": 12}).x
    lb = minimize_scalar(lambda x: dist(s_, np.exp(x))[0], bounds=(np.log(0.5), np.log(2.0)), method="bounded",
                         options={"xatol": 0.02, "maxiter": 10}).x
    s_ = minimize_scalar(lambda x: dist(x, np.exp(lb))[0], bounds=(0.9, 1.1), method="bounded", options={"xatol": 2e-3, "maxiter": 10}).x
    out["MFB*"] = dist(s_, np.exp(lb))
    out["MFB*_s"], out["MFB*_b"] = float(s_), float(np.exp(lb))
    STATE.pop("s"), STATE.pop("b")
    out["f1"] = float(np.median(r.labels["formants"][v, 0]))
    out["f0"] = float(np.median(r.labels["f0"][v]))
    return i, asp, out


if __name__ == "__main__":
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "summary" else 60
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    O.mkdir(parents=True, exist_ok=True)
    res = {True: [], False: []}
    i0 = 0
    if sys.argv[1:2] == ["summary"]:  # re-summarise identify.json (no rendering)
        res = {k == "True": v for k, v in json.loads((O / "identify.json").read_text()).items()}
        n_clips = 0
    with ProcessPoolExecutor(workers) as ex:
        while min(len(v) for v in res.values()) < n_clips:
            jobs = [(i, a) for i in range(i0, i0 + 2 * workers) for a in (True, False)]
            i0 += 2 * workers
            for i, asp, out in ex.map(one, jobs):
                if out is not None and len(res[asp]) < n_clips:
                    res[asp].append({"index": i, **out})
    if n_clips:
        (O / "identify.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
    # clips with no voiced frames have nothing to compare (NaN throughout): left out, and counted
    n_all = {k: len(v) for k, v in res.items()}
    res = {k: [r for r in v if np.isfinite(r["f0"])] for k, v in res.items()}
    L = ["D-081 I: VQT log-power difference against R (offset -0.3), RMS dB on voiced frames, median over clips "
         "(all bins / below 1.5 kHz)", f"variants: U = offset +0.1; MF / MFB = +0.1 with the cycle-mean F (and B) matched", ""]
    verdict = None
    for asp in (False, True):
        rows = res[asp]
        med = {m: np.median([r[m][0] for r in rows]) for m in (*VARIANTS, *REPORTED)}
        lo = {m: np.median([r[m][1] for r in rows]) for m in (*VARIANTS, *REPORTED)}
        L.append(f"{'with' if asp else 'without'} aspiration ({len(rows)} clips; {n_all[asp] - len(rows)} without voiced frames left out):")
        L += [f"  {m:8s} {med[m]:.3f} / {lo[m]:.3f}" for m in (*VARIANTS, *REPORTED)]
        L.append(f"  MFB* (best-fit F1 / B1 on top of MFB, reported): median F1 x {np.median([r['MFB*_s'] for r in rows]):.3f}, "
                 f"B1 x {np.median([r['MFB*_b'] for r in rows]):.2f}; MFB* / F1x1.01 per clip "
                 f"{np.median([r['MFB*'][0] / r['F1x1.01'][0] for r in rows]):.2f}, ratio of medians {med['MFB*'] / med['F1x1.01']:.2f}")
        rat = lambda m, k: np.median([r[m][k] / r["F1x1.01"][k] for r in rows])  # noqa: E731
        L.append(f"  per clip, MFB / F1x1.01: {rat('MFB', 0):.2f} / {rat('MFB', 1):.2f}; MF / F1x1.01: {rat('MF', 0):.2f} / "
                 f"{rat('MF', 1):.2f}; MFB / U: {np.median([r['MFB'][0] / r['U'][0] for r in rows]):.2f}")
        if not asp:
            x = med["MFB"] / med["F1x1.01"]
            verdict = "audible" if x >= 1 else "not audible" if x < 0.25 else "weak"
        L.append("")
    L.append(f"I (without aspiration, MFB against F1 x 1.01, all bins): {verdict}")
    (O / "identify.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
