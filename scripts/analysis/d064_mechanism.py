"""D-064 preparation (not registered; see the note under D-063): confirm the mechanism of the /h/ level mismatch.

Claim to test: `whisper_noise` holds the *source's* time-domain power constant across tilts, so a tilt change moves
power between bands; the level that survives to the output is that source weighted by the engine's tract, and the
three engines' tracts weight it differently. Hence the measured /h/ swing of +19 / -4 / -16 dB (as-is to -12).

Per engine x tilt, the /h/ onsets of one clip per seed: the whisper component's band energy at each /h/ segment
against the voiced component's band energy at the clip's loudest frame (a level ratio immune to the clip gain and to
the vowel's level). Also: the source-only in-band power ratio (`whisper_noise` on white noise), and for the cascade
and the tube the same ratio after the /h/ frame's tract (all-pole from the label formants; the interactive path's
captured impulse response h). If predicted tracks measured, the mechanism is the source normalisation being weighted
by the tract.

    python scripts/analysis/d064_mechanism.py [clips per cell, default 4]

Note (2026-11-01, after 7c8ded9): numbers from before the level fix were read from this script's output as they came;
after the fix the KL row first read +0.0 at every tilt because it took `cap["flows"][-1]`, and the fix renders the
as-is reference after the tilted render, so that flow is tilt-independent by construction. The KL branch now asks
`generate(..., keep_components=True)` for the rendered components instead, as the cascade and tube branches do.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

FS = 16000
BANDS = {"broadband": (0.0, 8000.0), "0.1-1k": (100.0, 1000.0), "1-5k": (1000.0, 5000.0),
         "1-3.5k": (1000.0, 3500.0)}


def band_pow(x, band, fs=FS) -> float:
    x = np.asarray(x, float)
    if len(x) < 8:
        return 0.0
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(spec[(f >= band[0]) & (f <= band[1])].sum())


def db(x) -> float:
    return 10.0 * np.log10(max(float(x), 1e-30))


def _clip(args):
    engine, t, i = args
    from dataclasses import replace

    import d063_h_noise as H
    from vocal_analysis.synth import interactive as I
    from vocal_analysis.synth import kl as KL
    from vocal_analysis.synth import kl_set as K
    from vocal_analysis.synth import sampler as S

    cap: dict = {"comps": None, "flows": []}

    def wrap(mod, name):
        orig = getattr(mod, name)

        def w(*a, **k):
            out = orig(*a, **k)
            cap["comps"] = out
            return out

        setattr(mod, name, w)

    if engine == "kl":
        orig_r = KL.render

        def r(*a, **k):
            out = orig_r(*a, **k)
            cap["flows"].append((k.get("flow") is not None, out["audio"]))
            return out

        K.kl.render = r
    elif engine == "tube":
        wrap(I, "synthesize")
        orig_tr = I.tract_responses

        def tr(*a, **k):
            out = orig_tr(*a, **k)
            cap["flows"].append(out.h)
            return out

        I.tract_responses = tr
    else:
        wrap(S, "synthesize")

    base = replace(H._load("d059_realness").synth_config(), p_h_onset=1.0, asp_tilt_db_oct=t, whisper_tilt_db_oct=t)
    if engine == "kl":
        c = K.generate(i, replace(K.KLSetConfig(), synth=replace(base, p_tube=1.0), p_lf_pulse=0.0), H.SEED,
                       keep_components=True)
    elif engine == "tube":
        c = S.generate(i, replace(base, p_tube=1.0, p_interactive=1.0), H.SEED)
    else:
        c = S.generate(i, replace(base, p_tube=0.0), H.SEED)

    hs = [(g["start_s"], g["end_s"]) for g in c.meta["segments"] if g.get("kind") == "h_onset" and
          g["end_s"] - g["start_s"] >= 0.06]
    x = np.asarray(c.audio, float)
    if engine == "kl":
        # the rendered components, not the last captured flow: D-064's level anchor renders the as-is reference
        # after the tilted one, so `cap["flows"][-1]` is tilt-independent by construction
        voiced = np.asarray(c.components["voiced"], float)
        whisper = np.asarray(c.components["whisper"], float)
        h_row = None
    else:
        voiced = np.asarray(cap["comps"]["voiced"], float)
        whisper = np.asarray(cap["comps"]["whisper"], float)
        h_row = cap["flows"][-1] if engine == "tube" and cap["flows"] else None

    vfr = np.lib.stride_tricks.sliding_window_view(np.asarray(voiced, float), 512)[::256]
    loud = int(np.argmax((vfr ** 2).sum(1))) * 256  # the loudest *voiced* frame, the level reference
    formants = c.labels.get("formants")
    bws = c.labels.get("bandwidths")
    fi = None
    if hs:
        mid = (hs[0][0] + hs[0][1]) / 2
        fi = int(np.argmin(np.abs(np.asarray(c.labels["t"]) - mid)))
    return {"engine": engine, "tilt": t, "seed": i, "x": x, "voiced": voiced, "whisper": whisper,
            "hs": hs, "loud": loud, "h_row": h_row,
            "F": None if formants is None else formants[fi], "B": None if bws is None else bws[fi]}


def _source_ratio(t):
    from vocal_analysis.synth.voice import whisper_noise

    rng = np.random.default_rng(7)
    w = rng.standard_normal(FS * 4 + 1)
    y0, yt = np.diff(w), whisper_noise(w, t)
    return {k: db(band_pow(yt, b) / band_pow(y0, b)) for k, b in BANDS.items()}


def _predicted(row, t):
    """The cascade's and the tube's predicted in-band ratio through the /h/ frame's tract."""
    from vocal_analysis.synth.voice import whisper_noise

    rng = np.random.default_rng(7)
    w = rng.standard_normal(len(row["x"]) + 1)
    y0, yt = np.diff(w), whisper_noise(w, t)
    if row["engine"] == "cascade":
        from vocal_analysis.synth.voice import Trajectories, tract_filter

        n = len(y0)
        c = n // 16 + 1  # control-rate, as the sampler's trajectories are
        F, B = np.asarray(row["F"], float), np.asarray(row["B"], float)
        ok = np.isfinite(F) & np.isfinite(B)
        F, B = F[ok][None, :], B[ok][None, :]
        tr = Trajectories(f0=np.full(c, 120.0), rd=np.ones(c), voice_amp=np.ones(c), whisper_amp=np.ones(c),
                          fric_amp=np.zeros(c), poles_f=np.tile(F, (c, 1)), poles_b=np.tile(B, (c, 1)),
                          zeros_f=np.zeros((c, 0)), zeros_b=np.zeros((c, 0)))
        y0, yt = tract_filter(y0, tr), tract_filter(yt, tr)
    elif row["engine"] == "tube" and row["h_row"] is not None:
        h = np.asarray(row["h_row"], float)

        def conv(v):
            n = len(v) + len(h)
            m = 1 << int(np.ceil(np.log2(n)))
            return np.fft.irfft(np.fft.rfft(v, m) * np.fft.rfft(h, m), m)[: len(v)]

        y0, yt = conv(y0), conv(yt)
    else:
        return None
    return {k: db(band_pow(yt, b) / band_pow(y0, b)) for k, b in BANDS.items()}


def _map(jobs):
    """Parallel when the sandbox allows process pipes, serial otherwise (the DSH sandbox denies them)."""
    from concurrent.futures import ProcessPoolExecutor

    try:
        with ProcessPoolExecutor(8) as ex:
            return list(ex.map(_clip, jobs))
    except (PermissionError, OSError):
        return [_clip(j) for j in jobs]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tilts = (None, 6.0, -12.0)
    print("source-only in-band power ratio vs as-is:")
    for t in tilts[1:]:
        r = _source_ratio(t)
        print(f"  tilt {t:+.0f}: " + "  ".join(f"{k} {v:+.1f} dB" for k, v in r.items()))
    print()
    for e in ("cascade", "tube", "kl"):
        rows = []
        for t in tilts:
            rows += _map([(e, t, i) for i in range(n)])
        print(f"{e}:  whisper component against the voiced component at the clip's loudest frame, per /h/ segment")
        med = {}
        for t in tilts:
            sel = [r for r in rows if r["tilt"] == t]
            lev = {}
            for k, bd in BANDS.items():
                vals = []
                for r in sel:
                    ref = band_pow(r["voiced"][r["loud"]:r["loud"] + 512], bd)
                    if ref <= 0:
                        continue
                    for a, b in r["hs"]:
                        c = int((a + b) / 2 * FS)
                        vals.append(db(band_pow(r["whisper"][c - 256:c + 256], bd) / ref))
                lev[k] = float(np.median(vals))
            med[t] = lev
            print(f"  tilt {str(t):>4}: " + "  ".join(f"{k} {v:+.1f}" for k, v in lev.items()))
        for t in tilts[1:]:
            d = {k: med[t][k] - med[None][k] for k in BANDS}
            pred = [_predicted(r, t) for r in rows if r["tilt"] == t for _ in r["hs"]]
            pr = ""
            if pred and pred[0]:
                pr = "  predicted " + "  ".join(f"{k} {np.median([p[k] for p in pred]):+.1f}" for k in BANDS)
            print(f"  swing {t:+.0f} vs as-is: " + "  ".join(f"{k} {v:+.1f}" for k, v in d.items()) + pr)


if __name__ == "__main__":
    main()
