"""Which invariant should the whisper level hold across tilt arms? Both candidates, measured.

(Moved from the scratchpad, 2026-11-02. Written while the anchor was still the band one; the broadband anchor
it argues for became the default in 7c8ded9, D-064.)

"as-is" as implemented matches the un-tilted render in 1-3.5 kHz (`REFERENCE_BAND_HZ`, D-063's primary
band). The alternative matches its broadband (time-domain) power, which is what the aspiration's HNR
holds. Renders each engine's /h/ clips with tilt None and with -12 dB/oct, then rescales the tilted
component by the broadband gain, and reports each variant's broadband and 1-3.5 kHz level against the
clip's own loudest 5 % of 512-sample frames (as d064_level.py does), in dB.

    python scripts/analysis/d064_level_choice.py [clips per cell, default 3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BAND = (1000.0, 3500.0)


def _frames(x, n=512, hop=256):
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, float), n)[::hop]


def _band_power(fr):
    w = np.hanning(fr.shape[-1])
    X = np.fft.rfft(fr * w, axis=-1)
    f = np.fft.rfftfreq(fr.shape[-1], 1 / 16000.0)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return (np.abs(X[..., m]) ** 2).sum(-1)


def _levels(x, audio):
    """component energy in each band, minus the clip's loudest 5 % of frames in the same band, dB"""
    fr, afr = _frames(x), _frames(audio)
    top_b = 10 * np.log10((afr ** 2).sum(1) + 1e-20)
    top_n = 10 * np.log10(_band_power(afr) + 1e-20)
    return {"bb": 10 * np.log10((fr ** 2).sum() + 1e-20) - np.percentile(top_b, 95),
            "band": 10 * np.log10(_band_power(fr).sum() + 1e-20) - np.percentile(top_n, 95)}


def _clip(engine, tilt, i):
    from dataclasses import replace

    from vocal_analysis.synth import kl_set, sampler
    from vocal_analysis.synth.sampler import SynthConfig

    cfg = SynthConfig(duration_s=(1.0, 1.0), p_h_onset=1.0, whisper_tilt_db_oct=tilt, p_tube=0.0)
    if engine == "kl":
        return kl_set.generate(i, replace(kl_set.KLSetConfig(), synth=replace(cfg, p_tube=1.0), p_lf_pulse=0.0),
                               5059, keep_components=True)
    if engine == "tube":
        return sampler.generate(i, replace(cfg, p_tube=1.0, p_interactive=1.0), 5059, keep_components=True)
    return sampler.generate(i, cfg, 5059, keep_components=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for engine in ("cascade", "tube", "kl"):
        rows, i, tries = [], 0, 0
        while len(rows) < 2 * n and tries < 60:
            a, b = _clip(engine, None, i), _clip(engine, -12.0, i)
            i += 1
            tries += 1
            if "whisper" not in a.components or "whisper" not in b.components:
                continue
            ref, tilt = a.components["whisper"], b.components["whisper"]
            g_bb = np.sqrt(np.mean(ref ** 2) / np.mean(tilt ** 2))
            for name, comp in (("as-is(1-3.5k)", tilt), ("broadband", tilt * g_bb)):
                la, lb = _levels(ref, a.audio), _levels(comp, a.audio)
                rows.append((name, lb["bb"] - la["bb"], lb["band"] - la["band"]))
        for name in ("as-is(1-3.5k)", "broadband"):
            sel = [r for r in rows if r[0] == name]
            print(f"{engine:8s} {name:14s} /h/ at -12 vs no tilt: broadband {np.median([r[1] for r in sel]):+.1f} dB, "
                  f"1-3.5 kHz {np.median([r[2] for r in sel]):+.1f} dB  (n={len(sel)})")


if __name__ == "__main__":
    main()
