"""Measure each codec and bitrate's band gain and coding noise for the training-time observability channel (D-027):
12 synthetic clips (train_v2's config, seed 777, a seed no set uses) through `codecs.apply` (delay-compensated,
fresh process per clip); per STFT bin (obs_store's analysis), the transfer H = sum Y X* / sum |X|^2 over every frame,
gain |H|^2, and coding noise sum |Y - H X|^2 / sum |X|^2 (relative to the input's power at that bin), then the mean
over +-1/12 octave around each NOISE_GRID point. The phone channel is measured the same way ("phone@0": D-027's
validation found a flat -38 dB mu-law guess not good enough). Writes vocal_analysis/synth/codec_table.json."""
import json
import sys

import numpy as np

from vocal_analysis.synth import codecs, obs_store as st
from vocal_analysis.synth.sampler import SynthConfig, generate

sys.stdout.reconfigure(encoding="utf-8")
clips = [generate(i, SynthConfig(), 777).audio.astype(np.float64) for i in range(12)]
win = st._window()
hop = 160


def stft(x):
    n = (len(x) - len(win)) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(len(win))[None, :]
    return np.fft.rfft(x[idx] * win, st.N_FFT, axis=1)


table = {}
from vocal_analysis.synth.channel import _phone  # noqa: E402

# the phone channel (band-pass, 8 kHz resampling, 8-bit mu-law) measured the same way, as "phone@0"
for name, (_, _, rates, _) in {"phone": (None, None, (0.0,), None), **codecs.CODECS}.items():
    for kbps in rates:
        sxy = sxx = see = 0.0
        ys = []
        for x in clips:
            xn = x * 0.5 / np.max(np.abs(x))
            y = _phone(xn) if name == "phone" else codecs.apply(xn, name, kbps, stream=False).astype(np.float64)
            ys.append(y)
        for x, y in zip(clips, ys):
            x = x * 0.5 / np.max(np.abs(x))
            X, Y = stft(x), stft(y[: len(x)])
            sxy = sxy + np.sum(Y * np.conj(X), 0)
            sxx = sxx + np.sum(np.abs(X) ** 2, 0)
        H = sxy / np.maximum(sxx, 1e-30)
        for x, y in zip(clips, ys):
            x = x * 0.5 / np.max(np.abs(x))
            X, Y = stft(x), stft(y[: len(x)])
            see = see + np.sum(np.abs(Y - H * X) ** 2, 0)
        gain = st._band_mean((np.abs(H) ** 2)[None, :], st.NOISE_GRID)[0]
        nsr = st._band_mean((see / np.maximum(sxx, 1e-30))[None, :], st.NOISE_GRID)[0]
        key = f"{name}@{kbps:g}"
        table[key] = {"grid": st.NOISE_GRID.tolist(), "gain_db": (10 * np.log10(np.maximum(gain, 1e-12))).round(2).tolist(),
                      "nsr_db": (10 * np.log10(np.maximum(nsr, 1e-12))).round(2).tolist()}
        g, r = np.array(table[key]["gain_db"]), np.array(table[key]["nsr_db"])
        at = lambda f, v: v[int(np.argmin(np.abs(st.NOISE_GRID - f)))]  # noqa: E731
        print(f"{key:14s} gain dB @1k/3k/5k/7k {at(1000, g):+6.1f} {at(3000, g):+6.1f} {at(5000, g):+6.1f} {at(7000, g):+6.1f}"
              f"   nsr dB {at(1000, r):+6.1f} {at(3000, r):+6.1f} {at(5000, r):+6.1f} {at(7000, r):+6.1f}", flush=True)
st.CODEC_TABLE.write_text(json.dumps(table, indent=0), encoding="utf-8")
print(f"wrote {st.CODEC_TABLE}")
