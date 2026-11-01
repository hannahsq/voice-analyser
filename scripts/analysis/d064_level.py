"""D-064 preparation (not registered; see the note under D-063): the synthetic /h/ level against the clip's loudest
frames, per engine and noise tilt, to check that the whisper / aspiration tilt switch changes only the noise's shape.
Clips come from d063_h_noise._synth_clip (every voiced segment with an /h/ onset). Per /h/ segment: the energy of
512 samples at its centre minus the 95th percentile of the clip's 512-sample frame energies (hop 256), dB, and the
same in 1-3.5 kHz; the median over segments is printed. Seen 2026-11-01 (8 clips each), as-is / -12 / +6 dB/oct
before the D-064 fix (legacy level policy): cascade -23.3 / -4.1 / -25.0, tube -32.4 / -36.3 / -32.5,
KL -31.1 / -47.2 / -31.2.

After the D-064 fix (same clips and seeds; broadband anchor, so the whisper component's power is now the as-is
render's at every tilt): broadband cascade -23.3 / -19.0 / -23.6, tube -32.4 / -32.3 / -32.1, KL -31.1 / -29.3 /
-30.8; 1-3.5 kHz cascade -17.5 / -32.1 / -17.6, tube -40.2 / -43.1 / -42.0, KL -37.6 / -35.5 / -34.0. The -12
level swing fell from 19.2 / -3.9 / -16.1 dB to +4.3 / +0.1 / +1.8 dB, and the engine spread at -12 from 43 to
13 dB. The residual +4.3 dB on the cascade is the measure's, not the component's: the clip gain is normalised to the
peak, and the low-frequency-weighted noise has a higher peak factor. In band the tilt still moves the level by its
shape: -14.6 dB at -12 on the cascade, so a D-064 arm at steep tilts will need a floor against the vowel's tail.

    python scripts/analysis/d064_level.py [clips per cell, default 8] [--serial]

`--serial` avoids the process pool (it is how this ran in the DSH sandbox, which denies the pipes a pool needs).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

BAND_HZ = (1000.0, 3500.0)


def _levels(res) -> tuple[list[float], list[float]]:
    """broadband and 1-3.5 kHz levels of each /h/ window against the clip's own loudest 5 %"""
    bb, bd = [], []
    for _, _, _, x, _, hs in res:
        x = np.asarray(x, float)
        fr = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
        top = np.percentile(10 * np.log10((fr ** 2).sum(1) + 1e-20), 95)
        w = np.hanning(512)
        f = np.fft.rfftfreq(512, 1 / 16000.0)
        sel = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
        band_frames = (np.abs(np.fft.rfft(fr * w, axis=-1)) ** 2)[:, sel].sum(1)
        top_bd = np.percentile(10 * np.log10(band_frames + 1e-20), 95)
        for a, b in hs:
            c = int((a + b) / 2 * 16000)
            win = x[c - 256:c + 256]
            bb.append(10 * np.log10((win ** 2).sum() + 1e-20) - top)
            bd.append(10 * np.log10((np.abs(np.fft.rfft(win * w)) ** 2)[sel].sum() + 1e-20) - top_bd)
    return bb, bd


def main():
    import d063_h_noise as H

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = int(args[0]) if args else 8
    serial = "--serial" in sys.argv
    for e in ("cascade", "tube", "kl"):
        for t in (None, -12.0, 6.0):
            jobs = [(e, t, i) for i in range(n)]
            if serial:
                res = [H._synth_clip(j) for j in jobs]
            else:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(8) as ex:
                    res = list(ex.map(H._synth_clip, jobs))
            bb, bd = _levels(res)
            print(f"{e} tilt {t}: /h/ level re the loudest 5 %: median {np.median(bb):.1f} dB broadband, "
                  f"{np.median(bd):.1f} dB in 1-3.5 kHz ({len(bb)} segments)")


if __name__ == "__main__":
    main()
