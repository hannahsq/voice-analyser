"""D-064 preparation (not registered; see the note under D-063): the synthetic /h/ level against the clip's loudest
frames, per engine and noise tilt, to check that the whisper / aspiration tilt switch changes only the noise's shape.
Clips come from d063_h_noise._synth_clip (every voiced segment with an /h/ onset). Per /h/ segment: the energy of
512 samples at its centre minus the 95th percentile of the clip's 512-sample frame energies (hop 256), dB; the
median over segments is printed. Seen 2026-11-01 (8 clips each), as-is / -12 / +6 dB/oct: cascade -23.3 / -4.1 /
-25.0, tube -32.4 / -36.3 / -32.5, KL -31.1 / -47.2 / -31.2.

    python scripts/analysis/d064_level.py [clips per cell, default 8]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def h_levels(res) -> list[float]:
    lev = []
    for _, _, _, x, _, hs in res:
        x = np.asarray(x, float)
        fr = np.lib.stride_tricks.sliding_window_view(x, 512)[::256]
        top = np.percentile(10 * np.log10((fr ** 2).sum(1) + 1e-20), 95)
        for a, b in hs:
            c = int((a + b) / 2 * 16000)
            lev.append(10 * np.log10((x[c - 256:c + 256] ** 2).sum() + 1e-20) - top)
    return lev


def main():
    from concurrent.futures import ProcessPoolExecutor

    import d063_h_noise as H

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for e in ("cascade", "tube", "kl"):
        for t in (None, -12.0, 6.0):
            with ProcessPoolExecutor(8) as ex:
                res = list(ex.map(H._synth_clip, [(e, t, i) for i in range(n)]))
            lev = h_levels(res)
            print(f"{e} tilt {t}: /h/ level re the loudest 5 %: median {np.median(lev):.1f} dB ({len(lev)} segments)")


if __name__ == "__main__":
    main()
