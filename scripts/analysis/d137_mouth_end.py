"""D-137: what makes CHIMERA's resonances differ from the transmission-line models (`tube.poles`, KL)? A uniform 3 cm^2
tube (17.5 and 15.6 cm, 44 sections) with only its lip end (last 4 sections) or only its epilarynx (first 4) set to
areas 0.15-6 cm^2; F1-F3 from `tube.poles`, KL's measured resonances and CHIMERA's network. Labels only.

    python scripts/analysis/d137_mouth_end.py     -> reports/analysis/d137/d137.txt   (needs $CHIMERA_DRIVER, as D-135)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/analysis/d137"
AREAS = (0.15, 0.3, 0.6, 1.2, 3.0, 6.0)
LENGTHS = (17.5, 15.6)
REGIONS = {"lips": slice(40, 44), "epilarynx": slice(0, 4)}


def main() -> None:
    from vocal_analysis.synth import kl

    spec = importlib.util.spec_from_file_location("d136", ROOT / "scripts/analysis/d136_label_check.py")
    d136 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d136)
    shapes, rows = {}, []
    for L in LENGTHS:
        for region, sl in REGIONS.items():
            for a in AREAS:
                A = np.full(44, 3.0)
                A[sl] = a
                name = f"L{L:g}_{region}_{a:g}"
                shapes[name] = (L, region, a, A)
                rows.append((name, L, A))
    ch = d136._chimera(rows)
    c = lambda x, y: 1200 * np.log2(x / y)  # noqa: E731
    lines = ["D-137: uniform 3 cm^2 tube, one region's area varied; F1 / F2 / F3 in Hz from tube.poles, and KL and CHIMERA "
             "minus tube.poles in cents", ""]
    for L in LENGTHS:
        for region in REGIONS:
            lines.append(f"{L:g} cm, {region} area varied:")
            for a in AREAS:
                n = f"L{L:g}_{region}_{a:g}"
                A = shapes[n][3]
                t = d136._tube(A, L)
                (fk, _), = kl.frame_poles([d136._shape(A, L)])
                k, h = c(fk[:3], t), c(ch[n], t)
                lines.append(f"  {a:5.2f} cm^2: tube {t[0]:5.0f} / {t[1]:5.0f} / {t[2]:5.0f} Hz; kl - tube {k[0]:+4.0f} / {k[1]:+4.0f} / "
                             f"{k[2]:+4.0f} c; chimera - tube {h[0]:+4.0f} / {h[1]:+4.0f} / {h[2]:+4.0f} c")
            lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "d137.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
