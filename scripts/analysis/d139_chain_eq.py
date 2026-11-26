"""D-139: does equalising the recording chain with the session's sweeps remove the dropped formant? D-138's takes and
measures, each take corrected by the inverse of a chain response deconvolved from a room-reference sweep. Inference only;
per-take numbers go to OUT (outside the repo), next to D-138's levels.json.

    python scripts/analysis/d139_chain_eq.py response REC OUT NAME SOURCE FILE#TAKE [FILE#TAKE ...]  -> OUT/resp_NAME.npy
    python scripts/analysis/d139_chain_eq.py run REC OUT FILES NAME [NAME ...]                     -> OUT/eq.json
    python scripts/analysis/d139_chain_eq.py report OUT                                            -> OUT/d139.txt
SOURCE is the played sweep file (relative to REC); FILES as D-138.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CENTRES = np.array([200 * 2 ** (k / 3) for k in range(15)])  # 200-5080 Hz
B = 2000


def _d138():
    spec = importlib.util.spec_from_file_location("d138", ROOT / "scripts/analysis/d138_real_balance.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ir(rec, src, fs):
    nf = 1 << int(np.ceil(np.log2(len(rec) + len(src))))
    S, R = np.fft.rfft(src, nf), np.fft.rfft(rec, nf)
    h = np.fft.irfft(R * np.conj(S) / (np.abs(S) ** 2 + 1e-3 * np.max(np.abs(S)) ** 2), nf)
    k = int(np.argmax(np.abs(h)))
    return h[max(0, k - int(0.005 * fs)):k + int(0.3 * fs)]  # direct sound + 300 ms


def _thirdoct(h, fs):
    nf = 1 << 16
    P = np.abs(np.fft.rfft(h, nf)) ** 2
    f = np.fft.rfftfreq(nf, 1 / fs)
    L = np.array([10 * np.log10(P[(f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))].mean()) for c in CENTRES])
    return L - np.interp(np.log2(1000), np.log2(CENTRES), L)


def cmd_response(rec: Path, out: Path, name: str, source: str, takes: list[str]) -> None:
    import soundfile as sf

    man = list(csv.DictReader(open(rec / "manifest.csv", encoding="utf-8-sig")))
    src, _ = sf.read(rec / source, always_2d=True)
    src = src[:, -1]  # mono or the right channel
    Ls = []
    for ft in takes:
        f, no = ft.split("#")
        r = [r for r in man if r["file"] == f and r["take_list_no"] == no][0]
        y, fs = sf.read(rec / f, always_2d=True)
        y = y.mean(1)
        nz = np.flatnonzero(np.abs(src) > 1e-4)
        s = src[nz[0]:nz[0] + int((float(r["end_s"]) - float(r["start_s"])) * fs)]  # the first sweep in the file
        a, b = int((float(r["start_s"]) - 0.3) * fs), int((float(r["end_s"]) + 0.6) * fs)
        Ls.append(_thirdoct(_ir(y[a:b], s, fs), fs))
    L = np.mean(Ls, 0)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"resp_{name}.npy", L)
    print(name, " ".join(f"{c:.0f}:{v:+.1f}" for c, v in zip(CENTRES, L)))


def equalise(y, L, fs=16000):
    """Zero-phase gain = inverse of the third-octave response, log-frequency interpolated, held outside its range."""
    Y = np.fft.rfft(y)
    f = np.maximum(np.fft.rfftfreq(len(y), 1 / fs), 1.0)
    g = -np.interp(np.log2(f), np.log2(CENTRES), L)
    x = np.fft.irfft(Y * 10 ** (g / 20), len(y))
    return 0.5 * x / (np.abs(x).max() + 1e-12)


def cmd_run(rec: Path, out: Path, files: list[str], names: list[str]) -> None:
    d138 = _d138()
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    fail = {k for k, t in lv.items() if d138._tfail(t) >= d138.FAIL}
    vowels = {lv[k]["vowel"] for k in fail}
    ctrl = {k for k, t in lv.items() if t["vowel"] in vowels and d138._tfail(t) < d138.OK}
    resp = {n: np.load(out / f"resp_{n}.npy") for n in names}
    dev, models = d138._models()
    res = []
    for name, r, y in d138._load(rec, files):
        if name not in fail | ctrl:
            continue
        F = np.array(lv[name]["F"])
        row = {"take": name, "vowel": lv[name]["vowel"], "group": "fail" if name in fail else "control"}
        for arm, x in [("O", y)] + [(n, equalise(y, resp[n])) for n in names]:
            rd, _ = d138._read(models, dev, x, F)
            n = int(len(x) / 160)
            Lv = np.array([d138.levels(x[i * 160 - 512:i * 160 + 512], F[:3]) for i in range(n)[d138._mid(n)]
                           if i * 160 >= 512 and i * 160 + 512 <= len(x)])
            row[arm] = {"read": rd, "L12": float(np.median(Lv[:, 0] - Lv[:, 1]))}
        res.append(row)
        print(name, flush=True)
    (out / "eq.json").write_text(json.dumps({"arms": ["O"] + names, "rows": res}, indent=1, ensure_ascii=False), encoding="utf-8")


def cmd_report(out: Path) -> None:
    d138 = _d138()
    E = json.loads((out / "eq.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(1390)
    lines = ["D-139 (private: per-take numbers; not for the repo): mean failure share per arm, teacher seed mean "
             "[95 % bootstrap over takes]; students; median L1-L2 (dB)"]
    for g in ("fail", "control"):
        rs = [r for r in E["rows"] if r["group"] == g]
        lines.append(f"  {g} takes (n = {len(rs)}):")
        for arm in E["arms"]:
            v = np.array([np.nanmean([r[arm]["read"][m]["fail"] for m in d138.TEACHERS]) for r in rs])
            s = np.array([np.nanmean([r[arm]["read"][m]["fail"] for m in d138.STUDENTS]) for r in rs])
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
            lines.append(f"    {arm:4s} teacher {v.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]; "
                         f"students {s.mean():.2f}; L1-L2 {np.median([r[arm]['L12'] for r in rs]):+.1f}; per take "
                         + " ".join(f"{x:.2f}" for x in v))
    (out / "d139.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "response":
        cmd_response(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6:])
    elif cmd == "run":
        cmd_run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","), sys.argv[5:])
    elif cmd == "report":
        cmd_report(Path(sys.argv[2]))
