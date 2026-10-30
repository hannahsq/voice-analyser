"""Time reversal across a change of speaker (D-050, probe 5; Hannah's design in considering.md).

A clip of speaker A joined to a clip of speaker B is analysed forwards and time-reversed (whole file, no windows, so
window boundaries do not move), and the reversed outputs are mapped back frame by frame. A frame's spectrum is
unchanged by reversal (the VQT kernels are symmetric Hann windows), so a frame-local estimate would not change; what
does change is which side of the join supplies the context. Same-speaker joins (two clips of A) are the control:
context use within one speaker is legitimate, so the leakage is the cross-speaker difference minus the same-speaker
one, by distance from the join. The joined signal is k * HOP + 1 samples long, so frame i maps to reversed frame k - i.
Clips: probe 1's cached originals (Zeroth-Korean val, numeric speaker ids).

    python -m vocal_analysis.eval.probes.reversal <ckpt> [...]   ($VA_DEVICE=cuda)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path("data/probes/counterfactual_v1")
REPORTS = Path("reports/probes/reversal")
HOP = 160
BINS = ((0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 10_000))
OUTS = ("F1", "F2", "F3", "VTL")


def joins() -> list[tuple[str, np.ndarray, int]]:
    """(kind, audio, join frame): every ordered pair of speakers (their first clips) and every ordered pair of one
    speaker's clips (D-050 amendment 2: the same-speaker control needs as many frames near the join)."""
    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    by: dict[str, list[np.ndarray]] = {}
    for it in meta["items"]:
        by.setdefault(it["group"], []).append(np.load(SRC / it["file"])["original"].astype(float))
    trim = lambda x: x[: len(x) // HOP * HOP]  # noqa: E731
    out = []
    spk = sorted(by)
    for a in spk:
        for b in spk:
            if a != b:
                out.append(("cross", *(_join(trim(by[a][0]), trim(by[b][0])))))
        for i, x in enumerate(by[a]):
            for k, y in enumerate(by[a]):
                if i != k:
                    out.append(("same", *(_join(trim(x), trim(y)))))
    return out


def _join(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    a, b = a / (np.max(np.abs(a)) + 1e-12), b / (np.max(np.abs(b)) + 1e-12)
    x = np.r_[a, b, 0.0]  # k * HOP + 1 samples
    return (0.9 * x).astype(np.float32), len(a) // HOP


def score(ckpt: str) -> dict:
    import torch

    from ...models.infer import analyse, load_teacher

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    rows = {kind: {o: [[] for _ in BINS] for o in OUTS} for kind in ("cross", "same")}
    for kind, x, j in joins():
        fw = analyse(model, x, dev, window_frames=None)
        rv = analyse(model, x[::-1].copy(), dev, window_frames=None)
        k = (len(x) - 1) // HOP
        n = k + 1
        back = lambda a: a[:n][::-1]  # noqa: E731  reversed frame k - i -> frame i
        son = (fw["sonorant_p"][:n] > 0.5) & (back(rv["sonorant_p"]) > 0.5)
        dist = np.abs(np.arange(n) - j)
        for o in OUTS:
            if o == "VTL":
                if "vtl_log2cm" not in fw:
                    continue
                d = 1200 * np.abs(fw["vtl_log2cm"][:n] - back(rv["vtl_log2cm"]))
            else:
                m = int(o[1]) - 1
                d = 1200 * np.abs(np.log2(fw["formants"][:n, m]) - np.log2(back(rv["formants"][:, m])))
            for bi, (lo, hi) in enumerate(BINS):
                sel = son & (dist >= lo) & (dist < hi) & np.isfinite(d)
                rows[kind][o][bi].extend(d[sel].tolist())
    return rows


def score_direction(ckpt: str) -> dict:
    """Direction arm (D-050 amendment 3): probe 1's reference resynthesis (exact formant labels), forward and reversed.
    Per item and formant: median |error| (cents) over the scored frames each way, and median |forward - reversed|."""
    import torch

    from ...models.infer import analyse, load_teacher
    from . import measures as M

    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    meta = json.loads((SRC / "meta.json").read_text(encoding="utf-8"))
    n_items = len(meta["items"])
    fwd, rev, diff = (np.full((n_items, 4), np.nan) for _ in range(3))
    for i, it in enumerate(meta["items"]):
        with np.load(SRC / it["file"]) as z:
            d = {key: z[key] for key in ("audio_0", "label_formants_0", "scored_0")}
        k = (len(d["audio_0"]) - 1) // HOP
        x = d["audio_0"][: k * HOP + 1].astype(np.float32)  # frame i <-> reversed frame k - i
        n = k + 1
        fw = analyse(model, x, dev, window_frames=None)["formants"][:n, :4]
        rv = analyse(model, x[::-1].copy(), dev, window_frames=None)["formants"][:n, :4][::-1]
        lab, sc = d["label_formants_0"][:n], d["scored_0"][:n]
        with np.errstate(all="ignore"):
            for m in range(4):
                ok = sc & np.isfinite(lab[:, m])
                if ok.any():
                    fwd[i, m] = np.median(np.abs(1200 * np.log2(fw[ok, m] / lab[ok, m])))
                    rev[i, m] = np.median(np.abs(1200 * np.log2(rv[ok, m] / lab[ok, m])))
                    diff[i, m] = np.median(np.abs(1200 * np.log2(fw[ok, m] / rv[ok, m])))
    groups = np.array([it["group"] for it in meta["items"]])
    res = {}
    for m in range(4):
        ok = np.isfinite(fwd[:, m]) & np.isfinite(rev[:, m])
        res[f"F{m + 1}"] = {"forward_error": float(np.median(fwd[ok, m])), "reversed_error": float(np.median(rev[ok, m])),
                            "reversed_minus_forward": list(M.group_bootstrap(rev[ok, m] - fwd[ok, m], groups[ok])),
                            "forward_vs_reversed": float(np.median(diff[ok, m])), "n_items": int(ok.sum())}
    return res


def report_direction(res: dict) -> str:
    lines = ["direction arm (amendment 3): probe 1's reference resynthesis, |error| against exact labels (cents, median over items)"]
    for o, v in res.items():
        p, lo, hi = v["reversed_minus_forward"]
        lines.append(f"  {o}  forward {v['forward_error']:.0f}  reversed {v['reversed_error']:.0f}  reversed - forward "
                     f"{p:+.1f} [{lo:+.1f}, {hi:+.1f}]  |forward - reversed| {v['forward_vs_reversed']:.0f}  ({v['n_items']} items)")
    return "\n".join(lines)


def report(name: str, rows: dict) -> tuple[str, dict]:
    res = {}
    lines = [f"Time reversal across a speaker change (D-050, probe 5): {name}",
             "median |forward - reversed| (cents) by distance from the join (frames); cross-speaker, same-speaker, difference"]
    for o in OUTS:
        if not rows["cross"][o][0]:
            continue
        parts = []
        for bi, (lo, hi) in enumerate(BINS):
            c, s = rows["cross"][o][bi], rows["same"][o][bi]
            mc, ms = (float(np.median(c)) if c else np.nan), (float(np.median(s)) if s else np.nan)
            res[f"{o}/{lo}-{hi}"] = {"cross": mc, "same": ms, "leak": mc - ms, "n_cross": len(c), "n_same": len(s)}
            parts.append(f"{lo}-{hi if hi < 10_000 else 'end'}: {mc:.0f}/{ms:.0f}/{mc - ms:+.0f}")
        lines.append(f"  {o:3s} " + "  ".join(parts))
    return "\n".join(lines), res


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not argv:
        print(__doc__)
        return 2
    REPORTS.mkdir(parents=True, exist_ok=True)
    for ckpt in argv:
        name = Path(ckpt).parent.name
        text, res = report(name, score(ckpt))
        res["direction"] = score_direction(ckpt)
        text += "\n" + report_direction(res["direction"])
        (REPORTS / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (REPORTS / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
