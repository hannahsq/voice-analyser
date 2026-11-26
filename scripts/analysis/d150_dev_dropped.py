"""D-150: is the dropped formant there on other real voices? Hillenbrand dev-tier tokens against their hand-checked
F1-F3 (steady state and 20 / 50 / 80 % of the vowel). CPU, inference only; dev tier only.

    python scripts/analysis/d150_dev_dropped.py run      -> reports/analysis/d150/d150.{json,txt}
    python scripts/analysis/d150_dev_dropped.py report   -> reports/analysis/d150/d150.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d150"
DECIDING = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
REPORTED = ("d098_base_b20k", "d098_heads_b20k")
HOP = 0.01


def _tokens():
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry

    mod = registry()["hillenbrand"]
    rows = TI.filter_rows(read_manifest(str(ROOT / "data/manifests/hillenbrand.csv")), ("dev",))
    for r in rows:
        it = mod.load(ROOT / "data/hillenbrand", r)
        yield r, it, load_audio(it)


def _reading(r, t):
    """The model's F1-F3 at time t: median over voiced frames within +-20 ms; None if fewer than 2."""
    i = int(round(t / HOP))
    k = np.arange(max(0, i - 2), min(len(r["voiced_p"]), i + 3))
    k = k[(r["voiced_p"][k] > 0.5) & np.all(np.isfinite(r["formants"][k, :3]), 1)]
    return np.median(r["formants"][k, :3], 0) if len(k) >= 2 else None


def cmd_run() -> None:
    import torch

    from vocal_analysis.models.infer import analyse, load_teacher

    torch.set_num_threads(6)
    dev = torch.device("cpu")
    models = {m: load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0] for m in DECIDING + REPORTED}
    res = []
    for row, it, y in _tokens():
        reads = {m: analyse(mod, y, dev) for m, mod in models.items()}
        frames = []
        for j, t in enumerate(it.times):
            ref = it.formants[j, :3]
            if not (np.isfinite(t) and np.all(np.isfinite(ref))):
                continue
            fr = {"kind": it.meta["frame_kinds"][j], "ref": ref.tolist()}
            for m in models:
                v = _reading(reads[m], t)
                fr[m] = None if v is None else v.tolist()
            frames.append(fr)
        res.append({"token": Path(row.path).stem.lower(), "speaker": row.speaker_id, "group": it.meta["group"],
                    "vowel": it.meta["vowel"], "f0": float(it.f0[0]), "ref_ss": it.formants[0, :3].tolist(), "frames": frames})
        print(".", end="", flush=True)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d150.json").write_text(json.dumps(res), encoding="utf-8")
    print()
    cmd_report()


def _token_stats(tok, m):
    """Share of the token's labelled frames failing (F2 read nearer the reference F3 than its F2), and of each kind:
    dropped F1 (F1 read nearer the reference F2 than its F1) and skipped F2 (F1 right, F2 at F3)."""
    fl, drop, skip = [], [], []
    for fr in tok["frames"]:
        v = fr.get(m)
        if v is None:
            continue
        ref = np.array(fr["ref"])
        f = abs(np.log(v[1] / ref[2])) < abs(np.log(v[1] / ref[1]))
        d = abs(np.log(v[0] / ref[1])) < abs(np.log(v[0] / ref[0]))
        fl.append(f)
        drop.append(f and d)
        skip.append(f and not d)
    return (float(np.mean(fl)), float(np.mean(drop)), float(np.mean(skip))) if fl else None


def cmd_report() -> None:
    T = json.loads((REP / "d150.json").read_text(encoding="utf-8"))
    for t in T:
        F = np.array(t["ref_ss"])
        t["cell"] = bool(350 <= F[0] <= 650 and F[1] / F[0] < 2.2)
    lines = ["D-150: the dropped formant on Hillenbrand's dev tier (hand-checked F1-F3 at steady state and 20 / 50 / 80 %)",
             "  a frame fails if the model's F2 is nearer the reference F3 than the reference F2; a token fails if half or more of "
             "its labelled frames do", "  cell: D-141's (reference steady-state F1 350-650 Hz, F2 / F1 < 2.2); f0 is the token's "
             "steady-state reference f0", ""]

    def block(sel, title):
        toks = [t for t in T if sel(t)]
        spk = len({t["speaker"] for t in toks})
        lines.append(f"  {title}: {len(toks)} tokens, {spk} speakers")
        out = {}
        for m in DECIDING + REPORTED:
            st = [(t, _token_stats(t, m)) for t in toks]
            st = [(t, s) for t, s in st if s is not None]
            if not st:
                continue
            ft = [t for t, s in st if s[0] >= 0.5]
            fr = np.mean([s[0] for _, s in st])
            kinds = (np.mean([s[1] for _, s in st]), np.mean([s[2] for _, s in st]))
            out[m] = (len(ft) / len(st), len({t["speaker"] for t in ft}), len(st))
            lines.append(f"    {m:18s} failing tokens {len(ft):3d} of {len(st)} ({len(ft) / len(st):.0%}), from {out[m][1]} "
                         f"speakers; failing frames {fr:.1%} (dropped F1 {kinds[0]:.1%}, skipped F2 {kinds[1]:.1%})")
        return out

    primary = block(lambda t: t["cell"] and t["f0"] <= 150, "primary: in the cell, f0 <= 150 Hz")
    block(lambda t: t["cell"] and 150 < t["f0"] <= 200, "in the cell, f0 150-200 Hz")
    block(lambda t: t["cell"] and t["f0"] > 200, "in the cell, f0 above 200 Hz")
    block(lambda t: not t["cell"] and t["f0"] <= 150, "outside the cell, f0 <= 150 Hz")
    block(lambda t: not t["cell"] and t["f0"] > 150, "outside the cell, f0 above 150 Hz")
    lines.append("")
    lines.append("  by group and vowel, primary block (5b's seed mean of the token failure share >= 0.5):")
    toks = [t for t in T if t["cell"] and t["f0"] <= 150]
    for key in ("group", "vowel"):
        cnt = {}
        for t in toks:
            s = [_token_stats(t, m) for m in DECIDING]
            if any(x is None for x in s):
                continue
            c = cnt.setdefault(t[key], [0, 0])
            c[1] += 1
            c[0] += int(np.mean([x[0] for x in s]) >= 0.5)
        lines.append("    " + key + ": " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(cnt.items())))
    # the registered reading on 5b's seed mean
    st = []
    for t in toks:
        s = [_token_stats(t, m) for m in DECIDING]
        if all(x is not None for x in s):
            st.append((t, np.mean([x[0] for x in s]) >= 0.5))
    nf = sum(f for _, f in st)
    share = nf / len(st) if st else float("nan")
    spk = len({t["speaker"] for t, f in st if f})
    if share >= 0.10 and spk >= 3:
        out = "PRESENT: the dropped formant appears on other voices; a dev-tier deciding measure exists"
    elif nf <= 1:
        out = "RARE: at most one dev token fails; the failure isn't a common pattern on these voices"
    else:
        out = "SOME: reported, Hannah decides"
    lines += ["", f"Reading (registered; 5b seed mean, primary block): {nf} of {len(st)} tokens ({share:.0%}) from {spk} speakers -> {out}"]
    (REP / "d150.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
