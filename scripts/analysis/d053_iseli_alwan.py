"""D-053: Iseli-Alwan validation with predicted F / B, by f0 band (plan §8.2, M3's checklist).

On clips whose source harmonics are known exactly (synthetic validation: the LF source's own H_k; resynthesised
validation: H_k measured on the recovered source the clip was rendered from), H1*-H2* and H2*-H4* are measured on
the clean audio and corrected (`backend.corrections.correct`, upper poles from the apparent VTL) with the formants
and bandwidths of each arm; the error is corrected minus the source's own value.

Arms: raw (uncorrected), true F + true B (the estimator's floor), predicted F + predicted B (the deliverable),
predicted F + Hawks-Miller B (common practice), predicted F + true B (bandwidth error removed), true F + predicted B
(formant error removed); and the deliverable with the model's own f0 for the harmonic fits ("full").

    python scripts/analysis/d053_iseli_alwan.py floor             (CPU; true labels only: arms raw and true F + B)
    python scripts/analysis/d053_iseli_alwan.py score <ckpt> ...  (VA_DEVICE=cuda for the models)
    python scripts/analysis/d053_iseli_alwan.py report
"""

import json
import sys
from pathlib import Path

import numpy as np

FS, HOP = 16000, 160
OUT = Path("reports/analysis/iseli_alwan")
CACHE = Path("cache/d053")
SETS = {"synthetic": "data/synth/train_v2/flat", "zeroth": "data/resynth/zeroth_korean_val_d16k/flat",
        "vocalset": "data/resynth/vocalset_val_d16k/flat"}
CLIPS = {"synthetic": 400, "zeroth": 600, "vocalset": 600}
FRAMES_PER_CLIP = 40
EDGE = 3  # frames dropped at each clip end (the harmonic fit needs signal on both sides)
K = 10
SEED = 5053  # used by no other set
ARMS = ("raw", "true", "pred", "pred_hm", "predF_trueB", "trueF_predB", "full")
MEASURES = ("H1H2", "H2H4")
N_BOOT = 1000
MIN_FRAMES = 500  # a band is reported with fewer, but flagged


def _arrays(name: str) -> dict:
    d = Path(SETS[name])
    load = lambda k: np.load(d / k, mmap_mode="r")  # noqa: E731
    out = {"audio": load("audio.npy"), "a_off": np.load(d / "audio_offsets.npy"), "f_off": np.load(d / "frame_offsets.npy")}
    for k in ("f0", "voiced", "sonorant", "formants", "bandwidths", "source_h_db"):
        out[k] = load(f"label_{k}.npy")
    return out


def _groups(name: str, n_clips: int) -> np.ndarray:
    """Bootstrap group per clip: the clip itself for synthetic sets; the speaker for resynthesised ones (from the
    shard's file paths: Zeroth's numeric speaker directory, VocalSet's singer directory). A shard whose files span two
    speakers gives "" (its clips can't be assigned, so they are left out)."""
    if name == "synthetic":
        return np.arange(n_clips).astype(str)
    root = Path(SETS[name]).parent
    g = []
    for p in sorted(root.glob("shard_*.json")):
        sh = json.loads(p.read_text(encoding="utf-8"))
        spk = {Path(f).parts[2] if name == "zeroth" else Path(f).parts[1] for f in sh["files"]}
        g += [spk.pop() if len(spk) == 1 else ""] * sh["n_clips"]
    assert len(g) == n_clips, (name, len(g), n_clips)
    return np.array(g)


def sample(name: str, a: dict) -> list[tuple[int, np.ndarray]]:
    """(clip id, label frame indices) for the set's sample; frames voiced, sonorant, with finite labels."""
    n = len(a["a_off"]) - 1
    if name == "synthetic":
        from vocal_analysis.train.data import split_ids

        ids = split_ids(n)[1]
    else:
        ids = np.arange(n)
    rng = np.random.default_rng(SEED)
    groups = _groups(name, n)
    ids = ids[groups[ids] != ""]
    ids = np.sort(rng.choice(ids, min(CLIPS[name], len(ids)), replace=False))
    out = []
    for c in ids:
        f0, f1 = int(a["f_off"][c]), int(a["f_off"][c + 1])
        sl = slice(f0, f1)
        ok = (np.asarray(a["voiced"][sl], bool) & np.asarray(a["sonorant"][sl], bool) & (np.asarray(a["f0"][sl]) > 0)
              & np.isfinite(a["formants"][sl][:, :4]).all(1) & np.isfinite(a["bandwidths"][sl][:, :4]).all(1)
              & np.isfinite(a["source_h_db"][sl][:, :4]).all(1))
        ok[:EDGE] = ok[-EDGE:] = False
        fr = np.flatnonzero(ok)
        if len(fr) > FRAMES_PER_CLIP:
            fr = np.sort(rng.choice(fr, FRAMES_PER_CLIP, replace=False))
        if len(fr):
            out.append((int(c), fr))
    return out


def _clip(a: dict, c: int) -> tuple[np.ndarray, slice]:
    x = np.asarray(a["audio"][a["a_off"][c]: a["a_off"][c + 1]], np.float32) / 32768.0
    return x, slice(int(a["f_off"][c]), int(a["f_off"][c + 1]))


def _measures(hc: np.ndarray) -> dict:
    return {"H1H2": hc[:, 0] - hc[:, 1], "H2H4": hc[:, 1] - hc[:, 3]}


def truth_and_floor(name: str) -> dict:
    """Per-frame truth, f0, groups, and the label-only arms (raw, true F + B), cached per set."""
    from vocal_analysis.backend.corrections import correct
    from vocal_analysis.backend.harmonics import harmonic_amplitudes

    f = CACHE / f"{name}_truth.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    a = _arrays(name)
    groups = _groups(name, len(a["a_off"]) - 1)
    rows = {k: [] for k in ("clip", "frame", "group", "f0", "F", "B", "h", "t_H1H2", "t_H2H4", "raw_H1H2", "raw_H2H4",
                            "true_H1H2", "true_H2H4")}
    for c, fr in sample(name, a):
        x, sl = _clip(a, c)
        f0 = np.asarray(a["f0"][sl], float)[fr]
        F = np.asarray(a["formants"][sl], float)[fr][:, :4]  # F1-F4; the upper poles come from the apparent VTL in every arm
        B = np.asarray(a["bandwidths"][sl], float)[fr][:, :4]
        s = np.asarray(a["source_h_db"][sl], float)[fr]
        h = harmonic_amplitudes(x, FS, fr * HOP / FS, f0, K)
        raw = _measures(h)
        tru = _measures(correct(h, f0, F, B, FS, upper_from_vtl=True))
        t = _measures(s)
        for k, v in (("clip", np.full(len(fr), c)), ("frame", fr), ("group", np.full(len(fr), groups[c], object)),
                     ("f0", f0), ("F", F), ("B", B), ("h", h)):
            rows[k].append(v)
        for m in MEASURES:
            rows[f"t_{m}"].append(t[m])
            rows[f"raw_{m}"].append(raw[m])
            rows[f"true_{m}"].append(tru[m])
    out = {k: np.concatenate(v) for k, v in rows.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def score(ckpt: str, log=print) -> None:
    """The model arms for one checkpoint on every set (the clean stored audio, windowed analysis as default)."""
    import torch

    from vocal_analysis.backend.corrections import correct
    from vocal_analysis.backend.harmonics import harmonic_amplitudes
    from vocal_analysis.models.infer import analyse, load_teacher
    from vocal_analysis.synth.filters import hawks_miller_bandwidth

    run = Path(ckpt).parent.name
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_teacher(ckpt, dev)
    for name in SETS:
        f = CACHE / f"{name}_{run}.npz"
        if f.exists():
            continue
        tr = truth_and_floor(name)
        a = _arrays(name)
        out = {f"{arm}_{m}": [] for arm in ARMS[2:] for m in MEASURES}
        out |= {"pF": [], "pB": [], "pf0": []}
        clips = tr["clip"]
        for c in np.unique(clips):
            idx = np.flatnonzero(clips == c)
            fr = tr["frame"][idx]
            x, _ = _clip(a, int(c))
            r = analyse(model, x, dev)
            n = len(r["f0"])
            assert fr.max() < n, (name, c, fr.max(), n)
            pF, pB, pf0 = np.asarray(r["formants"], float)[fr], np.asarray(r["bandwidths"], float)[fr], np.asarray(r["f0"], float)[fr]
            f0, F, B, h = tr["f0"][idx], tr["F"][idx], tr["B"][idx], tr["h"][idx]
            hm = hawks_miller_bandwidth(pF, f0[:, None])
            arms = {"pred": (h, f0, pF, pB), "pred_hm": (h, f0, pF, hm), "predF_trueB": (h, f0, pF, B),
                    "trueF_predB": (h, f0, F, pB)}
            hp = harmonic_amplitudes(x, FS, fr * HOP / FS, pf0, K)
            arms["full"] = (hp, pf0, pF, pB)
            for arm, (hh, ff, FF, BB) in arms.items():
                v = _measures(correct(hh, ff, FF, BB, FS, upper_from_vtl=True))
                for m in MEASURES:
                    out[f"{arm}_{m}"].append(v[m])
            out["pF"].append(pF)
            out["pB"].append(pB)
            out["pf0"].append(pf0)
        np.savez_compressed(f, **{k: np.concatenate(v) for k, v in out.items()})
        log(f"{run} {name}: {len(clips)} frames")


def _band(f0: np.ndarray) -> np.ndarray:
    from vocal_analysis.eval.metrics import f0_band_labels

    return f0_band_labels(f0)


def _boot(groups: np.ndarray, stat, rng) -> tuple[float, float, float]:
    """Point and 95 % group-bootstrap interval of stat(index array)."""
    u, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(u))]
    pt = stat(np.arange(len(groups)))
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(u), len(u))
        bs.append(stat(np.concatenate([members[i] for i in pick])))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def summarise(name: str, run: str | None) -> dict:
    tr = truth_and_floor(name)
    arms = {"raw": tr, "true": tr}
    if run is not None:
        with np.load(CACHE / f"{name}_{run}.npz") as z:
            md = {k: z[k] for k in z.files}
        arms |= {arm: md for arm in ARMS[2:]}
    band = _band(tr["f0"])
    rng = np.random.default_rng(SEED)
    res = {}
    for m in MEASURES:
        err = {arm: d[f"{arm}_{m}"] - tr[f"t_{m}"] for arm, d in arms.items()}
        for b in ("all", "<150", "150-300", "300-500", ">500"):
            sel = np.flatnonzero(np.ones(len(band), bool) if b == "all" else band == b)
            ok = sel[np.all([np.isfinite(e[sel]) for e in err.values()], axis=0)]
            if len(ok) < 20:
                continue
            g = tr["group"][ok]
            row = {"n": int(len(ok)), "groups": int(len(np.unique(g)))}
            for arm, e in err.items():
                ee = e[ok]
                row[arm] = {"mae": _boot(g, lambda i: np.median(np.abs(ee[i])), rng),
                            "bias": _boot(g, lambda i: np.median(ee[i]), rng)}
            if run is not None:
                for lab, a1, a2 in (("bandwidth_share", "pred", "predF_trueB"), ("formant_share", "pred", "trueF_predB"),
                                    ("model_vs_hm", "pred", "pred_hm")):
                    e1, e2 = np.abs(err[a1][ok]), np.abs(err[a2][ok])
                    row[lab] = _boot(g, lambda i: np.median(e1[i]) - np.median(e2[i]), rng)
            res[f"{m}/{b}"] = row
    return res


def _fmt(t) -> str:
    return f"{t[0]:5.2f} [{t[1]:5.2f}, {t[2]:5.2f}]"


def report(log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted({p.stem.split("_", 1)[1] for p in CACHE.glob("*_*.npz") if not p.stem.endswith("_truth")})
    lines = ["D-053: Iseli-Alwan validation with predicted F / B, by f0 band (plan §8.2). Error = corrected - source's own",
             "value, dB; median |error| and median error (bias) with 95 % bootstrap intervals (clips for synthetic, speakers",
             "for resynthesised sets). Shares: median |error| of predicted F + B minus that with the true bandwidths (or",
             "formants, or Hawks-Miller bandwidths) put back; positive = the predicted value costs that much.", ""]
    doc = {}
    for run in [None] + runs:
        for name in SETS:
            if not (CACHE / f"{name}_truth.npz").exists() or (run and not (CACHE / f"{name}_{run}.npz").exists()):
                continue
            s = summarise(name, run)
            doc[f"{run or 'labels'}/{name}"] = s
            lines.append(f"== {run or 'labels only'}: {name}")
            for key, row in s.items():
                flag = "" if row["n"] >= MIN_FRAMES else "  (few frames)"
                lines.append(f"  {key:16s} n {row['n']:6d} ({row['groups']} groups){flag}")
                for arm in ARMS:
                    if arm in row:
                        lines.append(f"    {arm:12s} |e| {_fmt(row[arm]['mae'])}  bias {_fmt(row[arm]['bias'])}")
                for lab in ("bandwidth_share", "formant_share", "model_vs_hm"):
                    if lab in row:
                        lines.append(f"    {lab:16s} {_fmt(row[lab])}")
            lines.append("")
    txt = "\n".join(lines)
    (OUT / "iseli_alwan.txt").write_text(txt, encoding="utf-8")
    (OUT / "iseli_alwan.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log(txt)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "floor":
        for name in SETS:
            tr = truth_and_floor(name)
            print(name, len(tr["clip"]), "frames")
        report()
    elif cmd == "score":
        for ck in sys.argv[2:]:
            score(ck)
        report()
    else:
        report()
