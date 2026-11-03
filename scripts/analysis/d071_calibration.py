"""D-071 stage A: uncertainty calibration of teacher v3.1's non-f0 heads on sets with exact labels (docs/decisions.md).

    python scripts/analysis/d071_calibration.py analyse <run>...   (GPU: per-frame z and probabilities, cached)
    python scripts/analysis/d071_calibration.py report <run>...    (CPU: scale factors, coverage, ECE; first run decides)

Sets: train_v2's validation clips (the trainer's split and channel, 512 clips), each `suite_v1` case, and the
resynthesised validation sets (formants and bandwidths only: their f0 and voicing are the tracker's, not exact). Each
item is the trainer's fixed validation crop at 400 frames (D-028's window, so train_v2's 2-4 s clips are whole),
analysed by `infer.analyse` as `va analyse` would.

Per frame: formant z = cents error / formant_sigma_cents (sonorant frames with a label); bandwidth z = (ln B - ln B_ref)
/ bandwidth_log_sd (the same frames); length z = (mu - label) / vtl_sigma (log2; train_v2's frames with D-035's label);
voicing and sonorant probabilities against their labels (synthetic sets only). Cells: set x f0 band (label f0;
"unvoiced" for whispered sonorant frames).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("reports/analysis/d071")
CACHE = Path("cache/d071")
SUITE = Path("data/synth/suite_v1")
CROP = 400
SEED = 5071
S_BAND = (0.87, 1.15)  # D-071: sigma scale factor flag, +-15 % symmetric in log
MIN_CLIPS = 20
Q1 = 0.6826894921370859  # P(|Z| < 1)


def _sets(run: Path):
    """name -> (kind, iterator of (audio, labels, n_valid, w_formant))."""
    from dataclasses import replace

    from vocal_analysis.features.vqt import N_MAX
    from vocal_analysis.synth.channel import ChannelConfig
    from vocal_analysis.synth.shards import ShardReader
    from vocal_analysis.train.data import SynthCrops, consolidate, split_ids
    from vocal_analysis.train.teacher import _saved_channel

    cfg = json.loads((run / "config.json").read_text())["train"]
    channel = _saved_channel(run)
    val_channel = replace(channel, gain_db=ChannelConfig().gain_db)

    def crops(flat, ids, synthetic):
        ds = SynthCrops(flat, ids, CROP, val_channel, seed=12345, fixed=True, vtl=synthetic)
        f_off = ds._open()["f_off"]
        for i in range(len(ds)):
            seg, lab = ds[i]
            cid = ds.clip_ids[i]
            yield seg[N_MAX // 2:], lab, min(CROP, int(f_off[cid + 1] - f_off[cid])), float(lab["w_formant"][0])

    flat = consolidate(cfg["data"], log=lambda *a: None)
    n = json.loads((flat / "done").read_text())["n_clips"]
    val_ids = split_ids(n)[1][: cfg["val_clips"]]
    out = {"train_v2_val": ("synthetic", lambda: crops(flat, val_ids, True))}

    def suite(case_dir):
        r = ShardReader(case_dir)
        for i in range(len(r)):
            audio, lab, _ = r[i]
            yield audio, lab, len(lab["t"]), 1.0

    for case in sorted(p for p in SUITE.iterdir() if (p / "index.json").exists()):
        out[f"suite:{case.name}"] = ("synthetic", lambda case=case: suite(case))
    for d in cfg["resynth_val"]:
        rflat = consolidate(d, log=lambda *a: None)
        m = json.loads((rflat / "done").read_text())["n_clips"]
        ids = np.sort(np.random.default_rng(0).permutation(m)[: cfg["val_clips"]])  # the trainer's
        out[f"resynth:{Path(d).name}"] = ("resynth", lambda rflat=rflat, ids=ids: crops(rflat, ids, False))
    return out


def analyse(runs: list[str]):
    import torch

    from vocal_analysis.models.infer import analyse as run_model, load_teacher

    dev = torch.device("cuda")
    CACHE.mkdir(parents=True, exist_ok=True)
    for r in runs:
        model = load_teacher(Path(r) / "ckpt_last.pt", dev)[0]
        for name, (kind, items) in _sets(Path(r)).items():
            path = CACHE / f"{Path(r).name}__{name.replace(':', '_')}.npz"
            if path.exists():
                continue
            rows: dict[str, list] = {k: [] for k in ("clip", "f0", "fz", "fok", "bz", "bok", "lz", "vp", "vl", "sp", "sl")}
            for ci, (audio, lab, nv, w) in enumerate(items()):
                res = run_model(model, np.asarray(audio, np.float32), dev)
                n = min(nv, len(res["formants"]), len(lab["f0"]))
                son = np.asarray(lab["sonorant"][:n], bool)
                ref = np.asarray(lab["formants"][:n, :4], float)
                fok = son[:, None] & np.isfinite(ref) & (ref > 0) & (w >= 0.5)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fz = 1200 * np.log2(res["formants"][:n, :4] / ref) / res["formant_sigma_cents"][:n, :4]
                    bref = np.asarray(lab["bandwidths"][:n, :4], float)
                    bok = fok & np.isfinite(bref) & (bref > 0)
                    bz = (np.log(res["bandwidths"][:n, :4]) - np.log(bref)) / res["bandwidth_log_sd"][:n, :4]
                lz = np.full(n, np.nan)
                if "vtl" in lab and "vtl_log2cm" in res:
                    y = np.asarray(lab["vtl"][:n], float)
                    lz = np.where(son & np.isfinite(y), (res["vtl_log2cm"][:n] - y) / res["vtl_sigma"][:n], np.nan)
                f0 = np.where(np.asarray(lab["voiced"][:n], bool), np.asarray(lab["f0"][:n], float), np.nan)
                keep = son | (kind == "synthetic")  # probabilities on every frame of synthetic sets
                for k, v in (("clip", np.full(n, ci)), ("f0", f0), ("fz", fz), ("fok", fok), ("bz", bz), ("bok", bok), ("lz", lz),
                             ("vp", res["voiced_p"][:n]), ("vl", np.asarray(lab["voiced"][:n], bool)),
                             ("sp", res["sonorant_p"][:n]), ("sl", son)):
                    rows[k].append(np.asarray(v)[keep])
            np.savez_compressed(path, kind=kind, **{k: np.concatenate(v) for k, v in rows.items()})
            print(f"{Path(r).name} {name}: {sum(len(v) for v in rows['clip'])} frames", flush=True)


def _scale(z: np.ndarray) -> float:
    """sigma scale factor making 1-sigma coverage nominal: the Q1 quantile of |z|"""
    return float(np.quantile(np.abs(z), Q1))


def _cell(z: np.ndarray, clip: np.ndarray, rng) -> dict | None:
    from scipy.stats import norm

    ok = np.isfinite(z)
    z, clip = z[ok], clip[ok]
    clips = np.unique(clip)
    if len(clips) < MIN_CLIPS:
        return None
    s = _scale(z)
    c1, c2 = float(np.mean(np.abs(z) <= 1)), float(np.mean(np.abs(z) <= 2))
    implied2 = lambda s: 2 * norm.cdf(2 / s) - 1  # noqa: E731
    by = {c: z[clip == c] for c in clips}
    bs_s, bs_d = [], []
    for _ in range(1000):
        zz = np.concatenate([by[c] for c in rng.choice(clips, len(clips))])
        sb = _scale(zz)
        bs_s.append(sb)
        bs_d.append(np.mean(np.abs(zz) <= 2) - implied2(sb))
    lo, hi = np.percentile(bs_s, [2.5, 97.5])
    dlo, dhi = np.percentile(bs_d, [2.5, 97.5])
    flag = (s < S_BAND[0] or s > S_BAND[1]) and not (lo <= 1 <= hi)
    tails = not (dlo <= 0 <= dhi) and abs(c2 - implied2(s)) > 0.01
    return {"s": s, "lo": lo, "hi": hi, "c1": c1, "c2": c2, "c2_implied": implied2(s), "rms": float(np.sqrt(np.mean(z ** 2))),
            "clips": len(clips), "frames": int(len(z)), "flag": bool(flag), "tails": bool(tails)}


def _ece(p: np.ndarray, y: np.ndarray) -> float:
    b = np.minimum((p * 10).astype(int), 9)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any()))


def report(runs: list[str]):
    from vocal_analysis.eval.metrics import f0_band_labels

    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["D-071 stage A: calibration on exact labels. s = sigma scale factor making 1-sigma coverage nominal (flag",
             f"outside {S_BAND[0]}-{S_BAND[1]} with the clip-bootstrap interval excluding 1, >= {MIN_CLIPS} clips); coverage at 1 / 2 sigma",
             "(+-15 % implies 0.615-0.750 / 0.918-0.979); 'tails': 2-sigma coverage off the s-implied value (interval excludes 0,",
             "beyond 0.01). Probabilities: ECE (10 bins; flag > 0.05). The first run decides; others reported.", ""]
    summary = {}
    for r in runs:
        rn = Path(r).name
        lines.append(f"== {rn}")
        flags = []
        for path in sorted(CACHE.glob(f"{rn}__*.npz")):
            d = dict(np.load(path, allow_pickle=True))
            name = path.stem.split("__", 1)[1]
            band = f0_band_labels(d["f0"])
            heads = [(f"F{k + 1}", np.where(d["fok"][:, k], d["fz"][:, k], np.nan)) for k in range(4)]
            heads += [(f"B{k + 1}", np.where(d["bok"][:, k], d["bz"][:, k], np.nan)) for k in range(4)]
            heads.append(("VTL", d["lz"]))
            for h, z in heads:
                for b in ["all", *sorted(set(band.tolist()))]:
                    m = np.ones(len(z), bool) if b == "all" else band == b
                    c = _cell(z[m], d["clip"][m], rng)
                    if c is None:
                        continue
                    mark = (" FLAG" if c["flag"] else "") + (" TAILS" if c["tails"] else "")
                    lines.append(f"  {name:32s} {h:4s} {b:9s} s {c['s']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]  cover {c['c1']:.3f} / "
                                 f"{c['c2']:.3f} (s-implied {c['c2_implied']:.3f})  rms z {c['rms']:.2f}  {c['clips']} clips{mark}")
                    if c["flag"] or c["tails"]:
                        flags.append(f"{name} {h} {b}:{mark}")
            if str(d["kind"]) == "synthetic":
                for h, p, y in (("voicing", d["vp"], d["vl"]), ("sonorant", d["sp"], d["sl"])):
                    e = _ece(p.astype(float), y.astype(float))
                    lines.append(f"  {name:32s} {h:9s} ECE {e:.3f}{' FLAG' if e > 0.05 else ''}")
                    if e > 0.05:
                        flags.append(f"{name} {h}: ECE {e:.3f}")
        summary[rn] = flags
        lines.append("")
    first = Path(runs[0]).name
    lines.append(f"FLAGS ({first}, deciding): " + ("none" if not summary[first] else ""))
    lines += [f"  {f}" + ("" if all(f in summary[Path(o).name] for o in runs[1:]) else "  (this seed only)") for f in summary[first]]
    text = "\n".join(lines)
    print(text)
    (OUT / "d071_stage_a.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"analyse": analyse, "report": report}[sys.argv[1]](sys.argv[2:])
