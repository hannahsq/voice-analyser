"""D-161: scoring and decision for the source-envelope arm (E, `d161_env_s{1,2}`) against its matched control (Z,
`d161_zero_s{1,2}`). CPU, inference only. The guards come from the GPU chain (`chains/d161.sh`); this script reads them.

    python scripts/analysis/d161_score.py run MODEL[,MODEL...]    -> cache/d161/score/... and the shared grid caches
    python scripts/analysis/d161_score.py private REC OUT FILES   -> OUT/d161_private.json (counts only reach the repo)
    python scripts/analysis/d161_score.py report [OUT]            -> reports/analysis/d161/d161.{txt,json}

Channel 4 modes on arm E's grids: "head" (the model as trained: its own predicted envelope), "truth" (the clip's true
LF levels fed through `source_fed`), "zero" (`source_zero`: the channel held at zero).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/analysis/d161"
CACHE = ROOT / "cache/d161/score"
FS = 16000
N_H = 40
F_MAX = 4000.0
ENV = {1: "d161_env_s1", 2: "d161_env_s2"}
ZERO = {1: "d161_zero_s1", 2: "d161_zero_s2"}
FIVEB = {1: "d040_5b_s1_b20k", 2: "d040_5b_s2_b20k"}
D160 = {1: "d160_s1", 2: "d160_s2"}
OTHERS = {s: (ZERO[s], FIVEB[s], f"d090_f5_s{s}", f"d090_f57_s{s}", D160[s]) for s in (1, 2)}
SYN = {"v2": ROOT / "data/synth/train_v2/flat", "v2w": ROOT / "data/synth/train_v2w/flat"}
RES = {"vocalset": ROOT / "data/resynth/vocalset_val_d16k_h40/flat", "zeroth": ROOT / "data/resynth/zeroth_korean_val_d16k_h40/flat"}
N_VAL = 512
CLEAN = 0.1


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _d160():
    """D-160's scoring functions, writing their per-model caches under cache/d161/score."""
    m = _mod("d160_score")
    m.CACHE = CACHE
    return m


def _dev():
    import torch

    torch.set_num_threads(6)
    return torch.device("cpu")


def _load(m, dev):
    from vocal_analysis.models.infer import load_teacher

    return load_teacher(str(ROOT / f"runs/{m}/ckpt_last.pt"), dev)[0]


def _wrap(md, mode, levels=None):
    """The model with channel 4 from the given true levels ("truth", (40,) dB) or held at zero ("zero")."""
    import torch

    class Fed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.md = md

        def forward(self, logpow):
            b, _, t = logpow.shape
            if mode == "zero":
                return self.md(logpow, source_zero=torch.ones(b, dtype=torch.bool, device=logpow.device))
            lv = torch.as_tensor(np.asarray(levels, np.float32), device=logpow.device)
            return self.md(logpow, source_fed=lv.expand(b, t, -1))

    return Fed().eval()


def _rd_levels(rd):
    from vocal_analysis.synth.glottal import source_h_db

    return source_h_db(np.array([rd]), N_H)[0]


# ----------------------------------------------------------------------------- per-model inference

def _grids_fed(m, md, dev):
    """Arm E's D-155, D-156, D-157 and D-158 grids with channel 4 from the truth and at zero (cache/d161/score)."""
    d147, d155, d156, d157, d158 = (_mod(n) for n in ("d147_branch_spread", "d155_b2_grid", "d156_upper_bw", "d157_rd_bw", "d158_lf_offline"))
    from vocal_analysis.models.infer import analyse

    d160 = _d160()
    for mode in ("truth", "zero"):
        out = CACHE / f"grids_{mode}_{m}.json"
        if out.exists():
            continue
        res = {}
        z = np.load(ROOT / "cache/d155/clips.npz")
        ks = d155._keys()
        f = d147.fail_by_clip(_wrap(md, mode, _rd_levels(1.0)), dev, [(None, z[d155._key(k)], k[5]) for k in ks])
        res["d155"] = {d155._key(ks[j]): v for j, v in f.items()}
        d160._d156_grid(f"{m}@{mode}", _wrap(md, mode, _rd_levels(1.0)), dev)
        res["d156"] = json.loads((CACHE / f"d156_{m}@{mode}.json").read_text(encoding="utf-8"))
        ks = d157._keys()
        res["d157"] = {}
        for rd in d157.RDS:
            sub = [k for k in ks if k[4] == rd]
            f = d147.fail_by_clip(_wrap(md, mode, _rd_levels(rd)), dev,
                                  [(None, d157._audio(k[6], k[1], k[2], k[3], k[4], k[5]), k[6]) for k in sub])
            res["d157"].update({d157._key(sub[j]): v for j, v in f.items()})
        coefs = {s[:3]: d158._coefs(s[3]) for s in d158.sources()}
        res["d158"] = {}
        for k in d158._keys():
            c = coefs[k[0][:3]]
            y = d158._audio(k[6], k[2], k[3], k[4], k[5], c).astype(np.float32)
            s = d158._score(analyse(_wrap(md, mode, 20 * np.log10(np.abs(c[:N_H]))), y, dev), y, k[6], k[3], k[4])
            if s is not None:
                res["d158"][d158._key(k)] = s
        out.write_text(json.dumps(res), encoding="utf-8")
        print(m, mode, "grids", flush=True)


def _absorption(m, md, dev):
    """D-155's grid, the model's own source levels: per clip the mean over D-147's frames (dB re H2)."""
    from vocal_analysis.models.infer import analyse

    out = CACHE / f"absorb_{m}.json"
    if out.exists():
        return
    d155 = _mod("d155_b2_grid")
    z = np.load(ROOT / "cache/d155/clips.npz")
    res = {}
    for k in d155._keys():
        y = z[d155._key(k)].astype(np.float32)
        r = analyse(md, y, dev)
        n = len(r["voiced_p"])
        i = np.arange(int(0.2 * n), int(0.8 * n))
        i = i[(r["voiced_p"][i] > 0.5) & np.isfinite(r["f0"][i]) & (i * 160 >= 512) & (i * 160 + 512 <= len(y))]
        if len(i):
            res[d155._key(k)] = np.mean(r["source_levels"][i], 0).tolist()
    out.write_text(json.dumps(res), encoding="utf-8")


def _val_source(m, md, dev):
    """Synthetic (train_v2 and train_v2w: the first 512 validation clips) and resynthesised validation (the trainer's
    512 per set): per voiced label frame the head's levels, the 40 true levels, f0 and (resynthesised) beta*."""
    from vocal_analysis.models.infer import analyse
    from vocal_analysis.train.data import split_ids

    out = CACHE / f"valsrc_{m}.npz"
    if out.exists():
        return
    acc = {}
    for name, flat in {**SYN, **RES}.items():
        a = np.load(flat / "audio.npy", mmap_mode="r")
        ao, fo = np.load(flat / "audio_offsets.npy"), np.load(flat / "frame_offsets.npy")
        n_clips = len(ao) - 1
        lab40 = np.load(flat / ("label_source_h_db40.npy" if name in SYN else "label_source_h_db.npy"), mmap_mode="r")
        voi = np.load(flat / "label_voiced.npy", mmap_mode="r")
        f0l = np.load(flat / "label_f0.npy", mmap_mode="r")
        bst = np.load(flat / "label_beta_star.npy", mmap_mode="r") if name in RES else None
        ids = split_ids(n_clips)[1][:N_VAL] if name in SYN else np.sort(np.random.default_rng(0).permutation(n_clips)[:N_VAL])
        P, Y, F0, B = [], [], [], []
        for i in ids:
            y = np.asarray(a[ao[i]:ao[i + 1]], np.float32) / 32767.0
            r = analyse(md, y, dev)
            nn = min(len(r["source_levels"]), int(fo[i + 1] - fo[i]))
            v = np.asarray(voi[fo[i]:fo[i] + nn]) > 0.5
            P.append(r["source_levels"][:nn][v])
            Y.append(np.asarray(lab40[fo[i]:fo[i] + nn])[v])
            F0.append(np.asarray(f0l[fo[i]:fo[i] + nn])[v])
            if bst is not None:
                B.append(np.asarray(bst[fo[i]:fo[i] + nn])[v])
        acc[f"{name}|P"], acc[f"{name}|Y"], acc[f"{name}|f0"] = (np.concatenate(x).astype(np.float32) for x in (P, Y, F0))
        if B:
            acc[f"{name}|beta"] = np.concatenate(B).astype(np.float32)
    np.savez_compressed(out, **acc)


def _vs_source(m, md, dev):
    """D-153's VocalSet files: the head's median H1-H2 and slope (dB per octave, H2-H10 fitted on log2 k) over the
    file's voiced frames, by singing technique (a sanity check)."""
    import soundfile as sf
    from math import gcd

    from scipy.signal import resample_poly

    from vocal_analysis.models.infer import analyse

    out = CACHE / f"vssrc_{m}.json"
    if out.exists():
        return
    units = _mod("d153_vocalset_arms")._units()
    k = np.log2(np.arange(2, 11))
    res = []
    for u in units:
        y, sr = sf.read(ROOT / "data/vocalset" / u["meta"]["path"], dtype="float64", always_2d=True)
        g = gcd(sr, FS)
        r = analyse(md, resample_poly(y.mean(1), FS // g, sr // g).astype(np.float32), dev)
        v = r["voiced_p"] > 0.5
        if not v.any():
            continue
        L = r["source_levels"][v]
        slope = np.polyfit(k, L[:, 1:10].T, 1)[0]
        res.append({"tech": u["meta"]["tech"], "h1h2": float(np.median(L[:, 0] - L[:, 1])), "slope": float(np.median(slope))})
    out.write_text(json.dumps(res), encoding="utf-8")


def cmd_run(models):
    dev = _dev()
    d160 = _d160()
    CACHE.mkdir(parents=True, exist_ok=True)
    for m in models:
        md = _load(m, dev)
        d160._vocalset(m, md, dev)
        print(m, "vocalset", flush=True)
        _mod("d155_b2_grid").cmd_run([m])
        _mod("d157_rd_bw").cmd_run([m])
        _mod("d158_lf_offline").cmd_run([m])
        _mod("d159_d065_bw")._readout([m])
        d160._d156_grid(m, md, dev)
        print(m, "grids", flush=True)
        if m in ENV.values():
            _grids_fed(m, md, dev)
        if md.cfg.source_head:
            _absorption(m, md, dev)
            _val_source(m, md, dev)
            _vs_source(m, md, dev)
        d160._val(m, md, dev)
        d160._hillenbrand(m, md, dev)
        print(m, "done", flush=True)


def cmd_private(rec: Path, out: Path, files: list[str]):
    """The private recording's takes (D-149's form): per-take failure for both arms; counts only reach the repo."""
    d138 = _mod("d138_real_balance")
    lv = {t["take"]: t for t in json.loads((out / "levels.json").read_text(encoding="utf-8"))}
    dev = _dev()
    arms = [*ENV.values(), *ZERO.values()]
    models = {m: _load(m, dev) for m in arms}
    res = {}
    for name, _, y in d138._load(rec, files):
        if name not in lv:
            continue
        t = lv[name]
        rd, _ = d138._read(models, dev, y, np.array(t["F"], float))
        f5 = float(np.mean([t["read"][m]["fail"] for m in FIVEB.values()]))
        res[name] = {"group": "fail" if f5 >= d138.FAIL else "ok" if f5 < d138.OK else "mid", "5b": f5,
                     **{m: rd[m]["fail"] for m in arms}}
    (out / "d161_private.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- report

def _shape_err(P, Y, f0):
    """Per frame: RMS over H2-H40 below 4 kHz of the error with its mean removed (dB); NaN with fewer than 2 harmonics."""
    k = np.arange(1, N_H + 1)
    ok = np.isfinite(Y) & np.isfinite(P) & (f0[:, None] * k[None, :] < F_MAX) & (k[None, :] >= 2)
    e = np.where(ok, P - np.nan_to_num(Y), 0.0)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(ok, e - (e.sum(1) / n)[:, None], 0.0)
        r = np.sqrt((e ** 2).sum(1) / n)
    return np.where(n >= 2, r, np.nan)


def _mean_shape():
    """The training mix's mean source shape (dB re H2), sampled: train_v2w's training frames (0.7) and the resynthesised
    training sets' frames with beta* weight (0.3), voiced frames."""
    out = CACHE / "mean_shape.json"
    if out.exists():
        return np.array(json.loads(out.read_text(encoding="utf-8")))
    parts = []
    for flat, lab, wkey, share in ((SYN["v2w"], "label_source_h_db40.npy", None, 0.7),
                                   (ROOT / "data/resynth/vocalset_train_d16k_h40/flat", "label_source_h_db.npy", "label_beta_star.npy", 0.15),
                                   (ROOT / "data/resynth/zeroth_korean_train_d16k_h40/flat", "label_source_h_db.npy", "label_beta_star.npy", 0.15)):
        Y = np.load(flat / lab, mmap_mode="r")
        idx = np.arange(0, len(Y), 997)
        y = np.asarray(Y[idx], float)
        v = np.asarray(np.load(flat / "label_voiced.npy", mmap_mode="r")[idx]) > 0.5
        w = np.ones(len(idx))
        if wkey:
            b = np.asarray(np.load(flat / wkey, mmap_mode="r")[idx], float)
            w = np.where(np.isfinite(b), np.clip(1 - 2 * np.abs(b), 0, 1), 0)
        rel = y - y[:, 1:2]
        ok = v & np.isfinite(rel[:, 1])
        m = np.nansum(rel[ok] * w[ok, None], 0) / np.maximum(np.sum(np.isfinite(rel[ok]) * w[ok, None], 0), 1e-9)
        parts.append((share, m))
    shape = sum(s * m for s, m in parts) / sum(s for s, _ in parts)
    out.write_text(json.dumps(shape.tolist()), encoding="utf-8")
    return shape


def _vs_means(m, units):
    d160 = _d160()
    return d160._vs_shares(m, units)


def _grid_fail(sc, keys, sel):
    v = [sc[k] for k in keys if sel(k) and k in sc]
    return float(np.mean(v)) if v else float("nan")


def cmd_report(out_private: Path | None = None):
    d153, d155, d156, d157, d158 = (_mod(n) for n in ("d153_vocalset_arms", "d155_b2_grid", "d156_upper_bw", "d157_rd_bw", "d158_lf_offline"))
    d160 = _d160()
    units = d153._units()
    null = json.loads((ROOT / "reports/analysis/d153/d153.json").read_text(encoding="utf-8"))["runs"]
    d160r = json.loads((ROOT / "reports/analysis/d160/d160.json").read_text(encoding="utf-8"))["vocalset"]
    boot = np.random.default_rng(0).integers(0, len(units), (1000, len(units)))
    lines = ["D-161: a predicted source envelope as a fourth stage-2 channel (arm E) against a matched zero channel (arm Z);",
             "warm restarts from 5b's step 16000 on train_v2w (4000 constant + 4000 decay steps)", ""]
    res = {}
    # manipulation check: the head's shape error against a constant shape
    shape = _mean_shape()
    lines.append("  Manipulation check: median per-frame shape error over H2-H40 below 4 kHz (dB, offset removed), synthetic validation;")
    lines.append("  learned if the arm's head is at most half the constant (training mean) shape's error on both seeds")
    man = {}
    for s in (1, 2):
        for m in (ENV[s], ZERO[s]):
            z = np.load(CACHE / f"valsrc_{m}.npz")
            row = {}
            for name in (*SYN, *RES):
                P, Y, f0 = z[f"{name}|P"].astype(float), z[f"{name}|Y"].astype(float), z[f"{name}|f0"].astype(float)
                C = np.broadcast_to(shape, P.shape)
                eh, ec = _shape_err(P, Y, f0), _shape_err(C, Y, f0)
                row[name] = {"head": float(np.nanmedian(eh)), "const": float(np.nanmedian(ec))}
                if name in RES:
                    cl = np.isfinite(z[f"{name}|beta"]) & (z[f"{name}|beta"] < CLEAN)
                    row[name]["head_clean"] = float(np.nanmedian(eh[cl])) if cl.any() else None
                    row[name]["const_clean"] = float(np.nanmedian(ec[cl])) if cl.any() else None
            Ps = np.concatenate([z[f"{n}|P"] for n in SYN]).astype(float)
            Ys = np.concatenate([z[f"{n}|Y"] for n in SYN]).astype(float)
            fs = np.concatenate([z[f"{n}|f0"] for n in SYN]).astype(float)
            eh, ec = np.nanmedian(_shape_err(Ps, Ys, fs)), np.nanmedian(_shape_err(np.broadcast_to(shape, Ps.shape), Ys, fs))
            row["synthetic"] = {"head": float(eh), "const": float(ec), "ratio": float(eh / ec)}
            res.setdefault("head", {})[m] = row
            lines.append(f"    {m:16s} synthetic {eh:.2f} dB vs constant {ec:.2f} (ratio {eh / ec:.2f}); " + "; ".join(
                f"{n} {row[n]['head']:.2f}/{row[n]['const']:.2f}" + (f" (clean {row[n]['head_clean']:.2f}/{row[n]['const_clean']:.2f})"
                                                                     if row[n].get("head_clean") is not None else "")
                for n in (*SYN, *RES)))
        man[s] = res["head"][ENV[s]]["synthetic"]["ratio"]
    learned = all(r <= 0.5 for r in man.values())
    lines.append(f"    arm E's ratio: seed 1 {man[1]:.2f}, seed 2 {man[2]:.2f} -> " + ("LEARNED" if learned else "NOT LEARNED"))
    # VocalSet sanity: H1-H2 and slope by technique
    lines += ["", "  VocalSet (D-153's files): the head's median H1-H2 / slope H2-H10 (dB per octave) by technique"]
    for s in (1, 2):
        for m in (ENV[s], ZERO[s]):
            p = CACHE / f"vssrc_{m}.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text(encoding="utf-8"))
            techs = sorted({r["tech"] for r in rows})
            lines.append(f"    {m:16s} " + "; ".join(
                f"{t} {np.median([r['h1h2'] for r in rows if r['tech'] == t]):+.1f}/{np.median([r['slope'] for r in rows if r['tech'] == t]):+.1f}"
                f" (n {sum(r['tech'] == t for r in rows)})" for t in techs))
    # deciding: VocalSet
    lines += ["", f"  VocalSet (D-153's measure, {len(units)} files): mean failure share; arm E against every other run of the seed;",
              "  arm E minus arm Z [95 %, paired bootstrap over files]"]
    verdict, res["vocalset"] = [], {}
    for s in (1, 2):
        e = _vs_means(ENV[s], units)
        zz = _vs_means(ZERO[s], units)
        others = {}
        for m in OTHERS[s]:
            if m == ZERO[s]:
                others[m] = float(zz.mean())
            elif m == D160[s]:
                others[m] = d160r[m]["mean"]
            else:
                others[m] = null[m]["mean"]
        d = e - zz
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        below = e.mean() < min(others.values())
        verdict.append("lower" if below and hi < 0 else "not lower" if not below else "mixed")
        f5 = _vs_means(FIVEB[s], units)
        dz = zz - f5
        lz, hz = np.percentile(dz[boot].mean(1), [2.5, 97.5])
        res["vocalset"][s] = {"env": float(e.mean()), "others": others, "env_minus_zero": [float(d.mean()), float(lo), float(hi)],
                              "zero_minus_5b": [float(dz.mean()), float(lz), float(hz)], "failing_env": int((e >= 0.5).sum()),
                              "failing_zero": int((zz >= 0.5).sum()), "verdict": verdict[-1]}
        lines.append(f"    seed {s}: arm E {e.mean():.3f} ({int((e >= 0.5).sum())} failing files); others " +
                     ", ".join(f"{k} {v:.3f}" for k, v in others.items()) +
                     f"; E - Z {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict[-1]}")
        vow = np.array([u["meta"]["vowel"] for u in units])
        lines.append("      failing files by vowel, 5b / Z / E: " + ", ".join(
            f"{v} {int((f5[vow == v] >= 0.5).sum())}/{int((zz[vow == v] >= 0.5).sum())}/{int((e[vow == v] >= 0.5).sum())}" for v in ("o", "u", "a")))
        lines.append(f"      arm Z against 5b (a second reading of D-160's data change, 4000 more steps): {dz.mean():+.3f} [{lz:+.3f}, {hz:+.3f}];"
                     f" D-160 {d160r[D160[s]]['mean']:.3f}")
        for m in (ENV[s], ZERO[s]):
            if (CACHE / f"vs_{m}.npz").exists():
                fail = e if m == ENV[s] else zz
                auc = d160._flag_auc(m, units, fail)
                lines.append(f"      flag B AUC, {m}: " + (f"{auc[0]:.2f} ({auc[1]} / {auc[2]})" if auc[0] is not None else f"too few ({auc[1]} / {auc[2]})"))
    if not learned:
        vs = "NOT LEARNED: the VocalSet result is reported but uninformative"
    elif verdict == ["lower", "lower"]:
        vs = "LOWER ON REAL VOICES"
    elif "not lower" in verdict:
        vs = "NOT LOWER"
    else:
        vs = "MIXED: reported, Hannah decides"
    # reported: the grids, channel 4 from the head / truth / zero
    k155 = [d155._key(k) for k in d155._keys()]
    kk155 = {d155._key(k): k for k in d155._keys()}
    lines += ["", "  Grids, channel 4 from the head / the truth / zero (arm E; arm Z and 5b with their own):",
              "    D-155 low f0 (110, 140): failure at B2 250-400 and at B2 90; D-157: failure by Rd (0.6 / 1.0 / 1.8 / 2.7), low f0, B1 360;",
              "    D-158: failure by anchor and Ra factor, low f0, B1 360"]
    res["grids"] = {}
    for s in (1, 2):
        rows = [(FIVEB[s], "own"), (ZERO[s], "own"), (ENV[s], "head"), (ENV[s], "truth"), (ENV[s], "zero")]
        for m, mode in rows:
            if mode in ("own", "head"):
                g155 = json.loads((ROOT / f"cache/d155/{m}.json").read_text(encoding="utf-8"))
                g157 = json.loads((ROOT / f"cache/d157/{m}.json").read_text(encoding="utf-8"))
                g158 = json.loads((ROOT / f"cache/d158/{m}.json").read_text(encoding="utf-8"))
            else:
                G = json.loads((CACHE / f"grids_{mode}_{m}.json").read_text(encoding="utf-8"))
                g155, g157, g158 = G["d155"], G["d157"], G["d158"]
            wide = _grid_fail(g155, k155, lambda k: kk155[k][1] <= 140 and kk155[k][3] >= 250)
            nar = _grid_fail(g155, k155, lambda k: kk155[k][1] <= 140 and kk155[k][3] == 90)
            k157 = {d157._key(k): k for k in d157._keys()}
            byrd = [_grid_fail(g157, list(k157), lambda k, rd=rd: k157[k][1] <= 140 and k157[k][2] == 360 and k157[k][4] == rd) for rd in d157.RDS]
            k158 = {d158._key(k): k for k in d158._keys()}
            by158 = []
            for rd in d158.ANCHORS:
                for p_, f_ in (("-", 1.0), ("Ra", 0.5), ("Ra", 3.0)):
                    v = [g158[k]["fail"] for k, kk in k158.items() if k in g158 and kk[0][0] == rd and kk[0][1] == p_ and kk[0][2] == f_
                         and kk[2] <= 140 and kk[3] == 360]
                    by158.append((rd, p_, f_, float(np.mean(v)) if v else float("nan")))
            res["grids"][f"{m}|{mode}"] = {"d155_wide": wide, "d155_b2_90": nar, "d157": byrd, "d158": by158}
            lines.append(f"    {m:16s} {mode:5s} D-155 {wide:.2f} ({nar:.2f}); D-157 " + " / ".join(f"{v:.2f}" for v in byrd) +
                         "; D-158 " + "  ".join(f"{rd:g}{'' if p_ == '-' else f' Ra x{f_:g}'} {v:.2f}" for rd, p_, f_, v in by158))
    # D-156 part B by mode
    lines += ["", "  D-156 part B (B2 90): failure / F3 or F4 error (c) by step"]
    for s in (1, 2):
        for m, lab in ((FIVEB[s], FIVEB[s]), (ZERO[s], ZERO[s]), (ENV[s], ENV[s]), (ENV[s], f"{ENV[s]}@truth"), (ENV[s], f"{ENV[s]}@zero")):
            p = CACHE / f"d156_{lab}.json" if "@" in lab or m != FIVEB[s] else ROOT / f"cache/d160/d156_{m}.json"
            if not p.exists():
                continue
            G = json.loads(p.read_text(encoding="utf-8"))
            cells = []
            for kk, steps in d156.STEPS.items():
                for b in steps:
                    rr = [g for g in G if g["k"] == kk and g["b"] == b]
                    cells.append(f"B{kk} {b:.0f}: {np.mean([g['fail'] for g in rr]):.2f}/{np.median([g[f'e{kk}'] for g in rr]):.0f}")
            lines.append(f"    {lab:22s} " + "  ".join(cells))
    # absorption
    lines += ["", "  Absorption, D-155's grid (true source fixed): median |change| of the head's levels (dB) at harmonics within one",
              "  bandwidth of F1 (B1 150 -> 700, B2 90) and of F2 (B2 90 -> 400, B1 150), and of its H1-H2; by f0"]
    res["absorption"] = {}
    for s in (1, 2):
        for m in (ENV[s], ZERO[s]):
            p = CACHE / f"absorb_{m}.json"
            if not p.exists():
                continue
            A = json.loads(p.read_text(encoding="utf-8"))
            row = {}
            for band, sel in (("f0 <= 150", lambda f: f <= 150), ("f0 > 150", lambda f: f > 150)):
                d1, d2, dh = [], [], []
                for (st, f0, b1, b2, sd, Fs) in d155._keys():
                    if not sel(f0):
                        continue
                    kk = np.arange(1, N_H + 1) * f0
                    if b1 == 150 and b2 == 90:
                        a = A.get(d155._key((st, f0, 700.0, 90.0, sd, Fs)))
                        b_ = A.get(d155._key((st, f0, 150.0, 90.0, sd, Fs)))
                        if a and b_:
                            near = np.abs(kk - Fs[0]) <= 700
                            d1.append(np.median(np.abs(np.array(a)[near] - np.array(b_)[near])) if near.any() else np.nan)
                            dh.append(abs((a[0] - a[1]) - (b_[0] - b_[1])))
                        a = A.get(d155._key((st, f0, 150.0, 400.0, sd, Fs)))
                        if a and b_:
                            near = np.abs(kk - Fs[1]) <= 400
                            d2.append(np.median(np.abs(np.array(a)[near] - np.array(b_)[near])) if near.any() else np.nan)
                row[band] = {"F1": float(np.nanmedian(d1)) if d1 else None, "F2": float(np.nanmedian(d2)) if d2 else None,
                             "H1H2": float(np.nanmedian(dh)) if dh else None}
            res["absorption"][m] = row
            lines.append(f"    {m:16s} " + "; ".join(f"{b}: F1 {r['F1']:.2f}, F2 {r['F2']:.2f}, H1-H2 {r['H1H2']:.2f}" for b, r in row.items()
                                                    if r["F1"] is not None))
    # resolvability, front vowels and Hillenbrand: D-160's measures
    lines += ["", "  synthetic validation (512 clips, clean): pair (Fk, Fk+1) unresolvable if spacing < mean bandwidth or < f0;",
              "  median |error| of the pair's two formants (c) and median spread (c); front vowels (label F2 > 1800): F2 / F3 |error| (c)"]
    for name in ("v2", "v2w"):
        for s in (1, 2):
            for m in (FIVEB[s], ZERO[s], ENV[s]):
                p = CACHE / f"val_{m}.npz" if m != FIVEB[s] else ROOT / f"cache/d160/val_{m}.npz"
                if not p.exists():
                    continue
                X = np.load(p)[name]
                Fl, Bl, f0, Fp, sg = X[:, :4], X[:, 4:8], X[:, 8], X[:, 9:13], X[:, 13:17]
                with np.errstate(invalid="ignore", divide="ignore"):
                    err = np.abs(np.log2(Fp / Fl)) * 1200
                    cells = []
                    for k in range(3):
                        un = ((Fl[:, k + 1] - Fl[:, k]) < (Bl[:, k] + Bl[:, k + 1]) / 2) | ((Fl[:, k + 1] - Fl[:, k]) < f0)
                        ok = np.all(np.isfinite(err[:, k:k + 2]), 1)
                        cells.append(f"F{k + 1}F{k + 2} res {np.median(err[ok & ~un, k:k + 2]):.0f}c/{np.median(sg[ok & ~un, k:k + 2]):.0f}c, "
                                     f"unres {np.median(err[ok & un, k:k + 2]):.0f}c/{np.median(sg[ok & un, k:k + 2]):.0f}c")
                    fv = Fl[:, 1] > 1800
                    cells.append(f"front F2 {np.nanmedian(err[fv, 1]):.0f}c F3 {np.nanmedian(err[fv, 2]):.0f}c")
                lines.append(f"    {name:3s} {m:16s} " + "; ".join(cells))
    lines += ["", "  Hillenbrand dev front vowels (iy ih ei eh ae): median |F2| / |F3| error against the hand-checked values (c)"]
    d150 = json.loads((ROOT / "reports/analysis/d150/d150.json").read_text(encoding="utf-8"))
    for s in (1, 2):
        for m in (FIVEB[s], ZERO[s], ENV[s]):
            if m in FIVEB.values():
                fr = [(f["ref"], f.get(m)) for t in d150 if t["vowel"] in d160.FRONT for f in t["frames"]]
            elif (CACHE / f"hill_{m}.json").exists():
                fr = [(f["ref"], f["read"]) for t in json.loads((CACHE / f"hill_{m}.json").read_text(encoding="utf-8")) for f in t["frames"]]
            else:
                continue
            fr = [(np.array(r), np.array(v)) for r, v in fr if v is not None]
            lines.append(f"    {m:16s} F2 {np.median([abs(np.log2(v[1] / r[1])) * 1200 for r, v in fr]):.0f} c, "
                         f"F3 {np.median([abs(np.log2(v[2] / r[2])) * 1200 for r, v in fr]):.0f} c (n {len(fr)})")
    # private counts
    if out_private is not None and (out_private / "d161_private.json").exists():
        P = json.loads((out_private / "d161_private.json").read_text(encoding="utf-8"))
        fail = [v for v in P.values() if v["group"] == "fail"]
        ok = [v for v in P.values() if v["group"] == "ok"]
        lines += ["", f"  a private recording: {len(fail)} takes 5b fails, {len(ok)} it reads; counts by run (fails < 0.5 / new failures >= 0.5):"]
        for m in (*ENV.values(), *ZERO.values()):
            lines.append(f"    {m}: reads {sum(v[m] < 0.5 for v in fail)} of {len(fail)} failing takes; {sum(v[m] >= 0.5 for v in ok)} new failures")
    # guards
    for a in ("env", "zero"):
        g = REP / f"decide_guards_{a}.txt"
        if g.exists():
            lines += ["", f"  guards, arm {'E' if a == 'env' else 'Z'} against 5b (D-051's, D-065's script; its primary line is D-065's bw criterion, reported only):"]
            lines += ["    " + ln for ln in g.read_text(encoding="utf-8").splitlines() if ln.strip()][:60]
    res["reading"] = {"learned": learned, "head_ratio": man, "vocalset": vs, "per_seed": verdict}
    lines += ["", f"Reading (registered): {vs}"]
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "d161.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (REP / "d161.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2].split(","))
    elif cmd == "private":
        cmd_private(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(","))
    else:
        cmd_report(Path(sys.argv[2]) if len(sys.argv) > 2 else None)
