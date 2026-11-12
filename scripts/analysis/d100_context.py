"""D-100 (descriptive): aVTL length reliability on dev voices with longer context, and bounds on the iVTL head's share.

Each item (a Hillenbrand dev word, men and women; a PTDB-TUG dev sentence, the first 40 per speaker) is read alone
(R1, analyse()'s default 4 s windows), inside a same-speaker concatenation through the default windows (R2) and the
same concatenation in one pass (R3); PTDB-TUG's sentences also alone in one pass (R0). Always the item's own frames
(Hillenbrand's reference frames; PTDB-TUG's voiced reference frames), the precision-weighted mean of vtl_score_joint.
Reliability as vtl_score_joint (between-speaker variance of speaker means / (that + within-speaker variance), mean of
men and women), paired speaker bootstrap. For models with an iVTL output, the within-speaker regression of aVTL on
iVTL: its R^2 and the reliability with that part removed (an upper bound on a perfect split's gain).

    python scripts/analysis/d100_context.py [name=ckpt ...] > reports/analysis/d100/context.txt

Per-model reads are cached in cache/d100/<name>.npz (skipped if present). Without arguments, D-100's models.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SR, HOP = 16000, 160
GAP = np.zeros(int(0.2 * SR), np.float32)
N_PTDB = 40
CACHE = Path("cache/d100")
MODELS = {r: f"runs/{r}/ckpt_last.pt" for r in (
    "d040_5b_s1_b20k", "d040_5b_s2_b20k", "d051_5c_s1_b20k", "d051_5c_s2_b20k", "d080_a0_s1", "d080_a0_s2",
    "d098_base_b20k", "d098_heads_b20k")}
GROUPS = {"5b": ("d040_5b_s1_b20k", "d040_5b_s2_b20k"), "5c": ("d051_5c_s1_b20k", "d051_5c_s2_b20k"),
          "A0": ("d080_a0_s1", "d080_a0_s2"), "D-098 base": ("d098_base_b20k",), "D-098 heads": ("d098_heads_b20k",),
          "D-099 narrow": ("d099_narrow_b20k",), "D-099 base s2": ("d099_base_s2_b20k",)}
READS = {"hillenbrand": ("R1", "R2", "R3"), "ptdb": ("R1", "R0", "R2", "R3")}
READ_NAMES = {"R1": "alone, 4 s windows", "R0": "alone, one pass", "R2": "in context, 4 s windows", "R3": "in context, one pass"}


def load_items():
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI
    from vocal_analysis.eval.datasets import load_audio, registry

    mods = registry()
    sets = {}
    hb = []
    for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), ("dev",)):
        item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
        g = item.meta.get("group", "?")
        ok = np.isfinite(item.times)
        if g in "mw" and ok.any():
            hb.append((load_audio(item, SR).astype(np.float32), item.times[ok], row.speaker_id, g))
    sets["hillenbrand"] = hb
    pt, count = [], {}
    for row in TI.filter_rows(read_manifest("data/manifests/ptdb_tug.csv"), ("dev",)):
        if count.get(row.speaker_id, 0) >= N_PTDB:
            continue
        item = mods["ptdb_tug"].load(Path("data/ptdb_tug"), row)
        v = np.isfinite(item.f0) & (item.f0 > 0) & np.isfinite(item.times)
        if v.sum() < 10:
            continue
        count[row.speaker_id] = count.get(row.speaker_id, 0) + 1
        sex = "w" if row.speaker_id.split(":")[-1].startswith("F") else "m"
        pt.append((load_audio(item, SR).astype(np.float32), item.times[v], row.speaker_id, sex))
    sets["ptdb"] = pt
    return sets


def concatenations(items, per):
    """same-speaker concatenations in a seeded order: list of (audio, [(item index, offset s)]); `per` items each (None = all)"""
    rng = np.random.default_rng(100)
    out = []
    for s in sorted({it[2] for it in items}):
        idx = [i for i, it in enumerate(items) if it[2] == s]
        idx = [idx[j] for j in rng.permutation(len(idx))]
        chunks = [idx] if per is None else [idx[k:k + per] for k in range(0, len(idx), per)]
        for ch in chunks:
            parts, offs, t = [], [], 0.0
            for i in ch:
                offs.append((i, t))
                parts += [items[i][0], GAP]
                t += (len(items[i][0]) + len(GAP)) / SR
            out.append((np.concatenate(parts[:-1]), offs))
    return out


def agg(mu, sig):
    w = 1 / sig ** 2
    k = np.isfinite(mu) & np.isfinite(w)
    return float((w[k] * mu[k]).sum() / w[k].sum()) if k.any() else np.nan


def read_item(r, times, offset=0.0):
    idx = np.clip(np.round((times + offset) / 0.01).astype(int), 0, len(r["vtl_log2cm"]) - 1)
    a = agg(r["vtl_log2cm"][idx], r["vtl_sigma"][idx])
    i = agg(r["vtl_inst_log2cm"][idx], r["vtl_inst_sigma"][idx]) if "vtl_inst_log2cm" in r else np.nan
    return a, i


def score_model(name, ck, sets, cats, dev):
    import torch

    from vocal_analysis.models.infer import analyse, load_teacher

    model = load_teacher(ck, dev)[0]
    res = {}
    for ds, items in sets.items():
        n = len(items)
        for rd in READS[ds]:
            res[f"{ds}/{rd}/a"], res[f"{ds}/{rd}/i"] = np.full(n, np.nan), np.full(n, np.nan)
        for j, (x, t, _, _) in enumerate(items):
            r = analyse(model, x, dev)
            res[f"{ds}/R1/a"][j], res[f"{ds}/R1/i"][j] = read_item(r, t)
            if "R0" in READS[ds]:
                r = analyse(model, x, dev, window_frames=None)
                res[f"{ds}/R0/a"][j], res[f"{ds}/R0/i"][j] = read_item(r, t)
        for audio, offs in cats[ds]:
            for rd, wf in (("R2", 400), ("R3", None)):
                r = analyse(model, audio, dev, window_frames=wf)
                for j, off in offs:
                    res[f"{ds}/{rd}/a"][j], res[f"{ds}/{rd}/i"][j] = read_item(r, items[j][1], off)
        print(f"  {name} {ds}: done", file=sys.stderr, flush=True)
    del model
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return res


# --- measures -------------------------------------------------------------------------------------------------------
def reliability(e, spk, draw):
    """draw: {sex: speaker list (with repeats)} -> (reliability, within SD c, w/m)"""
    rel, within, med = {}, [], {}
    for G, ss in draw.items():
        means = np.array([np.nanmean(e[spk == s]) for s in ss])
        dev2 = np.concatenate([e[spk == s] - np.nanmean(e[spk == s]) for s in ss])
        wv = np.nanmean(dev2 ** 2)
        rel[G] = np.var(means) / (np.var(means) + wv)
        within.append(dev2)
        med[G] = np.median(means)
    return (rel["m"] + rel["w"]) / 2, 1200 * np.sqrt(np.nanmean(np.concatenate(within) ** 2)), 2 ** (med["w"] - med["m"])


def residualise(a, i, spk):
    """aVTL minus its pooled within-speaker regression on iVTL; and that regression's R^2"""
    ac, ic = a.copy(), i.copy()
    for s in np.unique(spk):
        m = spk == s
        ac[m] -= np.nanmean(a[m])
        ic[m] -= np.nanmean(i[m])
    k = np.isfinite(ac) & np.isfinite(ic)
    b = (ic[k] @ ac[k]) / (ic[k] @ ic[k])
    r2 = 1 - np.sum((ac[k] - b * ic[k]) ** 2) / np.sum(ac[k] ** 2)
    return a - b * (i - np.array([np.nanmean(i[spk == s]) for s in spk])), r2, b


if __name__ == "__main__":
    from vocal_analysis.device import setup_device

    args = dict(a.split("=", 1) for a in sys.argv[1:]) or MODELS
    models = {k: v for k, v in args.items() if Path(v).exists() or (CACHE / f"{k}.npz").exists()}
    CACHE.mkdir(parents=True, exist_ok=True)
    sets = load_items()
    cats = {"hillenbrand": concatenations(sets["hillenbrand"], None), "ptdb": concatenations(sets["ptdb"], 2)}
    todo = [k for k in models if not (CACHE / f"{k}.npz").exists()]
    if todo:
        dev = setup_device()
        for k in todo:
            np.savez(CACHE / f"{k}.npz", **score_model(k, models[k], sets, cats, dev))
    R = {k: dict(np.load(CACHE / f"{k}.npz")) for k in models}
    groups = {g: ms for g, ms in GROUPS.items() if all(m in R for m in ms)}

    L = ["D-100: aVTL length reliability on dev voices with longer context, and the iVTL head's share (descriptive)",
         "reads: " + "; ".join(f"{k} {v}" for k, v in READ_NAMES.items()) + ". Same frames and averaging in every read.",
         "values: reliability [95 % paired speaker bootstrap, 2000]; seed means where a model has two seeds", ""]
    rng = np.random.default_rng(0)
    for ds, items in sets.items():
        spk = np.array([it[2] for it in items])
        sex = {s: it[3] for it in items for s in [it[2]]}
        spks = {G: sorted(s for s in set(spk) if sex[s] == G) for G in "mw"}
        draws = [{G: list(rng.choice(spks[G], len(spks[G]))) for G in "mw"} for _ in range(2000)]
        dur = np.median([len(it[0]) / SR for it in items])
        cl = np.median([len(a) / SR for a, _ in cats[ds]])
        L.append(f"== {ds} dev: {len(spks['m'])} men, {len(spks['w'])} women, {len(items)} items (median {dur:.2f} s); "
                 f"concatenations median {cl:.1f} s")

        def val(g, rd, draw=None, kind="raw"):
            """seed-mean (reliability, within SD, w/m) for a group"""
            out = []
            for m in groups[g]:
                a = R[m][f"{ds}/{rd}/a"]
                if kind == "resid":
                    a = residualise(a, R[m][f"{ds}/{rd}/i"], spk)[0]
                out.append(reliability(a, spk, draw or spks))
            return np.mean(out, 0)

        def ci(f):
            b = [f(d) for d in draws]
            return np.percentile(b, [2.5, 97.5])

        L.append(f"  {'model':14s} " + " | ".join(f"{rd:>22s}" for rd in READS[ds]) + "   (reliability; within SD c; w/m)")
        for g in groups:
            cells = []
            for rd in READS[ds]:
                rl, sd, wm = val(g, rd)
                cells.append(f"{rl:.3f} {sd:6.0f} c {wm:.3f}")
            L.append(f"  {g:14s} " + " | ".join(f"{c:>22s}" for c in cells))
        L.append("")
        L.append("  context at inference (read - R1, reliability):")
        for g in groups:
            parts = []
            for rd in READS[ds][1:]:
                d = val(g, rd)[0] - val(g, "R1")[0]
                lo, hi = ci(lambda dr: val(g, rd, dr)[0] - val(g, "R1", dr)[0])
                parts.append(f"{rd} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"    {g:14s} " + "; ".join(parts))
        L.append("")
        L.append("  between models (reliability, paired):")
        for a_, b_ in (("5c", "5b"), ("A0", "5b"), ("D-098 base", "5b"), ("D-098 base", "A0"), ("D-098 heads", "D-098 base"),
                       ("D-099 narrow", "5b"), ("D-098 base", "D-099 narrow"), ("D-099 base s2", "5b")):
            if a_ not in groups or b_ not in groups:
                continue
            parts = []
            for rd in READS[ds]:
                d = val(a_, rd)[0] - val(b_, rd)[0]
                lo, hi = ci(lambda dr: val(a_, rd, dr)[0] - val(b_, rd, dr)[0])
                parts.append(f"{rd} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L.append(f"    {a_ + ' - ' + b_:26s} " + "; ".join(parts))
        L.append("")
        L.append("  iVTL within a model: R^2 of aVTL on iVTL within speakers (slope); reliability with that part removed - raw:")
        for g in groups:
            if not all(np.isfinite(R[m][f"{ds}/R1/i"]).any() for m in groups[g]):
                continue
            parts = []
            for rd in READS[ds]:
                r2b = np.mean([residualise(R[m][f"{ds}/{rd}/a"], R[m][f"{ds}/{rd}/i"], spk)[1:] for m in groups[g]], 0)
                d = val(g, rd, kind="resid")[0] - val(g, rd)[0]
                lo, hi = ci(lambda dr: val(g, rd, dr, "resid")[0] - val(g, rd, dr)[0])
                parts.append(f"{rd} R^2 {r2b[0]:.2f} (b {r2b[1]:.2f}), {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {val(g, rd, kind='resid')[0]:.3f}")
            L.append(f"    {g:14s} " + "; ".join(parts))
        L.append("")
    print("\n".join(L))
