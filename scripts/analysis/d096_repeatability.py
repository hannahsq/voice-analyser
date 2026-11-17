"""D-096 (descriptive, no decision rule): how repeatable are v3.1's outputs within a speaker, and how small a change can a
recording show? PTDB-TUG dev (12 speakers), windowed analysis (analyse()'s default), v3.1 seed 2 (`d040_5b_s2_b20k`) with
seed 1 beside it.

Per utterance (every dev utterance): medians over the voiced sonorant frames of f0, F1-F4, the anatomical length, and
H1*-H2* (Iseli-Alwan corrected with the model's own F1-F4 and B1-B4; D-096 amendment 1) with raw H1-H2 beside it.
(1) within-speaker SD across utterances; (2) the same for pseudo-recordings of k = 1, 2, 5, 10, 20 utterances (200 draws
without replacement), the standard error of measurement (SEM) against recording length; (3) minimal detectable change at
95 %, 1.96 x sqrt(2) x SEM; (4) the channel term: the paired shift of the same utterances under D-050's channel-probe
conditions (`vocal_analysis/eval/probes/channel.py`, its cached audio), and the root sum of squares with (3). Amendment 1:
a crossed speaker + sentence + residual decomposition per output (sentences read by at least three dev speakers of the
sex), and the minimal detectable change for the same text (residual alone).

    score:  python scripts/analysis/d096_repeatability.py score [--no-channel] [name=ckpt ...]   (GPU: VA_DEVICE=cuda)
    report: python scripts/analysis/d096_repeatability.py report [name ...] > reports/analysis/d096/d096.txt   (CPU only)

Scoring caches per speaker, so an interrupted run resumes: cache/d096/<name>/<speaker>.npz (per-frame outputs at the
utterance's frames) and cache/d096/<name>/channel_<speaker>.npz (per file and condition, the eight item values). Without
arguments: v3.1's two seeds. The channel step needs data/probes/channel_v1 (`python -m vocal_analysis.eval.probes.channel
render`, already done for D-050).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore", message="(Mean of empty slice|All-NaN|Degrees of freedom|invalid value)")

FS, HOP = 16000, 160
CACHE = Path("cache/d096")
CHANNEL_CACHE = Path("data/probes/channel_v1")
MODELS = {r: f"runs/{r}/ckpt_last.pt" for r in ("d040_5b_s2_b20k", "d040_5b_s1_b20k")}  # primary first
OUTS = ("f0", "F1", "F2", "F3", "F4", "L", "H1*-H2*", "H1-H2")
UNITS = ("Hz", "Hz", "Hz", "Hz", "Hz", "cm", "dB", "dB")
N_LOG = 6  # outputs 0..5 are positive quantities, reported in cents (1200 log2); the H measures are in dB
KS = (1, 2, 5, 10, 20)
N_DRAWS = 200
N_BOOT = 2000
# CHOICE: an utterance needs at least this many voiced sonorant frames (d100 uses >= 10 voiced frames); fewer = excluded.
MIN_FRAMES = 10
# The size of change each output would need to show, from the outcomes literature (D-096 "Reported"): to be filled with
# cited values when written; nothing is invented here. {output name: (value, unit, citation)}
NEEDED: dict = {}


# --- scoring (GPU) ----------------------------------------------------------------------------------------------------
def item_values(fr: dict) -> np.ndarray:
    """eight utterance values: medians over the cached frames (Hz, Hz x4, cm, dB, dB)"""
    cols = [fr["f0"], *fr["F"].T, fr["L"], fr["H1H2c"], fr["H1H2"]]
    with np.errstate(all="ignore"):
        return np.array([np.nanmedian(c) if len(c) else np.nan for c in cols])


def read_frames(model, x, dev, times, ref_f0, sel=None):
    """analyse() the audio and read the frames at the reference's frame times. sel (bool over reference frames) is the
    frame set: None = decide it here (reference voiced and the model's sonorant_p > 0.5), else reuse it (channel arms)."""
    from vocal_analysis.backend.corrections import correct
    from vocal_analysis.backend.harmonics import harmonic_amplitudes
    from vocal_analysis.models.infer import analyse

    r = analyse(model, x, dev)
    idx = np.clip(np.round(times / 0.01).astype(int), 0, len(r["f0"]) - 1)
    if sel is None:
        # CHOICE: "voiced sonorant frames": PTDB-TUG has no sonorant labels, so the reference's voiced frames (f0 > 0,
        # the laryngograph-derived track, as D-100) AND the model's own sonorant_p > 0.5 (as the reversal probe).
        # Alternative: voiced by the model's voiced_p, or no sonorant condition (all reference-voiced frames).
        sel = np.isfinite(ref_f0) & (ref_f0 > 0) & (r["sonorant_p"][idx] > 0.5)
    ii = idx[sel]
    f0, F, B = r["f0"][ii], r["formants"][ii, :4], r["bandwidths"][ii, :4]
    if len(ii):
        # CHOICE: H1, H2 are fitted at the model's own f0 (the quantity that would be reported), not the reference's.
        h = harmonic_amplitudes(x.astype(np.float32), FS, ii * HOP / FS, f0, 2)
        hc = correct(h, f0, F, B, FS, upper_from_vtl=True)  # as D-053: upper poles from the apparent VTL
        h1h2, h1h2c = h[:, 0] - h[:, 1], hc[:, 0] - hc[:, 1]
    else:
        h1h2 = h1h2c = np.zeros(0)
    fr = {"f0": f0, "F": F, "B": B, "L": 2.0 ** r["vtl_log2cm"][ii], "son": r["sonorant_p"][ii], "vp": r["voiced_p"][ii],
          "H1H2": h1h2, "H1H2c": h1h2c}
    return fr, sel


def safe(spk: str) -> str:
    return spk.replace(":", "_")


def dev_rows_by_speaker():
    from vocal_analysis.data.manifest import read_manifest
    from vocal_analysis.eval import tiers as TI

    by = {}
    for row in TI.filter_rows(read_manifest("data/manifests/ptdb_tug.csv"), ("dev",)):  # dev tier only
        by.setdefault(row.speaker_id, []).append(row)
    return by


def score_main(name, model, dev, log=print):
    from vocal_analysis.eval.datasets import load_audio, registry

    mod = registry()["ptdb_tug"]
    (CACHE / name).mkdir(parents=True, exist_ok=True)
    for spk, rows in sorted(dev_rows_by_speaker().items()):
        out = CACHE / name / f"{safe(spk)}.npz"
        if out.exists():
            continue
        utt, ns, parts = [], [], {k: [] for k in ("f0", "F", "B", "L", "son", "vp", "H1H2", "H1H2c")}
        for row in rows:
            item = mod.load(Path("data/ptdb_tug"), row)
            fr, sel = read_frames(model, load_audio(item, FS), dev, item.times, item.f0)
            utt.append(Path(row.path).stem), ns.append(len(fr["f0"]))
            for k in parts:
                parts[k].append(fr[k])
        np.savez(out, spk=spk, utt=np.array(utt), n=np.array(ns), **{k: np.concatenate(v) for k, v in parts.items()})
        log(f"  {name} {spk}: {len(rows)} utterances", file=sys.stderr, flush=True)


def score_channel(name, model, dev, log=print):
    meta = __import__("json").loads((CHANNEL_CACHE / "meta.json").read_text(encoding="utf-8"))
    conds = meta["conditions"]
    by = {}
    for it in meta["items"]:
        if it["corpus"] == "ptdb_tug":
            by.setdefault(it["speaker"], []).append(it["file"])
    for spk, files in sorted(by.items()):
        out = CACHE / name / f"channel_{safe(spk)}.npz"
        if out.exists():
            continue
        val = np.full((len(files), len(conds), len(OUTS)), np.nan)
        nsel = np.zeros(len(files), int)
        for j, f in enumerate(files):
            d = np.load(CHANNEL_CACHE / f)
            sel = None
            for c in range(len(conds)):  # condition 0 is clean and fixes the frame set for every condition
                fr, sel = read_frames(model, d[f"audio_{c}"].astype(np.float32) / 32000, dev, d["times"], d["ref_f0"], sel)
                val[j, c] = item_values(fr)
            nsel[j] = int(sel.sum())
        np.savez(out, spk=spk, files=np.array(files), conds=np.array(conds), val=val, nsel=nsel)
        log(f"  {name} channel {spk}: {len(files)} files x {len(conds)} conditions", file=sys.stderr, flush=True)


def score(models: dict, channel: bool = True):
    import torch

    from vocal_analysis.device import setup_device
    from vocal_analysis.models.infer import load_teacher

    dev = setup_device()
    for name, ck in models.items():
        model = load_teacher(ck, dev)[0]
        score_main(name, model, dev, log=print)
        if channel:
            score_channel(name, model, dev, log=print)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# --- statistics (CPU) ---------------------------------------------------------------------------------------------------
def load_model(name):
    """-> list of items {spk, sex, sent, utt, fr: {field: array}, nfr} from the per-speaker caches"""
    items = []
    for f in sorted((CACHE / name).glob("*.npz")):
        if f.name.startswith("channel_"):
            continue
        z = np.load(f)
        spk, off = str(z["spk"]), np.concatenate([[0], np.cumsum(z["n"])])
        for j, u in enumerate(z["utt"]):
            sl = slice(off[j], off[j + 1])
            u = str(u)
            items.append({"spk": spk, "sex": "w" if spk.split(":")[-1].startswith("F") else "m",
                          "sent": u.split("_", 2)[-1],  # mic_F01_sa1 -> sa1: the sentence id within the corpus
                          "utt": u, "nfr": int(z["n"][j]), "fr": {k: z[k][sl] for k in ("f0", "F", "L", "H1H2", "H1H2c")}})
    return items


def frame_matrix(it):
    """(frames, 8) per-frame values in OUTS order"""
    fr = it["fr"]
    return np.column_stack([fr["f0"], fr["F"], fr["L"], fr["H1H2c"], fr["H1H2"]])


def cents(v):
    """native values (..., 8) -> cents for the first N_LOG outputs, NaN for the dB ones"""
    out = np.full(np.shape(v), np.nan)
    with np.errstate(all="ignore"):
        out[..., :N_LOG] = 1200 * np.log2(np.asarray(v)[..., :N_LOG])
    return out


def draw_sets(n_items: int, k: int, rng) -> np.ndarray:
    """(N_DRAWS, k) item indices, each draw without replacement"""
    return np.array([rng.choice(n_items, k, replace=False) for _ in range(N_DRAWS)])


def speaker_variances(mats, nfr, V, sets, how="pooled"):
    """per k: (variance over pseudo-recordings of each output in native units, same in cents, mean voiced duration s).
    k = 1 is the utterances themselves (CHOICE: exact, rather than 200 draws of one). mats: frame matrices, V: utterance
    values (n, 8), sets: {k: (draws, k) indices}."""
    out = {1: (np.nanvar(V, 0, ddof=1), np.nanvar(cents(V), 0, ddof=1), float(np.mean(nfr)) * 0.01)}
    for k, s in sets.items():
        if how == "pooled":
            # CHOICE: a pseudo-recording's value is the median over all voiced sonorant frames of its k utterances (what
            # analysing that recording would give). Alternative ("mean"): mean of the k utterance medians.
            with np.errstate(all="ignore"):
                rec = np.array([np.nanmedian(np.concatenate([mats[i] for i in d]), 0) for d in s])
        else:
            rec = np.nanmean(V[s], 1)
        out[k] = (np.nanvar(rec, 0, ddof=1), np.nanvar(cents(rec), 0, ddof=1), float(np.mean([nfr[d].sum() for d in s])) * 0.01)
    return out


def crossed(Y):
    """Y: (speakers I, sentences J), balanced, one observation per cell -> (var speaker, var sentence, var residual).
    Method-of-moments (expected mean squares), components floored at 0. The residual is the speaker x sentence
    interaction plus occasion noise (they cannot be separated without repeats)."""
    I, J = Y.shape
    g, ri, cj = Y.mean(), Y.mean(1), Y.mean(0)
    ms_s = J * np.sum((ri - g) ** 2) / (I - 1)
    ms_t = I * np.sum((cj - g) ** 2) / (J - 1)
    ms_e = np.sum((Y - ri[:, None] - cj[None, :] + g) ** 2) / ((I - 1) * (J - 1))
    return max(0.0, (ms_s - ms_e) / J), max(0.0, (ms_t - ms_e) / I), ms_e


def fmt_c(c, native, unit, i):
    """'23 c (4.1 Hz)', length 'x c (y cm, z %)', dB measures 'x dB'"""
    if i >= N_LOG:
        return f"{native:.2f} dB"
    if OUTS[i] == "L":
        return f"{c:.0f} c ({native:.2f} cm, {100 * (2 ** (c / 1200) - 1):.1f} %)"
    return f"{c:.0f} c ({native:.1f} Hz)"


def report(names):
    sys.stdout.reconfigure(encoding="utf-8")
    models = {n: load_model(n) for n in names if (CACHE / n).is_dir()}
    primary = list(models)[0]
    # CHOICE: an utterance is kept only if every model has >= MIN_FRAMES frames for it (so seeds are compared on the same
    # utterances and the same pseudo-recordings). Alternative: per-model eligibility.
    keys = [{(it["spk"], it["utt"]) for it in m if it["nfr"] >= MIN_FRAMES} for m in models.values()]
    keep = set.intersection(*keys)
    for n in models:
        models[n] = [it for it in models[n] if (it["spk"], it["utt"]) in keep]
    ref = models[primary]
    spks = {G: sorted({it["spk"] for it in ref if it["sex"] == G}) for G in "mw"}
    sexname = {"m": "men", "w": "women"}
    rng = np.random.default_rng(96)
    boots = {G: rng.integers(0, len(spks[G]), (N_BOOT, len(spks[G]))) for G in "mw"}  # shared by every model: paired
    L = ["D-096: within-speaker repeatability and minimal detectable change (PTDB-TUG dev, windowed analysis; descriptive)",
         f"models: {', '.join(models)} (primary first; the seed mean is the mean of the SEMs of the models listed)",
         f"speakers: {len(spks['m'])} men, {len(spks['w'])} women; utterances {len(ref)} "
         f"(>= {MIN_FRAMES} voiced sonorant frames in every model); median {np.median([it['nfr'] for it in ref]) * 0.01:.1f} s "
         "of voiced sonorant frames each",
         "values: cents (native units); [95 % speaker bootstrap, 2000, resampled within sex, same draws for every model];"
         " length also in cm and %; H1*-H2* and H1-H2 in dB", ""]

    # per model, speaker, k: variances
    S = {}
    for n, items in models.items():
        S[n] = {}
        for how in ("pooled", "mean"):
            for G in "mw":
                rows = []
                for s in spks[G]:
                    its = [it for it in items if it["spk"] == s]
                    mats = [frame_matrix(it) for it in its]
                    V = np.array([item_values(it["fr"]) for it in its])
                    nfr = np.array([it["nfr"] for it in its])
                    # the draws depend only on the speaker, k and the shared utterance count: paired across models
                    r2 = np.random.default_rng([96, int(s.split(":")[-1][1:]), G == "w"])
                    sets = {k: draw_sets(len(its), k, r2) for k in KS[1:]}
                    rows.append(speaker_variances(mats, nfr, V, sets, how))
                S[n][how, G] = rows  # list over speakers of {k: (var native, var cents, dur)}

    def sem(how, G, k, draw, which, nm=None):
        """SEM over the drawn speakers (equal weight each); seed mean over the listed models if nm is None"""
        out = []
        for n in ([nm] if nm else models):
            v = np.array([S[n][how, G][j][k][which] for j in draw])
            out.append(np.sqrt(np.nanmean(v, 0)))
        return np.mean(out, 0)

    def ci(how, G, k, which, nm=None, scale=1.0):
        b = np.array([sem(how, G, k, d, which, nm) for d in boots[G]])
        return np.nanpercentile(b, [2.5, 97.5], axis=0) * scale

    ident = {G: np.arange(len(spks[G])) for G in "mw"}
    for nm in [*models, None] if len(models) > 1 else [primary]:
        tag = nm or "seed mean"
        L.append(f"=== {tag} ===")
        for G in "mw":
            dur = {k: np.mean([S[n]["pooled", G][j][k][2] for n in ([nm] if nm else models) for j in range(len(spks[G]))]) for k in KS}
            L.append(f"-- {sexname[G]}: (1) within-speaker SD across utterances (k = 1); (2) SEM of k-utterance pseudo-recordings "
                     f"(median over their pooled frames; {N_DRAWS} draws); (3) MDC95 = 1.96 sqrt(2) SEM")
            L.append("   mean voiced sonorant duration per recording: " + ", ".join(f"k={k} {dur[k]:.0f} s" for k in KS))
            for i, o in enumerate(OUTS):
                for what, mult in (("SEM", 1.0), ("MDC95", 1.96 * np.sqrt(2))):
                    cells = []
                    for k in KS:
                        c = sem("pooled", G, k, ident[G], 1, nm)[i] * mult if i < N_LOG else np.nan
                        x = sem("pooled", G, k, ident[G], 0, nm)[i] * mult
                        lo, hi = ci("pooled", G, k, 1 if i < N_LOG else 0, nm, mult)[:, i]
                        cells.append(f"k={k}: {fmt_c(c, x, UNITS[i], i)} [{lo:.0f}, {hi:.0f}]" if i < N_LOG else
                                     f"k={k}: {x:.2f} dB [{lo:.2f}, {hi:.2f}]")
                    L.append(f"   {o:8s} {what:5s} " + "; ".join(cells))
                if o in NEEDED:
                    L.append(f"   {o:8s} change needed (literature): {NEEDED[o]}")
            if not NEEDED:
                L.append("   change each output would need to show: not yet filled (NEEDED is empty; to be cited when written)")
            # sensitivity: recording value as the mean of utterance medians
            L.append("   sensitivity (recording = mean of its utterances' medians), SEM at k = 5 and 20, cents / dB: " + "; ".join(
                f"{o} " + "/".join(f"{sem('mean', G, k, ident[G], 1 if i < N_LOG else 0, nm)[i]:.2f}" for k in (5, 20))
                for i, o in enumerate(OUTS)))
        L.append("")

    # (4) channel term
    L.append("=== (4) channel term: paired shift of the same utterances under D-050's channel-probe conditions "
             "(PTDB-TUG dev, 20 files per speaker, frames fixed by the clean file) ===")
    L.append("per condition: RMS shift [bootstrap], cents (native) or dB; then the root sum of squares with MDC95 (3) at k = 5 and k = 20")
    chan = {}
    for n in models:
        zs = [np.load(f) for f in sorted((CACHE / n).glob("channel_*.npz"))]
        if zs:
            chan[n] = {"spk": np.concatenate([["ptdb_tug:" + str(z["spk"]).split(":")[-1]] * len(z["files"]) for z in zs]),  # as the main caches' ids
                       "conds": list(zs[0]["conds"]),
                       "val": np.concatenate([z["val"] for z in zs]), "nsel": np.concatenate([z["nsel"] for z in zs])}
    for n, ch in chan.items():
        ok = ch["nsel"] >= MIN_FRAMES
        val = ch["val"][ok]
        spk = ch["spk"][ok]
        with np.errstate(all="ignore"):
            shift_nat = val[:, 1:] - val[:, :1]  # (files, conditions, 8)
            shift_c = cents(val[:, 1:]) - cents(val[:, :1])
        shift_dB = shift_nat.copy()
        shift_dB[..., :N_LOG] = np.nan
        L.append(f"--- {n}: {ok.sum()} files, conditions {len(ch['conds']) - 1} ---")
        for G in "mw":
            ss = [s for s in spks[G] if s in set(spk)]
            sidx = {s: np.where(spk == s)[0] for s in ss}
            L.append(f"  {sexname[G]}: shift RMS (mean signed in parentheses)")

            def rms(which, draw):
                per = [np.nanmean(which[sidx[ss[j]]] ** 2, 0) for j in draw]
                return np.sqrt(np.nanmean(per, 0))

            rc = {u: rms(shift_c if u == "c" else shift_nat, range(len(ss))) for u in ("c", "n")}
            bb = np.array([rms(shift_c, d) for d in rng.integers(0, len(ss), (N_BOOT, len(ss)))]) if ss else None
            bn = np.array([rms(shift_nat, d) for d in rng.integers(0, len(ss), (N_BOOT, len(ss)))]) if ss else None
            for c, cname in enumerate(ch["conds"][1:]):
                cells, rss = [], []
                for i, o in enumerate(OUTS):
                    allf = np.concatenate([sidx[s] for s in ss])
                    mean_c = np.nanmean(shift_c[allf, c, i]) if i < N_LOG else np.nanmean(shift_nat[allf, c, i])
                    if i < N_LOG:
                        lo, hi = np.nanpercentile(bb[:, c, i], [2.5, 97.5])
                        cells.append(f"{o} {rc['c'][c, i]:.0f}c ({rc['n'][c, i]:.1f}{'cm' if o == 'L' else 'Hz'}) "
                                     f"[{lo:.0f},{hi:.0f}] ({mean_c:+.0f})")
                    else:
                        lo, hi = np.nanpercentile(bn[:, c, i], [2.5, 97.5])
                        cells.append(f"{o} {rc['n'][c, i]:.2f}dB [{lo:.2f},{hi:.2f}] ({mean_c:+.2f})")
                    parts = []
                    for k in (5, 20):
                        mdc = sem("pooled", G, k, ident[G], 1 if i < N_LOG else 0, n)[i] * 1.96 * np.sqrt(2)
                        r = rc["c"][c, i] if i < N_LOG else rc["n"][c, i]
                        parts.append(f"{np.hypot(mdc, r):.1f}")
                    rss.append("/".join(parts))
                L.append(f"    {cname:16s} " + "; ".join(cells))
                L.append(f"    {'':16s} RSS with MDC95 at k=5/20 (cents; dB for H): " + "; ".join(f"{o} {r}" for o, r in zip(OUTS, rss)))
    L.append("")

    # amendment 1: content against voice
    L.append("=== Amendment 1: content against voice (crossed decomposition: speaker + sentence + residual) ===")
    L.append("sentences read by at least three dev speakers of the sex; cents (dB for H); SD of each component [95 % speaker bootstrap];"
             " MDC95 for different text = measure (3) at k = 1, for the same text = 1.96 sqrt(2) x residual SD")
    for n, items in models.items():
        L.append(f"--- {n} ---")
        for G in "mw":
            cnt = {}
            for it in items:
                if it["sex"] == G:
                    cnt.setdefault(it["sent"], set()).add(it["spk"])
            sents = sorted(s for s, v in cnt.items() if len(v) >= 3)
            L.append(f"  {sexname[G]}: sentences read by >= 3 of {len(spks[G])} speakers: {len(sents)} ({', '.join(sents[:6])}{' ...' if len(sents) > 6 else ''})")
            if len(sents) < 2:
                L.append("    fewer than two such sentences: no decomposition")
                continue
            vals = {(it["spk"], it["sent"]): item_values(it["fr"]) for it in items if it["sex"] == G and it["sent"] in sents}
            for i, o in enumerate(OUTS):
                def cell(s, t, i=i):
                    v = vals.get((s, t), np.full(len(OUTS), np.nan))[i]
                    return 1200 * np.log2(v) if i < N_LOG else v

                full = [s for s in spks[G] if all(np.isfinite(cell(s, t)) for t in sents)]
                if len(full) < 3:
                    L.append(f"    {o}: fewer than 3 speakers have every sentence")
                    continue
                Y = np.array([[cell(s, t) for t in sents] for s in full])
                pt = np.sqrt(crossed(Y))
                bs = np.array([np.sqrt(crossed(Y[d])) for d in rng.integers(0, len(full), (N_BOOT, len(full)))])
                u = "dB" if i >= N_LOG else "c"
                f = lambda j: f"{pt[j]:.1f} {u} [{np.percentile(bs[:, j], 2.5):.1f}, {np.percentile(bs[:, j], 97.5):.1f}]"  # noqa: E731
                tot = np.sum(pt ** 2)
                mdc_diff = sem("pooled", G, 1, ident[G], 1 if i < N_LOG else 0, n)[i] * 1.96 * np.sqrt(2)
                mdc_same = pt[2] * 1.96 * np.sqrt(2)
                lo, hi = np.percentile(bs[:, 2] * 1.96 * np.sqrt(2), [2.5, 97.5])
                L.append(f"    {o:8s} SD speaker {f(0)}, sentence {f(1)}, residual {f(2)}; sentence share of variance "
                         f"{pt[1] ** 2 / tot:.0%}, residual {pt[2] ** 2 / tot:.0%}; MDC95 different text {mdc_diff:.1f} {u}, "
                         f"same text {mdc_same:.1f} {u} [{lo:.1f}, {hi:.1f}]")
    # amendment 2: the pair estimate (a woman and a man reading the same sentences)
    L.append("")
    L.append("=== Amendment 2: content against voice from the woman-man pairs that read the same sentences ===")
    L.append("per pair and output: content variance = covariance across shared sentences of the two speakers' values (each centred on its "
             "speaker's mean); residual = each speaker's variance across those sentences minus it; means over pairs; "
             "[95 % bootstrap over sentences within pairs, 2000]; same-text MDC95 = 1.96 sqrt(2) x residual SD")
    for n, items in models.items():
        by = {}
        for it in items:
            by.setdefault(it["spk"], {})[it["sent"]] = item_values(it["fr"])
        pairs = [(w, m, sorted(set(by[w]) & set(by[m]))) for w in spks["w"] for m in spks["m"] if len(set(by[w]) & set(by[m])) >= 50]
        L.append(f"--- {n}: {len(pairs)} pairs (" + ", ".join(f"{w.split(':')[-1]} / {m.split(':')[-1]} {len(t)} sentences" for w, m, t in pairs) + ") ---")
        if not pairs:
            continue
        for i, o in enumerate(OUTS):
            conv = (lambda v: 1200 * np.log2(v)) if i < N_LOG else (lambda v: v)
            XY = []
            for w, m, t in pairs:
                with np.errstate(all="ignore"):
                    x, y = conv(np.array([by[w][s][i] for s in t])), conv(np.array([by[m][s][i] for s in t]))
                k = np.isfinite(x) & np.isfinite(y)
                XY.append((x[k], y[k]))

            def est(draws=None):
                c, rw, rm = [], [], []
                for j, (x, y) in enumerate(XY):
                    if draws is not None:
                        x, y = x[draws[j]], y[draws[j]]
                    cv = np.cov(x, y)
                    c.append(cv[0, 1]), rw.append(cv[0, 0] - cv[0, 1]), rm.append(cv[1, 1] - cv[0, 1])
                return np.array([np.mean(c), np.mean(rw), np.mean(rm)]), np.array(c)

            pt, per_pair = est()
            bs = np.array([est([rng.integers(0, len(x), len(x)) for x, _ in XY])[0] for _ in range(N_BOOT)])
            sd = lambda v: np.sqrt(np.maximum(v, 0.0))  # noqa: E731
            u = "dB" if i >= N_LOG else "c"
            f = lambda j, mult=1.0: (f"{mult * sd(pt[j]):.1f} {u} [{mult * sd(np.percentile(bs[:, j], 2.5)):.1f}, "  # noqa: E731
                                     f"{mult * sd(np.percentile(bs[:, j], 97.5)):.1f}]")
            share = pt[0] / (pt[0] + (pt[1] + pt[2]) / 2) if pt[0] + (pt[1] + pt[2]) / 2 > 0 else np.nan
            L.append(f"    {o:8s} content SD {f(0)}; residual SD women {f(1)}, men {f(2)}; content's share of the within-speaker variance "
                     f"{share:.0%}; same-text MDC95 women {f(1, 1.96 * np.sqrt(2))}, men {f(2, 1.96 * np.sqrt(2))}; content SD by pair "
                     + " ".join(f"{sd(c):.1f}" for c in per_pair))
    print("\n".join(L))


if __name__ == "__main__":
    cmd, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "score":
        flags = [a for a in rest if a.startswith("--")]
        args = dict(a.split("=", 1) for a in rest if not a.startswith("--")) or MODELS
        score(args, channel="--no-channel" not in flags)
    elif cmd == "report":
        report(rest or list(MODELS))
    else:
        print(__doc__)
        sys.exit(2)
