"""Long inputs and context carry-over on real speech (D-028), PTDB-TUG dev speakers (ptdb_egg_sub.npz frames).
1. Same speaker: each speaker's utterances joined in groups of 4 (0.3 s gaps of -80 dB noise; every part padded to
   whole 10 ms frames, so frames align exactly), so files of about 10 s.
   Closed-phase median |error| and bias (Hz, F1-F3, per sex) joined against each utterance analysed alone, on the
   same frames. Lines "JOINED <model> <sex> <F1> <F2> <F3>" give joined minus alone, median |error|, in % of alone.
2. Carry-over: for each dev speaker B, a partner A of the other sex and one of the same sex; files A A A B and B A A A
   (three of A's utterances). B's formants on the frames B alone calls sonorant (p > 0.5) against B alone: median signed shift in cents, F1-F3, signed so that positive is toward A (A's median formants above B's ->
   upward is positive). Lines "CARRY <model> <cross|same> <before|after> <F1> <F2> <F3>" (cents), "after" = B after A.
Device from VA_DEVICE (default cpu). Args: scratch dir, name=checkpoint pairs. Run from the main checkout."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.eval import tiers as T
from vocal_analysis.features.vqt import HOP
from vocal_analysis.models.infer import analyse, load_teacher

sys.stdout.reconfigure(encoding="utf-8")
S = Path(sys.argv[1])
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
D = Path("../voice-analyser/data")
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
tmap = T.load()
is_dev = np.array([T.tier_of(f"ptdb_tug:{s}", tmap) == "dev" for s in d["speaker"]])
GAP = np.random.default_rng(0).normal(0, 1e-4, int(0.3 * 16000)).astype(np.float32)  # 30 frames


def wav(spk, utt):
    x, sr = sf.read(D / "ptdb_tug" / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{utt}.wav")
    x = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
    pad = (-len(x)) % HOP  # whole frames, so every part of a joined file starts on a frame boundary
    return np.concatenate([x, np.random.default_rng(1).normal(0, 1e-4, pad).astype(np.float32)])


utts = {}  # speaker -> ordered utterances (that have closed-phase frames)
for s, u in zip(d["speaker"][is_dev], d["utt"][is_dev]):
    utts.setdefault(str(s), [])
    if u not in utts[str(s)]:
        utts[str(s)].append(str(u))
speakers = sorted(utts)
sex = {s: "f" if s[0] == "F" else "m" for s in speakers}
audio = {(s, u): wav(s, u) for s in speakers for u in utts[s]}
print(f"{len(speakers)} dev speakers, {sum(len(v) for v in utts.values())} utterances")


def frames_of(s, u):
    i = np.flatnonzero((d["speaker"] == s) & (d["utt"] == u))
    return i, np.round(d["t"][i] / 0.010).astype(int)


def run(m, parts):
    """Analyse the concatenation; return each part's formants (frames from the part's own start)."""
    xs, starts, pos = [], [], 0
    for k, x in enumerate(parts):
        if k:
            xs.append(GAP)
            pos += len(GAP)
        starts.append(pos)
        xs.append(x)
        pos += len(x)
    f = analyse(m, np.concatenate(xs), dev)["formants"][:, :3]
    return [f[st // HOP: st // HOP + len(x) // HOP + 1] for st, x in zip(starts, parts)]


for name, ck in (a.split("=", 1) for a in sys.argv[2:]):
    m = load_teacher(ck, dev)[0]
    res_alone = {k: analyse(m, x, dev) for k, x in audio.items()}
    alone = {k: r["formants"][:, :3] for k, r in res_alone.items()}
    son = {k: r["sonorant_p"] > 0.5 for k, r in res_alone.items()}
    # 1. same speaker, joined in fours
    rows = {"f": ([], []), "m": ([], [])}
    for s in speakers:
        us = utts[s]
        for g in range(0, len(us), 4):
            grp = us[g:g + 4]
            joined = run(m, [audio[(s, u)] for u in grp])
            for u, fj in zip(grp, joined):
                i, j = frames_of(s, u)
                fa = alone[(s, u)]
                ok = j < min(len(fa), len(fj))
                rows[sex[s]][0].append(fa[j[ok]] - d["cp"][i[ok], :3])
                rows[sex[s]][1].append(fj[j[ok]] - d["cp"][i[ok], :3])
    for sx, (ea, ej) in rows.items():
        ea, ej = np.concatenate(ea), np.concatenate(ej)
        ok = np.isfinite(ea) & np.isfinite(ej)
        ma = [np.median(np.abs(ea[ok[:, k], k])) for k in range(3)]
        mj = [np.median(np.abs(ej[ok[:, k], k])) for k in range(3)]
        ba = [np.median(ea[ok[:, k], k]) for k in range(3)]
        bj = [np.median(ej[ok[:, k], k]) for k in range(3)]
        print(f"  {name} {sx}: |err| alone {' '.join(f'{x:.1f}' for x in ma)}  joined {' '.join(f'{x:.1f}' for x in mj)};"
              f" bias alone {' '.join(f'{x:+.1f}' for x in ba)}  joined {' '.join(f'{x:+.1f}' for x in bj)}")
        print(f"JOINED {name} {sx} " + " ".join(f"{100 * (b - a) / a:+.2f}" for a, b in zip(ma, mj)), flush=True)
    # 2. carry-over
    med = {s: np.nanmedian(np.concatenate([alone[(s, u)][son[(s, u)]] for u in utts[s]]), 0) for s in speakers}
    shifts = {(kind, order): [] for kind in ("cross", "same") for order in ("after", "before")}
    for b in speakers:
        for kind in ("cross", "same"):
            cands = [a for a in speakers if a != b and (sex[a] != sex[b]) == (kind == "cross")]
            a = cands[speakers.index(b) % len(cands)]
            ub = utts[b][0]
            ua = utts[a][:3]
            for order, parts in (("after", [audio[(a, u)] for u in ua] + [audio[(b, ub)]]),
                                 ("before", [audio[(b, ub)]] + [audio[(a, u)] for u in ua])):
                res = run(m, parts)
                fb = res[-1] if order == "after" else res[0]
                fa = alone[(b, ub)]
                n = min(len(fa), len(fb))
                c = 1200 * np.log2(fb[:n] / fa[:n])[son[(b, ub)][:n]]
                sign = np.sign(med[a] - med[b])
                shifts[(kind, order)].append(c * sign)
    for (kind, order), v in shifts.items():
        v = np.concatenate(v)
        print(f"CARRY {name} {kind} {order} " + " ".join(f"{np.nanmedian(v[:, k]):+.1f}" for k in range(3))
              + "  (mean |shift| " + " ".join(f"{np.nanmean(np.abs(v[:, k])):.1f}" for k in range(3)) + ")", flush=True)
print(json.dumps({"gap_s": 0.3, "group": 4}))
