"""D-034 pre-registered (decisions.md, 2026-10-26): a fitted teacher/head F1 weight. From ensemble_feasibility's dumps
(eval_heads.py VA_DUMP): T = teacher v3, H = mean of H5's six seeds, in log frequency; baseline (c-bar) = 0.5 T + 0.5 H;
(w) = w T + (1 - w) H, w fitted leaving one PTDB-TUG dev speaker out on J(w) = mean over sexes of |median signed error|
+ 0.1 x RMSE (Hz); (m) = frame-wise median of teacher v3 and its two siblings and H5 seeds 0-2. Hillenbrand with w fitted
on all of PTDB-TUG dev. Speaker bootstrap (within sex, 2000 draws) for arm minus (c-bar). CPU only.
Args: dump dir, out txt."""
import sys
from pathlib import Path

import numpy as np

D, OUTF = Path(sys.argv[1]), sys.argv[2]
OUT = open(OUTF, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
LAM = 0.1


def load(feat, tag):
    d, dp = dict(np.load(D / f"{feat}_{tag}.npz", allow_pickle=True)), dict(np.load(D / f"P_{tag}.npz", allow_pickle=True))
    assert np.array_equal(d["ref"], dp["ref"], equal_nan=True)
    m = {"T": dp["m_teacher_v3"], "T3": np.mean([dp[f"m_{k}"] for k in ("teacher_v3", "tv3_s0", "tv3_s2")], 0),
         "H": np.mean([d[f"m_H5#{k}"] for k in range(6)], 0),
         "med": np.median(np.stack([dp["m_teacher_v3"], dp["m_tv3_s0"], dp["m_tv3_s2"]] + [d[f"m_H5#{k}"] for k in range(3)]), 0)}
    ok = np.isfinite(d["ref"]) & (d["ref"] > 0)
    for k in ("teacher_v3", "tv3_s0", "tv3_s2"):
        ok &= np.isfinite(dp[f"m_{k}"])
    for k in range(6):
        ok &= np.isfinite(d[f"m_H5#{k}"])
    return {k: v[ok] for k, v in m.items()}, d["ref"][ok], d["group"][ok], d["speaker"][ok]


def err(x, ref):
    return 2 ** x - ref


def bias(e, sex):
    return np.array([np.median(e[sex == "f"]), np.median(e[sex == "m"])])


def J(e, sex):
    return np.abs(bias(e, sex)).mean() + LAM * np.sqrt(np.mean(e ** 2))


def fit(T, H, ref, sex, lam=LAM):
    js = [np.abs(bias(err(w * T + (1 - w) * H, ref), sex)).mean() + lam * np.sqrt(np.mean(err(w * T + (1 - w) * H, ref) ** 2)) for w in GRID]
    return float(GRID[int(np.argmin(js))])


def loso(T, H, ref, sex, spk, lam=LAM):
    pred, ws = np.empty_like(T), {}
    for s in np.unique(spk):
        tr = spk != s
        w = fit(T[tr], H[tr], ref[tr], sex[tr], lam)
        ws[str(s)] = w
        pred[~tr] = w * T[~tr] + (1 - w) * H[~tr]
    return pred, ws


def boot(e_arm, e_base, sex, spk, n=2000, seed=20261026):
    rng = np.random.default_rng(seed)
    idx = {s: np.flatnonzero(spk == s) for s in np.unique(spk)}
    by = {g: [s for s in idx if sex[idx[s][0]] == g] for g in ("f", "m")}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[s] for g in ("f", "m") for s in rng.choice(by[g], len(by[g]))])
        da = np.abs(bias(e_arm[pick], sex[pick])) - np.abs(bias(e_base[pick], sex[pick]))
        out.append([da[0], da[1], da.mean()])
    out = np.array(out)
    return np.percentile(out, [2.5, 97.5], 0)


for feat in ("P", "v4"):
    M, ref, sex, spk = load(feat, "ptdb")
    Mh, refh, _, _ = load(feat, "hillenbrand")
    p(f"\n######## H5 on {feat}  (PTDB-TUG dev closed phase: {len(ref)} frames, {len(np.unique(spk))} speakers)")
    eT = err(M["T"], ref)
    bT, maeT = bias(eT, sex), np.mean(np.abs(err(Mh["T"], refh)))
    p(f"teacher v3 alone: women {bT[0]:+.1f}, men {bT[1]:+.1f}; Hillenbrand F1 MAE {maeT:.1f}")
    w_all = fit(M["T"], M["H"], ref, sex)
    w_all3 = fit(M["T3"], M["H"], ref, sex)
    arms = {}
    arms["(c-bar) 0.5"] = (0.5 * M["T"] + 0.5 * M["H"], 0.5 * Mh["T"] + 0.5 * Mh["H"])
    pw, ws = loso(M["T"], M["H"], ref, sex, spk)
    arms["(w) LOSO"] = (pw, w_all * Mh["T"] + (1 - w_all) * Mh["H"])
    arms["(m) median"] = (M["med"], Mh["med"])
    pw3, ws3 = loso(M["T3"], M["H"], ref, sex, spk)
    arms["(w) T3, reported"] = (pw3, w_all3 * Mh["T3"] + (1 - w_all3) * Mh["H"])
    pw0, ws0 = loso(M["T"], M["H"], ref, sex, spk, lam=0.0)
    w_all0 = fit(M["T"], M["H"], ref, sex, lam=0.0)
    arms["(w) no RMSE, reported"] = (pw0, w_all0 * Mh["T"] + (1 - w_all0) * Mh["H"])
    p(f"w fitted on all dev: {w_all:.2f} (T3: {w_all3:.2f}; no RMSE term: {w_all0:.2f})")
    p("LOSO w per held-out speaker: " + ", ".join(f"{k.split(':')[-1]} {v:.2f}" for k, v in ws.items()))
    p("  no RMSE term:            " + ", ".join(f"{k.split(':')[-1]} {v:.2f}" for k, v in ws0.items()))
    eb = err(arms["(c-bar) 0.5"][0], ref)
    p(f"\n{'arm':24s} {'women':>7s} {'men':>7s} {'RMSE':>6s} {'Hb MAE':>7s} | {'cut':>5s} {'rise':>5s} {'MAEcut':>6s} rule | vs (c-bar): women, men, mean |bias| [95 %]")
    for name, (x, xh) in arms.items():
        e = err(x, ref)
        b, mae = bias(e, sex), np.mean(np.abs(err(xh, refh)))
        cut, rise, mcut = abs(bT[0]) - abs(b[0]), abs(b[1]) - abs(bT[1]), maeT - mae
        ok = cut >= 5 and rise <= 5 and mcut >= -4
        if name.startswith("(c-bar)"):
            cmp = ""
        else:
            ci = boot(e, eb, sex, spk)
            d = np.abs(b) - np.abs(bias(eb, sex))
            cmp = "  ".join(f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]" for v, lo, hi in zip([d[0], d[1], d.mean()], ci[0], ci[1]))
        p(f"{name:24s} {b[0]:+7.1f} {b[1]:+7.1f} {np.sqrt(np.mean(e ** 2)):6.1f} {mae:7.1f} | {cut:+5.1f} {rise:+5.1f} {mcut:+6.1f} {'yes' if ok else 'no ':3s}  | {cmp}")
