"""What is different about the utterances every model reads worst? Per PTDB-TUG utterance (closed-phase subset): the
F1 error all models share (per frame, the mean over the models of F1 minus the closed-phase F1; then the utterance
median) against voice-quality measures: Praat (parselmouth) HNR, CPPS, jitter, shimmer; H1-H2 (uncorrected, at the
closed-phase frames); alpha ratio (0-1 kHz against 1-5 kHz); median f0; the EGG contact quotient. Spearman correlations
by sex, and the worst decile against the rest. Args: scratch dir, out file. Run from the m3 worktree."""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from scipy.stats import spearmanr

S, OUT = Path(sys.argv[1]), sys.argv[2]
D = Path("../voice-analyser/data")
MODELS = ["v2_s0", "v2_s1", "v2_s2", "v3b_s0", "v3b_s1", "v3b_s2", "v1_s0", "v1_s1", "v1_s2", "conf_v2_s0", "mix_gru_s0",
          "mix_conformer_s0", "mix_conformer_s1", "mix_conformer_s2"]


def wav(spk, utt):
    return D / "ptdb_tug" / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{utt}.wav"


def measures(job):
    spk, utt, times, f0s = job
    x, sr = sf.read(wav(spk, utt))
    x = x[:, 0] if x.ndim == 2 else x
    snd = parselmouth.Sound(x, sr)
    floor, ceil = (100, 600) if spk[0] == "F" else (60, 350)
    out = {}
    hnr = snd.to_harmonicity_cc(0.01, floor, 0.1, 1.0).values[0]
    out["HNR dB"] = float(np.median(hnr[hnr > -100])) if np.any(hnr > -100) else np.nan
    try:
        pc = call(snd, "To PowerCepstrogram", floor, 0.002, 5000.0, 50.0)
        out["CPPS dB"] = call(pc, "Get CPPS", "yes", 0.02, 0.0005, floor, ceil, 0.05, "parabolic", 0.001, 0.05,
                              "Exponential decay", "Robust")  # "Robust slow": ~50 s per file
    except Exception:
        out["CPPS dB"] = np.nan
    pp = call(snd, "To PointProcess (periodic, cc)", floor, ceil)
    out["jitter %"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    out["shimmer %"] = 100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    h12, low, high = [], 0.0, 0.0
    n = int(0.04 * sr)
    for t, f0 in zip(times, f0s):
        i = int(t * sr) - n // 2
        if i < 0 or i + n > len(x) or not np.isfinite(f0):
            continue
        seg = x[i:i + n] * np.hanning(n)
        spec = np.abs(np.fft.rfft(seg, 8 * n)) ** 2
        f = np.fft.rfftfreq(8 * n, 1 / sr)
        amp = lambda fc: spec[(f > 0.9 * fc) & (f < 1.1 * fc)].max()  # noqa: E731
        h12.append(10 * np.log10(amp(f0) / amp(2 * f0)))
        low += spec[(f >= 50) & (f < 1000)].sum()
        high += spec[(f >= 1000) & (f < 5000)].sum()
    out["H1-H2 dB"] = float(np.median(h12)) if h12 else np.nan
    out["alpha ratio dB"] = float(10 * np.log10(high / low)) if low > 0 else np.nan
    return spk, utt, out


if __name__ == "__main__":
    d = dict(np.load(S / "ptdb_egg_sub.npz"))
    models = [m for m in MODELS if m in d]
    shared = np.nanmean(np.stack([d[m][:, 0] for m in models]), 0) - d["cp"][:, 0]
    keys = list(dict.fromkeys(zip(d["speaker"], d["utt"])))
    idx = {k: [] for k in keys}
    for i, k in enumerate(zip(d["speaker"], d["utt"])):
        idx[k].append(i)
    jobs = [(s, u, d["t"][idx[(s, u)]], d["f0"][idx[(s, u)]]) for s, u in keys]
    with Pool(int(__import__('os').environ.get('VA_PROCS', '8'))) as pool:
        vq = {(s, u): o for s, u, o in pool.imap(measures, jobs, chunksize=8)}
    rows = []
    for k in keys:
        ii = idx[k]
        if not np.isfinite(shared[ii]).any():  # no frame with both a closed-phase F1 and model readings
            continue
        rows.append({"sex": d["sex"][ii[0]], "speaker": k[0], "err": np.nanmedian(shared[ii]), "frames": len(ii),
                     "f0 Hz": np.nanmedian(d["f0"][ii]), "EGG CQ": np.nanmedian(d["cq"][ii]),
                     "cycle - cp F1": np.nanmedian(d["cycle"][ii, 0] - d["cp"][ii, 0]), **vq[k]})
    names = ["f0 Hz", "EGG CQ", "cycle - cp F1", "HNR dB", "CPPS dB", "jitter %", "shimmer %", "H1-H2 dB", "alpha ratio dB"]
    with open(OUT, "w", encoding="utf-8") as fh:
        p = lambda *a: (print(*a, file=fh, flush=True), print(*a, flush=True))  # noqa: E731
        p(f"Shared F1 error ({len(models)} models; per frame, mean of model F1 minus closed-phase F1; utterance median), "
          f"PTDB-TUG subset, {len(rows)} utterances")
        r = np.array
        fr_ok = np.isfinite(shared) & np.isfinite(d["cq"])
        for sex in ("f", "m"):
            sel = [x for x in rows if x["sex"] == sex]
            err = r([x["err"] for x in sel])
            fm = fr_ok & (d["sex"] == sex)
            rho_f = spearmanr(shared[fm], d["cq"][fm])[0]
            p(f"\n{'women' if sex == 'f' else 'men'}: {len(sel)} utterances, shared error median {np.median(err):+.1f} Hz; "
              f"frame-level Spearman (shared error, EGG CQ) {rho_f:+.2f} over {fm.sum()} frames")
            worst = err >= np.percentile(err, 90)
            p(f"  {'measure':16s} {'Spearman with error':>20s} {'worst 10 %':>12s} {'the rest':>10s}")
            for nm in names:
                v = r([x[nm] for x in sel], float)
                ok = np.isfinite(v) & np.isfinite(err)
                rho, pv = spearmanr(v[ok], err[ok])
                p(f"  {nm:16s} {rho:+8.2f} (p {pv:7.1e}) {np.nanmedian(v[worst]):12.2f} {np.nanmedian(v[~worst]):10.2f}")
            spk = sorted({x["speaker"] for x in sel})
            p("  per speaker (median shared error, Hz): " + ", ".join(
                f"{s} {np.median([x['err'] for x in sel if x['speaker'] == s]):+.0f}" for s in spk))
