"""Does synthesis-time observability predict model error? 150 train_v2-config clips (seed 77, not a training seed),
conformer_v2_d16k, sonorant frames; |error| in cents per formant by observability bin, and model sigma alongside."""
import sys, numpy as np, torch
import vocal_analysis.synth.sampler as smp
from vocal_analysis.synth import observability as ob
from vocal_analysis.models.infer import analyse, load_teacher
cap = {}
def wrap(fn):
    def f(*a, **k):
        out = fn(*a, **k); cap["c"] = out; return out
    return f
smp.synthesize = wrap(smp.synthesize); smp.interactive_source.synthesize = wrap(smp.interactive_source.synthesize)
dev = torch.device("cuda"); model = load_teacher(sys.argv[1], dev)[0]
K = 4; O, E, S = [], [], []
for i in range(int(sys.argv[2])):
    c = smp.generate(i, smp.SynthConfig(), 77); L = c.labels
    snr = ob.component_snr(cap["c"], L["t"], L["voiced"])
    o, _ = ob.observability(L["f0"], L["voiced"], L["sonorant"], L["formants"][:, :K], L["bandwidths"][:, :K], snr)
    r = analyse(model, c.audio, dev)
    n = min(len(o), len(r["formants"]))
    err = 1200 * np.abs(np.log2(r["formants"][:n, :K] / L["formants"][:n, :K]))
    m = L["sonorant"][:n]
    O.append(o[:n][m]); E.append(err[m]); S.append(r["formant_sigma_cents"][:n, :K][m])
O, E, S = (np.concatenate(x) for x in (O, E, S))
bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
print(f"{len(O)} sonorant frames. Median |error| (cents) and share of frames by observability bin")
for k in range(K):
    row = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (O[:, k] >= a) & (O[:, k] < b) & np.isfinite(E[:, k])
        row.append(f"[{a:.1f},{b:.1f}) {np.median(E[sel, k]) if sel.sum() > 20 else np.nan:5.0f} ({100 * sel.mean():4.1f}%)")
    print(f"  F{k + 1}: " + "  ".join(row))
from scipy.stats import spearmanr
for k in range(K):
    ok = np.isfinite(E[:, k]) & np.isfinite(O[:, k]) & np.isfinite(S[:, k])
    print(f"  F{k + 1}: Spearman |err| vs -o {spearmanr(-O[ok, k], E[ok, k]).statistic:+.2f}, vs model sigma {spearmanr(S[ok, k], E[ok, k]).statistic:+.2f}, o vs sigma {spearmanr(-O[ok, k], S[ok, k]).statistic:+.2f}")
