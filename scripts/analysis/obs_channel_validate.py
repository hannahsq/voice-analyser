"""D-027 validation (pre-registered 2026-10-17): the loader's analytic channel on stored spectra (`obs_store`) against
a full recomputation, on fresh clips (train_v2's config, seed 31337, used by no set) with a random channel each.
Full recomputation: each component (scaled as the stored audio) through the same drawn channel (`apply_channel` with
the mix's noise scale, peaks and clip limit; no added noise or clipping on the components); everything in the
channelled mix that isn't the voiced or tract-noise component counts as noise; then `component_snr`.
Accepted when, over sonorant frames, per formant the median |o_analytic - o_full| <= 0.05 and Spearman >= 0.9, and no
channel stage (over the clips that have it) has a median |delta o| above 0.1.
Args: out file, number of clips (default 300)."""
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from vocal_analysis.features.vqt import HOP
from vocal_analysis.synth import interactive as it
from vocal_analysis.synth import observability as ob
from vocal_analysis.synth import obs_store as st
from vocal_analysis.synth import sampler as smp
from vocal_analysis.synth.channel import ChannelConfig, apply_channel, draw_channel, public_params

sys.stdout.reconfigure(encoding="utf-8")
OUT = open(sys.argv[1], "w", encoding="utf-8")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731

cap = {}
real_s, real_i = smp.synthesize, it.synthesize


def keep_s(*a, **k):
    cap["c"] = real_s(*a, **k)
    return cap["c"]


def keep_i(*a, **k):
    cap["c"] = real_i(*a, **k)
    return cap["c"]


smp.synthesize, smp.interactive_source.synthesize = keep_s, keep_i
an_all, full_all = [], []
stage_err = defaultdict(list)
for i in range(N):
    c = smp.generate(i, smp.SynthConfig(), 31337)
    L, comps, gain = c.labels, cap["c"], c.meta["gain"]
    t, voiced, son, f0 = L["t"], L["voiced"], L["sonorant"], L["f0"]
    F, B = L["formants"][:, :4], L["bandwidths"][:, :4]
    store = st.clean_spectra(comps, t, voiced, f0)
    x = c.audio.astype(np.float64)
    mask = np.repeat(voiced, HOP)[: len(x)]
    mask = np.pad(mask, (0, len(x) - len(mask)))
    ch = draw_channel(np.random.default_rng([31337, i]), len(x), ChannelConfig())
    y, used = apply_channel(x, ch, mask, codec_stream=False)
    params = public_params(ch) | {k: v for k, v in used.items() if k == "clip_loss"}
    cs = st.ClipSpectra(store["sig"], store["noise"], store["peak"], voiced, f0).channel(params)
    o_an, _ = ob.observability(f0, voiced, son, F, B, cs.snr_db_at)
    zero = np.zeros(len(x))
    v_ch = apply_channel(comps.get("voiced", zero)[: len(x)] * gain, ch, ref=used, noise=False, clip=False, codec_stream=False)[0]
    a_ch = apply_channel((comps.get("aspiration", zero) + comps.get("whisper", zero))[: len(x)] * gain, ch, ref=used,
                         noise=False, clip=False, codec_stream=False)[0]
    rest = y.astype(np.float64) - v_ch - a_ch
    snr_full = ob.component_snr({"voiced": v_ch, "aspiration": a_ch, "frication": rest, "total": y}, t, voiced)
    o_full, _ = ob.observability(f0, voiced, son, F, B, snr_full)
    ok = son[:, None] & np.isfinite(o_an) & np.isfinite(o_full)
    an_all.append(np.where(ok, o_an, np.nan))
    full_all.append(np.where(ok, o_full, np.nan))
    d = np.nanmedian(np.abs(o_an - o_full)[ok]) if ok.any() else np.nan
    stages = ["clean"] if params.get("clean") else [k for k in ("drr_db", "highpass_hz", "lowpass_hz", "eq", "noise", "phone", "clip_loss", "codec") if k in params]
    if "noise" in params:
        stages.append(f"noise:{params['noise']}")
    for s in stages:
        stage_err[s].append(d)
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{N}", flush=True)
A, Fu = np.concatenate(an_all), np.concatenate(full_all)
p(f"Observability channel validation (D-027): {N} clips, train_v2 config, seed 31337, random channel each")
fails = []
for k in range(4):
    m = np.isfinite(A[:, k]) & np.isfinite(Fu[:, k])
    med = float(np.median(np.abs(A[m, k] - Fu[m, k])))
    rho = float(spearmanr(A[m, k], Fu[m, k])[0])
    bias = float(np.median(A[m, k] - Fu[m, k]))
    p(f"  F{k + 1}: {m.sum()} frames, median |delta o| {med:.3f}, Spearman {rho:.3f}, median signed {bias:+.3f}, "
      f"mean o analytic {np.mean(A[m, k]):.3f} full {np.mean(Fu[m, k]):.3f}")
    if med > 0.05 or rho < 0.9:
        fails.append(f"F{k + 1}")
p("Per channel stage, median over its clips of the clip's median |delta o|:")
for s, v in sorted(stage_err.items()):
    v = np.array(v, float)
    med = float(np.nanmedian(v))
    p(f"  {s:14s} {len(v):4d} clips  {med:.3f}")
    if med > 0.1:
        fails.append(s)
p(f"failures: {fails or 'none'}")
p(f"DECISION: {'accepted' if not fails else 'not accepted'}")
