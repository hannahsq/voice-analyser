"""D-027, the pooling flaw (2026-10-18): the label's temporal pooling (exp(-|dt| / 50 ms)) treats every frame's evidence
as independent, but the source's harmonic structure repeats from frame to frame. On the clean Rd-sweep renders (Story's
10 vowels, 15 cm, f0 210 / 300, Rd 0.5 / 2.5, recording floor 45 dB) the F3 posterior SD in cents, median over vowels,
for the first label ("old", independent scatter) and the second ("new", measured source structure), pooled as the
label is and for one frame alone (context_s = 0). CPU, no arguments; prints the table."""
import numpy as np
from dataclasses import replace
from vocal_analysis.synth import tract, tube, voice, observability as ob
N = 16000
L, asc, phar = 15.0, (15 / 17.46) ** 2, (15 / 17.46) ** 0.3
C = N // voice.CTRL + 1
res = {}
for v, (q1, q2) in tube.STORY_VOWELS.items():
    sh = tube.build_shape(q1, q2, L, pharynx_ratio=phar, area_scale=asc)
    F, B = tube.poles(sh, tract.F_MAX_HZ)
    F, B = F[:tract.N_POLES], B[:tract.N_POLES]
    pf, pb = np.full((C, tract.N_POLES), np.nan), np.full((C, tract.N_POLES), np.nan)
    pf[:, :len(F)], pb[:, :len(F)] = F, B
    for f0 in (210.0, 300.0):
        for rd in (0.5, 2.5):
            traj = voice.Trajectories(f0=np.full(C, f0), rd=np.full(C, rd), voice_amp=np.ones(C), whisper_amp=np.zeros(C),
                                      fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
            comps = {k: c[:N] for k, c in voice.synthesize(traj, np.random.default_rng(0)).items()}
            t = np.arange(100) * 0.01
            snr = ob.component_snr(comps, t, np.ones(100, bool), rec_snr_db=45.0)
            for name, cfg in (("old pooled", replace(ob.ObservabilityConfig(), src_sd_db=None)), ("new pooled", ob.ObservabilityConfig()),
                              ("old frame", replace(ob.ObservabilityConfig(), src_sd_db=None, context_s=0.0)),
                              ("new frame", replace(ob.ObservabilityConfig(), context_s=0.0))):
                _, sd = ob.observability(np.full(100, f0), np.ones(100, bool), np.ones(100, bool), np.tile(F[:4], (100, 1)), np.tile(B[:4], (100, 1)), snr, cfg)
                res.setdefault((name, f0, rd), []).append(np.median(1200 / np.log(2) * sd[30:71, 2] / F[2]))
print("Rd-sweep renders: F3 posterior SD (cents), median over Story's vowels; pooled over frames vs one frame")
print(f"{'':16s}" + "".join(f"{n:>13s}" for n in ("old pooled", "new pooled", "old frame", "new frame")))
for f0 in (210.0, 300.0):
    for rd in (0.5, 2.5):
        print(f"f0 {f0:.0f} Rd {rd}:   " + "".join(f"{np.median(res[(n, f0, rd)]):13.1f}" for n in ("old pooled", "new pooled", "old frame", "new frame")))
