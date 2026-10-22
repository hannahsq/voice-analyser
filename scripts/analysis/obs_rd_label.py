"""D-027: the observability label on clean Rd-sweep renders. Story's 10 vowels in a 15 cm tract (closed-glottis poles),
1 s each at f0 120 / 210 / 300 / 400 Hz and Rd 0.5-2.5, no channel; the label's median o and posterior SD (cents) over
the middle 0.4 s, F1-F4, median over vowels. Default `ObservabilityConfig`; the floor from `component_snr`'s defaults.
CPU, no arguments; prints the table."""
import numpy as np
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
    for f0 in (120.0, 210.0, 300.0, 400.0):
        for rd in (0.5, 1.0, 1.5, 2.0, 2.5):
            traj = voice.Trajectories(f0=np.full(C, f0), rd=np.full(C, rd), voice_amp=np.ones(C), whisper_amp=np.zeros(C),
                                      fric_amp=np.zeros(C), poles_f=pf, poles_b=pb, zeros_f=np.zeros((C, 0)), zeros_b=np.zeros((C, 0)))
            comps = {k: v_[:N] for k, v_ in voice.synthesize(traj, np.random.default_rng(0)).items()}
            T = 100
            t = np.arange(T) * 0.01
            snr = ob.component_snr(comps, t, np.ones(T, bool))
            o, sd = ob.observability(np.full(T, f0), np.ones(T, bool), np.ones(T, bool), np.tile(F[:4], (T, 1)), np.tile(B[:4], (T, 1)), snr)
            cents = 1200 / np.log(2) * sd / F[:4]
            res.setdefault((f0, rd), []).append((np.median(o[30:71], 0), np.median(cents[30:71], 0)))
print("clean label on the Rd-sweep renders (Story's 10 vowels, 15 cm tract, no channel): median over vowels")
print("f0   Rd   o F1-F4                     posterior SD (cents) F1-F4")
for (f0, rd), v in sorted(res.items()):
    o = np.median([a for a, _ in v], 0); c = np.median([b for _, b in v], 0)
    print(f"{f0:4.0f} {rd:3.1f}  " + " ".join(f"{x:.3f}" for x in o) + "    " + " ".join(f"{x:6.1f}" for x in c))
