"""D-027 label, second definition (pre-registered 2026-10-18 before running): does the label with a recording floor and
measured source structure track where a model struggles? On 300 validation crops of train_v2 (the trainer's own
validation draws: seed 12345, fixed crops, channels on), teacher v3's per-frame |error| (cents; the model as training
runs it, on the crop with its context) against the label, on sonorant frames: Spearman(o, |error|) per formant, for
  old:  the first label (floor 60 dB under each frame's peak, independent 2 dB scatter);
  new:  recording floor (45 dB), the measured source structure (`ObservabilityConfig` defaults);
  new35 / new55: the new label with a 35 / 55 dB recording floor (sensitivity, reported).
Also the clean label on the Rd-sweep renders (Story's 10 vowels, 15 cm, f0 210 / 300, Rd 0.5 / 2.5): F3 posterior SD.
Rule: "calibrated" if for F3 and F4 Spearman(new) <= -0.15 and at least 0.05 below Spearman(old), and for F1 and F2
Spearman(new) is at most 0.05 above Spearman(old); and the channel validation (obs_channel_validate.py, unchanged rule)
accepts the new label. Otherwise "prior". Lines "DECISION: ..." (the channel validation is judged separately).
Device from VA_DEVICE (default cpu). Args: out file, checkpoint (teacher v3)."""
import os
import sys
from dataclasses import replace

import numpy as np
import torch
from scipy.stats import spearmanr

from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.bins import FORMANT_AXIS, decode
from vocal_analysis.models.infer import load_teacher
from vocal_analysis.synth import observability as ob
from vocal_analysis.synth import tract, tube, voice
from vocal_analysis.synth.channel import ChannelConfig
from vocal_analysis.train.data import SynthCrops, split_ids

sys.stdout.reconfigure(encoding="utf-8")
OUT = open(sys.argv[1], "w", encoding="utf-8")
p = lambda *a: (print(*a, file=OUT, flush=True), print(*a, flush=True))  # noqa: E731
dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
model = load_teacher(sys.argv[2], dev)[0]
vq = TorchVQT(dev)
flat = "data/synth/train_v2/flat"
_, val = split_ids(100000)
ids = val[:300]
NEW = ob.ObservabilityConfig()
OLD = replace(NEW, src_sd_db=None)
labels = {"old": (OLD, "frame", None), "new": (NEW, "recording", 45.0), "new35": (NEW, "recording", 35.0),
          "new55": (NEW, "recording", 55.0)}
sets = {}
for name, (cfg, floor, rec) in labels.items():
    sets[name] = SynthCrops(flat, ids, 200, ChannelConfig(), seed=12345, fixed=True, observability=True, obs_config=cfg,
                            obs_floor=floor, obs_rec_snr_db=rec)
O = {k: [] for k in labels}
E, SON, F0 = [], [], []
for i in range(len(ids)):
    x, lab = sets["old"][i]
    a = torch.from_numpy(x)[None].to(dev)
    with torch.inference_mode():
        out = model(vq.log_power_with_context(a, len(lab["f0"])))
    fm, _ = decode(out["formant_logits"][0].float().softmax(-1).cpu().numpy(), FORMANT_AXIS)
    E.append(np.abs(1200 * np.log2(fm[:, :4] / lab["formants"][:, :4])))
    SON.append(lab["sonorant"].astype(bool))
    F0.append(np.where(lab["voiced"].astype(bool), lab["f0"], np.nan))
    O["old"].append(lab["obs"])
    for name in ("new", "new35", "new55"):
        x2, lab2 = sets[name][i]  # the same fixed crop and channel
        assert np.array_equal(x, x2)
        O[name].append(lab2["obs"])
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(ids)}", flush=True)
E, SON, F0 = np.concatenate(E), np.concatenate(SON), np.concatenate(F0)
O = {k: np.concatenate(v) for k, v in O.items()}
p(f"D-027 label check: 300 validation crops of train_v2 (channels on), teacher v3 ({sys.argv[2]})")
p("Spearman(o, |error| in cents), sonorant frames with a label, F1 / F2 / F3 / F4:")
rho = {}
for name, o in O.items():
    r = []
    for k in range(4):
        m = SON & np.isfinite(o[:, k]) & np.isfinite(E[:, k])
        r.append(float(spearmanr(o[m, k], E[m, k])[0]))
    rho[name] = r
    p(f"  {name:6s} " + "  ".join(f"{x:+.3f}" for x in r))
p("Median o by f0 band (voiced), F3 / F4, old -> new:")
for lo, hi in ((0, 150), (150, 250), (250, 350), (350, 600)):
    m = (F0 >= lo) & (F0 < hi)
    if m.sum() > 200:
        p(f"  f0 {lo}-{hi}: F3 {np.nanmedian(O['old'][m, 2]):.3f} -> {np.nanmedian(O['new'][m, 2]):.3f}   "
          f"F4 {np.nanmedian(O['old'][m, 3]):.3f} -> {np.nanmedian(O['new'][m, 3]):.3f}   (median |error| F3 {np.nanmedian(E[m, 2]):.0f} cents)")

# Rd-sweep renders, clean
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
            for name, cfg, rec in (("old", OLD, None), ("new", NEW, 45.0)):
                snr = ob.component_snr(comps, t, np.ones(100, bool), rec_snr_db=rec)
                _, sd = ob.observability(np.full(100, f0), np.ones(100, bool), np.ones(100, bool), np.tile(F[:4], (100, 1)),
                                         np.tile(B[:4], (100, 1)), snr, cfg)
                res.setdefault((name, f0, rd), []).append(np.median(1200 / np.log(2) * sd[30:71, 2] / F[2]))
p("Rd-sweep renders (clean): F3 posterior SD, cents, median over vowels, old -> new:")
for f0 in (210.0, 300.0):
    for rd in (0.5, 2.5):
        p(f"  f0 {f0:.0f} Rd {rd}: {np.median(res[('old', f0, rd)]):.1f} -> {np.median(res[('new', f0, rd)]):.1f}")
ok = all(rho["new"][k] <= -0.15 and rho["new"][k] <= rho["old"][k] - 0.05 for k in (2, 3)) and \
    all(rho["new"][k] <= rho["old"][k] + 0.05 for k in (0, 1))
p(f"DECISION: {'calibrated' if ok else 'prior'} (the channel validation judged separately)")
