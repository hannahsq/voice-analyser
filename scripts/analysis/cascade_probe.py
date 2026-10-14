"""Cascade probe (D-026 fail branch, pre-registered 2026-10-15): how much of an upper formant's shift can the
cascaded heads carry? Head n sees the expected bins of heads < n (and is softly masked below head n-1), so a shift
in F1 / F2 can pass to F3 through the conditioning rather than through the shared encoder.

On PTDB-TUG dev frames with a closed-phase reference (ptdb_egg_sub.npz), the encoder runs once per utterance and the
heads run again with the expected bins passed on from chosen heads moved by +-30 cents (the shift reaches the later
heads' inputs and the mask; the moved heads' own outputs are unchanged). Slope = (F_k(+) - F_k(-)) / 60 cents,
median over frames, per sex:
  F2<F1, F3<F1, F3<F2 (one head moved), F3<F1F2 (both moved: a uniform scale shift), F4<F1F2F3.
v2m's d16k underread is uniform in cents (women -19 / -18 / -19), so F3<F1F2 is the share of F3's shift the cascade
can carry. Summary: CASCADE <model> <sex> <F3<F1F2>; DECISION: material if the mean of the v2m d16k models' women's
F3<F1F2 is >= 0.33, else small.
Args: scratch dir, out file, name=ckpt ... (CPU, $VA_THREADS threads)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from vocal_analysis.features.vqt import HOP
from vocal_analysis.models.bins import FORMANT_AXIS, decode
from vocal_analysis.models.infer import _vqt, load_teacher
from vocal_analysis.models.teacher import normalise

sys.stdout.reconfigure(encoding="utf-8")
S, OUT = Path(sys.argv[1]), sys.argv[2]
torch.set_num_threads(int(os.environ.get("VA_THREADS", "6")))
cpu = torch.device("cpu")
models = {a.split("=", 1)[0]: load_teacher(a.split("=", 1)[1], cpu)[0] for a in sys.argv[3:]}
D_CENTS = 30.0
D_BINS = D_CENTS / FORMANT_AXIS.cents_per_bin
PERTURB = {"F2<F1": ((0,), 1), "F3<F1": ((0,), 2), "F3<F2": ((1,), 2), "F3<F1F2": ((0, 1), 2), "F4<F1F2F3": ((0, 1, 2), 3)}
out = open(OUT, "w", encoding="utf-8")
p = lambda *a: (print(*a, file=out, flush=True), print(*a, flush=True))  # noqa: E731

tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(str(s) if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
keep = dev & np.all(np.isfinite(d["cp"][:, :3]), 1)
groups = {}
for i in np.nonzero(keep)[0]:
    groups.setdefault((d["speaker"][i], d["utt"][i]), []).append(i)


def wav(spk, utt):
    return Path("data/ptdb_tug") / ("FEMALE" if spk[0] == "F" else "MALE") / "MIC" / spk / f"{utt}.wav"


@torch.inference_mode()
def heads(model, x16):
    """Decoded F1-F4 (T, 4) per perturbation key: 'base' and (key, +1 / -1)."""
    nf = len(x16) // HOP + 1
    x = normalise(_vqt(cpu).log_power(torch.from_numpy(x16)[None], nf))
    h0 = model.f0_net(x)
    if model.cfg.comb:
        f0 = model.expected_f0(model.f0_head(h0).float())
        vp = torch.sigmoid(model.voicing_head(h0)[..., 0].float())
        x = torch.cat([x, model.comb(f0, vp).to(x.dtype)], 1)
    h = model.enc(x)
    runs = {"base": {}} | {(k, sg): {src: sg * D_BINS for src in srcs} for k, (srcs, _) in PERTURB.items() for sg in (1, -1)}
    res = {}
    for key, shift in runs.items():
        logits, means = [], []
        for n, head in enumerate(model.formant_heads):  # as Teacher.forward, with the passed-on means shifted
            inp = torch.cat([h] + [(m / FORMANT_AXIS.n_bins)[..., None].to(h.dtype) for m in means], -1) if means else h
            lg = head(inp).float()
            if means:
                below = (means[-1][..., None] - model.cfg.mask_margin_bins - model.fm_bins).clamp_min(0.0)
                lg = lg - model.cfg.mask_strength * torch.tanh(below / model.cfg.mask_margin_bins)
            logits.append(lg)
            means.append((lg.softmax(-1) * model.fm_bins).sum(-1) + shift.get(n, 0.0))
        res[key] = decode(torch.stack(logits, 2)[0].softmax(-1).numpy(), FORMANT_AXIS)[0]
    return res


audio = {}
for (spk, utt) in groups:
    x, sr = sf.read(wav(spk, utt))
    audio[spk, utt] = resample_poly(x[:, 0] if x.ndim == 2 else x, 16000, sr).astype(np.float32)
p(f"Cascade probe (D-026): PTDB-TUG dev, {keep.sum()} closed-phase frames in {len(groups)} utterances; "
  f"passed-on expected bins moved by +-{D_CENTS:.0f} cents")
p("Slope = change in the later formant (cents) per cent moved, median over frames")
summary = {}
for name, model in models.items():
    slopes = {k: np.full(len(d["t"]), np.nan) for k in PERTURB}
    for (spk, utt), idx in groups.items():
        r = heads(model, audio[spk, utt])
        j = np.clip(np.round(d["t"][idx] / 0.010).astype(int), 0, len(r["base"]) - 1)
        for k, (_, tgt) in PERTURB.items():
            up, dn = r[k, 1][j, tgt], r[k, -1][j, tgt]
            slopes[k][idx] = 1200 * np.log2(up / dn) / (2 * D_CENTS)
    for sex in ("f", "m"):
        m = keep & (d["sex"] == sex)
        vals = {k: np.nanmedian(v[m]) for k, v in slopes.items()}
        summary[name, sex] = vals["F3<F1F2"]
        p(f"  {name:24s} {sex}: " + "  ".join(f"{k} {v:+.3f}" for k, v in vals.items()))
p("")
for (name, sex), v in summary.items():
    p(f"CASCADE {name} {sex} {v:+.3f}")
v2m = [v for (name, sex), v in summary.items() if sex == "f" and name.startswith("v2m") and name.endswith("d16k")]
share = float(np.mean(v2m)) if v2m else float("nan")
p(f"v2m d16k women's F3<F1F2: {share:+.3f} (material if >= 0.33)")
p(f"DECISION: {'material' if share >= 0.33 else 'small'}")
