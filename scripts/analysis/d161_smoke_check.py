"""D-161, before training, item 5: the smoke run's checks.

1. 5b's step-16000 checkpoint, loaded into the widened model (fourth channel at zero, a fresh source head), gives
   5b's outputs exactly, in both arms (source_env "pred" and "zero"), on a few synthetic validation clips.
2. The smoke run's log (runs/d161_smoke/log.jsonl): the share of clips fed the true levels starts near the synthetic
   share and falls with the ramp; clips held at zero during the head's warm-up are the resynthesised ones (zero share
   = 1 - synthetic share) and none after it; the source loss's share of the total loss.

    python scripts/analysis/d161_smoke_check.py   -> reports/analysis/d161/smoke.txt (exit 1 on a failed check)
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.teacher import Teacher, TeacherConfig
    from vocal_analysis.train.teacher import _widen_input

    lines, ok = ["D-161 smoke checks"], True
    ck = torch.load(ROOT / "runs/d040_5b_s1/ckpt_016000.pt", map_location="cpu", weights_only=False)
    base_cfg = TeacherConfig(**ck["teacher_config"])
    base = Teacher(base_cfg).eval()
    base.load_state_dict(ck["model"])
    flat = ROOT / "data/synth/train_v2/flat"
    ao = np.load(flat / "audio_offsets.npy")
    audio = np.load(flat / "audio.npy", mmap_mode="r")
    n = len(ao) - 1
    dev = torch.device("cpu")
    vq = TorchVQT(dev)
    clips = [np.asarray(audio[ao[i] : min(ao[i + 1], ao[i] + 3 * 16000)], np.float32) / 32768.0 for i in (n - 512, n - 300, n - 50)]
    for env in ("pred", "zero"):
        wide = Teacher(dataclasses.replace(base_cfg, source_head=True, source_env=env)).eval()
        sd = {k: v.clone() for k, v in ck["model"].items()}
        _widen_input(sd, wide)
        missing, unexpected = wide.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("source_head.") for k in missing), (missing, unexpected)
        worst = 0.0
        with torch.no_grad():
            for y in clips:
                x = torch.from_numpy(y)[None]
                lp = vq.log_power(x, len(y) // 160 + 1)
                a, b = base(lp), wide(lp)
                for k in ("formant_logits", "logb_mean", "f0_logits", "voicing_logit", "sonorant_logit"):
                    worst = max(worst, float((a[k] - b[k]).abs().max()))
        good = worst < 1e-4
        ok &= good
        lines.append(f"  step 16000, source_env {env}: max |output difference| against 5b {worst:.2e} -> {'OK' if good else 'FAILED'}")
    log = [json.loads(s) for s in (ROOT / "runs/d161_smoke/log.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    rows = [r for r in log if "source_fed_clips" in r]
    if not rows:
        lines.append("  smoke log: no source_fed_clips rows -> FAILED")
        ok = False
    else:
        lines.append("  smoke log (per log interval): step, fed share, zero share, synthetic share, source loss, total loss")
        for r in rows:
            lines.append(f"    {r['step']:6d}  {r['source_fed_clips']:.3f}  {r['source_zero_clips']:.3f}  {r['source_synth_clips']:.3f}"
                         f"  {r.get('source', float('nan')):.4f}  {r.get('loss', float('nan')):.4f}")
        first, last = rows[0], rows[-1]
        ramp_ok = abs(first["source_fed_clips"] - first["source_synth_clips"]) < 0.1 and last["source_fed_clips"] < first["source_fed_clips"]
        warm = [r for r in rows if r["step"] <= 16000 + 300]
        zero_ok = all(abs(r["source_zero_clips"] - (1 - r["source_synth_clips"])) < 0.05 for r in warm) and \
            all(r["source_zero_clips"] == 0 for r in rows if r["step"] > 16000 + 300 + 10)
        share = float(np.mean([0.1 * r["source"] / r["loss"] for r in rows if r.get("source") and r.get("loss")]))
        lines += [f"  truth ramp: {'OK' if ramp_ok else 'FAILED'}; zero channel during the warm-up only: {'OK' if zero_ok else 'FAILED'}",
                  f"  source loss's share of the total (weight 0.1): {share:.1%} (reported)"]
        ok &= ramp_ok and zero_ok
    lines.append(f"Result: {'PASSES' if ok else 'FAILS'}")
    out = ROOT / "reports/analysis/d161/smoke.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
