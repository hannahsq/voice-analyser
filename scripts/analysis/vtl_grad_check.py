"""D-036, before registering (AGENTS: check a hypothesis can hold): the size of the VTL loss's gradient in the shared
trunk at the branch point, against the existing losses', to set its weight. Teacher v3's constant-phase checkpoint
(conformer_v2r2_s1/ckpt_016000) with D-035's trained T head attached (a fitted head, as the joint head will be by the
end of the gate's ramp; it was trained on the decayed encoder, so it is a stand-in). Training crops as the run draws
them (train_v2 + resynthesised sets, the run's channel; 200 frames). Per batch: the trunk's gradient norm from the
teacher loss (weights as trained) and from the VTL NLL at weight 1, and their cosine. Trunk = every parameter outside
the output heads. Args: D-035 head.pt, n batches [default 20]."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vocal_analysis.device import setup_device
from vocal_analysis.features.vqt import TorchVQT
from vocal_analysis.models.teacher import Teacher, TeacherConfig
from vocal_analysis.train.data import MixedCrops, SynthCrops, collate, consolidate, split_ids
from vocal_analysis.train.losses import teacher_loss
from vocal_analysis.train.teacher import HEAD_MODULES, _saved_channel

def main():
    HEAD, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
    RUN = Path("runs/conformer_v2r2_s1")
    dev = setup_device()
    print("loading", flush=True)
    ck = torch.load(RUN / "ckpt_016000.pt", map_location="cpu", weights_only=False)
    cfg = TeacherConfig(**{**ck["teacher_config"], "vtl_head": True})
    model = Teacher(cfg).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("vtl_head.") for k in missing)
    model.vtl_head.load_state_dict({k.removeprefix("net."): v for k, v in torch.load(HEAD, map_location="cpu", weights_only=False)["state"].items()})
    model.train()
    model.vtl_gate = 1.0
    tc = json.loads((RUN / "config.json").read_text())["train"]
    channel = _saved_channel(RUN)
    flat = consolidate(tc["data"])
    train_ids, _ = split_ids(json.loads((flat / "done").read_text())["n_clips"])
    parts, sizes = [SynthCrops(flat, train_ids, 200, channel, seed=4242, epoch_size=N * 32, vtl=True)], []
    for d in tc["resynth"]:
        rflat = consolidate(d)
        n = json.loads((rflat / "done").read_text())["n_clips"]
        parts.append(SynthCrops(rflat, np.arange(n), 200, channel, seed=4243 + len(sizes), epoch_size=N * 32, vtl=True,
                                tract_left_weight=tuple(tc["tract_left_weight"])))
        sizes.append(n)
    ds = MixedCrops(parts, [1 - tc["resynth_frac"]] + [tc["resynth_frac"] * n / sum(sizes) for n in sizes], N * 32, seed=4242)
    trunk = [p for n, p in model.named_parameters() if n.split(".")[0] not in HEAD_MODULES + ("vtl_head",)]
    vq = TorchVQT(dev)
    lw = tc.get("loss_weights") or {}
    rows = []
    for audio, lab in DataLoader(ds, 32, num_workers=0, collate_fn=collate):
        audio = audio.to(dev)
        lab = {k: v.to(dev) for k, v in lab.items()}
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out = model(vq.log_power_with_context(audio, lab["f0"].shape[1]))
        total_base, parts_ = teacher_loss(out, lab, {**lw, "vtl": 0.0})
        g_base = torch.autograd.grad(total_base, trunk, retain_graph=True, allow_unused=True)
        if "vtl" not in parts_:
            continue
        l_vtl, _ = teacher_loss(out, lab, {k: 0.0 for k in ("f0", "voicing", "formants", "bandwidth", "sonorant", "obs")} | {"vtl": 1.0})
        g_vtl = torch.autograd.grad(l_vtl, trunk, allow_unused=True)
        fb = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(g_base, trunk)])
        fv = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(g_vtl, trunk)])
        rows.append((fb.norm().item(), fv.norm().item(), (fb @ fv / (fb.norm() * fv.norm() + 1e-12)).item(), parts_["vtl"].item(),
                     int(torch.isfinite(lab["vtl"]).sum())))
        print(f"batch {len(rows)}: |g_base| {rows[-1][0]:.4f}  |g_vtl| {rows[-1][1]:.4f}  cos {rows[-1][2]:+.3f}  VTL NLL {rows[-1][3]:.2f}  "
              f"labelled frames {rows[-1][4]}", flush=True)
    r = np.array(rows)
    ratio = np.median(r[:, 1] / r[:, 0])
    print(f"\n{len(r)} batches: median |g_vtl| / |g_base| at weight 1 = {ratio:.3f} (IQR {np.percentile(r[:, 1] / r[:, 0], 25):.3f}-"
          f"{np.percentile(r[:, 1] / r[:, 0], 75):.3f}); median cosine {np.median(r[:, 2]):+.3f}; median VTL NLL {np.median(r[:, 3]):.2f}")
    print(f"weight for a 10 % share of the trunk gradient: {0.1 / ratio:.3g}")


if __name__ == "__main__":
    main()
