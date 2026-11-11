"""D-085 note (after the result): which loss parts drive the trunk's gradient once the VTL gate opens.

For each checkpoint, one fixed batch of short crops (train_v2's validation clips with length labels, 400 frames) and one of long crops
(train_long_v3u's held-out clips, 1600 frames), the gate at 1 as in training after step 1700. Per loss part: the
gradient norm on the trunk (f0_net and enc) of the part unweighted, the weight the run gave it at that step, and the
weighted norm. 5b's weights are the fixed ones (losses.WEIGHTS); D-085's length weights come from _AuxWeights, rebuilt
from the training log's per-part means over the 500 steps before the checkpoint (an approximation of its EMA).
CPU only.

    python scripts/analysis/d085_grad_check.py [--short N] [--long N] > reports/analysis/d085/grad_check.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CKPTS = [("5b s2, step 2000", "runs/d040_5b_s2/ckpt_002000.pt", "runs/d040_5b_s2"),
         ("D-085 base, step 2000", "runs/d085_base/ckpt_002000.pt", "runs/d085_base"),
         ("D-085 base, step 16000", "runs/d085_base/ckpt_016000.pt", "runs/d085_base"),
         ("D-085 heads, step 2000", "runs/d085_heads/ckpt_002000.pt", "runs/d085_heads")]
MAIN = ("f0", "voicing", "formants", "bandwidth", "sonorant", "formants_acoustic", "bandwidth_acoustic", "upper")


def run_weights(run: str, step: int) -> dict:
    """The weights the run trained with near `step`."""
    from vocal_analysis.train.losses import WEIGHTS

    cfg = json.loads((Path(run) / "config.json").read_text())["train"]
    w = {**WEIGHTS, **(cfg.get("loss_weights") or {})}
    if cfg.get("aux_budget") is None:
        return w
    rows = [json.loads(line) for line in open(Path(run) / "log.jsonl")]
    rows = [r for r in rows if "loss" in r and "eval" not in r and step - 500 < r["step"] <= step]
    mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    main = sum(w[k] * mean(k) for k in rows[0] if k in MAIN)
    concepts = cfg["aux_concepts"]
    share = cfg["aux_budget"] / (1 - cfg["aux_budget"]) * abs(main) / len(concepts)
    for ks in concepts.values():
        for k in ks:
            w[k] = share / len(ks) / max(abs(mean(k)), cfg["aux_floor"])
    return w


def batch(set_root, frames, n):
    from torch.utils.data import DataLoader

    from d085_score import val_channel
    from vocal_analysis.train.data import SynthCrops, collate, consolidate, split_ids

    import numpy as np

    flat = consolidate(set_root)
    val = split_ids(json.loads((flat / "done").read_text())["n_clips"])[1]
    labelled = np.load(flat / "clip_tube.npy") & ~np.load(flat / "clip_length_from_f0.npy")  # where the length labels are
    ids = val[labelled[val]][:n]
    ds = SynthCrops(flat, ids, frames, val_channel(), seed=12345, fixed=True, vtl=True, vtl_inst=True, acoustic=True)
    return next(iter(DataLoader(ds, n, num_workers=0, collate_fn=collate)))


if __name__ == "__main__":
    import torch

    from vocal_analysis.features.vqt import TorchVQT
    from vocal_analysis.models.infer import load_teacher
    from vocal_analysis.train.losses import teacher_loss

    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    n_short, n_long = int(args.get("--short", 8)), int(args.get("--long", 2))
    torch.manual_seed(0)
    dev = torch.device("cpu")
    vq = TorchVQT(dev)
    batches = {"short 400": batch("data/synth/train_v2", 400, n_short),
               "long 1600": batch("data/synth/train_long_v3u", 1600, n_long)}
    feats = {k: (vq.log_power_with_context(a, lab["f0"].shape[1]), lab) for k, (a, lab) in batches.items()}
    print(f"D-085 gradient check: trunk (f0_net + enc) gradient norm per loss part; {n_short} short crops of 400 frames "
          f"(train_v2 validation), {n_long} long crops of 1600 (train_long_v3u held-out), all clips with length labels (tube, length not drawn from f0); VTL gate 1; CPU, float32\n")
    for name, ck, run in CKPTS:
        model, meta = load_teacher(ck, dev)
        step = int(meta["step"])
        for attr in ("vtl_gate", "upper_gate", "opening_gate"):
            if hasattr(model, attr):
                setattr(model, attr, 1.0)
        w = run_weights(run, step)
        trunk = [p for n, p in model.named_parameters() if n.startswith(("f0_net.", "enc."))]
        print(f"{name} ({ck})")
        print(f"  {'part':20s} {'weight':>8s} | " + " | ".join(f"{k:^25s}" for k in feats))
        print(f"  {'':20s} {'':>8s} | " + " | ".join(f"{'unweighted':>12s} {'weighted':>12s}" for _ in feats))
        res = {}
        for kind, (lp, lab) in feats.items():
            out = model(lp)
            _, parts = teacher_loss(out, lab, w)  # its parts come back detached: each part's gradient from one-hot weights
            res[kind] = {}
            for k in parts:
                v, _ = teacher_loss(out, lab, {j: float(j == k) for j in w})
                if not v.requires_grad:  # no frames for this part in the batch
                    res[kind][k] = float("nan")
                    continue
                g = torch.autograd.grad(v, trunk, retain_graph=True, allow_unused=True)
                res[kind][k] = torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)).item()
            tot, _ = teacher_loss(out, lab, w)
            g = torch.autograd.grad(tot, trunk, allow_unused=True)
            res[kind]["total (weighted)"] = torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None)).item()
        for k in res["short 400"]:
            wk = 1.0 if k.startswith("total") else w[k]
            cells = []
            for kind in feats:
                u = res[kind].get(k, float("nan"))
                cells.append(f"{u:12.3f} {u * wk if not k.startswith('total') else u:12.3f}")
            print(f"  {k:20s} {wk:8.4f} | " + " | ".join(cells))
        print()
