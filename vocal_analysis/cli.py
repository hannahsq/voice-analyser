"""Command-line entry point: `va <command>` (or `python -m vocal_analysis`).

    va env                         torch/GPU check, including the MIOpen workaround
    va download <corpus>...|all    fetch, verify and extract public corpora into data.root
    va manifest <corpus>|all       scan a corpus under data.root and write its manifest
    va leakage                     run the leakage rules over data/manifests/
    va baselines [--only NAME...]  run/score baselines -> reports/baselines.md (cached in runs/)
    va config                      print the resolved config

Every command takes --config FILE and repeated --set key.sub=value overrides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import Config, config_to_dict, load_config


def cmd_config(cfg: Config, args: argparse.Namespace) -> int:
    print(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), end="")
    return 0


def cmd_env(cfg: Config, args: argparse.Namespace) -> int:
    import time

    import torch

    from .device import rnn_guard, setup_device

    dev = setup_device(cfg.device)
    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {dev}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        print(f"  {p.name} ({getattr(p, 'gcnArchName', '?')}), {p.total_memory / 2**30:.1f} GiB visible")
    conv = torch.nn.Conv2d(2, 16, 3, padding=1).to(dev)
    gru = torch.nn.GRU(16 * 32, 64, batch_first=True, bidirectional=True).to(dev)
    x = torch.randn(4, 2, 32, 50, device=dev)
    t = time.time()
    for bf16 in (False, True):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            h = conv(x).permute(0, 3, 1, 2).flatten(2)
            with rnn_guard():
                y, _ = gru(h)
            loss = y.float().pow(2).mean()
        loss.backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"  conv + BiGRU forward/backward (fp32, bf16): ok in {time.time() - t:.1f}s")
    return 0


def cmd_manifest(cfg: Config, args: argparse.Namespace) -> int:
    from .data.manifest import write_manifest
    from .data.pools import registry as pools
    from .eval.datasets import registry

    reg = registry() | pools()  # gold sets and training pools
    names = sorted(reg) if args.corpus == "all" else [args.corpus]
    status = 0
    for name in names:
        if name not in reg:
            print(f"unknown corpus {name!r}; known: {', '.join(sorted(reg))}", file=sys.stderr)
            return 2
        root = Path(args.root) if args.root else Path(cfg.data.root) / name
        if not root.exists():
            print(f"[{name}] skipped: {root} not found")
            status = status or (1 if args.corpus != "all" else 0)
            continue
        if name == "mdb_stem_synth" and not (root / "vocal_stems.txt").exists():
            print(f"[{name}] WARNING: no vocal_stems.txt; including instrument stems")
        try:
            rows = reg[name].scan(root)
        except NotImplementedError as e:
            print(f"[{name}] skipped: {e}")
            continue
        out = write_manifest(rows, Path(cfg.data.manifests) / f"{name}.csv")
        speakers = len({r.speaker_id for r in rows})
        print(f"[{name}] {len(rows)} files, {speakers} speakers, {sum(r.duration for r in rows) / 3600:.2f} h -> {out}")
    return status


def cmd_download(cfg: Config, args: argparse.Namespace) -> int:
    from .data.download import SOURCES, download

    names = sorted(SOURCES) if args.corpora == ["all"] else args.corpora
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"unknown corpus {unknown}; downloadable: {', '.join(sorted(SOURCES))}", file=sys.stderr)
        return 2
    log = lambda m: print(m, flush=True)  # noqa: E731
    for name in names:
        dest = download(name, Path(cfg.data.root), log=log)
        if name == "mdb_stem_synth" and not (dest / "vocal_stems.txt").exists():
            from .eval.datasets.mdb_stem_synth import write_vocal_allowlist

            write_vocal_allowlist(dest, log=log)
    return 0


def cmd_baselines(cfg: Config, args: argparse.Namespace) -> int:
    from .eval.report import build_report

    build_report(cfg, Path(args.out), args.only, args.force, log=lambda m: print(m, flush=True))
    return 0


def cmd_leakage(cfg: Config, args: argparse.Namespace) -> int:
    from .data.manifest import read_manifests
    from .eval.leakage import check_leakage

    rows = read_manifests(cfg.data.manifests)
    rep = check_leakage(rows)
    print(f"{len(rows)} rows checked: {'OK' if rep.ok else 'FAILED'}")
    for e in rep.errors:
        print("  " + e)
    return 0 if rep.ok else 1


def cmd_synth(cfg: Config, args: argparse.Namespace) -> int:
    from .synth.sampler import SynthConfig
    from .synth.shards import write_set, write_suite

    log = lambda m: print(m, flush=True)  # noqa: E731
    root = Path(cfg.data.root) / "synth"
    if args.what == "suite":
        write_suite(Path(args.out or root / "suite_v1"), args.n or 64, args.workers, log=log)
    else:
        write_set(Path(args.out or root / "train_v2"), args.n or 100_000, SynthConfig(), seed=args.seed,
                  workers=args.workers, log=log)
    return 0


def _parse_value(v: str):
    """int, float, None, a tuple (comma list) or str, for --train-set fields whose default is None."""
    if v.lower() == "none":
        return None
    if "," in v:
        return tuple(_parse_value(x) for x in v.split(","))
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from .models.teacher import TeacherConfig
    from .train.teacher import TrainConfig, train

    if args.branch_from:  # decay branch of a wsd run: everything else comes from the parent run's config
        import torch

        from .train.teacher import branch_config

        ck = Path(args.branch_from)
        step = torch.load(ck, map_location="cpu", weights_only=False)["step"]
        tc = branch_config(ck.parent, step, args.out or f"{ck.parent}_d{step // 1000:02d}k", args.decay_steps)
        tc.branch_from = str(ck)
        tc.workers = args.workers
    else:
        tc = TrainConfig(data=args.data, out=args.out or "runs/teacher_v2", steps=args.steps, batch=args.batch,
                         workers=args.workers)
    for kv in args.train_set:
        k, v = kv.split("=", 1)
        cur = getattr(tc, k)
        setattr(tc, k, _parse_value(v) if cur is None else type(cur)(v))
    mc = TeacherConfig(comb=not args.no_comb)
    for kv in args.model_set:  # new runs only: a resumed run or branch keeps its checkpoint's architecture
        k, v = kv.split("=", 1)
        cur = getattr(mc, k)
        if isinstance(cur, tuple):
            setattr(mc, k, tuple(int(x) for x in v.split(",")))
        else:
            setattr(mc, k, v.lower() in ("1", "true") if isinstance(cur, bool) else type(cur)(v))
    train(tc, mc, log=lambda m: print(m, flush=True))
    return 0


def cmd_eval(cfg: Config, args: argparse.Namespace) -> int:
    from .eval.teacher_eval import build_teacher_report

    build_teacher_report(cfg, args.ckpt, args.out, args.suite, log=lambda m: print(m, flush=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml if present)")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    p = argparse.ArgumentParser(prog="va", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("config", parents=[common]).set_defaults(fn=cmd_config)
    sub.add_parser("env", parents=[common]).set_defaults(fn=cmd_env)
    m = sub.add_parser("manifest", parents=[common])
    m.add_argument("corpus")
    m.add_argument("--root", default=None, help="corpus root (default: <data.root>/<corpus>)")
    m.set_defaults(fn=cmd_manifest)
    d = sub.add_parser("download", parents=[common])
    d.add_argument("corpora", nargs="+", help="corpus names, or 'all'")
    d.set_defaults(fn=cmd_download)
    sub.add_parser("leakage", parents=[common]).set_defaults(fn=cmd_leakage)
    b = sub.add_parser("baselines", parents=[common])
    b.add_argument("--only", nargs="+", default=None, help="baseline names to run (default: all)")
    b.add_argument("--force", action="store_true", help="ignore cached predictions")
    b.add_argument("--out", default="reports/baselines.md")
    b.set_defaults(fn=cmd_baselines)
    s = sub.add_parser("synth", parents=[common], help="generate synthetic training data or the hard-case suite")
    s.add_argument("what", choices=["train", "suite"])
    s.add_argument("--n", type=int, default=None, help="clips (train: total, default 100000; suite: per case, default 64)")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default=None, help="default: <data.root>/synth/{train_v2,suite_v1}")
    s.set_defaults(fn=cmd_synth)
    t = sub.add_parser("train", parents=[common], help="train a model on synthetic data")
    t.add_argument("model", choices=["teacher"])
    t.add_argument("--data", default="data/synth/train_v2")
    t.add_argument("--out", default=None, help="default: runs/teacher_v2, or <parent run>_dNNk for a branch")
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--branch-from", default=None, metavar="CKPT",
                   help="start a decay branch of a wsd run from this checkpoint (other options come from that run)")
    t.add_argument("--decay-steps", type=int, default=None, help="branch decay length (default: the run's, 4000)")
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--no-comb", action="store_true", help="ablation: no harmonic-comb input channel")
    t.add_argument("--train-set", action="append", default=[], metavar="KEY=VALUE", help="other TrainConfig fields")
    t.add_argument("--model-set", action="append", default=[], metavar="KEY=VALUE",
                   help="TeacherConfig fields for a new run, e.g. channels=16,32,48 gru_hidden=128")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval", parents=[common], help="evaluate a trained teacher against the gold sets and the suite")
    e.add_argument("ckpt")
    e.add_argument("--out", default="reports/teacher_v1.md")
    e.add_argument("--suite", default="data/synth/suite_v1")
    e.set_defaults(fn=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.config or ("configs/default.yaml" if Path("configs/default.yaml").exists() else None)
    cfg = load_config(path, args.overrides)
    return args.fn(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
