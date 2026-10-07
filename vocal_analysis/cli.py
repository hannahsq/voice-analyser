"""Command-line entry point: `va <command>` (or `python -m vocal_analysis`).

    va env                         torch/GPU check, including the MIOpen workaround
    va download <corpus>...|all    fetch, verify and extract public corpora into data.root
    va manifest <corpus>|all       scan a corpus under data.root and write its manifest
    va leakage                     run the leakage rules over data/manifests/
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
    from .eval.datasets import registry

    reg = registry()
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


def cmd_leakage(cfg: Config, args: argparse.Namespace) -> int:
    from .data.manifest import read_manifests
    from .eval.leakage import check_leakage

    rows = read_manifests(cfg.data.manifests)
    rep = check_leakage(rows)
    print(f"{len(rows)} rows checked: {'OK' if rep.ok else 'FAILED'}")
    for e in rep.errors:
        print("  " + e)
    return 0 if rep.ok else 1


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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.config or ("configs/default.yaml" if Path("configs/default.yaml").exists() else None)
    cfg = load_config(path, args.overrides)
    return args.fn(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
