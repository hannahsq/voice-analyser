"""Command-line entry point: `va <command>` (or `python -m vocal_analysis`).

    va env                         torch/GPU check, including the MIOpen workaround
    va download <corpus>...|all    fetch, verify and extract public corpora into data.root
    va manifest <corpus>|all       scan a corpus under data.root and write its manifest
    va leakage                     run the leakage rules over data/manifests/
    va tiers [--dev-copy IN OUT]   assign gold speakers to dev / semi / lockbox (eval/tiers.json); copy IN's dev rows to OUT
    va baselines [--only NAME...]  run/score baselines on dev speakers -> reports/suite/baselines_dev.md (cached in runs/)
    va resynth <pool> --teacher CKPT   analysis-resynthesis training set from a training pool (plan §5.2)
    va analyse FILE... [--plot]    analyse audio files with a teacher -> per-frame tracks, a summary and a plot
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


def _tiers_arg(value: str) -> tuple[str, ...]:
    """--tiers: dev by default; semi-private or lockbox speakers only when named (D-024), and said loudly."""
    tiers = tuple(sorted({t.strip() for t in value.split(",") if t.strip()}))
    if set(tiers) - {"dev"}:
        print(f"*** tiers {', '.join(tiers)}: includes speakers outside dev. Log this use in docs/decisions.md. ***", flush=True)
    return tiers


TIERS_HELP = "gold-speaker tiers to score, comma-separated (D-024): dev (default), semi, lockbox; beyond dev is a logged use"


def cmd_baselines(cfg: Config, args: argparse.Namespace) -> int:
    from .eval.report import build_report

    tiers = _tiers_arg(args.tiers)
    out = Path(args.out or f"reports/suite/baselines_{'+'.join(tiers)}.md")
    build_report(cfg, out, args.only, args.force, log=lambda m: print(m, flush=True), tiers=tiers)
    return 0


def cmd_tiers(cfg: Config, args: argparse.Namespace) -> int:
    from .data.manifest import read_manifest, read_manifests, write_manifest
    from .eval import tiers

    if args.dev_copy:  # a manifest folder restricted to dev speakers, for the measure scripts
        src, dst = map(Path, args.dev_copy)
        t = tiers.load()
        for path in sorted(src.glob("*.csv")):
            rows = read_manifest(path)
            kept = tiers.filter_rows(rows, ("dev",), t)
            write_manifest(kept, dst / path.name)
            print(f"{path.name}: {len(kept)} of {len(rows)} rows (dev speakers)")
        return 0
    if tiers.TIER_FILE.exists() and not args.force:
        print(f"{tiers.TIER_FILE} exists: the assignment is fixed once made (--force rewrites it)")
        return 1
    speakers = tiers.write(read_manifests(cfg.data.manifests))
    print(f"{len(speakers)} gold speakers assigned -> {tiers.TIER_FILE}")
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
        import dataclasses

        sc = SynthConfig()
        over = {}
        for kv in args.synth_set:  # scalar fields only
            k, v = kv.split("=", 1)
            cur = getattr(sc, k)
            over[k] = v.lower() in ("1", "true", "yes") if isinstance(cur, bool) else _parse_value(v) if cur is None else type(cur)(v)
        write_set(Path(args.out or root / "train_v2"), args.n or 100_000, dataclasses.replace(sc, **over), seed=args.seed,
                  workers=args.workers, log=log)
    return 0


def cmd_resynth(cfg: Config, args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from .data.manifest import manifest_hash, read_manifest
    from .data.pools import registry as pools
    from .resynth import corpus
    from .resynth.render import ResynthConfig

    if args.corpus not in pools():
        print(f"{args.corpus!r} is not a training pool; known: {', '.join(sorted(pools()))}", file=sys.stderr)
        return 2
    manifest = Path(cfg.data.manifests) / f"{args.corpus}.csv"
    ok = getattr(pools()[args.corpus], "resynth_ok", lambda r: True)  # e.g. VocalSet's lip trills
    rows = [r for r in read_manifest(manifest) if r.split == args.split and ok(r)]
    if args.limit:  # evenly spaced through the sorted manifest, so a trial spans singers and techniques
        rows = sorted(rows, key=lambda r: r.path)
        rows = [rows[int(i)] for i in np.unique(np.linspace(0, len(rows) - 1, min(args.limit, len(rows))).round())]
    rc = ResynthConfig(variants=args.variants)
    for kv in args.resynth_set:
        k, v = kv.split("=", 1)
        cur = getattr(rc, k)
        setattr(rc, k, v.lower() in ("1", "true", "yes") if isinstance(cur, bool) else _parse_value(v) if cur is None else type(cur)(v))
    log = lambda m: print(m, flush=True)  # noqa: E731
    teacher_sha = corpus.file_sha256(args.teacher)
    root = Path(cfg.data.root) / args.corpus
    from .models.infer import DEFAULT_WINDOW

    # windowed analyses (D-028, from 2026-10-27) apart from the whole-file ones made before
    adir = Path(cfg.data.root) / "resynth" / "analysis" / args.corpus / f"{teacher_sha}-w{DEFAULT_WINDOW}"
    from dataclasses import replace

    from .device import setup_device

    device = setup_device(replace(cfg.device, device=args.device) if args.device else cfg.device)
    corpus.analyse(rows, root, args.teacher, adir, device=device, log=log)
    step = torch.load(args.teacher, map_location="cpu", weights_only=False).get("step")
    out = Path(args.out or Path(cfg.data.root) / "resynth" / f"{args.corpus}_{args.split}")
    corpus.render(rows, root, adir, out, rc, seed=args.seed, workers=args.workers, log=log,
                  extra_index={"corpus": args.corpus, "split": args.split, "manifest_hash": manifest_hash(manifest),
                               "teacher": {"path": Path(args.teacher).as_posix(), "sha256": teacher_sha, "step": step}})
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
                         workers=args.workers, resynth=tuple(args.resynth), resynth_frac=args.resynth_frac,
                         resynth_val=tuple(args.resynth_val), optimizer=args.optimizer)
    for kv in args.train_set:
        k, v = kv.split("=", 1)
        cur = getattr(tc, k)
        # bools by name (bool("false") is True); tuples as comma lists (tuple(str) splits characters). No chain had
        # passed either before 2026-10-30 (D-051), so earlier commands parse as they did
        setattr(tc, k, _parse_value(v) if cur is None else v.lower() in ("1", "true", "yes") if isinstance(cur, bool)
                else tuple(_parse_value(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(v))
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

    tiers = _tiers_arg(args.tiers)
    run = Path(args.ckpt.split("#")[0].split("@w")[0]).parent.name
    out = args.out or f"reports/suite/{run}_{'+'.join(tiers)}.md"
    build_teacher_report(cfg, args.ckpt, out, args.suite, log=lambda m: print(m, flush=True), tiers=tiers)
    return 0


def cmd_analyse(cfg: Config, args: argparse.Namespace) -> int:
    from .analyse import analyse_files

    analyse_files(args.audio, args.model, args.window, args.out, tuple(args.format.split(",")), args.plot, args.mask,
                  args.channel, args.start, args.end, args.device, args.fmax, args.threshold, args.spectrogram,
                  not args.no_normalise,
                  log=lambda m: print(m, flush=True))
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
    tr = sub.add_parser("tiers", parents=[common], help="assign gold speakers to evaluation tiers (decisions D-024)")
    tr.add_argument("--dev-copy", nargs=2, metavar=("IN", "OUT"), help="write IN's manifests restricted to dev speakers to OUT")
    tr.add_argument("--force", action="store_true", help="rewrite an existing assignment")
    tr.set_defaults(fn=cmd_tiers)
    b = sub.add_parser("baselines", parents=[common])
    b.add_argument("--only", nargs="+", default=None, help="baseline names to run (default: all)")
    b.add_argument("--force", action="store_true", help="ignore cached predictions")
    b.add_argument("--out", default=None, help="default: reports/suite/baselines_<tiers>.md")
    b.add_argument("--tiers", default="dev", help=TIERS_HELP)
    b.set_defaults(fn=cmd_baselines)
    s = sub.add_parser("synth", parents=[common], help="generate synthetic training data or the hard-case suite")
    s.add_argument("what", choices=["train", "suite"])
    s.add_argument("--n", type=int, default=None, help="clips (train: total, default 100000; suite: per case, default 64)")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default=None, help="default: <data.root>/synth/{train_v2,suite_v1}")
    s.add_argument("--synth-set", action="append", default=[], metavar="KEY=VALUE",
                   help="other SynthConfig fields (train only, scalar fields)")
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
    t.add_argument("--optimizer", choices=["adamw", "schedulefree", "adan", "muon", "muon_adan"], default="adamw",
                   help="schedulefree: AdamW-ScheduleFree, no decay schedule or branches (D-018)")
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--no-comb", action="store_true", help="ablation: no harmonic-comb input channel")
    t.add_argument("--resynth", action="append", default=[], metavar="DIR", help="resynthesised training set (repeatable)")
    t.add_argument("--resynth-frac", type=float, default=0.3, help="fraction of training items from --resynth sets")
    t.add_argument("--resynth-val", action="append", default=[], metavar="DIR", help="held-out resynthesised set to score")
    t.add_argument("--train-set", action="append", default=[], metavar="KEY=VALUE", help="other TrainConfig fields")
    t.add_argument("--model-set", action="append", default=[], metavar="KEY=VALUE",
                   help="TeacherConfig fields for a new run, e.g. channels=16,32,48 gru_hidden=128")
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("resynth", parents=[common], help="analysis-resynthesis training set from a training pool")
    r.add_argument("corpus")
    r.add_argument("--teacher", required=True, metavar="CKPT", help="tracker for f0, voicing and formant tracks")
    r.add_argument("--split", default="train", choices=["train", "val"])
    r.add_argument("--out", default=None, help="default: <data.root>/resynth/<corpus>_<split>")
    r.add_argument("--variants", type=int, default=2, help="perturbations per recording")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--device", default=None, help="teacher device (default: auto)")
    r.add_argument("--limit", type=int, default=None, help="N recordings spread through the manifest (trial runs)")
    r.add_argument("--resynth-set", action="append", default=[], metavar="KEY=VALUE", help="other ResynthConfig fields")
    r.set_defaults(fn=cmd_resynth)
    e = sub.add_parser("eval", parents=[common], help="evaluate a trained teacher against the gold sets and the suite")
    e.add_argument("ckpt")
    e.add_argument("--out", default=None, help="default: reports/suite/<run dir>_<tiers>.md")
    e.add_argument("--tiers", default="dev", help=TIERS_HELP)
    e.add_argument("--suite", default="data/synth/suite_v1")
    e.set_defaults(fn=cmd_eval)
    a = sub.add_parser("analyse", parents=[common], help="analyse audio files: per-frame tracks, summary, optional plot")
    a.add_argument("audio", nargs="+", help="audio files (any rate soundfile reads; resampled to 16 kHz)")
    a.add_argument("--model", default=None, help="checkpoint (default: teacher v3.1)")
    a.add_argument("--window", default="crop", help="crop (training crop, 400 frames; default), whole, or a window in frames")
    a.add_argument("--out", default="va_analyse", help="output directory (default: va_analyse)")
    a.add_argument("--format", default="csv", help="comma list of csv, json, npz (a summary .json is always written)")
    a.add_argument("--plot", action="store_true", help="spectrogram with F1-F4 (+-sigma), f0 and tract length panels (.png)")
    a.add_argument("--mask", action="store_true", help="blank f0 where unvoiced and formants / lengths where non-sonorant")
    a.add_argument("--threshold", type=float, default=0.5, help="voicing / sonorant probability threshold (default 0.5)")
    a.add_argument("--channel", type=int, default=None, help="channel to analyse (default: the mean of all)")
    a.add_argument("--start", type=float, default=None, help="start of the excerpt, seconds")
    a.add_argument("--end", type=float, default=None, help="end of the excerpt, seconds")
    a.add_argument("--device", default="auto", help="auto (default), cpu or cuda")
    a.add_argument("--fmax", type=float, default=5000.0, help="plot's upper frequency, Hz (default 5000)")
    a.add_argument("--no-normalise", action="store_true",
                   help="analyse at the recorded level (default: peak-normalise to -12 dBFS, the middle of training's range)")
    a.add_argument("--spectrogram", default="wide", choices=("wide", "narrow"),
                   help="plot's spectrogram: wide (6 ms, formants; default) or narrow (30 ms, harmonics)")
    a.set_defaults(fn=cmd_analyse)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.config or ("configs/default.yaml" if Path("configs/default.yaml").exists() else None)
    cfg = load_config(path, args.overrides)
    return args.fn(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
