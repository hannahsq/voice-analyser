"""Config system: typed dataclasses loaded from YAML, with dotted CLI overrides.

    cfg = load_config("configs/default.yaml", overrides=["data.root=D:/corpora"])

Unknown keys are errors, so typos in YAML or overrides fail loudly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    root: str = "data"  # parent directory holding one subdirectory per corpus
    manifests: str = "data/manifests"


@dataclass
class DeviceConfig:
    device: str = "auto"  # auto | cuda | cpu
    # MIOpen is unreliable on the Windows ROCm build (D-002): runtime-compiled
    # kernels (RNN, batch norm) fail and some conv shapes crash. "all" disables
    # it everywhere (native kernels; bf16 is fastest); "rnn" only around RNN
    # calls; "none" leaves it on.
    disable_miopen: str = "all"


@dataclass
class FrameConfig:
    sample_rate: int = 16000
    hop_s: float = 0.010


@dataclass
class EvalConfig:
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    f0_bands_hz: list[float] = field(default_factory=lambda: [150.0, 300.0, 500.0])
    formant_tolerances: list[list[float]] = field(
        default_factory=lambda: [[0.20, 200.0], [0.25, 250.0], [0.30, 300.0]]
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    frames: FrameConfig = field(default_factory=FrameConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build(cls: type, values: dict[str, Any], path: str = "") -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown config keys at '{path or '<root>'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value or {}, f"{path}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    key, sep, value = override.partition("=")
    if not sep:
        raise ValueError(f"override must look like key.sub=value, got {override!r}")
    *parents, leaf = key.strip().split(".")
    node = raw
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for o in overrides or []:
        _apply_override(raw, o)
    return _build(Config, raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
