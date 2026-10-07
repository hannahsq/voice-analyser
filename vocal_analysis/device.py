"""Torch device selection, including the MIOpen workaround for Windows ROCm (D-002).

Default policy disables MIOpen (cuDNN flag) process-wide; see DeviceConfig.

torch is imported lazily so the rest of the package (metrics, manifests,
loaders) works in environments without it.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from .config import DeviceConfig

_rnn_guard_enabled = False


def setup_device(cfg: DeviceConfig | None = None):
    """Return a torch.device and apply the configured MIOpen policy."""
    import torch

    global _rnn_guard_enabled
    cfg = cfg or DeviceConfig()
    if cfg.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(cfg.device)

    is_rocm = torch.version.hip is not None
    if cfg.disable_miopen == "all" and is_rocm:
        torch.backends.cudnn.enabled = False
    _rnn_guard_enabled = cfg.disable_miopen == "rnn" and is_rocm and dev.type == "cuda"
    return dev


@contextlib.contextmanager
def rnn_guard() -> Iterator[None]:
    """Wrap RNN forward calls in this on ROCm (backward then follows the native path).

    MIOpen's RNN kernels are compiled at runtime and that fails on the Windows
    ROCm build, so RNNs must use PyTorch's native kernels. Convolutions use
    precompiled MIOpen kernels and are unaffected.
    """
    if not _rnn_guard_enabled:
        yield
        return
    import torch

    with torch.backends.cudnn.flags(enabled=False):
        yield
