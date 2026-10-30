"""`va analyse`: run a teacher on audio files and write per-frame tracks, a summary and optionally a plot.

Frames are 10 ms apart (frame i at t = i * 10 ms). Outputs per frame: f0 and its sigma (cents), voicing and sonorant
probabilities, F1-F4 and their sigmas (cents), B1-B4, and where the model has them the anatomical tract length (cm)
and the instantaneous length (larynx and lips included, D-051) with their sigmas. Values are the model's raw outputs on
every frame; `mask` blanks f0 where unvoiced and the formants, bandwidths and lengths where non-sonorant.

Windowing (D-028): "crop" analyses long files in windows of the training crop (400 frames, half-window hop, each
frame taken from the window where it is most central), the default since 2026-10-27; "whole" runs the file at once;
an integer sets the window in frames.
"""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path

import numpy as np

FS = 16000
HOP_S = 0.01
DEFAULT_MODEL = "runs/d040_5b_s2_b20k/ckpt_last.pt"  # teacher v3.1 (docs/models.md, tag teacher-v3.1)
FORMATS = ("csv", "json", "npz")


def parse_window(value: str) -> int | None:
    """"crop" -> the training crop (infer.DEFAULT_WINDOW), "whole" -> None (no windows), "N" -> N frames."""
    from .models.infer import DEFAULT_WINDOW

    if value == "crop":
        return DEFAULT_WINDOW
    if value == "whole":
        return None
    n = int(value)
    if n < 32:
        raise ValueError(f"window of {n} frames is shorter than the shortest training crop (32)")
    return n


def load(path: str | Path, channel: int | None = None, start: float | None = None, end: float | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 at 16 kHz (channels averaged unless one is chosen), cut to [start, end) seconds. -> (audio, source rate)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if channel is not None:
        if not 0 <= channel < x.shape[1]:
            raise ValueError(f"{path} has {x.shape[1]} channel(s); --channel {channel} is out of range")
        x = x[:, channel]
    else:
        x = x.mean(1)
    a = int(round((start or 0.0) * sr))
    b = int(round(end * sr)) if end is not None else len(x)
    x = x[a:b]
    if sr != FS:
        g = gcd(FS, sr)
        x = resample_poly(x, FS // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(x, np.float32), sr


def columns(r: dict, t0: float = 0.0, mask: bool = False, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """The analysis as named per-frame columns (times offset by t0, the excerpt's start)."""
    voiced, son = r["voiced_p"] > threshold, r["sonorant_p"] > threshold
    v = (lambda a: np.where(voiced, a, np.nan)) if mask else (lambda a: a)  # noqa: E731
    s = (lambda a: np.where(son, a, np.nan)) if mask else (lambda a: a)  # noqa: E731
    cols = {"t": r["t"] + t0, "f0": v(r["f0"]), "f0_sigma_cents": v(r["f0_sigma_cents"]),
            "voiced_p": r["voiced_p"], "sonorant_p": r["sonorant_p"]}
    for k in range(4):
        cols[f"F{k + 1}"] = s(r["formants"][:, k])
        cols[f"F{k + 1}_sigma_cents"] = s(r["formant_sigma_cents"][:, k])
    for k in range(4):
        cols[f"B{k + 1}"] = s(r["bandwidths"][:, k])
    if "vtl_log2cm" in r:
        cols["vtl_cm"], cols["vtl_sigma_log2"] = s(2.0 ** r["vtl_log2cm"]), s(r["vtl_sigma"])
    if "vtl_inst_log2cm" in r:
        cols["vtl_inst_cm"], cols["vtl_inst_sigma_log2"] = s(2.0 ** r["vtl_inst_log2cm"]), s(r["vtl_inst_sigma"])
    return cols


def summary(r: dict, threshold: float = 0.5) -> dict:
    """Medians over voiced (f0) and sonorant (formants, bandwidths) frames; tract lengths precision-weighted."""
    voiced, son = r["voiced_p"] > threshold, r["sonorant_p"] > threshold
    out = {"frames": int(len(r["t"])), "duration_s": float(len(r["t"]) * HOP_S), "voiced_frac": float(voiced.mean()),
           "sonorant_frac": float(son.mean())}
    if voiced.any():
        f0 = r["f0"][voiced]
        out["f0_hz"] = {"median": float(np.median(f0)), "p5": float(np.percentile(f0, 5)), "p95": float(np.percentile(f0, 95))}
    if son.any():
        out["formants_hz_median"] = np.nanmedian(r["formants"][son], 0).tolist()
        out["bandwidths_hz_median"] = np.nanmedian(r["bandwidths"][son], 0).tolist()
        for key in ("vtl", "vtl_inst"):
            if f"{key}_log2cm" in r:
                w = 1.0 / r[f"{key}_sigma"][son] ** 2
                mu = float(np.sum(w * r[f"{key}_log2cm"][son]) / w.sum())
                out[f"{key}_cm"] = {"value": 2.0 ** mu, "sigma_log2": float(np.sqrt(1.0 / w.sum()))}
    return out


def write(cols: dict[str, np.ndarray], summ: dict, meta: dict, stem: Path, formats: tuple[str, ...]) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if "csv" in formats:
        p = stem.with_suffix(".csv")
        np.savetxt(p, np.column_stack(list(cols.values())), delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
        written.append(p)
    if "npz" in formats:
        p = stem.with_suffix(".npz")
        np.savez_compressed(p, **cols)
        written.append(p)
    # the summary and the settings always go beside the tracks; the "json" format adds the frames to it
    p = stem.with_suffix(".json")
    doc = {"meta": meta, "summary": summ}
    if "json" in formats:
        doc["frames"] = {k: [float(x) if np.isfinite(x) else None for x in v] for k, v in cols.items()}
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    written.append(p)
    return written


SPECTROGRAMS = {"wide": 96, "narrow": 480}  # window, samples: 6 ms shows formants, 30 ms resolves harmonics


def plot(audio: np.ndarray, r: dict, path: Path, title: str = "", t0: float = 0.0, fmax: float = 5000.0,
         threshold: float = 0.5, spectrogram: str = "wide") -> Path:
    """Spectrogram (wideband by default) with F1-F4 (sonorant frames, +-sigma bands), f0 (voiced frames, +-sigma) and
    the tract lengths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from scipy.signal import stft

    t = r["t"] + t0
    voiced, son = r["voiced_p"] > threshold, r["sonorant_p"] > threshold
    has_vtl = "vtl_log2cm" in r
    rows = 3 if has_vtl else 2
    width = float(np.clip(4 + 2.5 * len(audio) / FS, 8, 40))
    fig, ax = plt.subplots(rows, 1, figsize=(width, 3.2 * rows), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1][:rows]}, constrained_layout=True)
    n = SPECTROGRAMS[spectrogram]
    shown = np.r_[audio[:1], audio[1:] - 0.97 * audio[:-1]]  # pre-emphasis for display only, as phonetics tools do
    f, ts, z = stft(shown, FS, window="hann", nperseg=n, noverlap=n - 32, nfft=max(512, n))  # 2 ms hop
    db = 20 * np.log10(np.abs(z) + 1e-9)
    keep = f <= fmax
    ax[0].pcolormesh(ts + t0, f[keep], db[keep], shading="auto", cmap="Greys", vmin=db.max() - 55, vmax=db.max() - 5)
    colours = ("tab:red", "tab:orange", "tab:green", "tab:blue")
    for k in range(4):
        fk = np.where(son, r["formants"][:, k], np.nan)
        sg = 2.0 ** (np.where(son, r["formant_sigma_cents"][:, k], np.nan) / 1200)
        ax[0].fill_between(t, fk / sg, fk * sg, color=colours[k], alpha=0.25, linewidth=0)
        ax[0].plot(t, fk, color=colours[k], linewidth=1.2, label=f"F{k + 1}")
    ax[0].set_ylim(0, fmax)
    ax[0].set_ylabel("Hz")
    ax[0].legend(loc="upper right", ncols=4, fontsize=8)
    ax[0].set_title(title)
    f0 = np.where(voiced, r["f0"], np.nan)
    s0 = 2.0 ** (np.where(voiced, r["f0_sigma_cents"], np.nan) / 1200)
    ax[1].fill_between(t, f0 / s0, f0 * s0, color="tab:purple", alpha=0.25, linewidth=0)
    ax[1].plot(t, f0, color="tab:purple", linewidth=1.2)
    ax[1].set_yscale("log")
    for axis in (ax[1].yaxis,):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
    ax[1].set_ylabel("f0 (Hz)")
    if has_vtl:
        for key, colour, label in (("vtl", "tab:brown", "anatomical"), ("vtl_inst", "tab:cyan", "instantaneous")):
            if f"{key}_log2cm" not in r:
                continue
            mu = np.where(son, r[f"{key}_log2cm"], np.nan)
            sg = np.where(son, r[f"{key}_sigma"], np.nan)
            ax[2].fill_between(t, 2 ** (mu - sg), 2 ** (mu + sg), color=colour, alpha=0.25, linewidth=0)
            ax[2].plot(t, 2 ** mu, color=colour, linewidth=1.2, label=label)
        ax[2].set_ylabel("tract length (cm)")
        ax[2].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("time (s)")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def analyse_files(paths: list[str], model_path: str | None = DEFAULT_MODEL, window: str = "crop", out_dir: str = "va_analyse",
                  formats: tuple[str, ...] = ("csv",), make_plot: bool = False, mask: bool = False, channel: int | None = None,
                  start: float | None = None, end: float | None = None, device: str = "auto", fmax: float = 5000.0,
                  threshold: float = 0.5, spectrogram: str = "wide", log=print) -> list[dict]:
    import time

    import torch

    from .models.infer import analyse, load_teacher

    bad = [f for f in formats if f not in FORMATS]
    if bad:
        raise ValueError(f"unknown format(s) {bad}; choose from {FORMATS}")
    wf = parse_window(window)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    model_path = model_path or DEFAULT_MODEL
    model, _ = load_teacher(model_path, dev)
    out = Path(out_dir)
    results = []
    for path in paths:
        audio, sr = load(path, channel, start, end)
        tic = time.time()
        r = analyse(model, audio, dev, window_frames=wf)
        dt = time.time() - tic
        t0 = start or 0.0
        cols, summ = columns(r, t0, mask, threshold), summary(r, threshold)
        meta = {"file": str(path), "source_rate_hz": sr, "start_s": t0, "end_s": end, "channel": channel, "model": model_path,
                "window": window, "window_frames": wf, "mask": mask, "threshold": threshold, "frame_hop_s": HOP_S}
        stem = out / Path(path).stem
        written = write(cols, summ, meta, stem, formats)
        if make_plot:
            written.append(plot(audio, r, stem.with_suffix(".png"), Path(path).name, t0, fmax, threshold, spectrogram))
        log(_describe(Path(path).name, summ, sr, dt, dev.type))
        log("  wrote " + ", ".join(str(p) for p in written))
        results.append({"file": str(path), "summary": summ, "written": [str(p) for p in written]})
    return results


def _describe(name: str, s: dict, sr: int, dt: float, dev: str) -> str:
    lines = [f"{name}: {s['duration_s']:.1f} s (source {sr} Hz), {s['frames']} frames in {dt:.2f} s on {dev}; "
             f"voiced {s['voiced_frac']:.0%}, sonorant {s['sonorant_frac']:.0%}"]
    if "f0_hz" in s:
        lines.append(f"  f0 median {s['f0_hz']['median']:.0f} Hz (5-95 % {s['f0_hz']['p5']:.0f}-{s['f0_hz']['p95']:.0f})")
    if "formants_hz_median" in s:
        lines.append("  F1-F4 median " + " / ".join(f"{v:.0f}" for v in s["formants_hz_median"]) + " Hz; B1-B4 "
                     + " / ".join(f"{v:.0f}" for v in s["bandwidths_hz_median"]) + " Hz")
    for key, label in (("vtl_cm", "tract length (anatomical)"), ("vtl_inst_cm", "tract length (instantaneous)")):
        if key in s:
            lines.append(f"  {label} {s[key]['value']:.1f} cm (sigma {s[key]['sigma_log2'] * 1200:.0f} cents)")
    return "\n".join(lines)
