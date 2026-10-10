"""Lossy codecs through ffmpeg for channel augmentation (plan §5.1).

Each call encodes and decodes in memory. Opus, AAC and MP3 go through ffmpeg's libraries in-process (PyAV):
about 20 ms per clip, where starting two ffmpeg processes costs about 330 ms and serialises across data-loader
workers on Windows (training was loader-bound at about 60 s of audio per second). PyAV's wheels have no AMR
encoders, so AMR still runs through the ffmpeg executable (raw float32 over pipes). AAC and MP3 are bit-identical
either way; Opus voip mode differs between the two builds' libopus versions (same codec and settings). Codecs add
delay (encoder priming, filter banks), so each (codec, bitrate) is calibrated once per process by
cross-correlating a fixed probe signal, and the decoded audio is shifted back by that many samples and
trimmed to the input length. Frame labels therefore stay aligned (tests/test_codecs.py checks to 1 sample).
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

FS = 16000

# name: (encoder args, muxer, bitrates in kbps, sample rate the codec runs at)
CODECS: dict[str, tuple[tuple[str, ...], str, tuple[float, ...], int]] = {
    "opus": (("-c:a", "libopus", "-application", "voip"), "ogg", (8.0, 12.0, 16.0, 24.0, 32.0), 16000),
    "aac": (("-c:a", "aac"), "adts", (16.0, 24.0, 32.0, 48.0), 16000),
    "mp3": (("-c:a", "libmp3lame"), "mp3", (16.0, 24.0, 32.0, 48.0, 64.0), 16000),
    "amr_wb": (("-c:a", "libvo_amrwbenc"), "amr", (6.6, 8.85, 12.65, 23.85), 16000),
    "amr_nb": (("-c:a", "libopencore_amrnb"), "amr", (4.75, 7.95, 12.2), 8000),
}
DEMUXER = {"adts": "aac"}  # where the demuxer name differs from the muxer
# in-process (PyAV) encoder name and options, for the codecs its wheels include
IN_PROCESS: dict[str, tuple[str, dict[str, str]]] = {
    "opus": ("libopus", {"application": "voip"}), "aac": ("aac", {}), "mp3": ("libmp3lame", {})}
MAX_LAG = 4000  # samples searched when calibrating


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """$VA_FFMPEG, then PATH, then winget's links folder (PATH changes only reach new shells)."""
    cands = [os.environ.get("VA_FFMPEG"), shutil.which("ffmpeg")]
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cands.append(str(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
    return next((c for c in cands if c and Path(c).exists()), None)


def available() -> bool:
    return ffmpeg_path() is not None


def _run(args: list[str], data: bytes) -> bytes:
    exe = ffmpeg_path()
    if exe is None:
        raise FileNotFoundError("ffmpeg not found: install it (winget install ffmpeg) or set VA_FFMPEG")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run([exe, "-hide_banner", "-loglevel", "error", *args], input=data, capture_output=True,
                         creationflags=flags, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(args)}): {res.stderr.decode(errors='replace').strip()}")
    return res.stdout


@functools.lru_cache(maxsize=1)
def _av():
    try:
        import av
    except ImportError:
        return None
    return av


def _roundtrip_in_process(x: np.ndarray, codec: str, kbps: float) -> np.ndarray:
    import io

    av = _av()
    enc, opts = IN_PROCESS[codec]
    fmt = CODECS[codec][1]
    buf = io.BytesIO()
    with av.open(buf, "w", format=fmt) as out:
        st = out.add_stream(enc, rate=FS, options=opts)
        st.bit_rate, st.layout = int(round(kbps * 1000)), "mono"
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(x, np.float32)[None, :], format="flt", layout="mono")
        frame.sample_rate = FS
        for packet in (*st.encode(frame), *st.encode(None)):
            out.mux(packet)
    buf.seek(0)
    ys = []
    with av.open(buf, "r", format=DEMUXER.get(fmt, fmt)) as inp:
        rs = av.AudioResampler(format="flt", layout="mono", rate=FS)
        for f in inp.decode(audio=0):
            ys.extend(g.to_ndarray().ravel() for g in rs.resample(f))
        ys.extend(g.to_ndarray().ravel() for g in rs.resample(None))
    return np.concatenate(ys) if ys else np.zeros(0, np.float32)


def _roundtrip(x: np.ndarray, codec: str, kbps: float, in_process: bool | None = None) -> np.ndarray:
    """in_process: None = PyAV where it has the codec (unless $VA_CODECS_IN_PROCESS=0), else the ffmpeg executable."""
    if in_process is None:
        in_process = codec in IN_PROCESS and _av() is not None and os.environ.get("VA_CODECS_IN_PROCESS", "1") != "0"
    if in_process:
        return _roundtrip_in_process(x, codec, kbps)
    enc, fmt, _, rate = CODECS[codec]
    raw = ["-f", "f32le", "-ar", str(FS), "-ac", "1"]
    coded = _run([*raw, "-i", "pipe:0", "-ar", str(rate), *enc, "-b:a", f"{int(round(kbps * 1000))}", "-f", fmt, "pipe:1"],
                 np.ascontiguousarray(x, np.float32).tobytes())
    return np.frombuffer(_run(["-f", DEMUXER.get(fmt, fmt), "-i", "pipe:0", *raw, "pipe:1"], coded), np.float32)


def _probe() -> np.ndarray:
    """1 s of speech-like signal: a 120 Hz pulse train through three resonances, gated, plus a little noise."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(0)
    n = FS
    x = np.zeros(n)
    x[:: FS // 120] = 1.0
    for f, b in ((600.0, 80.0), (1400.0, 100.0), (2600.0, 150.0)):
        r = np.exp(-np.pi * b / FS)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / FS), r * r], x)
    x *= (np.sin(np.pi * np.arange(n) / n * 3) ** 2)
    x += 0.01 * rng.standard_normal(n)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


@functools.lru_cache(maxsize=None)
def delay(codec: str, kbps: float) -> int:
    """Samples by which the decoded signal lags the input (constant per codec and bitrate)."""
    x = _probe().astype(float)
    y = _roundtrip(x, codec, kbps).astype(float)
    m = min(len(y), len(x) + MAX_LAG)
    y = np.pad(y[:m], (0, len(x) + MAX_LAG - m))
    nfft = 1 << int(np.ceil(np.log2(len(y) + len(x))))
    xc = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(x, nfft)), nfft)
    return int(np.argmax(xc[: MAX_LAG + 1]))


def apply(x: np.ndarray, codec: str, kbps: float) -> np.ndarray:
    """Encode and decode x (FS, peak <= 1) and return it time-aligned to x, same length."""
    d = delay(codec, kbps)
    y = _roundtrip(np.pad(np.asarray(x, np.float32), (0, 2048)), codec, kbps)  # padding: codecs may drop a tail frame
    y = y[d : d + len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)
