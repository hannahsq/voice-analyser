"""Lossy codecs through ffmpeg for channel augmentation (plan §5.1).

Each call encodes and decodes in memory. Opus, AAC and MP3 go through ffmpeg's libraries in-process (PyAV):
about 20 ms per clip, where starting two ffmpeg processes costs about 330 ms and serialises across data-loader
workers on Windows (training was loader-bound at about 60 s of audio per second). PyAV's wheels have no AMR
encoders, so AMR runs through the ffmpeg executable (raw float32 over pipes), encoding and decoding in one process
with ffmpeg's loopback decoder (`-dec`). Each loader process keeps one long-lived ffmpeg per AMR bitrate and streams
clips through it, separated by silence (`AmrStream`): a few ms per clip, against about 270 ms per process launch,
and launches serialise on Windows. A clip's output then depends on the codec state the previous clip left and on
(clips start on a frame boundary, as in a one-shot call); the first clip of a stream is bit-identical to a one-shot
call, and a stream's delay equals the one-shot delay. $VA_CODECS_STREAM=0 launches a process per clip instead.
AAC and MP3 are bit-identical either way; Opus voip mode differs between the two builds' libopus versions (same
codec and settings). Codecs add
delay (encoder priming, filter banks), so each (codec, bitrate) is calibrated once per process by
cross-correlating a fixed probe signal, and the decoded audio is shifted back by that many samples and
trimmed to the input length. Frame labels therefore stay aligned (tests/test_codecs.py checks to 1 sample).
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import threading
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
MAX_LAG = 4000  # samples searched when calibrating (MAX_LEAD the other way: output ahead of the input)
MAX_LEAD = 400


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """$VA_FFMPEG, then PATH, then winget's links folder (PATH changes only reach new shells). Symlinks are resolved:
    on Windows, launching through winget's link costs about 120 ms more per process."""
    cands = [os.environ.get("VA_FFMPEG"), shutil.which("ffmpeg")]
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cands.append(str(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
    found = next((c for c in cands if c and Path(c).exists()), None)
    return str(Path(found).resolve()) if found else None


def available() -> bool:
    return ffmpeg_path() is not None


def _run(args: list[str], data: bytes) -> bytes:
    exe = ffmpeg_path()
    if exe is None:
        raise FileNotFoundError("ffmpeg not found: install it (winget install ffmpeg) or set VA_FFMPEG")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for attempt in range(3):  # a launch occasionally hangs on Windows under many concurrent launches: retry
        try:
            res = subprocess.run([exe, "-hide_banner", "-loglevel", "error", *args], input=data, capture_output=True,
                                 creationflags=flags, check=False, timeout=60)
            break
        except subprocess.TimeoutExpired:
            if attempt == 2:
                raise
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


STREAMED = ("amr_wb", "amr_nb")  # codecs PyAV can't run: through long-lived ffmpeg streams
FRAME = 320  # AMR's 20 ms frame at FS (NB and WB): streamed clips start on a frame boundary, as one-shot clips do


def _loopback_args(codec: str, kbps: float) -> list[str]:
    """ffmpeg arguments: float32 in, encode, decode the encoded stream again in the same process, float32 out."""
    enc, _, _, rate = CODECS[codec]
    raw = ["-f", "f32le", "-ar", str(FS), "-ac", "1"]
    return [*raw, "-i", "pipe:0", "-map", "0:a", "-ar", str(rate), *enc, "-b:a", f"{int(round(kbps * 1000))}", "-f", "null", "-",
            "-dec", "0:0", "-filter_complex", f"[dec:0]aresample={FS},asetpts=N/SR/TB[o]", "-map", "[o]",
            "-flush_packets", "1", *raw, "pipe:1"]


class AmrStream:
    """One long-lived ffmpeg (`_loopback_args`) per codec and bitrate. Each clip is written followed by `gap` samples of
    silence, padded so every clip starts on a frame boundary (as in a one-shot call); output is read on a thread (so
    neither pipe fills) and cut out at the clip's position in the stream, the same layout as a one-shot call."""

    def __init__(self, codec: str, kbps: float, gap: int = 15 * FRAME):
        exe = ffmpeg_path()
        if exe is None:
            raise FileNotFoundError("ffmpeg not found: install it (winget install ffmpeg) or set VA_FFMPEG")
        self.proc = subprocess.Popen([exe, "-hide_banner", "-loglevel", "quiet", *_loopback_args(codec, kbps)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.gap, self.buf, self.base, self.pos = gap, bytearray(), 0, 0  # base: stream sample at buf[0]; pos: written
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while chunk := self.proc.stdout.read1(1 << 16):
            with self.cv:
                self.buf += chunk
                self.cv.notify_all()
        with self.cv:
            self.cv.notify_all()

    def _write(self, x: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(x, np.float32).tobytes())
        self.proc.stdin.flush()
        self.pos += len(x)

    def roundtrip(self, x: np.ndarray, timeout: float = 20.0) -> np.ndarray:
        start = self.pos  # a frame boundary: each clip's gap is padded to whole frames
        gap = self.gap + (-(len(x) + self.gap)) % FRAME
        self._write(np.concatenate([np.asarray(x, np.float32), np.zeros(gap, np.float32)]))
        lo, hi, ready = start - self.base, start + len(x) - self.base, start + len(x) + self.gap // 2 - self.base
        with self.cv:
            if not self.cv.wait_for(lambda: len(self.buf) >= 4 * ready or self.proc.poll() is not None, timeout)                     or len(self.buf) < 4 * ready:
                raise TimeoutError(f"AMR stream: {len(self.buf) // 4} of {ready} samples after {timeout:.0f} s")
            y = np.frombuffer(bytes(self.buf[4 * lo : 4 * hi]), np.float32).copy()
            del self.buf[: 4 * hi]
            self.base += hi
        return y

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(5)
        except Exception:
            self.proc.kill()


_streams: dict[tuple[str, float], AmrStream] = {}


def _roundtrip_stream(x: np.ndarray, codec: str, kbps: float) -> np.ndarray:
    """Through this process's stream for (codec, kbps); a failed stream is dropped and the clip goes one-shot."""
    key = (codec, float(kbps))
    try:
        if key not in _streams:
            _streams[key] = AmrStream(codec, kbps)
        return _streams[key].roundtrip(x)
    except (OSError, TimeoutError, ValueError):
        st = _streams.pop(key, None)
        if st is not None:
            st.close()
        return _roundtrip_executable(x, codec, kbps)


def _roundtrip_executable(x: np.ndarray, codec: str, kbps: float) -> np.ndarray:
    """AMR: one process per call (loopback). Its decoder drops the codec's priming samples, so AMR-NB's output leads its
    input slightly (a negative delay); otherwise it matches encoding and decoding in two processes through the AMR
    container, which the other codecs still use (only without PyAV)."""
    if codec in STREAMED:
        return np.frombuffer(_run(_loopback_args(codec, kbps), np.ascontiguousarray(x, np.float32).tobytes()), np.float32)
    enc, fmt, _, rate = CODECS[codec]
    raw = ["-f", "f32le", "-ar", str(FS), "-ac", "1"]
    coded = _run([*raw, "-i", "pipe:0", "-ar", str(rate), *enc, "-b:a", f"{int(round(kbps * 1000))}", "-f", fmt, "pipe:1"],
                 np.ascontiguousarray(x, np.float32).tobytes())
    return np.frombuffer(_run(["-f", DEMUXER.get(fmt, fmt), "-i", "pipe:0", *raw, "pipe:1"], coded), np.float32)


def _roundtrip(x: np.ndarray, codec: str, kbps: float, in_process: bool | None = None, stream: bool | None = None) -> np.ndarray:
    """in_process: None = PyAV where it has the codec (unless $VA_CODECS_IN_PROCESS=0), else the ffmpeg executable,
    through a long-lived stream (stream None: AMR unless $VA_CODECS_STREAM=0) or a process per call."""
    if in_process is None:
        in_process = codec in IN_PROCESS and _av() is not None and os.environ.get("VA_CODECS_IN_PROCESS", "1") != "0"
    if in_process:
        return _roundtrip_in_process(x, codec, kbps)
    if stream is None:
        stream = codec in STREAMED and os.environ.get("VA_CODECS_STREAM", "1") != "0"
    return _roundtrip_stream(x, codec, kbps) if stream else _roundtrip_executable(x, codec, kbps)


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
    """Samples by which the decoded signal lags the input (constant per codec and bitrate; negative if it leads)."""
    x = _probe().astype(float)
    y = _roundtrip(x, codec, kbps, stream=False).astype(float)  # a stream's first clip is bit-identical
    m = min(len(y), len(x) + MAX_LAG)
    y = np.pad(y[:m], (0, len(x) + MAX_LAG - m))
    nfft = 1 << int(np.ceil(np.log2(len(y) + len(x))))
    xc = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(x, nfft)), nfft)
    lags = np.r_[np.arange(-MAX_LEAD, 0), np.arange(MAX_LAG + 1)]
    return int(lags[np.argmax(np.r_[xc[-MAX_LEAD:], xc[: MAX_LAG + 1]])])


def apply(x: np.ndarray, codec: str, kbps: float, stream: bool | None = None) -> np.ndarray:
    """Encode and decode x (FS, peak <= 1) and return it time-aligned to x, same length. stream: see _roundtrip
    (False: the output depends only on x, not on what this process coded before)."""
    d = delay(codec, kbps)
    y = _roundtrip(np.pad(np.asarray(x, np.float32), (0, 2048)), codec, kbps, stream=stream)  # padding: codecs may drop a tail frame
    if d < 0:  # output leads the input: its first -d samples are lost (AMR-NB loopback: 1.25 ms)
        y, d = np.pad(y, (-d, 0)), 0
    y = y[d : d + len(x)]
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)
