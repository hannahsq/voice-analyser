"""Observability labels at training time (D-027): a compact store of each synthetic clip's clean spectra, and the
drawn channel applied to them analytically, so the loader can compute `observability.observability` for the audio
the model actually hears.

Store, per label frame (the STFT of `observability.component_snr`: 25 ms Hann, 1024-point FFT):
  sig   (88,) uint8: signal power on SIG_GRID (50 Hz x 2^(k/12)). Voiced frames: the harmonic peaks (the voiced
        component's strongest bin within one of k f0), interpolated in log frequency, so reading back at k f0 is
        exact up to interpolation. Unvoiced frames: the tract-filtered noise (aspiration, whisper).
  noise (45,) uint8: noise power per bin on NOISE_GRID (50 Hz x 2^(k/6)), the mean over +-1/12 octave. Voiced
        frames: aspiration, whisper, frication; unvoiced: frication and any voicing. No analysis floor (added after
        the channel).
  peak  float16: the frame's strongest bin (signal plus noise) in dB against REF, the clip's mean over voiced frames
        of the frame power (sum over bins). sig and noise are dB below that peak in Q_STEP steps (Q_MISSING: none).
Units throughout are STFT bin powers (|X|^2 of the windowed frame), the units `component_snr` compares.

The channel (`channel.draw_channel`'s parameters, in augment's order): the RIR's late part as noise at -DRR with the
clip's mean spectrum; high-pass, low-pass and EQ on signal and synthesis noise alike; added noise at its SNR (against
the voiced-sample power after those filters, as `apply_channel` scales it) with its kind's spectrum; the phone band
and mu-law noise; clipping as flat noise at the power it removed; a codec as its measured band gain and coding noise
(`codec_table`). Then the analysis floor, 60 dB under the frame's peak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfreqz

from .channel import _biquad_peaking

FS = 16000.0
WIN_S, N_FFT, SMOOTH_BINS, FLOOR_DB = 0.025, 1024, 3, 60.0
# Analysis floor (D-027, 2026-10-18): "recording" = white noise REC_SNR_DB under the clip's mean voiced frame power
# (constant over the clip, a recording's noise); "frame" = FLOOR_DB under each frame's strongest bin (the first label).
REC_SNR_DB = 45.0
SIG_GRID = 50.0 * 2.0 ** (np.arange(88) / 12.0)
NOISE_GRID = 50.0 * 2.0 ** (np.arange(45) / 6.0)
Q_STEP, Q_MAX, Q_MISSING = 0.5, 220, 255
BIN_HZ = FS / N_FFT
CODEC_TABLE = Path(__file__).with_name("codec_table.json")


def _window():
    w = int(round(WIN_S * FS))
    return np.hanning(w)


def enbw_bins() -> float:
    """Equivalent noise bandwidth of the window in bins: a harmonic's power summed over bins, against its peak bin."""
    w = _window()
    return float(N_FFT * np.sum(w ** 2) / np.sum(w) ** 2)


def quantise(db: np.ndarray) -> np.ndarray:
    q = np.clip(np.round(-np.asarray(db, float) / Q_STEP), 0, Q_MAX)
    return np.where(np.isfinite(db), q, Q_MISSING).astype(np.uint8)


def dequantise(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return np.where(q == Q_MISSING, -np.inf, -q.astype(float) * Q_STEP)


def _powers(comps: dict, frame_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(comps["total"])
    win = _window()
    k = np.arange(-(len(win) // 2), len(win) - len(win) // 2)
    idx = np.clip(np.clip(np.round(frame_t * FS).astype(int), 0, n - 1)[:, None] + k[None, :], 0, n - 1)
    zero = np.zeros(n)

    def power(x):
        return np.abs(np.fft.rfft(x[idx] * win, N_FFT, axis=1)) ** 2

    return (power(comps.get("voiced", zero)), power(comps.get("aspiration", zero) + comps.get("whisper", zero)),
            power(comps.get("frication", zero)))


def _band_mean(p: np.ndarray, grid: np.ndarray, half_oct: float = 1.0 / 12.0) -> np.ndarray:
    """(T, bins) -> (T, len(grid)): mean bin power within grid x 2^(+-half_oct), at least the nearest bin."""
    c = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
    lo = np.clip(np.floor(grid * 2 ** -half_oct / BIN_HZ).astype(int), 0, p.shape[1] - 1)
    hi = np.clip(np.ceil(grid * 2 ** half_oct / BIN_HZ).astype(int), lo + 1, p.shape[1])
    return (c[:, hi] - c[:, lo]) / (hi - lo)


def clean_spectra(comps: dict, frame_t: np.ndarray, voiced: np.ndarray, f0: np.ndarray) -> dict:
    """A clip's store entries from its rendered components (`voice.synthesize` / `interactive.synthesize`)."""
    V, TN, Fr = _powers(comps, frame_t)
    kern = np.ones(2 * SMOOTH_BINS + 1) / (2 * SMOOTH_BINS + 1)
    smooth = lambda p: np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, p)  # noqa: E731
    sig_bins = np.where(voiced[:, None], V, TN)
    noise_bins = smooth(np.where(voiced[:, None], TN + Fr, Fr + V))
    peak = np.maximum(np.max(sig_bins + noise_bins, axis=1), 1e-30)
    T = len(frame_t)
    sig = np.empty((T, len(SIG_GRID)))
    vmax = np.maximum(np.maximum(V, np.roll(V, 1, axis=1)), np.roll(V, -1, axis=1))
    unv = 10 * np.log10(np.maximum(_band_mean(smooth(TN), SIG_GRID, 1.0 / 24.0), 1e-30))
    lg = np.log2(SIG_GRID)
    for t in range(T):
        if voiced[t] and np.isfinite(f0[t]) and f0[t] > 0:
            h = f0[t] * np.arange(1, int(7999.0 // f0[t]) + 1)
            vals = 10 * np.log10(np.maximum(vmax[t, np.clip(np.round(h / BIN_HZ).astype(int), 0, V.shape[1] - 1)], 1e-30))
            sig[t] = np.interp(lg, np.log2(h), vals)
        else:
            sig[t] = unv[t]
    noise = 10 * np.log10(np.maximum(_band_mean(noise_bins, NOISE_GRID), 1e-30))
    frame_power = np.sum(sig_bins + noise_bins, axis=1)
    ref = np.mean(frame_power[voiced]) if voiced.any() else np.mean(frame_power)
    pk = 10 * np.log10(peak)
    return {"sig": quantise(sig - pk[:, None]), "noise": quantise(noise - pk[:, None]),
            "peak": (pk - 10 * np.log10(max(ref, 1e-30))).astype(np.float16)}


# ----------------------------------------------------------------------------- channel on the spectra


def _regrid(p: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(T, len(src)) powers -> (T, len(dst)), linear in log power over log frequency (flat beyond the ends)."""
    x = np.clip(np.log2(dst), np.log2(src[0]), np.log2(src[-1]))
    j = np.clip(np.searchsorted(np.log2(src), x) - 1, 0, len(src) - 2)
    w = (x - np.log2(src[j])) / (np.log2(src[j + 1]) - np.log2(src[j]))
    lp = np.log10(np.maximum(p, 1e-300))
    return 10 ** ((1 - w) * lp[..., j] + w * lp[..., j + 1])


def _h2(sos: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.abs(sosfreqz(sos, worN=np.asarray(f, float), fs=FS)[1]) ** 2


def _noise_shape(p: dict, f: np.ndarray) -> np.ndarray:
    """Relative power per bin of the drawn noise at f (any scale; normalised by the caller)."""
    kind = p["noise"]
    if kind == "white":
        return np.ones_like(f)
    if kind == "pink":
        return 1.0 / np.maximum(f, 20.0)
    if kind == "brown":
        return 1.0 / np.maximum(f, 20.0) ** 2
    if kind == "hum":  # lines at k x mains, amplitude ~ U(0.1, 1) / k (mean 0.55 / k), over 0.05 x white
        out = np.full_like(f, 0.05 ** 2)
        lines = p["mains_hz"] * np.arange(1, 8)
        line_bins = enbw_bins()  # a line's power spread like a harmonic's
        for k, fl in enumerate(lines, 1):
            out = out + np.where(np.abs(f - fl) <= BIN_HZ, (0.55 / k) ** 2 * N_FFT / 2 / line_bins, 0.0)
        return out
    if kind == "babble":
        out = np.zeros_like(f)
        for lo, hi in p["babble_bands"]:
            out = out + _h2(butter(2, [lo, hi], "bandpass", fs=FS, output="sos"), f)
        return out
    raise ValueError(kind)


@lru_cache(maxsize=1)
def codec_table() -> dict:
    return json.loads(CODEC_TABLE.read_text(encoding="utf-8")) if CODEC_TABLE.exists() else {}


def _codec(p: dict, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(power gain, coding noise relative to the signal's power) at f, from the measured table."""
    e = codec_table()[f"{p['codec']}@{p['codec_kbps']:g}"]
    g = np.asarray(e["grid"])
    gain = 10 ** (np.interp(np.log2(f), np.log2(g), e["gain_db"]) / 10)
    nsr = 10 ** (np.interp(np.log2(f), np.log2(g), e["nsr_db"]) / 10)
    return gain, nsr


class ClipSpectra:
    """A clip's stored spectra in linear bin power (relative to REF), with the channel applied by `channel`."""

    def __init__(self, sig_q: np.ndarray, noise_q: np.ndarray, peak_db: np.ndarray, voiced: np.ndarray, f0: np.ndarray,
                 floor: str = "recording", rec_snr_db: float = REC_SNR_DB):
        pk = np.asarray(peak_db, float)[:, None]
        self.floor_mode, self.rec_snr_db = floor, rec_snr_db
        self.sig = 10 ** ((dequantise(sig_q) + pk) / 10)  # (T, 88), at harmonic peaks on voiced frames
        self.noise = 10 ** ((dequantise(noise_q) + pk) / 10)  # (T, 45) per bin
        self.voiced, self.f0 = np.asarray(voiced, bool), np.asarray(f0, float)
        self.sig_noise_add = np.zeros_like(self.sig)  # channel noise sampled on the signal grid, per bin
        self.noise_on_sig = None

    def _floor(self, ts: np.ndarray) -> np.ndarray:
        """Per-bin analysis floor for frames ts (linear, the store's units: relative to the clip's mean voiced frame
        power summed over bins)."""
        if self.floor_mode == "recording":
            return np.full(len(ts), 10 ** (-self.rec_snr_db / 10) / (N_FFT // 2 + 1))
        on = self.noise_on_sig[ts] if self.noise_on_sig is not None else 0.0
        return np.maximum(np.max(self.sig[ts] + on, axis=1), 1e-300) * 10 ** (-FLOOR_DB / 10)

    def _sig_mean_per_bin(self) -> np.ndarray:
        """Mean signal power per bin on SIG_GRID: a voiced frame's harmonic peaks spread over the harmonic spacing."""
        spread = np.where(self.voiced & (self.f0 > 0), enbw_bins() * BIN_HZ / np.where(self.f0 > 0, self.f0, 1.0), 1.0)
        return self.sig * np.minimum(spread, 1.0)[:, None]

    def channel(self, p: dict, mask_frames: np.ndarray | None = None) -> "ClipSpectra":
        """Apply a drawn channel (in place; returns self). mask_frames: frames whose power sets the added noise's
        level (the voiced frames, as augment's speech mask)."""
        noise_on_sig = _regrid(self.noise, NOISE_GRID, SIG_GRID)
        self.noise_on_sig = noise_on_sig
        if p.get("clean"):
            return self
        fs_, fn_ = SIG_GRID, NOISE_GRID
        m = self.voiced if mask_frames is None else mask_frames
        m = m if m.any() else np.ones_like(m)
        add_s, add_n = np.zeros_like(self.sig), np.zeros_like(self.noise)  # channel noise per bin
        bins_s, bins_n = np.gradient(fs_) / BIN_HZ, np.gradient(fn_) / BIN_HZ  # bins each grid point stands for

        def frame_power(sig_mean, noise):  # (T,) sum over bins: signal from SIG_GRID, noise from NOISE_GRID
            return sig_mean @ bins_s + noise @ bins_n

        if "drr_db" in p:  # late reverberation: the clip's mean spectrum, -DRR
            r = 10 ** (-p["drr_db"] / 10)
            mean_s = np.mean(self._sig_mean_per_bin() + noise_on_sig, axis=0)
            add_s = add_s + r * mean_s[None, :]
            add_n = add_n + r * _regrid(mean_s[None, :], fs_, fn_)
        hs, hn = np.ones_like(fs_), np.ones_like(fn_)
        if "highpass_hz" in p:
            sos = butter(p["highpass_order"], p["highpass_hz"], "highpass", fs=FS, output="sos")
            hs, hn = hs * _h2(sos, fs_), hn * _h2(sos, fn_)
        if "lowpass_hz" in p:
            sos = butter(p["lowpass_order"], p["lowpass_hz"], "lowpass", fs=FS, output="sos")
            hs, hn = hs * _h2(sos, fs_), hn * _h2(sos, fn_)
        for f, g, q in p.get("eq", []):
            sos = _biquad_peaking(f, g, q)
            hs, hn = hs * _h2(sos, fs_), hn * _h2(sos, fn_)
        self.sig, self.noise, noise_on_sig = self.sig * hs, self.noise * hn, noise_on_sig * hs
        add_s, add_n = add_s * hs, add_n * hn
        if "noise" in p:  # time power 10^(-snr/10) of the speech's over the mask, after the filters (apply_channel)
            ps = float(np.mean(frame_power(self._sig_mean_per_bin(), self.noise + add_n)[m]))
            sh_s, sh_n = _noise_shape(p, fs_), _noise_shape(p, fn_)
            sh_full = _noise_shape(p, np.maximum(np.arange(N_FFT // 2 + 1) * BIN_HZ, 1.0))
            per_bin = ps * 10 ** (-p["snr_db"] / 10) / np.sum(sh_full)  # its power spread over the bins by shape
            add_s, add_n = add_s + per_bin * sh_s[None, :], add_n + per_bin * sh_n[None, :]
        if p.get("phone"):  # measured like a codec: band gain and mu-law noise relative to the input (codec_table)
            add_s, add_n, noise_on_sig = self._coded({"codec": "phone", "codec_kbps": 0.0}, add_s, add_n, noise_on_sig)
        if "clip_loss" in p:  # the power clipping removed (apply_channel measures it), as flat noise
            tot = float(np.mean(frame_power(self._sig_mean_per_bin(), self.noise + add_n)))
            c = p["clip_loss"] * tot / (N_FFT // 2 + 1)
            add_s, add_n = add_s + c, add_n + c
        if "codec" in p:
            add_s, add_n, noise_on_sig = self._coded(p, add_s, add_n, noise_on_sig)
        self.noise = self.noise + add_n
        self.noise_on_sig = noise_on_sig + add_s
        return self

    def _coded(self, p, add_s, add_n, noise_on_sig):
        """A measured stage (codec or phone): band gain on everything, plus coding noise relative to its input."""
        fs_, fn_ = SIG_GRID, NOISE_GRID
        gs, ns = _codec(p, fs_)
        gn, nn = _codec(p, fn_)
        lvl_s = self._sig_mean_per_bin() + noise_on_sig + add_s  # the stage's input per bin
        lvl_n = _regrid(lvl_s, fs_, fn_)
        self.sig, self.noise, noise_on_sig = self.sig * gs, self.noise * gn, noise_on_sig * gs
        return add_s * gs + ns * lvl_s, add_n * gn + nn * lvl_n, noise_on_sig

    def snr_db_at(self, t: int, freqs: np.ndarray) -> np.ndarray:
        """`observability`'s callback: signal over noise (plus the analysis floor) at freqs on frame t."""
        lf = np.log2(np.maximum(freqs, 1.0))
        s = 10 ** np.interp(lf, np.log2(SIG_GRID), np.log10(np.maximum(self.sig[t], 1e-300)))
        nz = self.noise_on_sig if self.noise_on_sig is not None else self.noise
        grid = SIG_GRID if self.noise_on_sig is not None else NOISE_GRID
        n = 10 ** np.interp(lf, np.log2(grid), np.log10(np.maximum(nz[t], 1e-300)))
        return 10 * np.log10(s / (n + self._floor(np.array([t]))[0]))


    def snr_db_rows(self, ts: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """`snr_db_at` for many frames at once: ts (P,), freqs (P, S) (NaN entries give NaN). The grids are regular in
        log2 frequency, so the interpolation is index arithmetic (np.interp's clamping at the ends included)."""
        lf = np.log2(np.maximum(freqs, 1.0))

        def interp(table, step):
            pos = np.clip((lf - np.log2(50.0)) / step, 0.0, table.shape[1] - 1.0)
            pos = np.where(np.isfinite(pos), pos, 0.0)
            i = np.minimum(pos.astype(int), table.shape[1] - 2)
            fr = pos - i
            rows = table[ts]
            lo, hi = np.take_along_axis(rows, i, 1), np.take_along_axis(rows, i + 1, 1)
            return 10 ** (lo + fr * (hi - lo))

        s = interp(np.log10(np.maximum(self.sig, 1e-300)), 1.0 / 12.0)
        on_sig = self.noise_on_sig is not None
        n = interp(np.log10(np.maximum(self.noise_on_sig if on_sig else self.noise, 1e-300)), 1.0 / 12.0 if on_sig else 1.0 / 6.0)
        out = 10 * np.log10(s / (n + self._floor(ts)[:, None]))
        return np.where(np.isfinite(freqs), out, np.nan)

# ----------------------------------------------------------------------------- the store for a written set


def config_from_index(index: dict):
    """The SynthConfig a set was written with (index.json; JSON lists back to tuples; fields added since keep their
    defaults, which reproduce earlier sets)."""
    from dataclasses import fields

    from .sampler import SynthConfig

    known = {f.name for f in fields(SynthConfig)}
    tup = lambda v: tuple(tup(x) for x in v) if isinstance(v, list) else v  # noqa: E731
    return SynthConfig(**{k: tup(v) for k, v in index["synth_config"].items() if k in known})


def _store_shard(args) -> str:
    path, indices, cfg, seed, expect = args
    from .sampler import generate
    from .shards import INT16_SCALE

    parts = {"sig": [], "noise": [], "peak": []}
    for j, i in enumerate(indices):
        c = generate(i, cfg, seed, keep_components=True)
        if j == 0 and expect is not None:  # the regenerated clip must be the stored one
            q = np.clip(np.round(c.audio * INT16_SCALE), -32768, 32767).astype(np.int16)
            if not np.array_equal(q, expect):
                raise RuntimeError(f"clip {i} does not reproduce the stored audio")
        s = clean_spectra(c.components, c.labels["t"], c.labels["voiced"], c.labels["f0"])
        for k in parts:
            parts[k].append(s[k])
    tmp = Path(path).with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.concatenate(v) for k, v in parts.items()})
    tmp.replace(path)
    return path


def write_store(set_dir: str | Path, workers: int = 8, log=print) -> Path:
    """Observability store for a written synthetic set, beside its flat arrays: flat/obs_sig.npy (frames, 88),
    obs_noise.npy (frames, 45), obs_peak.npy (frames,), in the flat arrays' frame order. Clips are regenerated
    (`generate(i, cfg, seed, keep_components=True)`); each shard's first clip is checked against the stored audio.
    Resumable (per-shard files under obs_parts/)."""
    from concurrent.futures import ProcessPoolExecutor

    from .shards import ShardReader

    root = Path(set_dir)
    reader = ShardReader(root, cache_shards=False)
    index = reader.index
    if index.get("kind") == "resynth":
        raise ValueError("resynthesised sets have no separate noise component (D-027): no store")
    cfg, seed, size = config_from_index(index), index["seed"], index["shard_size"]
    parts = root / "obs_parts"
    parts.mkdir(exist_ok=True)
    jobs = []
    flat_audio = root / "flat" / "audio.npy"
    if flat_audio.exists():  # the check reads the consolidated audio (what the loader reads; shards may be gone)
        audio, offs = np.load(flat_audio, mmap_mode="r"), np.load(root / "flat" / "audio_offsets.npy")
    for s, start in enumerate(range(0, index["n_clips"], size)):
        path = parts / f"obs_{s:05d}.npz"
        if not path.exists():
            if flat_audio.exists():
                expect = np.array(audio[offs[start]:offs[start + 1]])
            else:
                with np.load(reader.shards[s]) as z:
                    expect = z["audio"][z["audio_offsets"][0]:z["audio_offsets"][1]]
            jobs.append((str(path), list(range(start, min(start + size, index["n_clips"]))), cfg, seed, expect))
    log(f"{root}: observability store, {len(jobs)} shards to write")
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(_store_shard, jobs), 1):
            if k % 10 == 0 or k == len(jobs):
                log(f"  {k}/{len(jobs)} shards")
    flat = root / "flat"
    n_frames = int(np.load(flat / "frame_offsets.npy")[-1])
    files = sorted(parts.glob("obs_*.npz"))
    out = {"sig": np.lib.format.open_memmap(flat / "obs_sig.npy", "w+", np.uint8, (n_frames, len(SIG_GRID))),
           "noise": np.lib.format.open_memmap(flat / "obs_noise.npy", "w+", np.uint8, (n_frames, len(NOISE_GRID))),
           "peak": np.lib.format.open_memmap(flat / "obs_peak.npy", "w+", np.float16, (n_frames,))}
    pos = 0
    for f in files:
        with np.load(f) as z:
            n = len(z["peak"])
            for k in out:
                out[k][pos:pos + n] = z[k]
            pos += n
    if pos != n_frames:
        raise RuntimeError(f"store has {pos} frames, the set {n_frames}")
    for m in out.values():
        m.flush()
    return flat
