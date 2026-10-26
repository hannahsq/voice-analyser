"""D-034 student v4: isolated-word items from MFA alignments (mfa_prep.py), shaped like the gold sets' isolated words
(Hillenbrand's h-V-d files): one aligned word, faded in and out over 10 ms, with silence on both sides taken from the
same recording's aligned pauses (room tone, not digital zeros), as a short whole file. Its length is made a whole number
of eight frames (L = 48 .. 192 frames of 10 ms, 0.5-1.9 s) by extending the trailing silence, so items of one length
batch together and the student sees exactly the file, as at inference.
Words: 0.2-0.9 s, a dictionary word (not MFA's <unk> / spn), with an aligned pause of at least 0.1 s on both sides, so
both boundaries are set against silence (inside running speech MFA's boundaries can carry part of a neighbouring word:
Hannah heard "as leader" in one). Room tone: only 20 ms frames of the recording's pauses within 10 dB of their 10th
percentile and at least 20 dB below the word (MFA can label noise or speech as a pause), in runs of at least 40 ms; the
item is skipped if under 0.1 s qualifies. On each side with a pause, the 60 ms next to the word must be EDGE_DB below it,
so a "pause" that is really a reduced neighbouring word (Hannah heard /tə/ before "this") rejects the word.
Children's words are scarce (Hannah: relax as far as possible without teaching wrong things; the targets are always
Parakeet's on the same audio, so the limit is items that stop being a word against silence): candidates(sides=1) admits
a pause on one side only, the other side cut at MFA's boundary with a 25 ms fade (a soft onset rather than a chop).
Defaults (`candidates` arguments): LibriTTS-R both sides, 10 dB; speechocean762 one side, 6 dB."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vocal_analysis.features.vqt import HOP

SR = 16000
LMIN, LMAX = 48, 192


def textgrid_tier(path: Path, tier: str = "words"):
    """[(start, end, text)] of one interval tier of a (long-format) TextGrid"""
    out, inside, cur = [], False, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("name ="):
            inside = s.split("=", 1)[1].strip().strip('"') == tier
        elif inside and s.startswith("xmin ="):
            cur = {"xmin": float(s.split("=")[1])}
        elif inside and s.startswith("xmax =") and "xmin" in cur:
            cur["xmax"] = float(s.split("=")[1])
        elif inside and s.startswith("text =") and "xmax" in cur:
            out.append((cur["xmin"], cur["xmax"], s.split("=", 1)[1].strip().strip('"')))
            cur = {}
    return out


def read16k(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def candidates(corpus_dir: Path, aligned_dir: Path, sides: int = 2, edge_db: float = 10.0):
    """[(wav, word start, word end, [(pause start, pause end)], speaker, word, pause left, pause right, edge dB)] over a
    corpus: words with an aligned pause on `sides` (2: both, 1: at least one) sides"""
    out = []
    for tg in sorted(aligned_dir.rglob("*.TextGrid")):
        spk = tg.parent.name
        wav = corpus_dir / spk / f"{tg.stem}.wav"
        iv = textgrid_tier(tg)
        pauses = [(a + 0.02, b - 0.02) for a, b, t in iv if t == "" and b - a >= 0.1]
        if not pauses or not wav.exists():
            continue
        quiet = lambda k: 0 <= k < len(iv) and iv[k][2] == "" and iv[k][1] - iv[k][0] >= 0.1  # noqa: E731
        for k, (a, b, t) in enumerate(iv):
            lq, rq = quiet(k - 1), quiet(k + 1)
            if t and t not in ("<unk>", "spn") and 0.2 <= b - a <= 0.9 and lq + rq >= sides:
                out.append((str(wav), a, b, pauses, spk, t, lq, rq, edge_db))
    return out


def json_candidates(align_dir: Path, wavs: dict, sides: int = 2, edge_db: float = 10.0):
    """as `candidates`, from minimal-pairs' cached MFA alignments (one JSON per utterance: {"alignment": {"utterance_id",
    "words": [[word, start, end, phones], ...]}}), for the utterances in `wavs` ({utterance id: (audio path, speaker)});
    the gaps between words (and before the first and after the last, to the file's end) are the pauses"""
    out = []
    for f in sorted(align_dir.glob("*.json")):
        al = json.loads(f.read_text(encoding="utf-8")).get("alignment") or {}
        if al.get("utterance_id") not in wavs:
            continue
        wav, spk = wavs[al["utterance_id"]]
        dur = sf.info(str(wav)).duration
        iv, t0 = [], 0.0  # words and gaps as one interval tier, gaps with text ""
        for t, a, b, _ in al["words"]:
            if a - t0 > 1e-3:
                iv.append((t0, a, ""))
            iv.append((a, b, t))
            t0 = b
        if dur - t0 > 1e-3:
            iv.append((t0, dur, ""))
        pauses = [(a + 0.02, b - 0.02) for a, b, t in iv if t == "" and b - a >= 0.1]
        if not pauses:
            continue
        quiet = lambda k: 0 <= k < len(iv) and iv[k][2] == "" and iv[k][1] - iv[k][0] >= 0.1  # noqa: E731
        for k, (a, b, t) in enumerate(iv):
            lq, rq = quiet(k - 1), quiet(k + 1)
            if t and t not in ("<unk>", "spn") and 0.2 <= b - a <= 0.9 and lq + rq >= sides:
                out.append((str(wav), a, b, pauses, spk, t, lq, rq, edge_db))
    return out


def quiet_segments(x: np.ndarray, pauses, word_db: float):
    """the pauses' 20 ms frames within 10 dB of their 10th percentile and 20 dB below the word, as runs of >= 40 ms"""
    fr = [(int(a * SR) + k * 320) for a, b in pauses for k in range(int((b - a) * SR) // 320)]
    if not fr:
        return []
    db = np.array([10 * np.log10(np.mean(x[s:s + 320] ** 2) + 1e-12) for s in fr])
    ok = (db <= np.percentile(db, 10) + 10) & (db <= word_db - 20)
    segs, run = [], []
    for s, k in zip(fr, ok):
        if k and run and s == run[-1] + 320:
            run.append(s)
        else:
            if len(run) >= 2:
                segs.append(x[run[0]:run[-1] + 320])
            run = [s] if k else []
    if len(run) >= 2:
        segs.append(x[run[0]:run[-1] + 320])
    return segs


def room_tone(segs, n: int, rng) -> np.ndarray:
    """n samples of the recording's quiet pause segments, joined with 5 ms crossfades from seeded starting points"""
    out, f = np.zeros(0, np.float32), 80
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    while len(out) < n:
        s = segs[rng.integers(len(segs))]
        s = s[rng.integers(0, max(len(s) // 2, 1)):]
        if len(out) >= f and len(s) > f:
            s = s.copy()
            s[:f] = s[:f] * ramp + out[-f:] * (1 - ramp)
            out = out[:-f]
        out = np.concatenate([out, s])
    return out[:n]


def make_item(cand, rng) -> np.ndarray | None:
    """one isolated-word file of L frames (L a multiple of 8 in [LMIN, LMAX]); None if it can't fit"""
    wav, a, b, pauses = cand[:4]
    lq, rq, edge_db = cand[6:9] if len(cand) > 6 else (True, True, 20.0)
    x = read16k(wav)
    w = x[int(a * SR):int(b * SR)].copy()
    word_db = 10 * np.log10(np.mean(w ** 2) + 1e-12)
    fl, fr = (160 if lq else 400), (160 if rq else 400)  # 10 ms fades against silence, 25 ms where cut from speech
    w[:fl] *= np.linspace(0, 1, fl, dtype=np.float32)
    w[-fr:] *= np.linspace(1, 0, fr, dtype=np.float32)
    edge = int(0.06 * SR)
    for has_pause, seg in ((lq, x[max(int(a * SR) - edge, 0):int(a * SR)]), (rq, x[int(b * SR):int(b * SR) + edge])):
        if has_pause and (len(seg) < edge // 2 or 10 * np.log10(np.mean(seg ** 2) + 1e-12) > word_db - edge_db):
            return None  # this boundary is not against silence
    segs = quiet_segments(x, pauses, word_db)
    if sum(len(s) for s in segs) < 0.1 * SR:
        return None
    pre, post = (int(rng.uniform(0.15, 0.5) * SR) for _ in range(2))
    n = pre + len(w) + post
    L = max(LMIN, -(-(n // HOP + 1) // 8) * 8)
    if L > LMAX:
        return None
    post += (L - 1) * HOP + 1 - n  # exactly L frames: (L - 1) * HOP + 1 samples
    if post < 0:
        return None
    tone = room_tone(segs, pre + post, rng)
    return np.concatenate([tone[:pre], w, tone[pre:]]).astype(np.float32)
