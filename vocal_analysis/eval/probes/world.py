"""WORLD f0-change pairs (D-094; plan section 5.3's T2): one recording analysed once with WORLD (Harvest f0,
CheapTrick envelope, D4C aperiodicity; 16 kHz, 10 ms) and resynthesised with f0 times each factor, the envelope and
aperiodicity unchanged. Variants are compared with each other (the reference is WORLD's own resynthesis at factor 1),
never with the original. Needs `pyworld` (modified BSD).

Item sets (all held out from M4's pairs):
- "zeroth": the counterfactual probe's items (Zeroth-Korean non-gold validation, 10 speakers x 6 clips), original clips;
- "libritts": LibriTTS-R dev-clean (no run has trained on it), 20 women and 20 men, 6 utterances each, at most 6 s;
- "gate": train_v2 validation clips regenerated clean (the set's own config and seed), drawn in validation order to
  match the real items' median-f0 bands (< 150, 150-250, > 250 Hz) in count.
Frames: WORLD's voiced frames (f0 > 0 at analysis), the same for every variant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FACTORS = (0.84, 0.92, 1.0, 1.09, 1.19)
FS, FRAME_MS = 16000, 10.0
BANDS = ((0, 150), (150, 250), (250, 1e9))
LIBRI = Path("C:/github/datasets/libritts_r/LibriTTS_R")
LIBRI_SEED, LIBRI_PER, LIBRI_MAX_S, LIBRI_MIN_S = 5094, 6, 6.0, 3.0
ZEROTH_ROOT = Path("data/zeroth_korean")
OUT = Path("data/probes/world_v1")


def analyse_world(x: np.ndarray) -> dict:
    import pyworld as pw

    x = np.asarray(x, np.float64)
    f0, t = pw.harvest(x, FS, frame_period=FRAME_MS)
    sp = pw.cheaptrick(x, f0, t, FS)
    ap = pw.d4c(x, f0, t, FS)
    return {"f0": f0, "sp": sp, "ap": ap}


def variants(x: np.ndarray, factors=FACTORS) -> tuple[dict, np.ndarray]:
    """{factor: audio} (same length as x, peak 0.9) and WORLD's voiced mask on the 10 ms frame grid"""
    import pyworld as pw

    a = analyse_world(x)
    out = {}
    for r in factors:
        y = pw.synthesize(a["f0"] * r, a["sp"], a["ap"], FS, frame_period=FRAME_MS)[: len(x)]
        y = np.r_[y, np.zeros(max(0, len(x) - len(y)))]
        out[r] = (0.9 * y / (np.abs(y).max() + 1e-12)).astype(np.float32)
    return out, a["f0"] > 0


def band(f0_med: float) -> int:
    return next(k for k, (lo, hi) in enumerate(BANDS) if lo <= f0_med < hi)


def zeroth_sex() -> dict:
    """{speaker id: sex} from AUDIO_INFO's SPEAKERID and SEX columns only (names are never read into memory)"""
    out = {}
    with open(ZEROTH_ROOT / "AUDIO_INFO", encoding="utf-8", errors="replace") as f:
        cols = [c.strip() for c in f.readline().rstrip("\n").split("|")]
        si, xi = cols.index("SPEAKERID"), cols.index("SEX")
        for line in f:
            p = [c.strip() for c in line.rstrip("\n").split("|")]
            if len(p) > max(si, xi):
                out[p[si]] = p[xi].lower()[:1]
    return out


def libritts_items() -> list[tuple[str, str, str, np.ndarray]]:
    """(key, group, sex, audio 16 kHz) for dev-clean: 6 utterances per speaker, a seeded draw among those >= 3 s,
    each cut to its first 6 s. Speaker ids and recorded sex only (speakers.tsv's READER and GENDER columns)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    sex = {}
    for line in (LIBRI / "speakers.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3 and p[2] == "dev-clean":
            sex[p[0]] = "f" if p[1] == "F" else "m"
    rng = np.random.default_rng(LIBRI_SEED)
    items = []
    for spk in sorted(sex, key=int):
        files = sorted((LIBRI / "dev-clean" / spk).glob("*/*.wav"))
        long_ = [f for f in files if sf.info(f).duration >= LIBRI_MIN_S]
        for f in [long_[i] for i in sorted(rng.choice(len(long_), min(LIBRI_PER, len(long_)), replace=False))]:
            x, sr = sf.read(f)
            x = resample_poly(x, FS // 1000, sr // 1000)[: int(LIBRI_MAX_S * FS)]
            items.append((f"libritts:{f.stem}", f"libritts:{spk}", sex[spk], x.astype(np.float32)))
    return items


def gate_items(real_bands: list[int]) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    """(key, group, sex '?', audio, label formants) for clean train_v2 validation clips matched to real_bands' counts"""
    from ...synth.sampler import SynthConfig, generate
    from ...train.data import split_ids

    idx = json.loads(Path("data/synth/train_v2/index.json").read_text())
    cfg_d = dict(idx["synth_config"])
    cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_d.items() if k in SynthConfig.__dataclass_fields__})
    need = {b: real_bands.count(b) for b in range(len(BANDS))}
    out = []
    for i in split_ids(idx["n_clips"])[1]:
        if not any(need.values()):
            break
        clip = generate(int(i), cfg, idx["seed"])
        f0 = clip.labels["f0"][clip.labels["voiced"] > 0.5] if "f0" in clip.labels else np.array([])
        if not len(f0):
            continue
        b = band(float(np.median(f0)))
        if need[b] > 0:
            need[b] -= 1
            out.append((f"gate:{int(i)}", f"gate:{int(i)}", "?", clip.audio.astype(np.float32),
                        clip.labels["formants"][:, :4].astype(np.float32)))
    return out


def render(log=print) -> Path:
    """data/probes/world_v1 in the probe runner's layout (audio_c, scored_c, voiced_c, label_formants_c per item)."""
    from . import counterfactual as cf
    from .run import STORES

    OUT.mkdir(parents=True, exist_ok=True)
    conds = [{"arm": "ref"}] + [{"arm": "world", "r": r} for r in FACTORS if r != 1.0]
    order = [1.0] + [r for r in FACTORS if r != 1.0]
    meta = {"conditions": conds, "factors": order, "items": []}
    zsex = zeroth_sex()
    real = []
    store, root, part = STORES["zeroth_val"]
    for it in cf.load_items(store, root, 6, part):
        real.append((f"zeroth:{it.key}", f"zeroth:{it.group}", zsex.get(it.group, "?"), it.audio, None))
    real += [(k, g, s, x, None) for k, g, s, x in libritts_items()]

    def save(k, key, group, sex, x, lab, kind):
        vs, voiced = variants(x)
        arrays = {}
        for c, r in enumerate(order):
            n = len(vs[r]) // 160 + 1
            v = np.r_[voiced, np.zeros(max(0, n - len(voiced)), bool)][:n]
            arrays[f"audio_{c}"], arrays[f"scored_{c}"], arrays[f"voiced_{c}"] = vs[r], v, v
            lf = lab if lab is not None else np.full((n, 4), np.nan, np.float32)
            arrays[f"label_formants_{c}"] = lf[:n]
            arrays[f"label_bandwidths_{c}"] = np.full((n, 4), np.nan, np.float32)
        f0 = analyse_world(x)["f0"]
        np.savez_compressed(OUT / f"{kind}_{k:03d}.npz", **arrays)
        return {"file": f"{kind}_{k:03d}.npz", "key": key, "group": group, "sex": sex, "set": kind,
                "f0_median": float(np.median(f0[f0 > 0])) if (f0 > 0).any() else float("nan"), "beta": float("nan")}

    zeroth_bands = []
    for k, (key, group, sex, x, lab) in enumerate(real):
        m = save(k, key, group, sex, x, lab, key.split(":")[0])
        meta["items"].append(m)
        if m["set"] == "zeroth" and np.isfinite(m["f0_median"]):
            zeroth_bands.append(band(m["f0_median"]))
    log(f"world: {len(real)} real items")
    for k, (key, group, sex, x, lab) in enumerate(gate_items(zeroth_bands)):
        meta["items"].append(save(k, key, group, sex, x, lab, "gate"))
    log(f"world: {sum(m['set'] == 'gate' for m in meta['items'])} gate items")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return OUT
