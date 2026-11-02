"""D-064 scratch check: the pre-existing D-059 record, and the tilted-path datum.

The provenance of D-064's regression evidence (moved from the scratchpad, 2026-11-02). Two jobs:

    python scripts/analysis/d064_check.py clips [--n N] [--engines kl,tube,cascade]
    python scripts/analysis/d064_check.py tilt capture
    python scripts/analysis/d064_check.py tilt check

`clips` regenerates D-059's synthetic grid (d059_realness._clip: train_v2's synth config, seed
5059, cascade / interactive tube / KL) and diffs the audio and the voiced labels against
`cache/d059/clips`, which was written before 7c77891. A match therefore chains the code now to the
code that made the training data; a mismatch names the clip and stops.

`tilt` covers what the cache cannot: it stores audio and voiced only, and no clip with a tilt set.
Twelve clips (3 engines x tilts +6/-12 x 2 indices, `p_h_onset=1` so the whisper component is
always present) are hashed in full - audio, labels, every component - into
`runs/d064_tilt_datum.json`, so the fix can be shown to change exactly the whisper component.

Serial on purpose: this sandbox denies process pipes, so no pools.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "cache" / "d059" / "clips"
DATUM = ROOT / "runs" / "d064_tilt_datum.json"
TILTS = (6.0, -12.0)
INDICES = (0, 2)
ENGINES = ("cascade", "tube", "kl")


def _d059():
    spec = importlib.util.spec_from_file_location("d059", ROOT / "scripts" / "analysis" / "d059_realness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(x) -> str:
    a = np.asarray(x)
    if a.dtype.kind == "f":
        a = np.nan_to_num(a, nan=-1e30, posinf=1e30, neginf=-1e30)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes() + a.dtype.str.encode()).hexdigest()


def cmd_clips(n: int, engines: tuple[str, ...]) -> int:
    d059 = _d059()
    bad = 0
    for name in engines:
        ok = 0
        for i in range(n):
            p = CACHE / name / f"{i:04d}.npz"
            if not p.exists():
                print(f"{name}/{i:04d}: no cache entry (cache/d059 is {len(list((CACHE / name).glob('*.npz')))} clips)")
                bad += 1
                continue
            _, _, audio, voiced = d059._clip((name, i))
            z = np.load(p)
            if not np.array_equal(audio, z["audio"]):
                peak = float(np.max(np.abs(np.asarray(audio) - z["audio"])))
                print(f"{name}/{i:04d}: AUDIO DIFFERS (peak {peak:.3e}, n {len(audio)} vs {len(z['audio'])})")
                bad += 1
            elif not np.array_equal(voiced, z["voiced"]):
                print(f"{name}/{i:04d}: VOICED DIFFERS ({int((voiced != z['voiced']).sum())} frames)")
                bad += 1
            else:
                ok += 1
        print(f"{name}: {ok}/{n} clips identical to cache/d059")
    print("OK: D-059 reproduces exactly" if not bad else f"MISMATCHES: {bad}")
    return 1 if bad else 0


def _tilt_case(engine: str, tilt: float, index: int):
    from vocal_analysis.synth import kl_set, sampler
    from vocal_analysis.synth.sampler import SynthConfig

    cfg = SynthConfig(duration_s=(1.0, 1.0), p_h_onset=1.0, asp_tilt_db_oct=tilt, whisper_tilt_db_oct=tilt)
    if engine == "kl":
        kcfg = replace(kl_set.KLSetConfig(), synth=replace(cfg, p_tube=1.0))
        return kl_set.generate(index, kcfg, 5059, keep_components=True)
    tube = engine == "tube"
    return sampler.generate(index, replace(cfg, p_tube=1.0 if tube else 0.0, p_interactive=1.0 if tube else 0.0),
                            5059, keep_components=True)


def _record() -> dict:
    out = {}
    for engine in ENGINES:
        for tilt in TILTS:
            for index in INDICES:
                clip = _tilt_case(engine, tilt, index)
                out[f"{engine}_t{tilt:+.0f}_{index}"] = {
                    "n": int(len(clip.audio)),
                    "audio": _sha(clip.audio),
                    "labels": {k: _sha(clip.labels[k]) for k in sorted(clip.labels)},
                    "components": {k: _sha(v) for k, v in sorted(clip.components.items())},
                }
    return out


def cmd_tilt(mode: str) -> int:
    now = _record()
    if mode == "capture":
        DATUM.parent.mkdir(parents=True, exist_ok=True)
        DATUM.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{DATUM}: {len(now)} clips")
        return 0
    want = json.loads(DATUM.read_text(encoding="utf-8"))
    bad = 0
    for key in sorted(set(want) | set(now)):
        w, g = want.get(key), now.get(key)
        if w == g:
            continue
        bad += 1
        if w is None or g is None:
            print(f"{key}: {'new' if w is None else 'gone'}")
            continue
        fields = [f for f in ("n", "audio") if w[f] != g[f]]
        fields += [f"labels.{k}" for k in sorted(w["labels"]) if w["labels"][k] != g["labels"].get(k)]
        fields += [f"components.{k}" for k in sorted(w["components"]) if w["components"][k] != g["components"].get(k)]
        fields += [f"components.{k} gone" for k in sorted(g["components"]) if k not in w["components"]]
        print(f"{key}: {', '.join(fields)}")
    print("OK: tilted datum identical" if not bad else f"CHANGED: {bad} clips")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips")
    c.add_argument("--n", type=int, default=160)
    c.add_argument("--engines", default=",".join(ENGINES))
    t = sub.add_parser("tilt")
    t.add_argument("mode", choices=("capture", "check"))
    a = ap.parse_args()
    if a.cmd == "clips":
        return cmd_clips(a.n, tuple(e for e in a.engines.split(",") if e))
    return cmd_tilt(a.mode)


if __name__ == "__main__":
    raise SystemExit(main())
