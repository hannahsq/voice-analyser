"""D-070: is ArtSpeech's MRI audio usable for checking vocal-tract length? Dev speakers P1 and P7 only (D-069 amendment 1),
label-free, never compared with any MRI length; no per-speaker length is printed or committed (docs/decisions.md).

    python scripts/analysis/d070_artspeech_audio.py analyse    (GPU: teacher v3.1 seeds 2 and 1, per-session summaries)
    python scripts/analysis/d070_artspeech_audio.py report     (CPU: against PTDB-TUG and USC dev from D-068's cache)

Per session as D-068 (sonorant voiced frames, seed mean). Per speaker: within-speaker SD of per-session log length (%)
with a bootstrap over sessions, median length sigma, median formant sigma F1-F4.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ZIPS = Path("C:/github/datasets/isaieva2021_french_rtmri/download")
SPEAKERS = {"P1": "m", "P7": "f"}  # vtl-mri V-023 (DICOM PatientSex)
D068 = Path("cache/d068/files.json")
CACHE = Path("cache/d070")
OUT = Path("reports/analysis/d070")
MODELS = ("d040_5b_s2_b20k", "d040_5b_s1_b20k")
SEED = 5070


def _sessions():
    import soundfile as sf

    from vocal_analysis.eval.tiers import tier_of

    for spk in SPEAKERS:
        assert tier_of(f"artspeech:{spk}") == "dev", spk  # never a lockbox speaker
        with zipfile.ZipFile(ZIPS / f"{spk}.zip") as z:
            for k in range(1, 17):
                x, sr = sf.read(io.BytesIO(z.read(f"{spk}/OTHER/S{k}/DENOISED_SOUND_{spk}_S{k}.wav")))
                assert sr == 16000 and x.ndim == 1
                yield spk, k, x


def analyse():
    import torch

    from vocal_analysis.models.infer import analyse as run, load_teacher

    dev = torch.device("cuda")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "files.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    models = [load_teacher(f"runs/{m}/ckpt_last.pt", dev)[0] for m in MODELS]
    for spk, k, x in _sessions():
        key = f"{spk}_S{k}"
        if key in done:
            continue
        x = np.asarray(x, np.float32)
        x = 0.5 * x / (np.abs(x).max() + 1e-12)
        per = []
        for m in models:
            r = run(m, x, dev)
            on = (r["voiced_p"] > 0.5) & (r["sonorant_p"] > 0.5)
            per.append({"frac": float(on.mean()), "n": int(on.sum()),
                        "vtl": float(np.median(2 ** r["vtl_log2cm"][on])) if on.any() else None,
                        "vtl_sigma": float(np.median(r["vtl_sigma"][on])) if on.any() else None,
                        "Fsig": [float(np.median(r["formant_sigma_cents"][on, j])) if on.any() else None for j in range(4)]})
        ok = all(p["vtl"] is not None for p in per)
        done[key] = {"speaker": spk, "sex": SPEAKERS[spk], "n": per[0]["n"], "frac": float(np.mean([p["frac"] for p in per])),
                     "vtl": float(np.mean([p["vtl"] for p in per])) if ok else None,
                     "vtl_sigma": float(np.mean([p["vtl_sigma"] for p in per])) if ok else None,
                     "Fsig": [float(np.mean([p["Fsig"][j] for p in per])) if ok else None for j in range(4)]}
        print(key, flush=True)
    path.write_text(json.dumps(done))
    print(f"analysed {len(done)} sessions", flush=True)


def _within(groups: list[np.ndarray]) -> float:
    """pooled within-speaker SD of log length, % (as D-068's _rel)"""
    return 100 * float(np.sqrt(np.mean([np.var(np.log(v), ddof=1) for v in groups if len(v) >= 2])))


def _ref(corpus: str, sx: str):
    d = json.loads(D068.read_text())
    rows = [v for v in d.values() if v["corpus"] == corpus and v["sex"] == sx and v["vtl"] is not None and v["n"] >= 50]
    by: dict[str, list] = {}
    for v in rows:
        by.setdefault(v["speaker"], []).append(v["vtl"])
    return _within([np.array(v) for v in by.values()]), float(np.median([v["vtl_sigma"] for v in rows]))


def report():
    rng = np.random.default_rng(SEED)
    d = json.loads((CACHE / "files.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["D-070: ArtSpeech MRI audio, dev speakers P1 (man) and P7 (woman), 16 sessions each, label-free.",
             "Teacher v3.1 seeds 2 and 1, seed mean per session; sonorant voiced frames. No per-speaker length is shown.",
             "References: PTDB-TUG dev and USC dev of the same sex (D-068's cache, same models).", ""]
    verdict = []
    for spk, sx in SPEAKERS.items():
        rows = [v for k, v in d.items() if v["speaker"] == spk and v["vtl"] is not None and v["n"] >= 50]
        L = np.array([v["vtl"] for v in rows])
        w = _within([L])
        bs = [_within([rng.choice(L, len(L))]) for _ in range(1000)]
        sig = float(np.median([v["vtl_sigma"] for v in rows]))
        fs = [np.median([v["Fsig"][j] for v in rows]) for j in range(4)]
        frac = np.median([v["frac"] for k, v in d.items() if v["speaker"] == spk])
        (wp, sp), (wu, su) = _ref("ptdb", sx), _ref("usc", sx)
        ok = w <= 1.5 * wp and sig <= 1.5 * sp
        verdict.append("usable" if ok else ("not usable" if w > 2 * wp else "doubtful"))
        who = "man" if sx == "m" else "woman"
        lines += [f"{spk} ({who}): {len(rows)} sessions (of {sum(v['speaker'] == spk for v in d.values())}; < 50 sonorant voiced frames left out)",
                  f"  sonorant voiced fraction (median session) {frac:.2f}; VTL sigma (median, log2) {sig:.3f}; formant sigma F1-F4 (cents) "
                  + " / ".join(f"{x:.0f}" for x in fs),
                  f"  within-speaker SD {w:.2f} % [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]",
                  f"  against PTDB-TUG {who}: within-SD ratio {w / wp:.2f} ({wp:.2f} %), sigma ratio {sig / sp:.2f} ({sp:.3f}) -> {verdict[-1]}",
                  f"  against USC {who} (reported): within-SD ratio {w / wu:.2f} ({wu:.2f} %), sigma ratio {sig / su:.2f} ({su:.3f})", ""]
    overall = "usable" if all(v == "usable" for v in verdict) else ("not usable" if "not usable" in verdict else "doubtful")
    lines.append(f"VERDICT: ArtSpeech's MRI audio is {overall} for checking vocal-tract length")
    text = "\n".join(lines)
    print(text)
    (OUT / "d070.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"analyse": analyse, "report": report}[sys.argv[1]]()
