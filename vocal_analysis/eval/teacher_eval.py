"""Evaluate a trained teacher (plan M1/M2 acceptance): gold sets alongside the baselines, the synthetic
hard-case suite, and the sim-to-real gap per f0 band.

The teacher is wrapped as a FormantBaseline and an F0Baseline, so it goes through the same
caching, scoring and report code as Praat, PESTO and the rest.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import numpy as np

from ..baselines.f0 import F0Baseline
from ..baselines.formants import N_FORMANTS, FormantBaseline
from ..config import Config
from .metrics import f0_band_labels

TOL = (0.20, 200.0)


def _ckpt_id(path: str) -> str:
    """Content hash of the checkpoint; "<ckpt>#ema" (its averaged weights) gets its own id."""
    h = hashlib.sha256()
    with open(path.removesuffix("#ema"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12] + ("-ema" if path.endswith("#ema") else "")


@functools.lru_cache(maxsize=2)
def _model(path: str):
    from ..device import setup_device
    from ..models.infer import load_teacher

    dev = setup_device()
    model, ck = load_teacher(path, dev)
    return model, dev, ck


def _analyse(path: str, x: np.ndarray, sr: int) -> dict:
    from ..models.infer import analyse

    model, dev, _ = _model(path)
    return analyse(model, x, dev, sr)


def teacher_baselines(ckpt: str | Path, name: str = "teacher") -> tuple[FormantBaseline, F0Baseline]:
    from ..models.infer import eval_bf16

    ckpt = str(ckpt)
    version = _ckpt_id(ckpt) + ("" if eval_bf16() else "-fp32")  # cached predictions are kept apart by precision
    step = _model(ckpt)[2].get("step", "?")

    def formants(x, sr, times, meta):
        res = _analyse(ckpt, x, sr)
        idx = np.clip(np.round(times / 0.010).astype(int), 0, len(res["t"]) - 1)
        out = np.full((len(times), N_FORMANTS), np.nan)
        out[:, :4] = res["formants"][idx]
        return out

    def f0(x, sr):
        res = _analyse(ckpt, x, sr)
        return res["t"], res["f0"], res["voiced_p"]

    desc = f"teacher {Path(ckpt.removesuffix('#ema')).parent.name}{' (EMA)' if ckpt.endswith('#ema') else ''} step {step} (sha {version}), trained on synthetic data only"
    return (FormantBaseline(name, desc, formants, version=version),
            F0Baseline(name, desc + "; voiced if voicing p >= 0.5", f0, voicing_threshold=0.5, version=version))


def suite_metrics(ckpt: str | Path, suite_dir: str | Path, log=print) -> dict:
    """Per hard case: DR(20 %, 200 Hz) and MAE for F1-F4 on sonorant frames, f0 RPA on voiced frames
    (None if there are none), voicing accuracy on all frames, and F1-F3 DR by f0 band (reference case)."""
    from ..synth.shards import ShardReader

    out, pooled = {}, {"err": [], "ref": [], "f0": []}
    for case_dir in sorted(p for p in Path(suite_dir).iterdir() if (p / "index.json").exists()):
        reader = ShardReader(case_dir)
        errs, refs, f0c, vok = [], [], [], []
        for i in range(len(reader)):
            audio, lab, _ = reader[i]
            res = _analyse(str(ckpt), audio, 16000)
            n = min(len(res["t"]), len(lab["t"]))
            son, voiced = lab["sonorant"][:n], lab["voiced"][:n]
            errs.append(res["formants"][:n][son] - lab["formants"][:n, :4][son])
            refs.append(np.c_[lab["formants"][:n, :4][son], lab["f0"][:n][son]])
            f0c.append(1200 * np.log2(res["f0"][:n][voiced] / lab["f0"][:n][voiced]))
            vok.append((res["voiced_p"][:n] >= 0.5) == voiced)
        e, r, c = np.concatenate(errs), np.concatenate(refs), np.concatenate(f0c)
        row = {"n_frames": int(len(e)), "f0_rpa": float(np.mean(np.abs(c) <= 50)) if len(c) else None,
               "voicing_acc": float(np.mean(np.concatenate(vok)))}
        for k in range(4):
            ok = np.isfinite(r[:, k])
            row[f"F{k + 1}_dr"] = float(np.mean((np.abs(e[ok, k]) <= TOL[0] * r[ok, k]) & (np.abs(e[ok, k]) <= TOL[1])))
            row[f"F{k + 1}_mae_hz"] = float(np.mean(np.abs(e[ok, k])))
        out[case_dir.name] = row
        if case_dir.name == "reference":
            pooled["err"].append(e), pooled["ref"].append(r)
        rpa = "-" if row["f0_rpa"] is None else f"{row['f0_rpa']:.3f}"
        log(f"  suite {case_dir.name}: F1 DR {row['F1_dr']:.3f} F2 DR {row['F2_dr']:.3f} f0 RPA {rpa} voicing {row['voicing_acc']:.3f}")
    e, r = np.concatenate(pooled["err"]), np.concatenate(pooled["ref"])
    bands = f0_band_labels(r[:, 4])
    by_band = {}
    for b in sorted(set(bands.tolist())):
        m = bands == b
        by_band[b] = {f"F{k + 1}": float(np.mean((np.abs(e[m, k]) <= TOL[0] * r[m, k]) & (np.abs(e[m, k]) <= TOL[1])))
                      for k in range(3)} | {"n": int(m.sum())}
    return {"cases": out, "reference_by_f0_band": by_band}


def _sections(suite: dict, name: str):
    def make(report: dict) -> list[str]:
        from .report import _label_key

        L = ["", "## Synthetic hard-case suite (fixed seed, `data/synth/suite_v1`)", "",
             "DR is within (20 %, 200 Hz) on sonorant frames (voiced, plus whispered in `whisper`); f0 RPA within "
             "50 cents on voiced frames; voicing accuracy over all frames. The channel is clean except in `low_snr`, "
             "`phone` and `codec`.", "",
             "| Case | Frames | F1 DR | F2 DR | F3 DR | F4 DR | F1 MAE Hz | F2 MAE Hz | f0 RPA | Voicing acc |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for case, r in suite["cases"].items():
            rpa = "–" if r["f0_rpa"] is None else f"{100 * r['f0_rpa']:.1f}"
            L.append(f"| {case} | {r['n_frames']:,} | {100 * r['F1_dr']:.1f} | {100 * r['F2_dr']:.1f} | {100 * r['F3_dr']:.1f} | "
                     f"{100 * r['F4_dr']:.1f} | {r['F1_mae_hz']:.0f} | {r['F2_mae_hz']:.0f} | {rpa} | {100 * r['voicing_acc']:.1f} |")
        L += ["", "## Sim-to-real gap by f0 band", "",
              "Formant DR (20 %, 200 Hz) % on the synthetic `reference` case versus Hillenbrand, by f0 band "
              "(Hillenbrand uses each token's steady-state f0).", "",
              "| f0 band | Synthetic F1 / F2 / F3 | Hillenbrand F1 / F2 / F3 | Gap F1 / F2 / F3 (points) |", "|---|---|---|---|"]
        hb = report["formants"].get(name, {}).get("by_f0_band", {})
        for band in sorted(set(suite["reference_by_f0_band"]) | set(hb), key=_label_key):
            s = suite["reference_by_f0_band"].get(band)
            h = hb.get(band)
            fs_ = " / ".join(f"{100 * s[f'F{k}']:.1f}" for k in (1, 2, 3)) if s else "–"
            fh = " / ".join(f"{100 * h[f'F{k}']['dr']:.1f}" for k in (1, 2, 3)) if h and all(f"F{k}" in h for k in (1, 2, 3)) else "–"
            gap = (" / ".join(f"{100 * (s[f'F{k}'] - h[f'F{k}']['dr']):+.1f}" for k in (1, 2, 3))
                   if s and h and all(f"F{k}" in h for k in (1, 2, 3)) else "–")
            L.append(f"| {band} | {fs_} | {fh} | {gap} |")
        return L
    return make


def build_teacher_report(cfg: Config, ckpt: str | Path, out: str | Path, suite_dir: str | Path, name: str = "teacher",
                         log=print) -> Path:
    from .report import build_report

    fb, f0b = teacher_baselines(ckpt, name)
    log("[suite]")
    suite = suite_metrics(ckpt, suite_dir, log)
    return build_report(cfg, Path(out), None, False, log, extra_formant={name: fb}, extra_f0={name: f0b},
                        title=f"Teacher evaluation: {Path(ckpt).parent.name}", extra_sections=_sections(suite, name))
