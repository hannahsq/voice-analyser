"""Baseline report (plan §6.5): `va baselines` -> reports/suite/baselines_<tiers>.md (+ .json); dev speakers by default (D-024).

Predictions are cached per (baseline, corpus) under runs/baselines/ and reused
while the manifest hash and baseline version are unchanged; long runs
checkpoint so they can resume. Every report records its provenance (plan §0).
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import Config, config_to_dict
from ..data.manifest import manifest_hash, read_manifest
from . import metrics as M
from .datasets import load_audio, registry

FORMANT_CORPUS = "hillenbrand"
F0_CORPORA = ("ptdb_tug", "mir1k", "mdb_stem_synth")
GROUP_NAMES = {"m": "men", "w": "women", "b": "boys", "g": "girls"}
CHECKPOINT_EVERY_S = 300.0


# --------------------------------------------------------------------------- caching


def _cache_path(cache_dir: Path, baseline: str, corpus: str, tiers: tuple[str, ...] | None = None) -> Path:
    return cache_dir / f"{baseline}__{corpus}{_tier_tag(tiers)}.npz"


def _tier_tag(tiers: tuple[str, ...] | None) -> str:
    return "" if tiers is None else "__" + "+".join(sorted(tiers))


def _key(key: dict, tiers: tuple[str, ...] | None) -> dict:
    return key if tiers is None else {**key, "tiers": sorted(tiers)}


def _rows(mpath: Path, tiers: tuple[str, ...] | None) -> list:
    """The manifest's rows; with `tiers`, only gold speakers in those tiers (D-024). None = every speaker, as the
    analysis scripts use it (they apply VA_TIERS themselves)."""
    rows = read_manifest(mpath)
    if tiers is None:
        return rows
    from .tiers import filter_rows

    return filter_rows(rows, tiers)


def _load_cache(path: Path, key: dict) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if json.loads(str(z["_key"])) != key:
            return None
        return {k: z[k] for k in z.files if k != "_key"}


def _save(path: Path, key: dict, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, _key=json.dumps(key), **arrays)
    tmp.replace(path)


def _concat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


# --------------------------------------------------------------------------- running baselines


def run_formant_baseline(b, cfg: Config, cache_dir: Path, force: bool, log: Callable,
                         tiers: tuple[str, ...] | None = None) -> dict:
    corpus = FORMANT_CORPUS
    mpath = Path(cfg.data.manifests) / f"{corpus}.csv"
    key = _key({"baseline": b.name, "version": b.version, "corpus": corpus, "manifest": manifest_hash(mpath)}, tiers)
    path = _cache_path(cache_dir, b.name, corpus, tiers)
    if not force and (cached := _load_cache(path, key)) is not None:
        return cached
    mod = registry()[corpus]
    root = Path(cfg.data.root) / corpus
    parts, t0, audio_s = [], time.time(), 0.0
    rows = _rows(mpath, tiers)
    for i, row in enumerate(rows):
        item = mod.load(root, row)
        x = load_audio(item, cfg.frames.sample_rate)
        audio_s += len(x) / cfg.frames.sample_rate
        ok = np.isfinite(item.times)
        est = np.full(item.formants.shape, np.nan)
        if ok.any():
            est[ok] = b.predict(x, cfg.frames.sample_rate, item.times[ok], item.meta)
        n = len(item.times)
        parts.append({
            "ref": item.formants,
            "est": est,
            "speaker": np.array([row.speaker_id] * n),
            "group": np.array([GROUP_NAMES.get(item.meta.get("group"), "?")] * n),
            "token_f0": np.full(n, item.f0[0]),
            "frame_kind": np.array(list(item.meta.get("frame_kinds", ("steady",)))[:n]),
        })
        if (i + 1) % 400 == 0:
            log(f"  [{b.name}/{corpus}] {i + 1}/{len(rows)}")
    out = _concat(parts)
    out["compute_s"] = np.array(time.time() - t0)
    out["audio_s"] = np.array(audio_s)
    _save(path, key, out)
    return out


def _align(ref_t: np.ndarray, est_t: np.ndarray, *est: np.ndarray, max_dt: float) -> list[np.ndarray]:
    """Nearest estimate frame for each reference time; NaN where none is within max_dt."""
    if len(est_t) < 2:
        return [np.full(len(ref_t), np.nan) for _ in est]
    right = np.clip(np.searchsorted(est_t, ref_t), 1, len(est_t) - 1)
    left = right - 1
    idx = np.where(ref_t - est_t[left] <= est_t[right] - ref_t, left, right)
    far = np.abs(est_t[idx] - ref_t) > max_dt
    return [np.where(far, np.nan, e[idx]) for e in est]


def run_f0_baseline(b, corpus: str, cfg: Config, cache_dir: Path, force: bool, log: Callable,
                    tiers: tuple[str, ...] | None = None) -> dict:
    from ..baselines.f0 import HOP_S

    mpath = Path(cfg.data.manifests) / f"{corpus}.csv"
    key = _key({"baseline": b.name, "version": b.version, "corpus": corpus, "manifest": manifest_hash(mpath)}, tiers)
    path = _cache_path(cache_dir, b.name, corpus, tiers)
    if not force and (cached := _load_cache(path, key)) is not None:
        return cached

    partial = path.with_suffix(".partial.npz")
    parts, done, compute_s, audio_s = [], 0, 0.0, 0.0
    if not force and (prev := _load_cache(partial, key)) is not None:
        done, compute_s, audio_s = int(prev.pop("n_done")), float(prev.pop("compute_s")), float(prev.pop("audio_s"))
        parts = [prev]
        log(f"  [{b.name}/{corpus}] resuming after {done} files")

    mod = registry()[corpus]
    root = Path(cfg.data.root) / corpus
    rows = _rows(mpath, tiers)
    last_ckpt = time.time()
    for i in range(done, len(rows)):
        row = rows[i]
        item = mod.load(root, row)
        x = load_audio(item, cfg.frames.sample_rate)
        t0 = time.time()
        est_t, est_f0, est_p = b.predict(x, cfg.frames.sample_rate)
        compute_s += time.time() - t0
        audio_s += len(x) / cfg.frames.sample_rate
        f0, prob = _align(item.times, est_t, est_f0, est_p, max_dt=HOP_S / 2 + 1e-6)
        parts.append({"ref": item.f0.astype(float), "est": f0, "prob": prob, "speaker": np.array([row.speaker_id] * len(f0))})
        if (i + 1) % 250 == 0:
            log(f"  [{b.name}/{corpus}] {i + 1}/{len(rows)}  (RTF {compute_s / audio_s:.3f})")
        if time.time() - last_ckpt > CHECKPOINT_EVERY_S:
            parts = [_concat(parts)]
            _save(partial, key, {**parts[0], "n_done": np.array(i + 1), "compute_s": np.array(compute_s), "audio_s": np.array(audio_s)})
            last_ckpt = time.time()
    out = _concat(parts)
    out["compute_s"], out["audio_s"] = np.array(compute_s), np.array(audio_s)
    _save(path, key, out)
    partial.unlink(missing_ok=True)
    return out


LABELS_VERSION = "1"


def run_frame_labels(corpus: str, cfg: Config, cache_dir: Path, force: bool, log: Callable,
                     tiers: tuple[str, ...] | None = None) -> np.ndarray | None:
    """Per-frame labels (manifest order, same frames as run_f0_baseline) from the corpus's frame_labels(), if any."""
    mod = registry()[corpus]
    if not hasattr(mod, "frame_labels"):
        return None
    mpath = Path(cfg.data.manifests) / f"{corpus}.csv"
    key = _key({"labels": "frame_labels", "version": LABELS_VERSION, "corpus": corpus, "manifest": manifest_hash(mpath)}, tiers)
    path = cache_dir / f"labels__{corpus}{_tier_tag(tiers)}.npz"
    if not force and (cached := _load_cache(path, key)) is not None:
        return cached["labels"]
    root = Path(cfg.data.root) / corpus
    rows = _rows(mpath, tiers)
    parts = []
    for i, row in enumerate(rows):
        parts.append(mod.frame_labels(root, row, mod.load(root, row)))
        if (i + 1) % 1000 == 0:
            log(f"  [labels/{corpus}] {i + 1}/{len(rows)}")
    labels = np.concatenate(parts)
    _save(path, key, {"labels": labels})
    return labels


# --------------------------------------------------------------------------- scoring


def _ci(sums, stat, cfg: Config) -> dict:
    r = M.bootstrap_sums(sums, stat, cfg.eval.bootstrap_resamples, seed=cfg.eval.bootstrap_seed)
    return {"value": r.estimate, "low": r.low, "high": r.high}


def _ratio(num: str, den: str) -> Callable:
    return lambda s: s[num] / s[den]


def score_formants(res: dict, cfg: Config) -> dict:
    tols = [tuple(t) for t in cfg.eval.formant_tolerances]
    det_keys = [f"det_{r:g}_{a:g}" for r, a in tols]
    out = {"overall": {}, "by_group": {}, "by_f0_band": {}, "by_frame_kind": {}}
    bands = M.f0_band_labels(res["token_f0"], cfg.eval.f0_bands_hz)
    strata = {"by_group": res["group"], "by_f0_band": bands, "by_frame_kind": np.where(res["frame_kind"] == "steady", "steady", "20/50/80%")}
    for k in range(res["ref"].shape[1]):
        name = f"F{k + 1}"
        fs = M.formant_frame_stats(res["est"][:, k], res["ref"][:, k], tols)
        _, sums = M.group_sums(fs, res["speaker"])
        if sums["n_ref"].sum() == 0:
            continue
        tot = {key: float(v.sum()) for key, v in sums.items()}
        out["overall"][name] = {
            "n": int(tot["n_ref"]),
            "missing_rate": 1 - tot["n_both"] / tot["n_ref"],
            "mae_hz": _ci(sums, _ratio("abs_err", "n_both"), cfg),
            "mae_pct": tot["abs_pct"] / tot["n_both"],
            "rmse_hz": float(np.sqrt(tot["sq_err"] / tot["n_both"])),
            **{f"dr_{key[4:]}": (_ci(sums, _ratio(key, "n_ref"), cfg) if i == 0 else tot[key] / tot["n_ref"]) for i, key in enumerate(det_keys)},
            "gross_rate": tot["gross"] / tot["n_ref"],
        }
        for sname, labels in strata.items():
            for lab in sorted(set(labels.tolist())):
                m = labels == lab
                n = fs["n_ref"][m].sum()
                if n:
                    out[sname].setdefault(lab, {})[name] = {"n": int(n), "dr": float(fs[det_keys[0]][m].sum() / n), "mae_hz": float(fs["abs_err"][m].sum() / max(fs["n_both"][m].sum(), 1))}
    return out


def _ptdb_sex(speaker: str) -> str:
    return "female" if speaker.split(":", 1)[1].startswith("F") else "male"


SPEAKER_SPLITS = {"ptdb_tug": _ptdb_sex}  # corpora whose speaker ids encode a group worth reporting


LOW_F0_HZ = 85.0  # "modal, low": well-defined f0 that trackers still miss (D-009)


def _rpa_subsets(fs: dict, res: dict, corpus: str, cfg: Config, labels: np.ndarray | None) -> dict:
    """RPA on all frames, on modal phonation (EGG-labelled corpora, D-009), and per speaker group."""
    spk = res["speaker"].astype(str)
    masks = {}
    if labels is not None:
        modal = labels == "modal"
        masks = {"all frames": np.ones(len(spk), bool), "modal": modal, f"modal, ref < {LOW_F0_HZ:g} Hz": modal & (res["ref"] < LOW_F0_HZ)}
    if corpus in SPEAKER_SPLITS:
        group = np.array([SPEAKER_SPLITS[corpus](s) for s in np.unique(spk)])[np.unique(spk, return_inverse=True)[1]]
        for g in sorted(set(group.tolist())):
            masks[g] = group == g
            if labels is not None:
                masks[f"{g}, modal"] = (group == g) & modal
    out = {}
    for label, m in masks.items():
        _, sums = M.group_sums({k: fs[k] * m for k in ("n_voiced", "rpa")}, spk)
        sums = {k: v[sums["n_voiced"] > 0] for k, v in sums.items()}
        out[label] = {"n": int(sums["n_voiced"].sum()), "rpa": _ci(sums, _ratio("rpa", "n_voiced"), cfg)}
    return out


def _phonation_outcomes(res: dict, ev: np.ndarray, labels: np.ndarray | None) -> dict:
    """Where each estimate lands, per phonation category: at the reference, at 2x or 1/2 of it, or elsewhere."""
    if labels is None:
        return {}
    voiced = res["ref"] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 1200 * np.log2(res["est"] / res["ref"])
    out = {}
    from .phonation import CATEGORIES

    for cat in CATEGORIES:
        m = voiced & (labels == cat)
        if not m.any():
            continue
        at = {k: float(np.mean(np.abs(c[m] - off) <= 50)) for k, off in (("at_ref", 0), ("at_2x", 1200), ("at_half", -1200))}
        out[cat] = {"n": int(m.sum()), **at, "other": 1 - sum(at.values()), "judged_voiced": float(ev[m].mean())}
    return out


def score_f0(res: dict, threshold: float, cfg: Config, corpus: str = "", labels: np.ndarray | None = None) -> dict:
    ev = np.nan_to_num(res["prob"], nan=0.0) >= threshold
    fs = M.pitch_frame_stats(res["est"], res["ref"], ev)
    _, sums = M.group_sums(fs, res["speaker"])
    tot = {k: float(v.sum()) for k, v in sums.items()}
    f1 = lambda s: 2 * s["tp"] / (2 * s["tp"] + s["fp"] + s["fn"])  # noqa: E731
    bands = M.f0_band_labels(res["ref"], cfg.eval.f0_bands_hz)
    by_band = {}
    for lab in sorted(set(bands.tolist()) - {"unvoiced"}):
        m = bands == lab
        by_band[lab] = {"n": int(fs["n_voiced"][m].sum()), "rpa": float(fs["rpa"][m].sum() / fs["n_voiced"][m].sum())}
    return {
        "n_voiced": int(tot["n_voiced"]),
        "n_frames": int(len(res["ref"])),
        "n_speakers": int(len(sums["n_voiced"])),
        "rpa": _ci(sums, _ratio("rpa", "n_voiced"), cfg),
        "rca": _ci(sums, _ratio("rca", "n_voiced"), cfg),
        "octave_error_rate": tot["octave"] / tot["n_voiced"],
        "gross_error_rate": tot["gross"] / tot["n_voiced"],
        "voicing_precision": tot["tp"] / max(tot["tp"] + tot["fp"], 1),
        "voicing_recall": tot["tp"] / max(tot["tp"] + tot["fn"], 1),
        "voicing_f1": _ci(sums, f1, cfg),
        "rpa_by_f0_band": by_band,
        "rpa_subsets": _rpa_subsets(fs, res, corpus, cfg, labels),
        "phonation_outcomes": _phonation_outcomes(res, ev, labels),
        "rtf": float(res["compute_s"] / res["audio_s"]),
    }


# --------------------------------------------------------------------------- provenance and rendering


def provenance(cfg: Config) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    versions = {}
    for pkg in ("numpy", "scipy", "torch", "praat-parselmouth", "pesto-pitch", "torchcrepe"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    manifests = {p.stem: manifest_hash(p) for p in sorted(Path(cfg.data.manifests).glob("*.csv"))}
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "submodules": git("submodule", "status"),
        "python": platform.python_version(),
        "versions": versions,
        "manifests": manifests,
        "config": config_to_dict(cfg),
    }


def _fmt_ci(d: dict, scale: float = 1.0, digits: int = 1) -> str:
    return f"{d['value'] * scale:.{digits}f} [{d['low'] * scale:.{digits}f}, {d['high'] * scale:.{digits}f}]"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


_GROUP_ORDER = ["men", "women", "boys", "girls"]


def _label_key(label: str) -> tuple:
    """Order f0 bands numerically ('<150' < '150-300' < '>500'), groups by _GROUP_ORDER."""
    if label in _GROUP_ORDER:
        return (0, _GROUP_ORDER.index(label))
    head = label.lstrip("<>").split("-")[0]
    try:
        return (1, float(head) + (0.5 if label.startswith(">") else 0.0) - (0.5 if label.startswith("<") else 0.0))
    except ValueError:
        return (2, label)


def _tiers_line(tiers: list[str] | None) -> str:
    if tiers is None:
        return "Speakers: every gold speaker (written before the evaluation tiers, D-024)."
    line = f"Speakers: tiers {', '.join(tiers)} (D-024)."
    others = [("semi-private" if t == "semi" else t) for t in tiers if t != "dev"]
    if others:
        line += f" **Includes {' and '.join(others)} speakers: a use to log in docs/decisions.md.**"
    return line


def render(report: dict) -> str:
    prov, cfg = report["provenance"], report["provenance"]["config"]
    tols = cfg["eval"]["formant_tolerances"]
    t0 = f"{tols[0][0] * 100:g} %, {tols[0][1]:g} Hz"
    L = [
        "# Baseline report",
        "",
        f"Generated {prov['generated']} by `va baselines` at commit `{prov['git_commit']}`"
        + (" (with uncommitted changes)" if prov["git_dirty"] else "")
        + ". Don't edit by hand; rerun the command.",
        "",
        f"Intervals are 95 % speaker-bootstrap percentile intervals ({cfg['eval']['bootstrap_resamples']} resamples, whole speakers resampled). "
        f"Detection rate (DR) counts a frame as correct when its error is within **both** bounds (decisions D-004); missing estimates count as misses. "
        f"Gross error means more than 30 % off, or missing.",
        "",
        _tiers_line(report.get("tiers")),
    ]
    if report["formants"]:
        L += _render_formants(report, tols, t0)
    if report["f0"]:
        L += _render_f0(report)

    L += ["", "## Provenance", "", "| | |", "|---|---|"]
    L.append(f"| Commit | `{prov['git_commit']}`{' (dirty)' if prov['git_dirty'] else ''} |")
    for sub in prov["submodules"].splitlines():
        L.append(f"| Submodule | `{sub.strip()}` |")
    L.append(f"| Python | {prov['python']} |")
    for pkg, v in prov["versions"].items():
        L.append(f"| {pkg} | {v} |")
    for m, h in prov["manifests"].items():
        L.append(f"| Manifest `{m}` | `{h}` |")
    L.append("")
    return "\n".join(L)


def _render_formants(report: dict, tols: list, t0: str) -> list[str]:
    L = [
        "",
        "## Formants: Hillenbrand et al. (1995)",
        "",
        f"{report.get('formant_set', {}).get('tokens', 1668):,} tokens from {report.get('formant_set', {}).get('speakers', 139)} speakers. "
        "Each token has 4 reference frames (steady state; 20/50/80 % of the vowel); F4 exists only at steady state. "
        "These are hand-checked measurements, and Hillenbrand is never trained on.",
        "",
    ]
    for name, desc in report["formant_baselines"].items():
        L.append(f"- **{name}**: {desc}")
    L += ["", f"| Baseline | Formant | n | MAE Hz | MAE % | RMSE Hz | DR ({t0}) % | DR ({tols[1][0] * 100:g} %, {tols[1][1]:g} Hz) % | DR ({tols[2][0] * 100:g} %, {tols[2][1]:g} Hz) % | Gross % | Missing % |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    dk = [f"dr_{r:g}_{a:g}" for r, a in tols]
    for name, sc in report["formants"].items():
        for fk, s in sc["overall"].items():
            L.append(
                f"| {name} | {fk} | {s['n']} | {_fmt_ci(s['mae_hz'], digits=0)} | {s['mae_pct']:.1f} | {s['rmse_hz']:.0f} | "
                f"{_fmt_ci(s[dk[0]], 100)} | {_pct(s[dk[1]])} | {_pct(s[dk[2]])} | {_pct(s['gross_rate'])} | {_pct(s['missing_rate'])} |"
            )
    for sname, title in (("by_group", "speaker group"), ("by_f0_band", "f0 band (the token's steady-state f0)"), ("by_frame_kind", "frame position")):
        L += ["", f"### DR ({t0}) % by {title}", ""]
        first = next(iter(report["formants"].values()))
        labels = sorted(first[sname], key=_label_key)
        formants = list(first["overall"])
        L.append("| Baseline | " + " | ".join(f"{lab} {fk}" for lab in labels for fk in formants[:3]) + " |")
        L.append("|---|" + "---|" * (len(labels) * 3))
        for name, sc in report["formants"].items():
            cells = [f"{100 * sc[sname][lab][fk]['dr']:.1f}" if fk in sc[sname].get(lab, {}) else "–" for lab in labels for fk in formants[:3]]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("n frames per column: " + ", ".join(f"{lab} {first[sname][lab]['F1']['n']}" for lab in labels if "F1" in first[sname][lab]) + ".")
    return L


def _render_f0(report: dict) -> list[str]:
    L = ["", "## f0", ""]
    for name, info in report["f0_baselines"].items():
        L.append(f"- **{name}**: {info['description']}. Trained on: {', '.join(info['trained_on']) or 'none of our gold sets'}.")
    L += [
        "",
        "RPA (raw pitch accuracy) and RCA (raw chroma accuracy) use a 50-cent tolerance and count every reference-voiced frame, whatever the estimated voicing decision. "
        "The voicing thresholds are fixed in advance (not tuned on gold data). "
        "**†** marks a baseline whose training data includes that corpus, so its score there is optimistic and not a fair test.",
        "",
        "| Corpus | Baseline | Speakers | Voiced frames | RPA % | RCA % | Octave err % | GPE % | Voicing P % | Voicing R % | Voicing F1 % | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for corpus, by_b in report["f0"].items():
        for name, s in by_b.items():
            dag = " †" if corpus in report["f0_baselines"][name]["trained_on"] else ""
            L.append(
                f"| {corpus} | {name}{dag} | {s['n_speakers']} | {s['n_voiced']:,} | {_fmt_ci(s['rpa'], 100)} | {_fmt_ci(s['rca'], 100)} | "
                f"{_pct(s['octave_error_rate'])} | {_pct(s['gross_error_rate'])} | {_pct(s['voicing_precision'])} | {_pct(s['voicing_recall'])} | "
                f"{_fmt_ci(s['voicing_f1'], 100)} | {s['rtf']:.3f} |"
            )
    L += ["", "### RPA % by reference f0 band", ""]
    bands = sorted({b for by_b in report["f0"].values() for s in by_b.values() for b in s["rpa_by_f0_band"]}, key=_label_key)
    L.append("| Corpus | Baseline | " + " | ".join(bands) + " |")
    L.append("|---|---|" + "---|" * len(bands))
    for corpus, by_b in report["f0"].items():
        for name, s in by_b.items():
            cells = [f"{100 * s['rpa_by_f0_band'][b]['rpa']:.1f} (n={s['rpa_by_f0_band'][b]['n']:,})" if b in s["rpa_by_f0_band"] else "–" for b in bands]
            L.append(f"| {corpus} | {name} | " + " | ".join(cells) + " |")

    PHON_DESC = {
        "modal": "regular vibration at the reference f0",
        "ref_error_2x": "folds cycle at 2 × the reference: the reference halved",
        "alternating": "alternate cycles differ (period doubling): the reference and 2 × are both defensible",
        "aperiodic": "irregular phonation (creak): no single f0",
        "weak_egg": "too little laryngograph signal; the reference is unreliable",
    }
    for corpus, by_b in report["f0"].items():
        first = next(iter(by_b.values()))
        if first.get("rpa_subsets"):
            labels = list(first["rpa_subsets"])
            L += ["", f"### RPA % on {corpus} subsets", ""]
            if first.get("phonation_outcomes"):
                L += [
                    "*Modal* frames are those where the laryngograph shows regular vibration at the reference f0 (decisions D-009): "
                    "the f0 is well defined there, so this is the fairest accuracy figure. Headline numbers use all frames.",
                    "",
                ]
            L.append("| Baseline | " + " | ".join(labels) + " |")
            L.append("|---|" + "---|" * len(labels))
            for name, s in by_b.items():
                L.append(f"| {name} | " + " | ".join(_fmt_ci(s["rpa_subsets"][lab]["rpa"], 100) for lab in labels) + " |")
            L += ["", "Voiced reference frames: " + ", ".join(f"{lab} {first['rpa_subsets'][lab]['n']:,}" for lab in labels) + "."]
        if first.get("phonation_outcomes"):
            L += [
                "",
                f"### {corpus}: where estimates land, by laryngograph phonation category",
                "",
                "Each voiced reference frame is categorised from the laryngograph signal alone (`eval/phonation.py`). "
                "Cells give % of frames within 50 cents of the reference / of 2 × / of ½ the reference, and % the baseline judged voiced. "
                "Outside *modal* the f0 is ambiguous or the reference unreliable, so these are descriptions, not accuracies.",
                "",
                "| Category | Share | " + " | ".join(by_b) + " |",
                "|---|---|" + "---|" * len(by_b),
            ]
            total = sum(o["n"] for o in first["phonation_outcomes"].values())
            for cat, o in first["phonation_outcomes"].items():
                cells = []
                for name, s in by_b.items():
                    q = s["phonation_outcomes"].get(cat)
                    cells.append(f"{100 * q['at_ref']:.0f} / {100 * q['at_2x']:.0f} / {100 * q['at_half']:.0f}; voiced {100 * q['judged_voiced']:.0f}" if q else "–")
                L.append(f"| **{cat}**: {PHON_DESC.get(cat, '')} | {100 * o['n'] / total:.1f} % | " + " | ".join(cells) + " |")
    return L


def build_report(cfg: Config, out_path: Path, only: list[str] | None, force: bool, log: Callable = print,
                 extra_formant: dict | None = None, extra_f0: dict | None = None, title: str = "Baseline report",
                 extra_sections: Callable[[dict], list[str]] | None = None, tiers: tuple[str, ...] = ("dev",)) -> Path:
    """Score baselines (and any extra models, e.g. a trained teacher) on the gold sets and write the report.

    tiers: which gold speakers are scored (D-024): dev by default; semi-private or lockbox speakers only when named,
    and each such use is logged in docs/decisions.md.

    extra_formant/extra_f0 are {name: FormantBaseline/F0Baseline} scored alongside the registered
    baselines; extra_sections(report) returns markdown lines inserted before the provenance.
    """
    from ..baselines import F0_BASELINES, FORMANT_BASELINES
    from ..device import setup_device

    setup_device(cfg.device)
    cache_dir = Path("runs") / "baselines"
    want = lambda name: only is None or name in only  # noqa: E731
    tiers = tuple(sorted(set(tiers)))
    report = {"provenance": provenance(cfg), "tiers": list(tiers), "formant_baselines": {}, "formants": {}, "f0_baselines": {},
              "f0": {}}

    fs_rows = _rows(Path(cfg.data.manifests) / f"{FORMANT_CORPUS}.csv", tiers)
    report["formant_set"] = {"tokens": len(fs_rows), "speakers": len({r.speaker_id for r in fs_rows})}
    for name, b in {**FORMANT_BASELINES, **(extra_formant or {})}.items():
        if not want(name):
            continue
        log(f"[{name}] {FORMANT_CORPUS}")
        res = run_formant_baseline(b, cfg, cache_dir, force, log, tiers)
        report["formant_baselines"][name] = b.description
        report["formants"][name] = score_formants(res, cfg)
    labels: dict[str, np.ndarray | None] = {}
    for name, b in {**F0_BASELINES, **(extra_f0 or {})}.items():
        if not want(name):
            continue
        report["f0_baselines"][name] = {"description": b.description, "trained_on": list(b.trained_on)}
        for corpus in F0_CORPORA:
            log(f"[{name}] {corpus}")
            res = run_f0_baseline(b, corpus, cfg, cache_dir, force, log, tiers)
            if corpus not in labels:
                labels[corpus] = run_frame_labels(corpus, cfg, cache_dir, force, log, tiers)
            report["f0"].setdefault(corpus, {})[name] = score_f0(res, b.voicing_threshold, cfg, corpus, labels[corpus])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render(report).replace("# Baseline report", f"# {title}", 1)
    if extra_sections:
        head, sep, tail = text.partition("\n## Provenance")
        text = head + "\n".join(extra_sections(report)) + "\n" + sep + tail
    out_path.write_text(text, encoding="utf-8")
    out_path.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    log(f"wrote {out_path} and {out_path.with_suffix('.json')}")
    return out_path
