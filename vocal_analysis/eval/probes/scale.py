"""Uniform scale ("tape speed") probe (D-050 amendment 2, probe 11; Hannah): a file resampled to R Hz and read as 16 kHz.

Every frequency scales by r = 16000 / R (f0, formants, bandwidths) and time by 1 / r, so against log2 r the ideal
slope is 1 for f0, F1-F4 and B1-B4 and -1 for VTL (in this probe's sense: a uniformly scaled voice is not a physically
plausible speaker). The resampling starts from the stored file, so PTDB-TUG (48 kHz) keeps the full 0-8 kHz band in
both directions and its reference f0 scales exactly; Hillenbrand (16 kHz) is scaled up only (r > 1), and its formant
references scale exactly. Files: probe 3's (Hillenbrand dev, every file; PTDB-TUG dev, 20 per speaker).

    python -m vocal_analysis.eval.probes.scale render
    python -m vocal_analysis.eval.probes.scale score <ckpt> [...]   ($VA_DEVICE=cuda)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from . import measures as M
from .channel import PTDB_PER_SPEAKER, _rows

CACHE = Path("data/probes/scale_v1")
REPORTS = Path("reports/probes/scale")
FS = 16000
R_PTDB = (0.85, 0.92, 1.0, 1.09, 1.18)
R_HILLENBRAND = (1.0, 1.09, 1.18)  # 16 kHz files: scaling down would leave the band above 8000 * r empty


def rates(rs: tuple[float, ...]) -> list[tuple[int, float]]:
    """(R, exact r) per nominal r: R is a whole number of Hz, and r = 16000 / R exactly."""
    return [(int(round(FS / r)), FS / int(round(FS / r))) for r in rs]


def render(log=print) -> Path:
    from ..datasets import load_audio, registry

    CACHE.mkdir(parents=True, exist_ok=True)
    reg = registry()
    by_spk: dict[str, list] = {}
    for r in _rows("ptdb_tug"):
        by_spk.setdefault(r.speaker_id, []).append(r)
    rows = {"hillenbrand": _rows("hillenbrand"),
            "ptdb_tug": [rs[int(i)] for _, rs in sorted(by_spk.items())
                         for i in np.linspace(0, len(rs) - 1, min(PTDB_PER_SPEAKER, len(rs)))]}
    meta = {"rates": {"hillenbrand": rates(R_HILLENBRAND), "ptdb_tug": rates(R_PTDB)}, "items": []}
    for corpus, rs in rows.items():
        for k, row in enumerate(rs):
            item = reg[corpus].load(Path("data") / corpus, row)
            arrays = {}
            for c, (sr, _) in enumerate(meta["rates"][corpus]):
                y = load_audio(item, sr=sr).astype(float)  # resampled from the stored file, then read as 16 kHz
                arrays[f"audio_{c}"] = np.round(y / (np.max(np.abs(y)) + 1e-12) * 32000).astype(np.int16)
            arrays["times"] = item.times
            arrays["ref_f0"] = item.f0 if item.f0 is not None else np.full(len(item.times), np.nan)
            arrays["ref_formants"] = item.formants if item.formants is not None else np.full((len(item.times), 4), np.nan)
            name = f"{corpus}_{k:04d}.npz"
            np.savez_compressed(CACHE / name, **arrays)
            meta["items"].append({"file": name, "corpus": corpus, "speaker": f"{corpus}:{row.speaker_id}"})
            if (k + 1) % 200 == 0:
                log(f"  {corpus} {k + 1}/{len(rs)}")
        log(f"{corpus}: {len(rs)} files")
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return CACHE


OUTS = [f"F{k}" for k in range(1, 5)] + [f"B{k}" for k in range(1, 5)] + ["VTL", "f0"]


def score(ckpt: str, log=print) -> dict:
    """Per item and condition: medians of log2 F1-F4, B1-B4, f0 and VTL (log2 cm) at the reference frames mapped to
    their scaled times (Hillenbrand: frames with a formant reference; PTDB-TUG: reference-voiced frames); Hillenbrand
    mean |error| (Hz) against the scaled formant references; PTDB-TUG counts of voiced frames and of f0 within
    50 cents of the scaled reference."""
    import torch

    from ...models.infer import analyse, load_teacher

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    items = meta["items"]
    nc = max(len(v) for v in meta["rates"].values())
    Y = {o: np.full((len(items), nc), np.nan) for o in OUTS}
    err = np.full((len(items), nc, 4), np.nan)
    pt = np.full((len(items), nc, 2), np.nan)
    for i, it in enumerate(items):
        d = np.load(CACHE / it["file"])
        times, rf0, rfm = d["times"], d["ref_f0"], d["ref_formants"][:, :4]
        for c, (_, r) in enumerate(meta["rates"][it["corpus"]]):
            res = analyse(model, d[f"audio_{c}"].astype(np.float32) / 32000, dev)
            idx = np.clip(np.round(times / r / 0.01).astype(int), 0, len(res["f0"]) - 1)
            if it["corpus"] == "hillenbrand":
                sel = np.isfinite(rfm).any(1)
            else:
                sel = np.isfinite(rf0) & (rf0 > 0)
            if not sel.any():
                continue
            with np.errstate(all="ignore"):
                for k in range(4):
                    Y[f"F{k + 1}"][i, c] = np.nanmedian(np.log2(res["formants"][idx[sel], k]))
                    Y[f"B{k + 1}"][i, c] = np.nanmedian(np.log2(res["bandwidths"][idx[sel], k]))
                Y["f0"][i, c] = np.nanmedian(np.log2(res["f0"][idx[sel]]))
                if "vtl_log2cm" in res:
                    Y["VTL"][i, c] = np.nanmedian(res["vtl_log2cm"][idx[sel]])
                if it["corpus"] == "hillenbrand":
                    err[i, c] = np.nanmean(np.abs(res["formants"][idx][:, :4] - rfm * r), 0)
                else:
                    f0 = res["f0"][idx[sel]]
                    pt[i, c] = [sel.sum(), np.sum(np.abs(1200 * np.log2(f0 / (rf0[sel] * r))) <= 50)]
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(items)}")
    return {"meta": meta, "Y": Y, "err": err, "pt": pt}


IDEAL = {**{f"F{k}": 1.0 for k in range(1, 5)}, **{f"B{k}": 1.0 for k in range(1, 5)}, "VTL": -1.0, "f0": 1.0}


def analyse_scores(res: dict) -> dict:
    meta, Y = res["meta"], res["Y"]
    items = meta["items"]
    out = {}
    for corpus in ("hillenbrand", "ptdb_tug"):
        sel = np.array([it["corpus"] == corpus for it in items])
        rs = meta["rates"][corpus]
        nc = len(rs)
        x = np.log2([r for _, r in rs])
        groups = np.array([it["speaker"] for it in items])[sel]
        for o in OUTS:
            y = Y[o][sel][:, :nc]
            if np.isfinite(y).sum() < 10 or (o == "f0" and corpus == "hillenbrand"):
                continue
            out[f"{corpus}/slope/{o}"] = M.summarise_slopes(y, x, groups).__dict__ | {"ideal": IDEAL[o]}
        ref = [c for c, (_, r) in enumerate(rs) if r == 1.0][0]
        for c, (sr, r) in enumerate(rs):
            if c == ref:
                continue
            if corpus == "hillenbrand":
                e = res["err"][sel]
                for k in range(4):
                    d = e[:, c, k] - e[:, ref, k]
                    ok = np.isfinite(d)
                    out[f"{corpus}/r{r:.3f}/F{k + 1}/abs_error_change_hz"] = list(M.group_bootstrap(d[ok], groups[ok], stat=np.nanmean))
            else:
                p = res["pt"][sel]
                rpa = lambda cc: p[:, cc, 1].sum() / max(np.nansum(p[:, cc, 0]), 1)  # noqa: E731
                out[f"{corpus}/r{r:.3f}/f0/rpa_change_points"] = float(100 * (rpa(c) - rpa(ref)))
        if corpus == "ptdb_tug":
            p = res["pt"][sel]
            out["ptdb_tug/clean/f0/rpa"] = float(100 * p[:, ref, 1].sum() / max(np.nansum(p[:, ref, 0]), 1))
        else:
            out["hillenbrand/clean/F/abs_error_hz"] = np.nanmean(res["err"][sel][:, ref], 0).tolist()
    return out


def report(name: str, a: dict) -> str:
    lines = [f"Uniform scale probe (D-050, probe 11): {name}",
             "slopes against log2 r: point [95 % speaker bootstrap]; shuffle null 2.5-97.5 %; ideal",
             f"clean: Hillenbrand |error| F1-F4 " + " / ".join(f"{v:.0f}" for v in a["hillenbrand/clean/F/abs_error_hz"])
             + f" Hz; PTDB-TUG RPA {a['ptdb_tug/clean/f0/rpa']:.1f} %"]
    for k, v in a.items():
        if "/slope/" in k:
            lines.append(f"  {k:28s} {v['point']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]  null [{v['null_low']:+.3f}, "
                         f"{v['null_high']:+.3f}]  ideal {v['ideal']:+.0f}")
    for k, v in a.items():
        if k.endswith("abs_error_change_hz"):
            lines.append(f"  {k:40s} {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] Hz")
        elif k.endswith("rpa_change_points"):
            lines.append(f"  {k:40s} {v:+.1f} points")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if argv[:1] == ["render"]:
        render()
        return 0
    if argv[:1] == ["score"]:
        REPORTS.mkdir(parents=True, exist_ok=True)
        for ckpt in argv[1:]:
            name = Path(ckpt).parent.name
            res = score(ckpt)
            np.savez_compressed(REPORTS / f"{name}_raw.npz", err=res["err"], pt=res["pt"],
                                **{f"Y_{o}": v for o, v in res["Y"].items()})
            a = analyse_scores(res)
            (REPORTS / f"{name}.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
            text = report(name, a)
            (REPORTS / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            print(text)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
