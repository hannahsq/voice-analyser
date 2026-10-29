"""Channel probe (D-050, probe 3): real dev gold audio degraded by fixed channels and paired against the clean file.

Hillenbrand dev (every file; formant references at steady state and 20 / 50 / 80 %): each formant's change against
the clean file (invariance, ideal 0), the change in its absolute error against the hand-checked reference, and the
change in the model's own sigma (where evidence is lost, sigma should grow rather than the estimate drift).
PTDB-TUG dev (PTDB_PER_SPEAKER files per speaker, spread through the list): f0 RPA (50 cents) and voicing recall on
the reference's voiced frames, against the clean file.

Conditions are deterministic (each draws from a generator seeded by the file and the condition) and cached as int16
under data/probes/channel_v1/, so every model hears the same audio. Gold audio is only evaluated here, on dev speakers.

    python -m vocal_analysis.eval.probes.channel render
    python -m vocal_analysis.eval.probes.channel score <ckpt> [...]   ($VA_DEVICE=cuda)
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt

from ...data.manifest import read_manifest
from ...synth import channel as ch
from ...synth import codecs
from ..datasets import load_audio, registry
from ..tiers import filter_rows
from . import measures as M

FS = 16000
CACHE = Path("data/probes/channel_v1")
REPORTS = Path("reports/probes/channel")
PTDB_PER_SPEAKER = 20


def _noise(kind: str, snr_db: float):
    def f(x, rng):
        n = ch._noise(rng, kind, len(x), FS)
        p_x, p_n = np.mean(x**2), np.mean(n**2) + 1e-20  # whole-file power (files are mostly speech)
        return x + n * np.sqrt(p_x / p_n / 10 ** (snr_db / 10))
    return f


def _codec(codec: str, kbps: float):
    def f(x, rng):
        peak = np.max(np.abs(x)) + 1e-12
        return codecs.apply(x / peak * 0.9, codec, kbps, stream=False) * peak / 0.9
    return f


def _eq(f0: float, gain_db: float, q: float = 1.0):
    return lambda x, rng: sosfilt(ch._biquad_peaking(f0, gain_db, q), x)


def _reverb(t60: float):
    return lambda x, rng: fftconvolve(x, ch.synth_rir(rng, t60))[: len(x)]


CONDITIONS = {
    "clean": lambda x, rng: x,
    **{f"codec_{c}_{k:g}": _codec(c, k) for c, k in (("opus", 16), ("aac", 32), ("mp3", 32), ("amr_wb", 12.65), ("amr_nb", 7.4))},
    "phone": lambda x, rng: ch._phone(x),
    "lowpass_4k": lambda x, rng: sosfiltfilt(butter(8, 4000.0, fs=FS, output="sos"), x),
    "eq_1k_+6": _eq(1000.0, 6.0), "eq_1k_-6": _eq(1000.0, -6.0), "eq_3k_+6": _eq(3000.0, 6.0),
    **{f"reverb_{t:g}": _reverb(t) for t in (0.2, 0.4, 0.8)},
    **{f"noise_{k}_{s}": _noise(k, s) for k in ("white", "pink", "babble") for s in (30, 20, 10, 5)},
}


def _rows(corpus: str) -> list:
    return filter_rows(read_manifest(Path("data/manifests") / f"{corpus}.csv"), ("dev",))


def render(log=print) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    reg = registry()
    rows = {"hillenbrand": _rows("hillenbrand")}
    by_spk: dict[str, list] = {}
    for r in _rows("ptdb_tug"):
        by_spk.setdefault(r.speaker_id, []).append(r)
    rows["ptdb_tug"] = [rs[int(i)] for _, rs in sorted(by_spk.items())
                        for i in np.linspace(0, len(rs) - 1, min(PTDB_PER_SPEAKER, len(rs)))]
    meta = {"conditions": list(CONDITIONS), "items": []}
    for corpus, rs in rows.items():
        for k, r in enumerate(rs):
            item = reg[corpus].load(Path("data") / corpus, r)
            x = load_audio(item).astype(float)
            arrays = {}
            for c, (name, fn) in enumerate(CONDITIONS.items()):
                rng = np.random.default_rng([zlib.crc32(r.path.encode()), c])
                y = np.asarray(fn(x, rng), float)
                peak = np.max(np.abs(y)) + 1e-12
                arrays[f"audio_{c}"] = np.round(y / peak * 32000).astype(np.int16)  # each condition at its own peak
            arrays["times"] = item.times
            arrays["ref_f0"] = item.f0 if item.f0 is not None else np.full(len(item.times), np.nan)
            arrays["ref_formants"] = item.formants if item.formants is not None else np.full((len(item.times), 4), np.nan)
            name = f"{corpus}_{k:04d}.npz"
            np.savez_compressed(CACHE / name, **arrays)
            meta["items"].append({"file": name, "corpus": corpus, "speaker": f"{corpus}:{r.speaker_id}",
                                  "group": item.meta.get("group")})
            if (k + 1) % 200 == 0:
                log(f"  {corpus} {k + 1}/{len(rs)}")
        log(f"{corpus}: {len(rs)} files")
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return CACHE


def score(ckpt: str, log=print) -> dict:
    """Per item and condition: Hillenbrand, log2 F1-F4, |error| (Hz) and sigma at the reference frames (median over
    them); PTDB-TUG, counts of reference-voiced frames, within 50 cents, and judged voiced."""
    import torch

    from ...models.infer import analyse, load_teacher

    meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
    dev = torch.device(os.environ.get("VA_DEVICE", "cpu"))
    model = load_teacher(ckpt, dev)[0]
    nc = len(meta["conditions"])
    items = meta["items"]
    hf = np.full((len(items), nc, 4), np.nan)
    he, hs = hf.copy(), hf.copy()
    pt = np.full((len(items), nc, 3), np.nan)
    for i, it in enumerate(items):
        d = np.load(CACHE / it["file"])
        times, rf0, rfm = d["times"], d["ref_f0"], d["ref_formants"]
        for c in range(nc):
            r = analyse(model, d[f"audio_{c}"].astype(np.float32) / 32000, dev)
            idx = np.clip(np.round(times / 0.01).astype(int), 0, len(r["f0"]) - 1)
            if it["corpus"] == "hillenbrand":
                est, sig = r["formants"][idx, :4], r["formant_sigma_cents"][idx, :4]
                ok = np.isfinite(rfm[:, :4])
                with np.errstate(all="ignore"):
                    hf[i, c] = np.nanmedian(np.where(ok, np.log2(est), np.nan), 0)
                    he[i, c] = np.nanmedian(np.where(ok, np.abs(est - rfm[:, :4]), np.nan), 0)
                    hs[i, c] = np.nanmedian(np.where(ok, sig, np.nan), 0)
            else:
                v = np.isfinite(rf0) & (rf0 > 0)
                f0, vp = r["f0"][idx], r["voiced_p"][idx]
                pt[i, c] = [v.sum(), np.sum(v & (np.abs(1200 * np.log2(f0 / np.where(v, rf0, 1.0))) <= 50)), np.sum(v & (vp > 0.5))]
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(items)}")
    return {"conds": meta["conditions"], "items": items, "hf": hf, "he": he, "hs": hs, "pt": pt}


def analyse_scores(res: dict) -> dict:
    conds, items = res["conds"], res["items"]
    hil = np.array([it["corpus"] == "hillenbrand" for it in items])
    spk = np.array([it["speaker"] for it in items])
    out = {}
    for c, name in enumerate(conds[1:], 1):
        for k in range(4):
            d = res["hf"][hil, c, k] - res["hf"][hil, 0, k]
            e = res["he"][hil, c, k] - res["he"][hil, 0, k]
            s = np.log2(res["hs"][hil, c, k] / res["hs"][hil, 0, k])
            ok = np.isfinite(d)
            if ok.sum() > 10:
                out[f"{name}/F{k + 1}/change_cents"] = [1200 * v for v in M.group_bootstrap(d[ok], spk[hil][ok], stat=np.nanmedian)]
                out[f"{name}/F{k + 1}/abs_error_change_hz"] = list(M.group_bootstrap(e[ok], spk[hil][ok], stat=np.nanmean))
                out[f"{name}/F{k + 1}/sigma_log2_change"] = float(np.nanmedian(s[ok]))
        p = res["pt"][~hil]
        rpa = lambda cc: p[:, cc, 1].sum() / max(p[:, cc, 0].sum(), 1)  # noqa: E731
        rec = lambda cc: p[:, cc, 2].sum() / max(p[:, cc, 0].sum(), 1)  # noqa: E731
        out[f"{name}/f0/rpa_change_points"] = 100 * (rpa(c) - rpa(0))
        out[f"{name}/f0/voicing_recall_change_points"] = 100 * (rec(c) - rec(0))
    p = res["pt"][~hil]
    out["clean/f0/rpa"] = 100 * p[:, 0, 1].sum() / max(p[:, 0, 0].sum(), 1)
    out["clean/F/abs_error_hz"] = np.nanmean(res["he"][hil, 0, :], 0).tolist()
    return out


def report(name: str, a: dict, n_h: int, n_p: int) -> str:
    lines = [f"Channel probe (D-050, probe 3): {name}",
             f"Hillenbrand dev {n_h} files; PTDB-TUG dev {n_p} files. Clean: |error| F1-F4 "
             + " / ".join(f"{v:.0f}" for v in a["clean/F/abs_error_hz"]) + f" Hz; PTDB RPA {a['clean/f0/rpa']:.1f} %",
             "per condition: F1-F4 change against clean, cents (median, [95 % speaker bootstrap]); mean |error| change, Hz;"
             " sigma log2 change; PTDB RPA and voicing recall change, points"]
    for c in [k.split("/")[0] for k in a if k.endswith("/f0/rpa_change_points")]:
        fs = " ".join(f"F{k}{a[f'{c}/F{k}/change_cents'][0]:+.0f}[{a[f'{c}/F{k}/change_cents'][1]:+.0f},{a[f'{c}/F{k}/change_cents'][2]:+.0f}]"
                      for k in range(1, 5) if f"{c}/F{k}/change_cents" in a)
        es = "/".join(f"{a[f'{c}/F{k}/abs_error_change_hz'][0]:+.0f}" for k in range(1, 5) if f"{c}/F{k}/abs_error_change_hz" in a)
        ss = "/".join(f"{a[f'{c}/F{k}/sigma_log2_change']:+.2f}" for k in range(1, 5) if f"{c}/F{k}/sigma_log2_change" in a)
        lines.append(f"  {c:18s} {fs} | err {es} Hz | sigma {ss} | f0 RPA {a[f'{c}/f0/rpa_change_points']:+.1f}, "
                     f"voicing {a[f'{c}/f0/voicing_recall_change_points']:+.1f}")
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
            np.savez(REPORTS / f"{name}_raw.npz", hf=res["hf"], he=res["he"], hs=res["hs"], pt=res["pt"])
            a = analyse_scores(res)
            (REPORTS / f"{name}.json").write_text(json.dumps(a, indent=1), encoding="utf-8")
            n_h = sum(it["corpus"] == "hillenbrand" for it in res["items"])
            text = report(name, a, n_h, len(res["items"]) - n_h)
            (REPORTS / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            print(text)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
