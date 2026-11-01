"""Golden clips: the synthesiser's output on a small branch- and draw-covering matrix, hashed exactly.

Any change to the audio, the labels or the meta of a covered case fails, so a refactor of the
generator cannot move a default silently. The matrix covers every branch of the generator, plus
the draw and length space that decides which branch a real clip takes: 1-3 s clips, fixed short
and long tracts, the population and f0-coupled length draws, and a fixed /h/ level. It is kept
honest by `test_cases_cover_every_config_field`: every `SynthConfig` / `KLSetConfig` field must be
either set by a case or listed in `IGNORED_FIELDS` with a reason, so adding an option forces a
decision instead of leaving a gap behind a green suite.

Deliberate regeneration (the only way it may change):

    python tests/test_synth_golden.py write --reason "D-064: whisper level fix"

The manifest records the commit, the environment (python / numpy / scipy / numba) and every
regeneration's reason. When the environment differs the test says so first and still fails: a
skip would be exactly the false confidence this file exists to prevent.

The cascade and tube cases are the frozen contract (train_v1, train_v2 and suite_v1 sit on
them). KL has never shipped, so its cases record current behaviour and may be regenerated
freely while it is still moving.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import fields as dc_fields
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from vocal_analysis.synth import channel as channel_mod
from vocal_analysis.synth import kl_set, sampler
from vocal_analysis.synth.channel import ChannelConfig
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import INT16_SCALE

MANIFEST = Path(__file__).with_name("data") / "synth_golden.json"
FORMAT = 1
BASE = SynthConfig(duration_s=(1.0, 1.0))

CHANNEL = dict(p_clean=0.0, p_rir=1.0, t60_s=(0.3, 0.3), p_noise=0.0, p_highpass=1.0, highpass_hz=(80.0, 80.0),
               p_eq=1.0, eq_gain_db=3.0, p_phone=0.0, p_clip=0.0, gain_db=(-1.0, -1.0), p_codec=0.0)

# Each case names the branches it must reach (`expect`): a case that silently stops exercising
# its branch is a gap in the matrix, so that is asserted before the hashes are compared.
CASES = [
    dict(name="cascade_default", engine="cascade", index=0, seed=101,
         expect=dict(components=("voiced", "aspiration"), not_components=("whisper", "frication"))),
    dict(name="tube_default", engine="tube", index=0, seed=102,
         expect=dict(components=("voiced", "aspiration"))),
    dict(name="kl_default", engine="kl", index=0, seed=103, kl_cfg={},
         expect=dict(components=("voiced", "aspiration"))),
    # aspiration tilt with HNR normalisation, and the whisper component (a whisper gap) with no tilt
    dict(name="cascade_asp_tilt", engine="cascade", index=1, seed=104,
         cfg=dict(asp_tilt_db_oct=-6.0, hnr_db=(10.0, 30.0), sonorant_includes_whisper=False),
         expect=dict(components=("aspiration", "whisper"))),
    # the /h/ onset path: whisper tilt set, every voiced segment led by an /h/
    dict(name="cascade_whisper_tilt", engine="cascade", index=2, seed=105,
         cfg=dict(whisper_tilt_db_oct=-12.0, p_h_onset=1.0),
         expect=dict(components=("whisper",), h_onset=True)),
    dict(name="tube_whisper_clip", engine="tube", index=1, seed=106,
         cfg=dict(p_whisper_clip=1.0),
         expect=dict(components=("whisper",), voiced_any=False)),
    dict(name="cascade_frication", engine="cascade", index=3, seed=107,
         cfg=dict(gap_kinds={"frication": 1.0}, gap_s=(0.3, 0.3), voiced_seg_s=(0.3, 0.3)),
         expect=dict(components=("frication",))),
    dict(name="cascade_coupled", engine="cascade", index=4, seed=108, cfg=dict(p_coupling=1.0)),
    dict(name="tube_return", engine="tube", index=2, seed=109,
         cfg=dict(interactive_return=True, interactive_sq=(1.5, 1.5))),  # a fixed speed quotient
    dict(name="tube_extras", engine="tube", index=3, seed=110, cfg=dict(p_nasal=1.0, p_subglottal=1.0)),
    dict(name="tube_articulation", engine="tube", index=4, seed=113, cfg=dict(articulation="sustained")),
    dict(name="kl_area_return", engine="kl", index=1, seed=111, kl_cfg=dict(area_return=True)),
    # the channel and the int16 quantisation write_set stores
    dict(name="cascade_channel", engine="cascade", index=5, seed=112, channel=CHANNEL, channel_seed=999),
    # draw and length breadth: the cases above are all 1 s at one drawn tract size, while the
    # D-059 grid the training sets came from runs 2-4 s over every tract length and both tract types
    dict(name="cascade_long", engine="cascade", index=0, seed=120, cfg=dict(duration_s=(3.0, 3.0)),
         expect=dict(components=("voiced", "aspiration"), min_segments=2)),
    dict(name="tube_long_tract", engine="tube", index=1, seed=121,
         cfg=dict(length_cm=(18.5, 18.5), p_length_from_f0=0.0), expect=dict(length_cm=18.5)),
    dict(name="cascade_short_tract", engine="cascade", index=2, seed=122,
         cfg=dict(length_cm=(11.5, 11.5), p_length_from_f0=0.0), expect=dict(length_cm=11.5)),
    # the two length branches that do not use `length_cm` uniformly
    dict(name="cascade_population", engine="cascade", index=3, seed=123,
         cfg=dict(length_dist="population", p_length_from_f0=0.0)),
    dict(name="cascade_f0_length", engine="cascade", index=4, seed=124,
         cfg=dict(p_length_from_f0=1.0, low_f0_hz=(120.0, 120.0), p_high_f0=0.0)),
    # the /h/ level draw with no tilt: the path the D-064 fix must leave alone
    dict(name="cascade_h_amp", engine="cascade", index=0, seed=125,
         cfg=dict(p_h_onset=1.0, h_amp=(3.0, 3.0)), expect=dict(h_onset=True, components=("whisper",))),
    dict(name="kl_long", engine="kl", index=2, seed=126, cfg=dict(duration_s=(1.5, 1.5)), kl_cfg={},
         expect=dict(components=("voiced", "aspiration"))),
]


# --------------------------------------------------------------------------- hashing


def _sha(x) -> str:
    """sha256 of the array's bytes and dtype; float arrays get NaN/inf canonicalised first, so a
    library change to the NaN payload cannot read as a regression."""
    a = np.asarray(x)
    if a.dtype.kind == "f":
        a = np.nan_to_num(a, nan=-1e30, posinf=1e30, neginf=-1e30)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes() + a.dtype.str.encode()).hexdigest()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return {"array": o.tolist(), "dtype": o.dtype.str}
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _sha_json(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=_jsonable).encode()).hexdigest()


def _sha_meta(meta: dict) -> str:
    # `config` is the whole SynthConfig as a dict: the case's overrides are the covered part and
    # the rest is the default, which every hash here pins anyway. Leaving it out keeps a new
    # config field from moving every meta hash for no behavioural reason.
    return _sha_json({k: v for k, v in meta.items() if k != "config"})


# --------------------------------------------------------------------------- rendering


def _clip(case: dict):
    cfg = replace(BASE, **case.get("cfg", {}))
    if case["engine"] == "kl":
        kcfg = replace(kl_set.KLSetConfig(synth=replace(cfg, p_tube=1.0)), **case.get("kl_cfg", {}))
        return kl_set.generate(case["index"], kcfg, case["seed"], keep_components=True)
    tube = case["engine"] == "tube"
    return sampler.generate(case["index"], replace(cfg, p_tube=1.0 if tube else 0.0, p_interactive=1.0 if tube else 0.0),
                            case["seed"], keep_components=True)


def _expect(case: dict, clip) -> None:
    """The case must reach the branches it claims, or the matrix has a silent gap."""
    exp = case.get("expect", {})
    for k in exp.get("components", ()):
        assert k in clip.components, f"{case['name']}: expected a {k} component, got {sorted(clip.components)}"
    for k in exp.get("not_components", ()):
        assert k not in clip.components, f"{case['name']}: unexpected {k} component"
    if "voiced_any" in exp:
        assert bool(clip.labels["voiced"].any()) is exp["voiced_any"], f"{case['name']}: voiced frames"
    if exp.get("h_onset"):
        assert any(s.get("kind") == "h_onset" for s in clip.meta["segments"]), f"{case['name']}: no h_onset segment"
    if "min_segments" in exp:
        n = len(clip.meta["segments"])
        assert n >= exp["min_segments"], f"{case['name']}: {n} segments, wanted at least {exp['min_segments']}"
    if "length_cm" in exp:
        got = float(clip.meta["length_cm"])
        assert abs(got - exp["length_cm"]) < 0.01, f"{case['name']}: tract length {got} cm, wanted {exp['length_cm']}"


def _record(case: dict) -> dict:
    clip = _clip(case)
    _expect(case, clip)
    assert np.isfinite(clip.audio).all(), f"{case['name']}: non-finite audio"
    rec = {
        "n": int(len(clip.audio)),
        "audio": _sha(clip.audio),
        "labels": {k: _sha(clip.labels[k]) for k in sorted(clip.labels)},
        "meta": _sha_meta(clip.meta),
        "components": {k: _sha(v) for k, v in sorted(clip.components.items())},
    }
    if "channel" in case:
        y, info = channel_mod.augment(clip.audio, np.random.default_rng(case["channel_seed"]),
                                      ChannelConfig(**case["channel"]))
        q = np.clip(np.round(y * INT16_SCALE), -32768, 32767).astype(np.int16)
        rec["channel"] = {"n": int(len(y)), "audio": _sha(q), "info": _sha_json(info)}
    return rec


# --------------------------------------------------------------------------- manifest


def _env() -> dict:
    import platform

    import numpy
    import scipy

    try:
        import numba

        nb = numba.__version__
    except ImportError:  # pragma: no cover - numba is a hard dependency of the synth
        nb = None
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "numba": nb}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _write(reason: str) -> None:
    old = _manifest() if MANIFEST.exists() else {}
    man = {
        "format": FORMAT,
        "written": date.today().isoformat(),
        "commit": _commit(),
        "env": _env(),
        "cases": {c["name"]: {**_record(c), "reason": reason} for c in CASES},
        "history": old.get("history", []) + [{"date": date.today().isoformat(), "commit": _commit(), "reason": reason}],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _diff(want, got, path="") -> list[str]:
    out = []
    if isinstance(want, dict) and isinstance(got, dict):
        for k in sorted(set(want) | set(got)):
            out += _diff(want.get(k, "<missing>"), got.get(k, "<missing>"), f"{path}.{k}" if path else k)
    elif want != got:
        out.append(f"{path}: manifest {want!r} != current {got!r}")
    return out


# --------------------------------------------------------------------------- coverage audit

# fields a case must not set itself: the engine wrapper pins them (cascade/tube/KL)
SET_BY_ENGINE = {"p_tube", "p_interactive"}


def _ignored(reason: str, *names: str) -> dict[str, str]:
    return {n: reason for n in names}


IGNORED_FIELDS = {
    **_ignored("f0, vibrato, jitter and shimmer draws: change the values, not a branch; the default cases "
               "hash their labels and source", "p_high_f0", "low_f0_hz", "high_f0_hz", "f0_limits_hz", "rd",
               "rd_wander_sd", "jitter", "shimmer_db", "vibrato_rate_hz", "vibrato_extent_cents",
               "p_vibrato_wide", "p_vibrato", "loudness_sd_db"),
    **_ignored("phrasing and contour draws: covered at their defaults by every case's labels and segments; the "
               "branches they gate (gaps, h_onset, whisper clip) are set explicitly where they matter",
               "hop_s", "target_every_s", "transition_s", "contour_probs_low", "contour_probs_high",
               "p_register_step"),
    **_ignored("tract prior and length draws: change values only; `tube_articulation` and `kl_area_return` cover "
               "the tract-side branches, and the length draws that select a branch are cases",
               "length_f0_slope_cm", "length_f0_sd_cm", "length_f0_cap_hz", "length_redraw", "vowel_dist",
               "p_uniform_formants", "bandwidth_f0_hz", "bw_factor", "tube_bw_factor", "p_f1_band",
               "f1_band_f0_hz", "f1_band_ratio", "p_cluster", "formant_constraint", "p_tube_constrict"),
    **_ignored("glottal and coupling scalars: values only; the coupling and interactive branches are cases, "
               "and the /h/ level draw has its own case", "glottal_peak_area_cm2", "glottal_gap_offset",
               "coupling_scale", "coupling_full_poles", "interactive_coupled", "p_interactive_coupled",
               "interactive_closure", "interactive_sub_gain"),
    **_ignored("KL-only draws and the sustained-articulation timing: values only, `kl_*` and "
               "`tube_articulation` cover their branches", "p_lf_pulse", "artic_hold_s", "artic_transition_s",
               "artic_pg_cm", "artic_pm_cm"),
}

IGNORED_KL_FIELDS = {
    **_ignored("values only: the waveguide geometry and its draws", "p_nasal_target", "port_cm2", "subglottal",
               "sq", "p_lf_pulse", "p_larynx", "p_lips", "oversample"),
}


def test_cases_cover_every_config_field():
    """A new option must be either exercised by a case or listed with a reason: otherwise the
    matrix can go stale while the suite stays green."""
    used = {"duration_s"} | SET_BY_ENGINE | {k for c in CASES for k in c.get("cfg", {})}
    names = {f.name for f in dc_fields(SynthConfig)}
    missing = names - used - set(IGNORED_FIELDS)
    stale = (used | set(IGNORED_FIELDS)) - names
    assert not missing, f"SynthConfig fields with no case and no reason: {sorted(missing)}"
    assert not stale, f"cases/IGNORED_FIELDS name fields that no longer exist: {sorted(stale)}"

    kl_used = {"synth"} | {k for c in CASES for k in c.get("kl_cfg", {})}
    kl_names = {f.name for f in dc_fields(kl_set.KLSetConfig)}
    kl_missing = kl_names - kl_used - set(IGNORED_KL_FIELDS)
    kl_stale = (kl_used | set(IGNORED_KL_FIELDS)) - kl_names
    assert not kl_missing, f"KLSetConfig fields with no case and no reason: {sorted(kl_missing)}"
    assert not kl_stale, f"kl_cfg/IGNORED_KL_FIELDS name fields that no longer exist: {sorted(kl_stale)}"


# --------------------------------------------------------------------------- the goldens


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_clip(case):
    man = _manifest()
    assert man["format"] == FORMAT, "the manifest predates this test's format; regenerate it"
    want = man["cases"].get(case["name"])
    assert want is not None, f"{case['name']} is not in the manifest: run write --reason ..."
    diffs = _diff({k: v for k, v in want.items() if k != "reason"}, _record(case))
    if diffs:
        note = ""
        if man["env"] != _env():
            note = (f"\nThe environment differs from the manifest's, which can move the last bits:\n"
                    f"  manifest: {man['env']}\n  current:  {_env()}")
        raise AssertionError(
            f"{case['name']} differs from the golden (written {man['written']} at {man['commit']}):\n  "
            + "\n  ".join(diffs[:12]) + note
            + "\nIf the change is intended, regenerate with a reason:\n"
              "  python tests/test_synth_golden.py write --reason \"D-xxx: what changed\"")


def test_manifest_lists_exactly_the_cases():
    man = _manifest()
    assert sorted(man["cases"]) == sorted(c["name"] for c in CASES)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write the golden-clip manifest.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="re-render every case and record its hashes")
    w.add_argument("--reason", required=True, help="why the goldens change; recorded in the manifest")
    args = ap.parse_args()
    if args.cmd == "write":
        _write(args.reason)
        print(f"{MANIFEST}: {len(CASES)} cases, {_env()}")
        sys.exit(0)
