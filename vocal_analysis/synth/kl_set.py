"""Labelled clips through the Kelly-Lochbaum waveguide (`kl.py`) with its subglottal and nasal branches (`branches.py`):
a time-domain reference beside the training synthesisers, for checking a model on physics it was not trained on.

`generate(index, cfg, seed)` has `sampler.generate`'s output (SynthClip: audio, labels, meta; written by
`shards.write_set(..., generator=kl_set.generate)`), and reuses its draws for the voice (f0 contour, Rd, tract length),
phrasing (voiced segments, whisper, frication, loudness, target times), speaker anatomy and articulatory targets
(Story's area functions, `tube.sample_target`). It leaves `sampler.generate` unchanged, so training sets reproduce.

Differences from the interactive tube clips:
- the tract is the waveguide, with the trachea below the glottis (the glottal flow solved with both loads) and a nasal
  branch at the speaker's port; the sampler's pole-zero extras are not used (both are physical here);
- each target is nasalised with chance p_nasal_target, its port opening log-uniform in port_cm2; the opening follows
  the articulation's transitions;
- larynx height and lip protrusion are drawn once per clip (the waveguide's section count is fixed within a clip);
- the source is the interactive source's area pulse (`interactive.glottal_area_track`), with aspiration and whisper
  as flow noise at a rigid glottis through the same waveguide;
- labels are the training definition on the waveguide's own discretised tube (`kl.frame_labels`: `tube.poles`, and
  with the port open the oral roots followed as it opens), at quarter-step keyframes along the articulatory path,
  interpolated in log frequency as `sampler._interactive_tract` does. A formant whose identity is a convention (it
  crossed a nasal pole, `branches.branched_poles`) is NaN in `formants` / `bandwidths` and True in `formants_mixed`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import branches, coupling, interactive, kl, sampler, tract, tube
from .sampler import N_LABEL_FORMANTS, N_SOURCE_HARMONICS, SynthClip
from .voice import CTRL, FS, Trajectories, tilted_noise


def _base() -> sampler.SynthConfig:
    return replace(sampler.SynthConfig(), p_tube=1.0, hnr_db=(10.0, 35.0))


@dataclass
class KLSetConfig:
    synth: sampler.SynthConfig = field(default_factory=_base)  # voice, phrasing, speaker and target draws
    p_nasal_target: float = 0.25
    port_cm2: tuple[float, float] = (0.05, 1.0)  # log-uniform opening of a nasalised target
    subglottal: bool = True
    sq: tuple[float, float] = (0.9, 2.0)  # Rosenberg speed quotient, uniform per clip
    p_lf_pulse: float = 0.3  # per clip: the LF-derived area pulse instead
    # Rosenberg pulses close with an exponential return in area at LF's Ta for the clip's Rd (`interactive.area_pulse`
    # ret; D-062), instead of the quarter-cosine's corner. No draw; False reproduces earlier sets
    area_return: bool = False
    p_larynx: float = 0.3  # per clip: a larynx height change (tube.PG_RANGE, scaled)
    p_lips: float = 0.3  # per clip: a lip protrusion change (tube.PM_RANGE, scaled)
    oversample: int = kl.OVERSAMPLE

    @property
    def hop_s(self) -> float:
        return self.synth.hop_s


def _state(p: float, shapes: list, ports: np.ndarray) -> tuple[tube.TubeShape, float]:
    """The tract at fractional target index p: area function blended as the interactive source does, port linear."""
    j = int(min(np.floor(p), len(shapes) - 1))
    w = p - j
    if j >= len(shapes) - 1 or w <= 0.0:
        return shapes[j], float(ports[j])
    return interactive.blend_shapes(shapes[j], shapes[j + 1], w), float((1 - w) * ports[j] + w * ports[j + 1])


def generate(index: int, cfg: KLSetConfig | None = None, seed: int = 0, keep_components: bool = False) -> SynthClip:
    cfg = cfg or KLSetConfig()
    sc, os = cfg.synth, cfg.oversample
    v = sampler._voice_and_length(index, sc, seed)
    rng, n, f0, rd, length = v.rng, v.n, v.f0, v.rd, v.length
    voice_env, whisper, fric, h_env, segments, _, loud, tt = sampler._phrasing(rng, sc, n)
    T = len(tt)

    spk = tube.sample_speaker(rng, length)
    glottis = coupling.sample_glottis(rng, length, sc.glottal_peak_area_cm2, sc.glottal_gap_offset)
    sq = None if rng.random() < cfg.p_lf_pulse else float(rng.uniform(*cfg.sq))
    rel = length / tube.STORY_LENGTH_CM
    pg = float(rng.uniform(*tube.PG_RANGE) * rel) if rng.random() < cfg.p_larynx else 0.0
    pm = float(rng.uniform(*tube.PM_RANGE) * rel) if rng.random() < cfg.p_lips else 0.0
    q_draw = (tube.vowel_q_sampler(tube.CONVERSATIONAL_VOWELS, tube.vowel_centres(length))
              if sc.vowel_dist == "conversational" else None)
    drawn = [tube.sample_target(rng, spk, tract.N_POLES, tract.F_MAX_HZ, sc.p_tube_constrict, q_draw=q_draw,
                                pg_cm=pg, pm_cm=pm) for _ in range(T)]
    shapes = [sampler._target_shape(spk, d[2]) for d in drawn]
    ports = np.array([float(np.exp(rng.uniform(*np.log(cfg.port_cm2)))) if rng.random() < cfg.p_nasal_target else 0.0
                      for _ in range(T)])
    dx = kl.section_length(os)
    nas = branches.nasal(length, dx, rng)
    sub = branches.subglottal(length, dx, rng) if cfg.subglottal else None
    port_cm = branches.port_position_cm(shapes[0], rng)
    nsect = kl.n_sections(shapes, os)
    widths = rng.uniform(*sc.transition_s, T - 1) * 1000
    path = sampler._raised_cosine_steps(n, np.array(tt[1:]), np.arange(T, dtype=float), widths)  # per control point

    # labels: quarter-step keyframes along the path, log-interpolated (as sampler._interactive_tract)
    kps = sampler.KEYFRAMES_PER_STEP
    n_kf = min(int(np.floor(path.max() * kps + 1e-9)) + 2, (T - 1) * kps + 1)
    kf_f = np.full((n_kf, tract.N_POLES), np.nan)
    kf_b = np.full((n_kf, tract.N_POLES), np.nan)
    kf_m = np.zeros((n_kf, tract.N_POLES), bool)
    for q in range(n_kf):
        shape, port = _state(q / kps, shapes, ports)
        try:
            (F, B, mixed), = kl.frame_labels([shape], os, nasal=nas, port_cm=port_cm, port_cm2=port, n_sect=nsect)
        except RuntimeError:  # rare: this keyframe's labels stay NaN (masked)
            continue
        k = min(len(F), tract.N_POLES)
        kf_f[q, :k], kf_b[q, :k], kf_m[q, :k] = F[:k], B[:k], mixed[:k]
    hop_ctrl = int(round(sc.hop_s * 1000))
    fi = np.arange(0, n - 1, hop_ctrl)
    pk = path[fi] * kps
    block = np.minimum(np.floor(pk).astype(np.int64), n_kf - 1)
    w = np.where(block < n_kf - 1, pk - block, 0.0)
    nxt = np.minimum(block + 1, n_kf - 1)
    formants = np.exp((1 - w)[:, None] * np.log(kf_f[block]) + w[:, None] * np.log(kf_f[nxt]))
    bws = np.exp((1 - w)[:, None] * np.log(kf_b[block]) + w[:, None] * np.log(kf_b[nxt]))
    mixed = kf_m[block] | (kf_m[nxt] & (w > 0)[:, None])
    formants[mixed], bws[mixed] = np.nan, np.nan

    # source and render
    whisper_amp = whisper * loud * rng.uniform(0.3, 1.5) + (h_env * loud * rng.uniform(*sc.h_amp) if h_env.any() else 0.0)
    traj = Trajectories(f0=f0, rd=rd, voice_amp=voice_env * loud, whisper_amp=whisper_amp,
                        fric_amp=fric * loud * rng.uniform(0.3, 1.5), poles_f=np.zeros((n, 0)), poles_b=np.zeros((n, 0)),
                        zeros_f=np.zeros((n, 0)), zeros_b=np.zeros((n, 0)))
    jitter, shimmer = rng.uniform(*sc.jitter), rng.uniform(*sc.shimmer_db)
    hnr = None if sc.hnr_db is None else rng.uniform(*sc.hnr_db)
    src = interactive.glottal_area_track(traj, glottis, sq, rng, jitter, shimmer, FS, ret=cfg.area_return)
    ns = traj.n_samples
    frames = ns // kl.HOP + 1
    ms = np.minimum(np.arange(frames) * kl.HOP // CTRL, n - 1)
    states = [_state(p, shapes, ports) for p in path[ms]]
    common = dict(oversample=os, subglottal=sub, nasal=nas, port_cm=port_cm, port_cm2=np.array([s[1] for s in states]),
                  n_sect=nsect)
    frame_shapes = [s[0] for s in states]
    args = (frame_shapes, src["area"], glottis.pressure, glottis.depth_cm, glottis.length_cm)
    # KL audio is d/dt of the radiated volume velocity; / FS gives the interactive source's per-sample difference
    out = {"voiced": kl.render(*args, **common)["audio"][:ns] / FS * src["shimmer"][:ns], "cycles": src["cycles"]}
    va = src["va"][:ns]
    if hnr is not None:
        # the waveguide radiates (d/dt) its flow, so the cascade's source tilt t is a flow tilt of t - 6 dB/oct; None:
        # the first difference of white noise as flow (radiated, about +12 dB/oct: every set before D-062)
        white = rng.standard_normal(ns + 1)
        t = sc.asp_tilt_db_oct
        noise = (np.diff(white) if t is None else tilted_noise(white[:ns], t - 6.0, FS)) * (0.25 + 0.75 * src["pulse"][:ns]) * va
        asp = kl.render(*args, flow=noise, **common)["audio"][:ns] / FS
        on = va > 0.5 * va.max() if va.max() > 0 else np.zeros(ns, bool)
        if on.any():
            ev, ea = np.mean(out["voiced"][on] ** 2), np.mean(asp[on] ** 2)
            if ea > 0:
                asp *= np.sqrt(ev / ea / 10 ** (hnr / 10))
        out["aspiration"] = asp
    from .voice import _per_sample

    wa = _per_sample(traj.whisper_amp)[:ns]
    if wa.any():
        white = rng.standard_normal(ns + 1)
        t = sc.whisper_tilt_db_oct  # as the aspiration's: a flow tilt of t - 6 (the waveguide radiates)
        noise = np.diff(white) if t is None else tilted_noise(white[:ns], t - 6.0, FS)
        out["whisper"] = kl.render(*args, flow=noise * wa, **common)["audio"][:ns] / FS * 0.05
    fa = _per_sample(traj.fric_amp)[:ns]
    if fa.any():
        from scipy.signal import butter, sosfilt

        sos = butter(4, rng.uniform(2000, 5000), "highpass", fs=FS, output="sos")
        out["frication"] = sosfilt(sos, rng.standard_normal(ns)) * fa * 0.05
    out["total"] = sum(x for k, x in out.items() if k != "cycles")
    audio = out["total"]
    gain = 0.5 / (np.max(np.abs(audio)) + 1e-12)
    audio = audio * gain

    voiced = voice_env[fi] >= 0.5
    whispered = np.maximum(whisper, h_env)[fi] >= 0.5
    labels = {
        "t": fi / 1000.0,
        "f0": f0[fi],
        "voiced": voiced,
        "sonorant": voiced | whispered if sc.sonorant_includes_whisper else voiced,
        "whisper": whispered,
        "formants": formants[:, :N_LABEL_FORMANTS],
        "bandwidths": bws[:, :N_LABEL_FORMANTS],
        "rd": rd[fi],
        "source_h_db": np.full((len(fi), N_SOURCE_HARMONICS), np.nan),  # not an LF source
        "length_cm": np.full(len(fi), length),
        "length_inst_cm": np.full(len(fi), nsect * dx),  # the waveguide's tube, fixed within the clip
        "formants_mixed": mixed[:, :N_LABEL_FORMANTS],
    }
    meta = {
        "index": index, "seed": seed, "fs": FS, "duration_s": len(audio) / FS, "length_cm": length,
        "length_group": v.length_group, "length_from_f0": v.length_from_f0, "contour": v.contour,
        "register_steps": v.register_steps, "segments": segments, "jitter": jitter, "shimmer_db": shimmer,
        "hnr_db": hnr, "gain": gain, "tract": "kl",
        "kl": {"speaker": {k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in asdict(spk).items()},
               "targets": [d[2] for d in drawn], "glottis": asdict(glottis), "pulse": "lf" if sq is None else "rosenberg", "area_return": bool(cfg.area_return and sq is not None),
               "area_sq": sq, "pg_cm": pg, "pm_cm": pm, "n_sections": nsect, "oversample": os,
               "port_cm": port_cm, "port_cm2_targets": ports.tolist(),
               "nasal": {k: (x.tolist() if isinstance(x, np.ndarray) else x) for k, x in asdict(nas).items()},
               "subglottal": None if sub is None else {k: (x.tolist() if isinstance(x, np.ndarray) else x)
                                                       for k, x in asdict(sub).items()}},
        "config": asdict(cfg),
    }
    return SynthClip(audio.astype(np.float32), labels, meta, out if keep_components else None)
