import numpy as np

from vocal_analysis.synth import observability as ob
from vocal_analysis.synth import sampler as smp

F = np.array([[500.0, 1500.0, 2500.0, 3500.0]])
B = np.array([[60.0, 90.0, 120.0, 180.0]])
FRAME = ob.ObservabilityConfig(context_s=0.0)


def flat(level_db):  # signal follows the formants over a flat floor; level_db: SNR off-resonance
    return lambda t, f: level_db + ob.resonance_db(f[:, None], F[0], B[0]).sum(1) - ob.resonance_db(f[:, None], F[0], B[0]).sum(1).mean()


def run(f0, snr, voiced=True, sonorant=True, cfg=FRAME):
    return ob.observability(np.array([f0]), np.array([voiced]), np.array([sonorant]), F, B, snr, cfg)[0][0]


def test_sparse_harmonics_lower_f1():
    assert run(600.0, flat(30))[0] < run(100.0, flat(30))[0] - 0.2


def test_noise_lowers_observability():
    quiet, loud = run(150.0, flat(-40)), run(150.0, flat(40))
    assert np.all(quiet <= loud + 1e-9) and quiet[1] < loud[1] - 0.2


def test_low_pass_removes_formants_above_cutoff():
    lp = lambda t, f: np.where(f < 3000.0, flat(30)(t, f), -80.0)  # noqa: E731
    full, cut = run(120.0, flat(30)), run(120.0, lp)
    assert cut[3] < 0.1 < full[3]
    assert abs(cut[0] - full[0]) < 0.05


def test_absent_and_non_sonorant_are_nan():
    assert np.all(np.isnan(run(150.0, flat(30), voiced=False, sonorant=False)))
    Fm = F.copy()
    Fm[0, 3] = np.nan
    o, _ = ob.observability(np.array([150.0]), np.array([True]), np.array([True]), Fm, B, flat(30), FRAME)
    assert np.isnan(o[0, 3]) and np.all(np.isfinite(o[0, :3]))


def test_context_pools_neighbouring_frames():
    T = 21
    f0 = np.linspace(560.0, 640.0, T)  # harmonics sweep past F1
    args = (f0, np.ones(T, bool), np.ones(T, bool), np.repeat(F, T, 0), np.repeat(B, T, 0), flat(30))
    alone = ob.observability(*args, FRAME)[0][:, 0]
    pooled = ob.observability(*args, ob.ObservabilityConfig(context_s=0.05))[0][:, 0]
    assert np.all(pooled >= alone - 1e-9) and np.median(pooled) > np.median(alone)


def test_component_snr_on_a_sampler_clip():
    cap = {}
    real = smp.synthesize

    def keep(*a, **k):
        cap["c"] = real(*a, **k)
        return cap["c"]

    smp.synthesize = keep
    try:
        c = smp.generate(3, smp.SynthConfig(p_tube=0.0), 5)
    finally:
        smp.synthesize = real
    L = c.labels
    snr = ob.component_snr(cap["c"], L["t"], L["voiced"])
    o, sd = ob.observability(L["f0"], L["voiced"], L["sonorant"], L["formants"][:, :4], L["bandwidths"][:, :4], snr)
    son = L["sonorant"] & np.all(np.isfinite(L["formants"][:, :4]), 1)
    assert son.any() and np.all(np.isfinite(o[son])) and np.all((o[son] > 0) & (o[son] <= 1))
    assert np.all(np.isnan(o[~L["sonorant"]]))


def _clip_with_components(index=3, cfg=None):
    cap = {}
    real = smp.synthesize

    def keep(*a, **k):
        cap["c"] = real(*a, **k)
        return cap["c"]

    smp.synthesize = keep
    try:
        c = smp.generate(index, cfg or smp.SynthConfig(p_tube=0.0), 5)
    finally:
        smp.synthesize = real
    return c, cap["c"]


def test_store_quantisation_roundtrip():
    from vocal_analysis.synth import obs_store as st

    db = np.array([0.0, -0.26, -37.4, -130.0, -np.inf])
    back = st.dequantise(st.quantise(db))
    assert np.allclose(back[:3], db[:3], atol=0.25) and back[3] == -st.Q_MAX * st.Q_STEP and back[4] == -np.inf


def test_store_clean_matches_component_snr():
    from vocal_analysis.synth import obs_store as st

    c, comps = _clip_with_components()
    L = c.labels
    s = st.clean_spectra(comps, L["t"], L["voiced"], L["f0"])
    cs = st.ClipSpectra(s["sig"], s["noise"], s["peak"], L["voiced"], L["f0"]).channel({"clean": True})
    ref = ob.component_snr(comps, L["t"], L["voiced"])
    args = (L["f0"], L["voiced"], L["sonorant"], L["formants"][:, :4], L["bandwidths"][:, :4])
    o_store, _ = ob.observability(*args, cs.snr_db_at)
    o_ref, _ = ob.observability(*args, ref)
    m = np.isfinite(o_store) & np.isfinite(o_ref)
    assert m.any() and np.median(np.abs(o_store[m] - o_ref[m])) < 0.02


def test_store_channel_low_pass_and_noise():
    from vocal_analysis.synth import obs_store as st

    c, comps = _clip_with_components()
    L = c.labels
    s = st.clean_spectra(comps, L["t"], L["voiced"], L["f0"])
    args = (L["f0"], L["voiced"], L["sonorant"], L["formants"][:, :4], L["bandwidths"][:, :4])

    def o(params):
        cs = st.ClipSpectra(s["sig"], s["noise"], s["peak"], L["voiced"], L["f0"]).channel(params)
        return ob.observability(*args, cs.snr_db_at)[0]

    clean = o({"clean": True})
    phone = o({"clean": False, "phone": True, "gain_db": 0.0})
    noisy = o({"clean": False, "noise": "white", "snr_db": 0.0, "gain_db": 0.0})
    f4 = np.isfinite(clean[:, 3]) & (L["formants"][:, 3] > 3600)
    assert f4.any() and np.median(phone[f4, 3]) < 0.1 < np.median(clean[f4, 3])
    m = np.isfinite(clean) & np.isfinite(noisy)
    assert np.all(noisy[m] <= clean[m] + 1e-6) and np.median(clean[m] - noisy[m]) > 0.05
