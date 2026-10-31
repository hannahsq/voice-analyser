import numpy as np
import pytest

from vocal_analysis.synth import branches as br
from vocal_analysis.synth import kl, tube

DX = kl.section_length()


def _oral(vowel: str, **kw):
    s = tube.build_shape(*tube.STORY_VOWELS[vowel], 17.5, **kw)
    a = kl.uniform_areas(s)
    return s, tube.TubeShape(a, np.full(len(a), DX))


def test_subglottal_resonances_are_calibrated_and_scale_with_size():
    """About 620 / 1390 / 2210 Hz at Story's speaker's size (adult men's reported values); a shorter tract's are
    higher by the size ratio."""
    F, _ = br.subglottal_resonances(br.subglottal(tube.STORY_LENGTH_CM, DX))
    assert np.all(np.abs(F[:3] / np.array([620.0, 1390.0, 2210.0]) - 1) < 0.05)
    Fs, _ = br.subglottal_resonances(br.subglottal(14.5, DX))
    assert np.all(np.abs(Fs[:3] / F[:3] / (tube.STORY_LENGTH_CM / 14.5) - 1) < 0.05)


def test_port_follows_the_pharynx():
    """The port sits at the pharynx / oral boundary, so a longer pharynx (same total length) puts it higher; the
    per-speaker jitter is bounded."""
    lo = br.port_position_cm(tube.build_shape(*tube.STORY_VOWELS["A"], 17.5, pharynx_ratio=0.85))
    hi = br.port_position_cm(tube.build_shape(*tube.STORY_VOWELS["A"], 17.5, pharynx_ratio=1.15))
    assert hi > lo + 1.0
    s = tube.build_shape(*tube.STORY_VOWELS["A"], 17.5)
    rng = np.random.default_rng(0)
    x = np.array([br.port_position_cm(s, rng) for _ in range(200)]) - br.port_position_cm(s)
    assert np.abs(x).max() <= br.PORT_JITTER_CM and x.std() > 0.1


def test_closed_port_leaves_the_oral_poles():
    s, oral = _oral("A")
    j = br.port_junction(br.port_position_cm(s), len(oral.areas), DX)
    fo, bo, fn, _, _ = br.branched_poles(oral, j, 0.0, br.nasal(17.5, DX))
    ft, bt = tube.poles(oral, 5000.0)
    assert np.allclose(fo, ft) and np.allclose(bo, bt) and len(fn) == 0


@pytest.mark.parametrize("vowel", ["A", "i", "u"])
def test_waveguide_matches_the_branched_model(vowel):
    """With the port open, the waveguide's measured poles (three-way junction, nasal passage, sinus, nostrils) are the
    continuous branched model's, oral and nasal, within 1 % (below 4.5 kHz; a closed vowel's broad F1 within 4 %,
    as without the nose)."""
    s, oral = _oral(vowel)
    nas, pc = br.nasal(17.5, DX), br.port_position_cm(s)
    (fk, _), = kl.frame_poles([s], nasal=nas, port_cm=pc, port_cm2=0.3, f_max_hz=5000.0)
    fo, _, fn, _, _ = br.branched_poles(oral, br.port_junction(pc, len(oral.areas), DX), 0.3, nas)
    ft = np.sort(np.r_[fo, fn])
    ft = ft[ft < 4500.0]
    assert len(fk) >= len(ft)
    err = np.abs(fk[: len(ft)] / ft - 1)
    assert err[0] < 0.04 and np.all(err[1:] < 0.01)


def test_entangled_labels_go_to_the_more_oral_root():
    """Where an oral and a nasal root are entangled (/u/'s F1 crosses the nose's lowest pole; /A/'s F1 starts degenerate
    with a nasal pole), the label sits on the root with the larger oral energy share, and is mixed only when neither is
    clearly oral. Unentangled labels are not mixed."""
    nas = br.nasal(17.5, DX)
    for vowel in ["u", "A", "i"]:
        s, oral = _oral(vowel)
        j = br.port_junction(br.port_position_cm(s), len(oral.areas), DX)
        fo, bo, fn, bn, mixed = br.branched_poles(oral, j, 0.3, nas)
        phi_o = [br.oral_fraction(-np.pi * b + 2j * np.pi * f, oral, j, 0.3, nas) for f, b in zip(fo[:3], bo[:3])]
        phi_n = [br.oral_fraction(-np.pi * b + 2j * np.pi * f, oral, j, 0.3, nas) for f, b in zip(fn, bn)]
        for k in range(3):
            if not mixed[k]:
                assert phi_o[k] >= br.ORAL_CLEAR
        assert np.median(phi_n) < 0.5  # the nose's own poles are mostly nasal


def test_trachea_passes_dc_and_the_nose_radiates():
    n = 8000
    frames = n // kl.HOP + 1
    shapes = [tube.build_shape(*tube.STORY_VOWELS["A"], 17.5)] * frames
    ag = np.full(n, 0.1)
    plain = kl.render(shapes, ag, 8000.0, 0.3, 1.0)
    sub = kl.render(shapes, ag, 8000.0, 0.3, 1.0, subglottal=br.subglottal(17.5, DX))
    assert abs(sub["ug"][-2000:].mean() / plain["ug"][-2000:].mean() - 1) < 0.01  # the lungs drive DC whole
    ag = kl.glottal_area(np.full(frames, 120.0), n, kl.FS, 0.12)
    s = shapes[0]
    closed = kl.render(shapes, ag, 8000.0, 0.3, 1.0, nasal=br.nasal(17.5, DX), port_cm=br.port_position_cm(s))
    opened = kl.render(shapes, ag, 8000.0, 0.3, 1.0, nasal=br.nasal(17.5, DX), port_cm=br.port_position_cm(s),
                       port_cm2=0.5)
    base = kl.render(shapes, ag, 8000.0, 0.3, 1.0)
    assert np.allclose(closed["audio"], base["audio"], atol=1e-6 * np.abs(base["audio"]).max())
    assert np.isfinite(opened["audio"]).all() and not np.allclose(opened["audio"], base["audio"])
