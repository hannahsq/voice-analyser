# The three synthesisers

Three paths in `vocal_analysis/synth/` make voiced sound. They differ in how the glottis and the vocal tract interact, and that difference matters for the F1 overread (`docs/f1_bias_register.md`). All three label F1 the same way: the **closed-glottis resonance**, the tract's pole with the glottis shut (D-023). Whatever moves the resonance while the glottis is open, the label stays put.

| | Cascade (`voice.py`) | Interactive tube (`interactive.py`, D-026) | KL waveguide (`kl.py`) |
|---|---|---|---|
| Source | LF pulse (Rd), fixed | flow solved per sample from glottal area and the acoustic load | flow solved per sample at the glottal end of the waveguide |
| Tract | resonator cascade: formants sampled from priors, or a tube's poles (`p_tube`) | `tube.py`'s lossy tube, as impulse responses | the same Story area functions as `tube.py`, as a Kelly-Lochbaum waveguide |
| Source-tract interaction | none by default. Optional (D-023, `p_coupling`): poles moved by a set amount, F(t) = F (1 + df a(t)) | physical: the tract's and the subglottal system's impedance feed back into the flow | physical, in the time domain, with walls, losses and a subglottal tract |
| Closed-phase gap | `gap_from_open_quotient` (coupled clips only) | the area pulse's gap | `glottal_area(..., gap=)` |
| Labels | the cascade's own formants, or `tube.poles` | `tube.poles` | `frame_labels`: `tube.poles` of the discretised tube |
| Trained on | yes: every training set (`train_v2` for teacher v3 and v3.1, with **no coupling**) | only in D-026's test sets (v2i, v2t and others) | never; an evaluation reference (D-060) |
| Golden tests | frozen contract | (through the tube) | current behaviour, may be regenerated |

## Cascade

The workhorse. It uses additive synthesis: each harmonic's amplitude is the LF source spectrum times the cascade's response at that frequency. That makes it exact and band-limited, but a pole can't move within a period. D-023's coupling switches to a time-domain path (`filters.filter_tv`) so poles can move within each cycle. How far each pole moves comes from physics: `coupling.open_shifts` gives the quasi-static pole with the glottis held at its peak area, which matches Barney et al.'s +13 % for F1. The movement over the cycle is a linear interpolation with the opening a(t) = g + (1 - g) sin^2(pi t / te) in the open phase, and g in the closed phase. The real physics isn't linear in the area (D-075). Bandwidths move the same way, B(t) = B (1 + (bm - 1) a(t)). Coupling was tested once (`train_v2c`, D-023) and is off in every set a released teacher trained on.

## Interactive tube

Level-1 interaction after Titze: the glottal flow comes from the glottal area, the lung pressure and the load. The load is the tract's input impedance in series with the subglottal system. The tract shapes the source (formant ripple, skewing, open-phase F1 and B1 changes) instead of a fixed pulse being filtered. The tract itself is the closed-glottis tube, so its labels are exactly `tube.poles`.

## KL waveguide

A time-domain acoustic simulation, ported from voice-dataset-labeler with three fixes (2026-10-30/31; module docstring). Pressure waves travel along roughly 65 sections. Walls are mass-spring-damper elements, boundary-layer losses are fitted filters, and the glottis is a quadratic flow law solved against the waveguide's own pressure. A coupled subglottal tract sits below it. No parameter sets how far F1 moves when the glottis opens; the movement comes from the simulation. Its labels are the rigid-glottis poles of the same tube, and agree with `tube.poles` within 1 % (D-058). It reproduced OPENGLOT II's growth of the overread with the gap (D-060), and is admitted as an interaction reference.

## How they relate

`tube.py` isn't a renderer itself. It's the tract model the cascade's tube clips, the interactive source, and KL all share. That's why D-058 and D-060 can compare them on the same Story vowels and labels. The interactive tube and KL show the same direction and size of gap growth in teacher v3.1's F1 (D-058). Neither was in v3.1's training data.
