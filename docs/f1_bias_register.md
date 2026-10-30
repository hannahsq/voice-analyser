# Women's F1 overread: what has been tried

A register of every hypothesis tested for the upward F1 bias that synthetically trained models show on real voices (largest for women and children), with where it was tested and what it showed. The decision log (`docs/decisions.md`) holds the detail; this file is the index. Add a row whenever an entry tests a cause or a fix.

**The bias as it stands (teacher v3, `conformer_v2r2_s1_d16k`):** PTDB-TUG dev, women: +6.2 Hz against the closed-phase reference [+2.5, +10.8], +1.9 against D-044's constrained fit R [−2.3, +6.4], +20 against QCP-FB; men +6.6 / +5.2 (D-044). About 84–90 % of the error is shared by every synthetically trained model (D-024). On renders it is absent (D-019, D-047, D-048), and on identity resynthesis of real singing too (+0.0 Hz, VocalSet val; `reports/analysis/sensitivity/resynth_identity_full_seeds3.txt`).

## Is the bias real, or the reference's?

| Question | Where | Finding | Verdict |
|---|---|---|---|
| Hillenbrand's LPC labels read women and children high | D-012, D-019 | LPC references do snap to harmonics (44 % within 0.1 spacing), but the models' errors are smallest on those frames, and the bias grows for men too | Not the main cause; Hillenbrand is not used for the bias since |
| Closed-phase LPC reference drifts | D-043 | Stands for women's speech (+2.9 Hz on renders); drifts up with a closure gap (+8 at gap 0.10) and a large open quotient (+12 at OQ 0.70); fails above ~450 Hz f0 | Reference usable; in breathy voices the true overread is likely understated |
| An independent reference (inverse filtering) | D-044 | R (constrained closed-phase fit) robust to leaky closure on renders; on PTDB-TUG women teacher v3 − R +1.9 [−2.3, +6.4] (vs reference +6.2), men +5.2; R is noisy (odd/even SD 49 Hz) and carries teacher v3's F2–F4. J (joint LF fit) fails on renders (+30 Hz) | Bias smaller against R, not resolved |
| QCP-FB as a third reference | D-046 | Unbiased on renders (+2 to +4 Hz) and better in leaky cells, but missed its gate narrowly (precision, F3); not adopted | Context only |
| The references' drift explains the D_a correlations | D-045 addendum | closed-phase − R tracks neither H1–H2 nor HNR | No |

## Causes in the training data

| Hypothesis | Where | Finding | Verdict |
|---|---|---|---|
| Label definition / prior pull on in-distribution data | D-019 | Synthetic validation signed bias −3 to +1 Hz in every f0 band; within ±10 Hz in every (f0, F1) cell to 260 Hz | No |
| Pull towards the synthetic F1 prior's centre | D-020, D-022 | Decoupling raised the bias (+8 women); population-based tract distributions (train_v3b) raised it further (+26 vs +15) while the prior's centre fell | Refuted |
| The glottal source (real sources, exact labels) | D-019 | Resynthesised VocalSet clips: +6 / +9 / +6 Hz for teacher v2 models, not growing with training | Mostly not |
| Glottal coupling and source–tract interaction missing | D-023, D-026, D-027 | Interactive / coupled sets (v2i, v2m, v2s, v2t) cut women's closed-phase bias by ~10–15 Hz ("fix"), but cost F3 / F4 and overshoot at full size (underread −9 to −16 at f0 ≥ 300); not adopted for teacher v3 | Real lever, unusable as built |
| Real-voice resynthesis in training | D-025 | VocalSet + Zeroth resynthesis (30 %): women's closed-phase bias −5.7 / −5.9 Hz, replicated on semi-private (teacher v3) | Partial fix, adopted |
| Training length | D-019, D-025 | Bias grows with training on synthetic data (Hillenbrand women +21 → +30 Hz, d06k → d16k); longer M3 runs lowered closed-phase bias | Interacts with data |
| Harmonic locking (steady excitation samples the envelope at harmonics) | D-034 sweeps, VocalSet locking | Total staircase on steady, noise-free synthetic vowels; none on real singing (locking excess ≤ 0) | Not the cause on real voices |
| Context length / long inputs | D-028 | Signed bias moves ≤ 1.6 Hz | No |
| Aspiration *level* | D-047 | Paired renders 5–60 dB: women 40 − 5 dB +0.7 [+0.1, +1.3] Hz | No |
| Aspiration *shape* (real band periodic fraction) | D-048 | −6 dB/oct at 10 dB matches real women's profile; shape-matched − reference −2.0 [−2.7, −1.4] Hz | No |
| Wider real bandwidths pulling F1 up | D-050 probes 1, 7 | Real recordings read B1 / B2 1.4–1.8 × the synthesis prior (men more than women); but scaling every bandwidth × 0.7–2 on resynthesised speech moves F1–F4 by ≤ ±0.003 log2 per log2 in every model | No (on resynthesised speech) |
| Real speech dynamics and real source (resynthesis, exact labels) | D-049 | Zeroth val speech resynthesis: teacher v3 women −0.0 [−0.8, +0.7] Hz, men +0.7; the same-seed synthetic-only model *underreads* (−5.4 / −4.1) | No: the overread needs the real tract or F1's definition |

## What the model's features see

| Question | Where | Finding | Verdict |
|---|---|---|---|
| Shared across architectures and seeds | D-024 | 84–90 % of the error common to all models | A data / definition cause, not an architecture one |
| Pretrained real-speech features (Parakeet) | D-033 | Frozen Parakeet layer 5 + a head trained on synthetic labels carries almost none of it; labeller v3 + H5×6 cuts women's |bias| 9.1 → 2.3 (pooled) | Features learned from real speech avoid it |
| Distilling that property into a small student | D-034 | Students (v2, v4) don't reach their noise control; the property did not survive distillation | No |
| Frame-level synthetic/real gap | D-045 | D_h separates perfectly (AUC 1.0) but real-ness doesn't track the overread (Spearman −0.01 / +0.01); HNR tracks it weakly (+0.11, all references), mostly in the most harmonic frames | No linear frame-level link |
| What makes frames look real | D-045 follow-ups | Renders too periodic below 2 kHz; codec, reverberation, higher f0 look real; resynthesised clips sit between real and synthetic; no band occlusion removes the separation | Realism lead, not a bias lead |

## Open

- ~~**Dynamics**~~: ruled out for the overread by D-049 (real speech dynamics and source, exact labels: no bias, either sex).
- **Source pulse shape**: resynthesis keeps the real (inverse-filtered) source and shows no bias (D-049), which makes the source an unlikely cause; another family of synthetic pulse shapes remains untested.
- **The real vocal tract** (real bandwidths, wall vibration, subglottal coupling and the open-phase shift): what resynthesis replaces, so now the main suspect with the definition below.
- **F1's definition**: our labels are the closed-glottis resonance (D-023); real voices' effective resonance over the whole cycle includes the open-phase coupling shift. A head for whole-cycle effective F1 (`considering.md`) would show how much of the overread is the definition.
  - *Once the F1 issue is fixed or accepted* (Hannah, 2026-10-30; the comparison is slightly unfair either way, but the F1 question comes first): restructure the reports' F1 comparisons. Split the baselines into those with no side information (Praat defaults, sweep) and with it (FastTrack: speaker group and vowel span), and judge F1 mainly on non-LPC references, since Hillenbrand's LPC labels favour Praat on F1 by construction (D-008, D-012).
- **Energy between F1 and F2** (a lead from D-050 probe 3, not yet a test): a ±6 dB peak at 1 kHz moves F1 by about ±12 cents in every model, and cutting it lowers Hillenbrand F1 error by 2–6 Hz; noise lowers F1 and its error the same way. If the models read energy in the F1–F2 valley as F1 evidence, real voices with a fuller valley (breathier, less damped) would read high. Test idea in `considering.md`.
- **Children**: D-045's children's arm (Samrómur licence check) not done.
