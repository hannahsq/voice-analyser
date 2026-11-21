# The reconstruction loss: what has been tried

A register of every entry that tests M4's reconstruction loss on real audio (plan section 5.3): its instrument, the training arms that used it, and its side effects, with where each was tested and what it showed. The decision log (`docs/decisions.md`) holds the detail; this file is the index, as `f1_bias_register.md` is for the F1 overread. Add a row whenever an entry tests the loss, an arm trained with it, or a candidate change to it.

**The loss as it stands (D-109, D-114; `vocal_analysis/train/reconstruction.py`, `model_recon_loss`):** per voiced, sonorant 10 ms frame with at least 4 valid harmonics, the model's F1-F4 (a differentiable local mean of the heat-map) and B1-B4 render a cascade of two-pole resonators, with F5-F8 continued from F4 by the uniform-tube rule (detached) and an LF source at the best Rd of a grid (0.3-2.7, no gradient through the choice). The render is compared at the harmonics k f0 (labeller's f0, up to 4000 Hz) with targets from a least-squares harmonic fit, SNR-gated at 10 dB (D-104); one level per frame is profiled out, then a Huber loss (delta 6 dB) per harmonic, averaged over the frame's harmonics and the batch's frames. Weight 0.024-0.025: a quarter of the formant term's encoder gradient at the branch point. Only F and B receive gradient, so whatever else a real recording holds lands in them (`considering.md`, 2026-11-17: what the renderer cannot cover).

**Where it stands (2026-11-17):** no arm with the loss is a candidate. It pulls against 5b's compression of women's F1 at both ends (D-121), pushes F1 away from 2 f0 where the reference asks for nothing (D-119, D-121), and costs the length head's reliability, more in context (D-115 to D-118). D-123 (draft) proposes changes to the loss and a check that should decide among them before any training; D-122 (draft) proposes the rule M4's arms are judged by.

## The instrument (renders with known labels; no training)

| Question | Where | Finding | Verdict |
|---|---|---|---|
| Where does the loss put F1 when everything else is exact? Renderer, harmonic targets, profiled source | D-102 | Biased by the source model; with the exact source F1 is recovered at every f0 | The renderer is right; the source model is what matters |
| An LF source by Rd in the renderer, Rd estimated | D-103 | Gate fails: the high-harmonic targets measure aspiration noise (up to +22 dB at 4-5 kHz at low HNR) | Targets need a noise gate |
| Noise-gated harmonic targets (SNR 10 dB, noise from half-integer sinusoids fitted jointly) | D-104 | Gate fails on spread only; the bias is gone and plain cascade clips pass | Adopted: the targets every arm uses |
| Free nuisance sections (poles and zeros per frame) | D-105 | Gate fails: each pole sat beside its zero and made narrow bumps that stood in for F1 | Free sections compete with F1 |
| SoFilNet's nuisance structure (Meka, Ramesh & Kodukula 2026, arXiv 2610.18120, section III-C: two high poles above F4, two broad poles, two bounded zeros; the ranges it does not state were our choices) | D-106 | Gate fails: better than D-104 below 150 Hz (F1 \|.\| 15 c), worse above | Not adopted |
| Extra sections held per clip | D-107 | Gate fails broadly: the sections take up the residual but don't tighten F1 | Not adopted; no nuisance sections in the training arms |
| Can v3.1's frozen trunk read the extra sections? | D-108 | Partly: presence AUC 0.83 nasal, 0.74 subglottal | A gate by the model's own reading is untried (`considering.md` 2026-11-17) |
| Higher-pole correction and the 4000 Hz limit (D-109's pre-checks) | D-109 | F1-F4 alone bias F1 down at high f0 (to 5000 Hz: medians +10 / -10 / -60 c); F5-F8 from F4 restore D-104's level (to 4000 Hz: \|.\| 15 / 15 / 30 c) | Adopted. Real-audio misfit about 3 dB per harmonic against 1.2 synthetic, with a shape: +2 dB below 250 Hz, -2 to -2.7 at 500-750, +1 to +2 at 2.5-4 kHz |

## Training arms (5b's decay branch, seeds paired with 5b's own branches)

| Arm | Where | Finding | Verdict |
|---|---|---|---|
| The loss on VocalSet and Zeroth originals, weight 0.0247 | D-109 | Not a candidate (seed 1 only gains; length guard fails narrowly). Women's closed-phase F1 bias +4.3 / +4.1 -> +11.9 / +14.6 Hz; PTDB-TUG F1 slope 1.04 -> 1.13 / 1.07 | The loss moves real F1, up on PTDB-TUG |
| The loss on VocalSet alone (Zeroth dropped), weight 0.0240 | D-114 | Not a candidate (no gain on both seeds; length guard fails, [-0.029, -0.009]). Stretch gone: PTDB-TUG women's contrast +6.2 Hz (D-109 +28.1). Pre-checks: on Zeroth the loss prefers the stretch for several reasons at once (the source's shape at H1-H2 about a third, a fixed shape per recording about a third, the rest unplaced) | Connected speech carried the stretch; the reliability cost stays without it |

## What the arms' shift is (inference; no training)

| Question | Where | Finding | Verdict |
|---|---|---|---|
| Is D-109's shift the open-phase (acoustic F1) shift? | D-110 | Not coupling-like: on PTDB-TUG women it grows with F1 (-6 Hz below 450 Hz, +22 above 600); none on Hillenbrand | A stretch by vowel on one corpus: the real-audio misfit, not coupling |
| D-109's shift in one voice and one recording chain (Hannah's sung vowels) | D-111 | Mixed: egressive open vowels +5.1 Hz, below the +8 bar | Style against chain still open |
| D-109's shift across speaking styles in one session | D-112 | Read sentences show PTDB-TUG's pattern (contrast +20); sustained vowels noisy; whisper flat | Comes with read speech |
| Does Hillenbrand's recording chain block the shift? | D-113 | No single feature; the whole spectral balance across the F1 region moves it by about a third | Not band edges or noise |
| Is the preferred F1 shift locked to the harmonics or to frequency? Branch-point scan by F1 band x F1 / f0 | D-119 | Mixed by the rule (D -0.14 [-0.36, +0.09] VocalSet, -0.02 [-0.06, +0.01] Zeroth). The preferred F1 jumps by 150-300 c across F1 = 2 f0 in every F1 band of both corpora | The loss pushes F1 away from the second harmonic |
| Is v3.1's F1 pulled towards 2 f0 on PTDB-TUG women, and do the arms correct it? | D-121 | No pull once the cells are not chosen on the reference (K +48 / +22 / +6 c by binning). 5b reads +53 to +74 c high at 1.5-2 f0 and -24 to -40 c low at 3.5 f0 and above; D-114 lowers \|error\| by 1-8 c in every cell, both seeds | The loss opposes 5b's compression at both ends and adds a push between 2 and 3.5 f0 that the reference does not ask for |

## Candidate changes to the loss (scans; no training)

| Question | Where | Finding | Verdict |
|---|---|---|---|
| Which change to the loss removes its jump across 2 f0, and is what is left what the closed-phase reference asks for? A, C, A+C, B', B'-fixed (D reported) scanned on the stores, the synthetic control and PTDB-TUG dev | D-123 | Unreadable by the rule (B''s +50 c code check: its borrowed offset passes the identifiable frames' F1 error to the hard frames). Outside the rule: no candidate removes the jump (base +143 to +301 c; closest B' +78 to +225); the harmonic-number terms halve the push below 2 f0 (Zeroth -186 -> -52 c) but leave the push up above it; on PTDB-TUG women they remove a push that was right in one 2-2.5 f0 cell (against the closed phase; weaker under the acoustic F1, D-123's note of 2026-11-21); D and B'-fixed change nothing | The source's shape carries the lower half of the jump, not the upper; the upper half's cause is untested (bandwidths, nasal frames) |

## The length cost

| Question | Where | Finding | Verdict |
|---|---|---|---|
| Where does the branch arms' Hillenbrand length-reliability cost come from? | D-115 | Loss-specific but small: men's between-speaker variance falls, women's vowel part rises | Specific to the loss |
| Is it in the arms' own formants? | D-116 | The formants keep men's speakers apart about as well as any 5b branch | Not in the formants |
| Does refitting the VTL head on the frozen trunk recover it? | D-117 | Not recovered: refit -0.018 [-0.035, -0.002] against -0.019 joint (f5 null +0.011) | In the trunk's features |
| Does it remain when the words are read in context? | D-118 | Kept, and larger: d114 - 5b -0.019 [-0.028, -0.009] words alone, -0.055 [-0.068, -0.043] in 4 s windows, -0.071 [-0.085, -0.058] in one pass; none on PTDB-TUG sentences | Larger in context |

## Open

- **D-122** (draft, not registered): the rule M4's training arms are judged by (gold-set gains against the f5 null arm, the PTDB-TUG RPA guard restored at -1.0, the length guard also read in context). Waiting for Hannah.
- **D-123** (done 2026-11-17, unreadable by its rule; see the table above. At registration D became reported only, since under the 10 dB gate it can move a harmonic by at most 0.46 dB, and B'-sweep was deferred to its own entry): changes to the loss, each scored by D-119's scan before any training (the jump across 2 f0 under a third of D-109's; synthetic shift within the control's spread with half the curvature kept; on PTDB-TUG dev, the sign of minus 5b's error): A, the source's shape per crop; B, no F1 gradient below about 2.5 f0 (since 2026-11-17 the floor, not a candidate: it passes the stores' criterion by construction); C, the channel's curve per recording; A+C per recording, D, the noise's power out of the targets, and B', the source's shape borrowed from identifiable frames of the same recording and f0 bin with F1's gradient kept on hard frames (all added 2026-11-17); a length-consistency term as its own entry. The check's stated goal is the cause of the push, not the fix. Waiting for Hannah.
- **Not yet in any entry**: the rest of the renderer-coverage list (`considering.md`, 2026-11-17), with a smoothness penalty on log F tracks (SoFilNet, Meka, Ramesh & Kodukula 2026, eq. 24) as the cheapest lever on the length cost.
