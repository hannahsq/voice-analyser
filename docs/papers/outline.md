# Paper outlines

Living outlines for write-ups of this project (Hannah, 2026-10-28). Each paper lists its claims, the evidence each claim needs and where that evidence stands: **done** (a decision or report exists), **running**, or **missing**. Numbers are not copied here; they live in `decisions.md` and `reports/`, and a draft cites the D-xxx entry. Nothing here changes what is run: a claim that needs a new experiment gets its own pre-registered entry first.

Common to every paper:
- **Pre-registration as evidence.** Claims rest on decisions registered before running, so the decision log's git history is part of the record: archived with a DOI (Zenodo) and linked when a paper goes out.
- **Evaluation tiers.** Dev decides, semi-private confirms (each use logged), lockbox opens once at milestone acceptance. No paper reports lockbox numbers before then.
- **Data and licences.** Training data are documented, licensed corpora (CC BY 4.0 pools; `ATTRIBUTION.md`); evaluation sets are never trained on. Weights are CC BY 4.0.
- **AI assistance.** Most of the code, analysis scripts and drafting were done with an AI assistant (Claude) under Hannah's direction; the venue's disclosure policy is read early and followed. Authors are people; the assistant is acknowledged, not listed.

---

## Paper A: evaluating formant and vocal-tract-length estimators

*Likely first.* Most formant-tracker papers report one mean absolute error on one gold set. The argument: an estimator meant for measurement must also respond to the right changes and ignore the wrong ones, on real voices, with uncertainty that grows when evidence is missing.

| Claim | Evidence | Status |
|---|---|---|
| Single-set MAE hides biases that matter (women's and children's F1, closed vs open phase) | closed-phase references (D-019, D-024), windowed references (D-031 follow-up), per-group bias with speaker bootstraps | done |
| Run-to-run and head-seed variance on real voices is large, so single-seed comparisons mislead | D-024, D-026, D-036's seeds, D-037's noise floor | done |
| An f0-invariance probe with a shuffle null bounds f0 leakage; detectability alone rewards noisy heads | the probe (D-020), D-036's noise-calibrated guard | done |
| Counterfactual resynthesis tests equivariance (length, single formants) and invariance (f0, quality) on realistic voices | `considering.md` probes | missing |
| Channel robustness: codecs, band-limiting, reverberation, noise, denoiser artefacts; σ grows where evidence is lost | `considering.md` probes; σ calibration | missing |
| Time reversal measures context use; with a change of speaker, it measures leakage across speakers more sensitively than references can | `considering.md` (Hannah's speaker-change extension) | missing |
| Pre-registration with tiers is practical for a one-person project | the log itself; amendments and slips recorded honestly (e.g. D-036's run note, D-037's amendments) | done |

Figures: slopes per probe and output with null bands; σ against error under each degradation; the forward-minus-reversed profile around a speaker change.
Venue ideas: Interspeech (deadline usually early in the year; check), JASA Express Letters (no deadline).

---

## Paper B: an analyser trained on synthetic and resynthesised speech

f0, voicing, F1–F4 with bandwidths and calibrated σ, and anatomical vocal-tract length, from a teacher trained without real formant labels.

| Claim | Evidence | Status |
|---|---|---|
| A synthesiser with Story area functions, glottal coupling and channel/codec augmentation is enough to train a formant tracker that holds up on real voices | D-010 to D-025; M1–M3 results | done (dev) |
| Resynthesised real recordings close part of the synthetic-to-real gap | D-016, D-025 | done (dev) |
| Bandwidths and σ are usable, not decorative | σ coverage, observability label (D-027) | partly done |
| Compared fairly with Praat and published neural trackers on the same references | baselines (D-007, D-008); a like-for-like set still to fix | partly done |
| A jointly trained length head reads length better than formant-derived apparent length, without costing formants | D-035, D-036, D-037 | done (dev); item 5 retrain pending |
| Final numbers | teacher v3.1 (roadmap item 5) and M4's v4, lockbox at milestone | missing |

Figures: synthetic pipeline; error by f0 band and group; reliability of length by group; examples where the trackers disagree.

---

## Paper C: vocal-tract length from speech, checked against MRI

| Claim | Evidence | Status |
|---|---|---|
| Midsagittal glottis-to-lips lengths can be measured at scale from real-time MRI with a contour model trained on few hand frames | Annot-16 labels (D-035), Hannah's corrected glottis ends; contour model (roadmap item 6) | missing |
| Learnt length estimates track anatomy across about 70 speakers, two labs and two languages (USC 75, ArtSpeechMRIfr) | MRI dev speakers; ArtSpeechMRIfr as an independent check; lockbox at milestone | missing |
| Calibration (level, women / men ratio) against anatomy, not group norms | cross-fitted calibration on dev MRI speakers | missing |
| Which normalisation of vowels by length works best (learnt length vs formant-derived vs Lobanov) | D-037, D-039 vowel classification (reported only) | partly done |

Constraints: Annot-16 forbids redistribution, so per-speaker values from it are not published; the USC 75 data are CC BY 4.0. If MRI data are used for many decisions, a use budget applies (each use logged).

---

## Open questions about writing up

- Order: A needs only the probes; B and C need milestones. A first, unless the MRI work moves faster.
- Preprints: arXiv needs an endorsement for a first submission in a category; Zenodo needs none and gives a DOI.
- Collaborators (optional): someone in voice science or a voice clinic could check framing, and later the consented before/after work.
