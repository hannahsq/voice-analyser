# Real-time vocal analysis pipeline: project plan (MK II)

Planning document for implementation in a local code session. Defaults marked *(proposed)* are starting points, not commitments; anything still open is collected in §13. Treat this as a living document: when the implementation diverges from it, update the plan and record why in a decisions log.

## 0. Notes for the coding session

- Work milestone by milestone (§12). A milestone is done when its acceptance criteria pass and the metrics report has been regenerated.
- **Never train on gold-set audio**, including in unlabelled, self-supervised and resynthesis pools. Splits are speaker-disjoint and enforced by an automated test (§6.4).
- Every component that has an exact answer on synthetic data gets a unit test against that answer.
- Every training run records its config, git commit, data manifest hashes and eval report.
- When this plan turns out to be wrong, fix the plan and log the decision rather than silently diverging.

## 1. Goals and scope

**Primary goal:** a voice analysis pipeline that runs in real time on suitable hardware and robustly estimates, per frame:

| Quantity | Notes |
|---|---|
| f0 + voicing | Speech and singing, ~50–1600 Hz |
| F1–F4 | Vocal tract resonance frequencies, with calibrated uncertainty |
| B1–B4 | Resonance bandwidths (needed for corrections; lower confidence) |
| H1, H2, H3 | Harmonic *amplitudes* (their frequencies are just k·f0), raw and formant-corrected |
| Spectral slope | Measure set in §8.3 |
| Vocal tract length | Apparent (per frame) and anatomical (aggregate), see §8.4 |

**Domains:** speech and singing, including classical singing with high f0, vibrato and register (M1/M2) transitions.

**Real time:** causal streaming inference with a small, bounded lookahead (§10). Offline analysis uses the same output schema, produced by a larger bidirectional teacher model.

**Scope assumptions for v1:** solo voice (no accompaniment), mono input, one voice per stream.

**Non-goals for v1:** accompaniment removal or source separation, polyphony, speaker identification, clinical voice assessment.

**Hardware:**
- Training: AMD Ryzen AI MAX+ 395 with 128 GB unified memory (§11).
- Real-time inference: CPU-first; target devices to be decided (§13).

## 2. Design principles

1. **Learn only what is ill-posed.** f0/voicing and formants (with bandwidths) are learned. Harmonic amplitudes, corrected amplitudes, slope measures and VTL are computed deterministically from those outputs. This keeps the labelling problem small, makes the back end unit-testable, and makes errors attributable to a stage.
2. **Use labels whose correctness doesn't depend on a formant tracker.** Exact labels come from procedural synthesis and from analysis–resynthesis of real recordings. Unlabelled real audio contributes through physics-based self-supervision. Classical trackers are candidate generators, never ground truth.
3. **Gold sets gate everything.** Independent evaluation sets that are never trained on decide whether a training round is accepted.
4. **Uncertainty is an output.** Every learned quantity carries a calibrated uncertainty that propagates into derived measures and the UI.
5. **Teacher for data, student for real time.** A larger offline bidirectional teacher drives the data engine; a small causal student is distilled from it for deployment. Real-time constraints never limit the quality of the data engine.

## 3. System overview

```
DATA ENGINE
  [A] Procedural synthesis ............. exact labels (F, B, f0, voicing, VTL, source H1–H2)
  [B] Analysis–resynthesis of real audio  exact labels by construction
  [C] Raw real audio ................... self-supervised losses (reconstruction, equivariance)
  [D] Classical trackers ............... candidates + gated refinement only (never ground truth)
          |
          v
  TEACHER (offline, bidirectional) --eval--> GOLD SETS (never trained on; gate every round)
          |   ↻ teacher re-tracks real audio -> feeds [B] and [D] in the next round
          v   distillation + supervised losses
  STUDENT (causal, streaming, small)

REAL-TIME INFERENCE
  mic -> streaming VQT (+ f0 harmonic-comb channel) -> STUDENT -> f0, voicing, F1–F4, B1–B4, sigma
      -> BACK END: harmonic LS fit -> H1–H3, corrections (H1*–H2*), slope, VTL, causal smoothing
      -> outputs (per frame + running aggregates, all with confidence)
```

| Component | Type | On real-time path? | Section |
|---|---|---|---|
| Synthetic generator | deterministic | no | §5.1 |
| Resynthesis labeller | deterministic, uses teacher tracks | no | §5.2 |
| Self-supervised objectives | training losses | no | §5.3 |
| Candidate generator and Viterbi | deterministic, uses teacher | no | §5.4 |
| Teacher model | learned | no (offline analysis mode) | §7.2 |
| Student model | learned | yes | §7.3 |
| Streaming VQT front end | deterministic | yes | §7.1, §10.2 |
| f0 module (PESTO for v1) | learned, pretrained | yes | §7.1 |
| Back end | deterministic | yes | §8 |
| Eval harness | deterministic | no | §6 |

## 4. Definitions and output schema

- **Formant (Fn):** frequency of the n-th resonance (pole) of the vocal tract transfer function, not the n-th spectral peak. Synthetic and resynthesised labels follow this definition, so the model learns resonances.
- **Bandwidth (Bn):** −3 dB bandwidth of the n-th resonance.
- **f0:** fundamental frequency of the glottal source. **Voicing:** probability that the frame is periodic phonation.
- **Hk:** amplitude (dB) of the k-th harmonic (at k·f0) in the radiated signal, from a least-squares harmonic fit (§8.1).
- **Hk\*:** Hk corrected for the boost from nearby formants (Iseli–Alwan, §8.2). H1\*–H2\* estimates the source's own H1–H2.
- **An:** amplitude of the strongest harmonic near Fn (for H1–A1, H1–A2, H1–A3).
- **Apparent VTL:** per-frame length from formant spacing (§8.4). It varies with articulation (larynx height, lip configuration).
- **Anatomical VTL:** speaker trait; robust aggregate of apparent VTL over many frames and vowels, calibrated against MRI-measured lengths.
- **Frame timing** *(proposed)*: internal sample rate 16 kHz, hop 10 ms. In streaming mode the output for frame t is emitted at wall-clock time t + L, where L is the student lookahead. Timestamps always refer to the analysed frame, not the emission time.

### Per-frame output schema

| Field | Unit | Notes |
|---|---|---|
| `t` | s | analysed frame time |
| `f0`, `f0_sigma` | Hz, cents | |
| `voiced_p` | 0–1 | |
| `sonorant_p` | 0–1 | formant outputs are meaningful (voiced sonorant) |
| `F1`…`F4`, `F1_sigma`…`F4_sigma` | Hz, cents | heat-map mean and spread |
| `B1`…`B4`, `B1_sigma`…`B4_sigma` | Hz, log units | |
| `H1`, `H2`, `H3` | dB | raw harmonic amplitudes |
| `H1c`, `H2c`, `H4c` | dB | formant-corrected amplitudes |
| `H1H2`, `H1H2c`, `H1H2c_ci` | dB | corrected difference with interval |
| `H1A1`, `H1A2`, `H1A3` | dB | |
| `H2H4c`, `H4H2kc`, `H2kH5k` | dB | H2\*–H4\*, H4\*–H2kHz\*, H2kHz\*–H5kHz |
| `slope_db_oct` | dB/octave | regression over harmonic peaks |
| `vtl_app_cm`, `vtl_app_ci` | cm | per-frame apparent VTL |

Per stream or utterance: `vtl_anat_cm` (with interval), running medians of the per-frame measures, and optionally CPP/HNR.

## 5. Data engine

### 5.1 Procedural synthesis (exact labels)

Purpose: exact labels everywhere, including regions real data rarely covers (high f0, F1 near a harmonic, close formants, strong vibrato, breathy and pressed extremes). Precedent for the high-pitch case: Mikušová (2021) trained a CNN on synthetic high-pitched vowels and tested it on recordings through plastic tubes and a 3D-printed vocal tract.

**Source**
- LF glottal flow derivative parameterised by Rd, Rd ∈ [0.3, 2.7] *(proposed)*, with occasional abrupt Rd and f0 changes to mimic register transitions.
- Aspiration noise modulated by glottal opening; harmonics-to-noise ratio 5–40 dB.
- Jitter 0–2 %, shimmer 0–1 dB.
- f0 contours: mostly borrowed from real speech and singing (PESTO tracks from the real pools, randomly transposed), plus synthetic glides and steps. Range 50–1600 Hz, oversampling f0 > 400 Hz.
- Vibrato: rate 4–8 Hz, semi-extent 0–150 cents (bulk 40–100 cents), with onset delay and slow rate/extent drift.

**Tract**
- Preferred: a lossy tube model driven by randomised area functions (Story's parametric model: neutral shape plus modes), scaled to length L ∈ [10, 20] cm. Compute the transfer function with wall losses and lip radiation, and extract poles to get exact Fn and Bn. L is the VTL label.
- Fallback and mixing: a cascade of second-order resonators with formants sampled from a VTL-scaled vowel space and bandwidths from Hawks & Miller × log-uniform(0.6, 1.6).
- Optional articulatory realism: VocalTractLab.
- Extras: poles above F4 (F5–F8) consistent with L; nasal pole–zero pairs in ~200–1200 Hz with random coupling; weak subglottal coupling (roughly 600/1400/2200 Hz in adults, scaled by size); radiation as a first difference.

**Channel augmentation** (each applied with some probability; some samples stay clean)
- Room impulse responses, measured and simulated (e.g. pyroomacoustics), T60 0.1–1.0 s.
- Additive noise, SNR 5–40 dB.
- Mic and EQ: high-pass 40–300 Hz, low-pass 4–8 kHz, random parametric EQ ±6 dB.
- Codecs via ffmpeg (Opus, AAC, MP3) at varied bitrates.
- Gain changes; rare mild clipping.

**Labels emitted per frame:** Fn and Bn (n ≤ 6), f0, voicing, sonorant mask, L, Rd, source-only harmonic amplitudes (ground truth for H1\*–H2\*), channel parameters.

**Throughput:** pre-generate the expensive parts (tube model, RIR convolution) to shards; apply cheap augmentation on the fly (§11).

**Tests:** filter poles equal the emitted labels; on clean settings, generated harmonic amplitudes equal source × |H(k·f0)| × channel to numerical precision; output is deterministic under a fixed seed.

**Hard-case suite:** a fixed-seed synthetic benchmark covering f0 > 400 Hz, F1 within 10 % of H1 or H2, |F3 − F2| < 300 Hz, nasalisation, breathy voice (Rd > 2), vibrato > 100 cents, low SNR and phone-like channels. Run it every round as a regression test.

### 5.2 Analysis–resynthesis labeller (exact labels on real-sourced audio)

Idea, after MDB-stem-synth's f0 analysis/synthesis approach: a noisy label on real audio becomes an exact label on slightly modified real audio.

Pipeline:
1. Estimate f0 and glottal closure instants on real audio (PESTO; EGG where available).
2. Inverse-filter to recover the source (glottal flow derivative): IAIF first, QCP later.
3. Get formant tracks from the current best tracker (teacher, Viterbi over candidates, gated LP refinement; see §5.4) and smooth them (DCT or Savitzky–Golay).
4. Perturb the tracks:
   - global VTL factor α ~ log-uniform(0.8, 1.25);
   - slowly varying per-formant offsets (σ ≈ 3–5 %);
   - bandwidths from Hawks & Miller × log-uniform(0.6, 1.6);
   - F5–F6 consistent with α, and an optional nasal pole–zero pair.
5. Re-filter the source through a time-varying cascade of resonators with exactly those poles. Interpolate per sample (or in small blocks) in (F, B) space rather than in LPC coefficients, so the filter stays stable.
6. Emit the audio and exact labels, plus source-only harmonic amplitudes measured on the source signal, which are ground truth for H1\*–H2\*.

QC: reject segments where inverse filtering fails, for example when the recovered source spectrum still shows strong peaks at the original formant positions. Log residual-ripple metrics per segment.

Caveats:
- Residual ripple from the original tract stays in the source and becomes label noise. Mitigate with large perturbations and a parametric-source variant that fits LF/Rd per period instead of using the residual.
- Source–tract interaction is lost.

Uses: training data with real prosody, voice quality and coarticulation; validation of the Iseli–Alwan correction against exact source H1–H2.

### 5.3 Self-supervised objectives on raw real audio

**Reconstruction (analysis-by-synthesis).** Precedent: FormantNet (Lilley & Bunnell 2021). It predicts pole and zero parameters, renders a spectral envelope with a fixed formant synthesiser, and trains on the mismatch with the input envelope. It matched DeepFormants on VTR-TIMIT without using hand-corrected data. Lessons from that paper:
- Six poles plus one zero converged best. With more poles, the extras got squeezed into gaps (e.g. between F2 and F3). So give the model explicit nuisance poles and a zero, keeping the F1–F4 heads clean.
- F1 suffered from residual f0 interference. So at high f0, compare at harmonic positions (below).

Design *(proposed)*:
- Renderer: cascade all-pole (F1–F4 plus 2 nuisance poles), one nuisance zero, a source tilt model and radiation; differentiable magnitude response in dB.
- Target: for low f0, a smoothed envelope. For high f0 (above ~300 Hz), the observed harmonic amplitudes at k·f0 from the LS harmonic fit, compared with the rendered response sampled at k·f0. This avoids an envelope-estimation step exactly where envelope estimation is least reliable.

**Equivariance and invariance (PESTO-style).** PESTO learns pitch from pairs of inputs made by translating VQT frames, without annotations. Formants allow three transforms:

| Transform | How | Constraint |
|---|---|---|
| T1 uniform scale | translate VQT frames by k bins (crop edges), on the fly | f0 and all Fn shift by k bins; Bn scale accordingly |
| T2 f0 change, envelope fixed | WORLD resynthesis with f0 × β (two variants β1, β2) | Fn invariant; f0 shifts by log(β2/β1) |
| T3 envelope warp, f0 fixed | WORLD resynthesis with the spectral envelope warped by α (two variants α1, α2) | Fn shift by log(α2/α1); f0 invariant |

- Always compare two resynthesised variants with each other, never original against resynthesis. That keeps the constraints exact and makes vocoder artefacts symmetric. Precompute T2/T3 pairs offline.
- Loss on heat-maps: cross-entropy between one variant's distribution shifted by the known offset and the other variant's distribution, per head, alternating which side gets the stop-gradient.
- T2 is the constraint that directly teaches "formants are not harmonics".
- Equivariance alone can be satisfied by features that aren't formants (a spectral centroid satisfies all three transforms). Always train these losses jointly with supervised losses, and watch the gold metrics.

### 5.4 Classical trackers: candidates and gated refinement only

Why trackers are not ground truth:
1. **Correlated failures.** LPC-family methods all snap to harmonics at high f0, so agreement between trackers is not correctness.
2. **Selection bias.** Keeping only frames where trackers agree keeps the easy frames and throws away exactly the hard cases.

Roles they do play:
- **Candidate generator:** Praat Burg via Parselmouth across a sweep of ceilings (4000–8000 Hz, step 250) and formant counts, with FastTrack-style smoothness-based selection; QCP-FB if an implementation is available. Store the per-frame candidate (F, B) sets.
- **Viterbi decoding** over candidates, with teacher heat-maps as emission scores and smoothness transition costs. Produces refined tracks for resynthesis (§5.2) and pseudo-labels.
- **Gated LP refinement:** snap a teacher estimate to a nearby LP peak (Gowda et al. 2021; Alku et al. 2023; Kadiri et al. 2025), but only when f0 < ~250 Hz *(proposed)* and the peak lies within the teacher's 1σ interval. At high f0 the nearest LP peak is usually a harmonic.
- **Pseudo-labels:** weight by posterior confidence; never hard-filter. Hard cases come from §5.1 and §5.2 instead.
- **Baselines** (§6.5): Praat defaults, FastTrack, FormantNet and/or DeepFormants.

### 5.5 Active learning and hand correction

This is the safe version of "bootstrap a model to validate labels": train on §5.1–5.3, then use the teacher to audit tracker output and to choose which frames get hand-corrected.
- Rank the queue by teacher–candidate disagreement × teacher uncertainty, stratified by f0 band and vowel.
- Export review packets (audio snippet, spectrogram, harmonic markers, candidate and teacher tracks). Store corrections as per-frame overrides with reviewer and timestamp.
- Corrected frames become a small, high-value labelled set (e.g. personal singing data). Hold a portion out as gold.

Avoid training on tracker labels and then "validating" those labels with the resulting model. That mostly certifies the trackers' biases.

### 5.6 Data pools

Training pools (check licences; record the licence of every item in the manifest):
- **Speech:** LibriTTS-R, VCTK, Common Voice (for channel diversity); the TIMIT training set if licensed, excluding all TIMIT test-set speakers.
- **Singing:** VocalSet, GTSinger, M4Singer, OpenSinger, NUS-48E.
- **Children's speech:** check availability and licences (e.g. CSLU Kids, MyST, PF-STAR).
- **Own recordings:** speech and singing on a fixed mic setup.
- **Augmentation:** OpenSLR RIRS_NOISES, MUSAN, DEMAND; simulated RIRs.

Exclusions: all gold-set audio and speakers (§6.4).

## 6. Evaluation

### 6.1 Gold sets (never trained on)

| Target | Set | Notes |
|---|---|---|
| F1–F3 tracks | VTR-TIMIT test split | 516 utterances in total (324 train / 192 test). F1–F3 were auto-tracked, then hand-corrected, so some bias toward the seeding tracker remains. F4 is not corrected. Audio requires TIMIT (LDC licence). |
| Formants, cross-domain | Hillenbrand et al. 1995 | Men, women and children; hand-checked steady-state measurements; public audio. DeepFormants models trained on VTR-TIMIT degraded on this set, which makes it the key cross-domain check. |
| f0, speech | PTDB-TUG | Laryngograph reference. |
| f0, singing | MIR-1K, MDB-stem-synth | MDB-stem-synth labels are exact by resynthesis. |
| VTL | USC speech-MRI corpora (e.g. the 75-speaker real-time MRI set) | Audio plus images. Per-speaker VTL may need measuring from the images (glottis-to-lips centreline) unless already published. |
| High-f0 resonances | Physical rig (§6.3) | Real acoustics, exact labels. |
| H1\*–H2\* | Resynthesised held-out speakers (§5.2) | Exact source H1–H2. |
| Personal | Hand-corrected own recordings (§5.5) | Small; held-out portion only. |

### 6.2 Metrics

- **Formants:** per-formant MAE (Hz and %), RMSE, and detection rate within (20 %, 200 Hz), also reported at (25 %, 250 Hz) and (30 %, 300 Hz); swap/gross-error rate (|error| > 30 %). Reference points on VTR-TIMIT vowels and semivowels at (20 %, 200 Hz): QCP-FBCOV ≈ 85 % for each of F1–F3; FormantNet ≈ 80 / 94 / 90 %.
- **f0:** raw pitch accuracy (within 50 cents), raw chroma accuracy, octave and gross error rates, voicing precision/recall/F1.
- **Bandwidths:** log error, on synthetic, resynthesised and physical sets only.
- **H1\*–H2\*:** error in dB against exact source H1–H2, by f0 band.
- **VTL:** MAE against MRI (cm); within-speaker SD of anatomical VTL across utterances (a stability check that needs no labels).
- **Calibration:** reliability diagrams and ECE per head; empirical coverage of 50/80/95 % intervals.
- **Real time:** algorithmic latency, compute per hop, real-time factor, maximum absolute streaming-vs-batch difference.
- **Teacher–student gap** on all of the above.

Stratify everything by f0 band (< 150, 150–300, 300–500, > 500 Hz), vowel or phone class, speaker group and channel condition. Report bootstrap confidence intervals with speakers as the resampling unit, so changes between rounds can be judged against noise.

### 6.3 Physical rig (optional; high value for singing)

- 3D-print vocal tracts from published area functions (e.g. Story's vowel area functions) and couple a small driver at the glottis end.
- Measure each tract's resonances with a sine sweep. These are the exact labels.
- Drive the tract with synthetic glottal pulse trains at any f0 (including > 500 Hz), with vibrato and varying Rd. Record through the actual practice mic and room.
- For your own tract: Fujimura & Lindqvist-style sweep-tone measurement (external vibrator at the neck, glottis closed).

### 6.4 Leakage rules (enforced by tests)

- Splits are speaker-disjoint. Gold-set speakers and audio are excluded from every training pool, including unlabelled, self-supervised and resynthesis pools.
- If TIMIT is used, all TIMIT test-set speakers are excluded from training.
- If MIR-1K or MDB-stem-synth are used as gold sets, they are excluded from training.
- Manifests carry `split`, `speaker_id`, `corpus` and `licence`, and a test fails on any overlap.

### 6.5 Baselines (regenerated with one command)

- Praat defaults.
- Praat multi-ceiling sweep with FastTrack-style selection, both as specified in §5.4 (4000–8000 Hz for every talker) and FastTrack-faithful (per-talker-group ranges; decisions D-008).
- FormantNet and/or DeepFormants, if they run.
- PESTO and CREPE for f0.
- TVLP (time-varying LP with temporal smoothing) from `voice-dataset-labeler` (decisions D-007). The MK I Whisper probe is not scored.

## 7. Models

### 7.1 Front end

- **VQT** *(proposed)*: 16 kHz input, hop 10 ms, ~48 bins/octave from ~40 Hz to ~7.5 kHz (≈ 360 bins). Choose the variable-Q γ so the longest window is ≤ 64–80 ms. That resolves harmonics for f0 ≥ ~60 Hz and caps front-end latency (§10.1). Use log magnitude with streamable normalisation (running statistics, or none).
- Why log frequency: pitch changes and tract scaling are both translations there, so convolution weights shared along frequency match the physics, and transform T1 (§5.3) costs nothing.
- **f0 module:** PESTO (pretrained) for v1. The 2025 real-time version describes a streamable VQT; check the package's streaming support and its licence. Optionally add a joint f0 head later, with PESTO as its teacher.
- **Harmonic-comb conditioning channel:** from the f0 posterior, render Gaussian bumps at k·f0 (k = 1…K) on the VQT axis, weighted by voicing probability, and concatenate them as an extra input channel. It stays consistent under T1, and it tells the formant network where the harmonics are so it doesn't mistake them for resonances.

### 7.2 Teacher (offline, bidirectional)

- Encoder: a 2-D conv stack over (frequency, time) with no early frequency pooling, followed by a bidirectional temporal model (BiGRU or conformer) with ±1–2 s of context.
- Size: up to ~10 M parameters *(proposed)*.
- Used for: re-tracking real audio (§5.2, §5.4), ranking the active-learning queue (§5.5), distillation targets (§7.3), and the offline analysis mode.

### 7.3 Student (real-time, causal)

- Same front end, a conv stack that is causal in time, and a causal temporal model (GRU with carried state, or a TCN with cached activations). Left context of at least 300–500 ms, so vibrato can be exploited (§9).
- Lookahead L ∈ {0, 1, 2} frames (0–20 ms) *(proposed; ablate)*, implemented as delayed output.
- Size: ≤ 1–2 M parameters. Compute budget: ≤ ~10 % of one modern CPU core at 100 frames/s *(proposed)*.
- Training: supervised losses on §5.1 and §5.2 data, distillation from teacher heat-maps on large real pools (KL), and equivariance T1.
- Evidence that causality is affordable for formants: in FormantNet, a single-layer unidirectional LSTM did as well as a BiLSTM, and deeper models didn't help.

### 7.4 Heads (teacher and student)

- **f0 (if joint):** categorical over ~320 bins (20 cents, 40–1600 Hz), plus a voicing logit.
- **Formants F1–F4:** per-formant categorical heat-maps on a shared log-frequency axis (~100–7000 Hz, 20-cent bins, ≈ 370 bins) with Gaussian-blurred targets. Decode with a local weighted mean around the mode; σ comes from the local spread.
  - Cascade the heads: head n conditions on the outputs of heads < n. Shrem, Kreuk & Keshet use a shared encoder with one decoder per formant, conditioned on the lower formants, and report better cross-domain behaviour. Cascading enforces ordering and reduces F2/F3 swaps.
  - Softly mask bins below the previous formant's mode minus a margin.
- **Bandwidths B1–B4:** Gaussian on log B (mean and log-variance).
- **Sonorant/voiced mask:** marks where formant outputs are meaningful.
- **Nuisance parameters** (used only by the reconstruction loss): 2 poles (F, B, amplitude), 1 zero, source tilt (or Rd), and gain.

### 7.5 Losses

| Loss | Data | Notes |
|---|---|---|
| Heat-map CE (f0, Fn) | synthetic, resynthesised | Gaussian-blurred targets (σ ≈ 25–50 cents); masked by sonorant |
| Bandwidth NLL | synthetic, resynthesised | Gaussian in the log domain |
| Pseudo-label CE | real audio + candidates (§5.4) | weighted by posterior confidence; never hard-filtered |
| Reconstruction | raw real audio | §5.3; harmonic-sampled at high f0 |
| Equivariance T1/T2/T3 | real (+ synthetic) | §5.3 |
| Ordering | all | soft penalty if Fn+1 < Fn + margin (when not enforced structurally) |
| Smoothness | all | small penalty on the 2nd difference of expected log Fn within voiced runs |
| Distillation | real pools | KL(student ‖ teacher) per head; MSE on bandwidth means |

Loss weights are configurable. Every new loss lands together with an ablation (§12).

### 7.6 Why not frozen pretrained encoders

AST tokenises 16 × 16 mel patches with a 10-frame stride, so tokens are ~100 ms apart and each one merges 16 mel bins. AudioSet classification also never rewards resolving a resonance to within tens of Hz. Small from-scratch models on a log-frequency representation have the right inductive bias and are cheap enough for real time.

## 8. Deterministic back end

### 8.1 Harmonic amplitudes

- Least-squares fit of K harmonics at k·f0 (f0 from the model) over a pitch-adaptive window of ~3 periods, causal in streaming mode. Optionally include a linear f0 chirp term within the window.
- Outputs H1…HK (dB) and An (the strongest harmonic within ±max(f0/2, 15 % of Fn) of Fn).
- Test: exact recovery (< 0.1 dB) on clean, stationary synthetic frames.

### 8.2 Formant corrections (Iseli–Alwan)

Remove each formant's boost from the harmonic amplitudes, modelling the formant as a second-order resonator normalised to unity gain at DC:

```
r_i = exp(-pi * B_i / fs)      theta_i = 2*pi*F_i / fs      w = 2*pi*f / fs

G_i(f) [dB] = 20*log10( (1 - 2*r_i*cos(theta_i) + r_i^2)
                        / sqrt( (1 - 2*r_i*cos(theta_i - w) + r_i^2)
                              * (1 - 2*r_i*cos(theta_i + w) + r_i^2) ) )

H*(f) = H(f) - sum_{i=1..N} G_i(f)
```

- Cross-check against the VoiceSauce implementation before trusting it.
- Bandwidths: use the model's predictions, falling back to the Hawks & Miller (1995) formula (take the coefficients from the paper or the VoiceSauce source).
- Uncertainty: Monte Carlo over (Fn, Bn) samples drawn from the heat-maps gives intervals for H1\*–H2\* and the other corrected measures. This matters most at high f0, where F1 sits near H1 or H2 and the correction depends on exactly the least reliable estimates.
- Validation: on resynthesised data, the source's own H1–H2 is exact ground truth (§5.2).

### 8.3 Spectral slope measures

- Kreiman/Garellek set: H1\*–H2\*, H2\*–H4\*, H4\*–H2kHz\*, H2kHz\*–H5kHz, where H2kHz and H5kHz are the harmonics nearest 2 kHz and 5 kHz.
- Regression slope (dB/octave) of harmonic amplitudes against log2 frequency over a configurable band (default 0–5 kHz).
- Optional: CPP, HNR, alpha ratio, Hammarberg index.
- Caveat: all of these depend on the recording chain. Mic low-frequency roll-off suppresses H1, and distance and room colouration change the tilt. The learned front end can be made channel-robust; these measurements can't. Either use a fixed setup for comparisons over time, or measure the mic's response and correct for it (store a per-device calibration curve).

### 8.4 Vocal tract length

Per frame, fit the uniform-tube spacing through the origin with weights w_n = 1/σ_n² (σ in Hz):

```
x_n = (2n - 1) / 2
dF  = sum_n(w_n * F_n * x_n) / sum_n(w_n * x_n^2)
L [cm] = c / (2 * dF),  c ~= 35000 cm/s   =>   L ~= 17500 / dF[Hz]
```

- **Apparent VTL:** per frame from F1–F4. Optionally up-weight F3 and F4, which depend less on vowel identity.
- **Anatomical VTL:** a robust aggregate (e.g. weighted median) over many sonorant frames and vowels, reported with an interval.
- **Calibration:** fit a correction from formant-derived estimates to MRI-measured lengths. Uniform-tube estimates are biased; side branches such as the piriform sinuses pull the upper formants down. Start from Lammert & Narayanan's work on short-time VTL estimation.
- **Check that needs no labels:** within-speaker SD of anatomical VTL across utterances.
- Classical technique typically lowers the larynx, so apparent VTL changes during singing. Expose that rather than smoothing it away.

### 8.5 Causal smoothing for display

- Per-track Kalman filter in log frequency (random-walk or constant-velocity), using each frame's predicted σ² as the measurement noise. Confident frames update strongly; uncertain frames barely move the track. Reset on voicing onsets.
- Alternative: a one-euro filter. Raw, unsmoothed values always remain available.

## 9. High f0 and singing

- **Information limit:** at f0 = 600 Hz the envelope is only sampled every 600 Hz, and LPC/cepstral methods find harmonics rather than resonances. A single steady frame carries no information between harmonics.
- **Where extra information comes from:**
  - *Vibrato.* A ±50-cent vibrato at 600 Hz sweeps the 5th harmonic across ~175 Hz, so each harmonic's amplitude modulation over a cycle traces the local envelope slope. Exploiting this needs ≥ 300–500 ms of context and vibrato in the synthetic data.
  - *Breathiness.* Aspiration noise excites the tract between harmonics.
  - *Priors.* The same singer's lower-pitched passages, and vowel identity.
- **Honest output:** sometimes the right answer is "F1 is somewhere between H1 and H2". Keep the heat-map spread, and never display a confident point estimate without its uncertainty.
- **Register (M1/M2) work:** H1\*–H2\* is the natural indicator. At high pitch, though, F1 near H1 or H2 dominates the raw H1–H2, and the correction relies on the least reliable F1/B1 estimates. Always report H1\*–H2\* with its interval.
- **Singer's formant cluster:** in classical voices (especially lower ones) F3–F5 bunch together around ~2.5–3.5 kHz, and "which peak is F4" becomes ill-defined. Consider an extra cluster-centre output, and evaluate F3/F4 identity separately in this regime.

## 10. Real-time design

### 10.1 Latency budget *(proposed, for visual feedback)*

| Stage | Budget |
|---|---|
| Audio I/O buffer | 5–10 ms |
| Front-end window delay | ≤ ~32–40 ms for the lowest bins (half the longest window); less at higher bins |
| Student lookahead L | 0–20 ms |
| Compute per 10 ms hop | ≤ 1–2 ms |
| Display smoothing (effective) | 10–30 ms |
| **Total (perceived)** | **≲ 80 ms** |

To reduce front-end delay later, use asymmetric (low-delay) analysis windows for the low-frequency bins.

### 10.2 Streaming implementation

- Streaming VQT via cached (strided) convolutions, as in PESTO's real-time variant.
- The student's state (conv caches, GRU hidden state) is explicit input and output. A Python `StreamingAnalyzer` owns the state and ring buffers.
- The back end runs once per hop on ring buffers; the harmonic fit needs ~3 periods of audio.
- Tests: streaming output equals batch causal output (max abs difference < 1e-5) on random audio across chunk sizes; ONNX output equals PyTorch output.

### 10.3 Export and runtime

- Export the student to ONNX with state I/O and run it with ONNX Runtime on CPU *(proposed)*. A native (C++/Rust) port can come later if needed.
- Benchmark on each target device: per-hop latency distribution (p50/p99), real-time factor, CPU usage.

### 10.4 Demo app (prototype)

- Python: sounddevice (PortAudio) input → `StreamingAnalyzer` → live plots (pyqtgraph or similar).
- Show the spectrogram with harmonic markers, f0, F1–F4 with uncertainty bands, H1\*–H2\* with its interval, slope, and apparent/anatomical VTL.
- Record session audio and outputs to disk so the offline teacher can re-analyse them.

## 11. Training infrastructure (Ryzen AI MAX+ 395, 128 GB)

- **CPU (16 Zen 5 cores):** DataLoader workers for cheap on-the-fly augmentation, WORLD resynthesis for T2/T3 pairs, and tube-model synthesis to shards.
- **128 GB unified memory:** cache resynthesised corpora and feature shards in RAM; batch size is not a constraint for models this size.
- **GPU:** verified with AMD's ROCm 10 / torch 2.13 gfx1151 wheels (conv, matmul, FFT, bf16 autocast). MIOpen is unreliable on Windows (runtime-compiled kernels fail, some conv shapes crash), so it's disabled and training uses native kernels in bf16, which is the fastest measured configuration (decisions D-002). CPU and GPU share memory bandwidth on this platform, so profile whether synthesis or training is the bottleneck.
- **Precision:** try bf16 autocast and fall back to fp32 if numerics misbehave. The models are small enough that fp32 is affordable.
- **Fallback:** student-sized models are trainable on the CPU if the GPU stack misbehaves.
- **Data format:** shards (e.g. WebDataset tar files or numpy memmaps) with manifests; regeneration is deterministic under a fixed seed.
- **Tracking:** local (TensorBoard or MLflow). Every run logs its config, git commit, manifest hashes and eval report.

## 12. Milestones

### M0: Foundations
- [x] Repo skeleton, config system, CLI entry points.
- [ ] Manifests for all corpora (`path, corpus, speaker_id, split, sr, duration, licence`) and the leakage test (§6.4). *Gold sets and the leakage test done; training-pool manifests come with those corpora (M2).*
- [x] Gold-set loaders: Hillenbrand, PTDB-TUG, MIR-1K, MDB-stem-synth (all verified on real data). VTR-TIMIT deferred: no LDC licence.
- [x] Metrics (§6.2) with unit tests on hand-computed cases; bootstrap intervals by speaker.
- [x] Baselines (§6.5) written to `reports/baselines.md` by a single command (`va baselines`). Formants: Praat defaults, §5.4 sweep, FastTrack-faithful, TVLP. f0: PESTO, CREPE, Praat autocorrelation, plus PTDB-TUG phonation subsets from the laryngograph (decisions D-008, D-009).

**Accept when:** the baseline report regenerates end to end and every baseline, including the earlier TVLP tracker, is scored on the same footing.

### M1: Synthetic generator, back end v1, teacher v1
- [ ] Generator (§5.1) with label emission, sampler configs and channel augmentation; tests (poles equal labels, determinism).
- [ ] Hard-case synthetic suite.
- [ ] Back end v1 (§8.1–8.4) with synthetic tests: harmonic fit < 0.1 dB on clean frames; corrected H1–H2 matches source H1–H2 when the true F/B are supplied (< 0.5 dB *(proposed)*).
- [ ] Teacher v1 trained on synthetic data only.

**Accept when:** teacher v1 beats Praat defaults on Hillenbrand (MAE and detection rate; VTR deferred, no TIMIT licence), and the sim-to-real gap is documented per f0 band. See §13 #14 on how hard this bar is.

### M2: Resynthesis labeller, teacher v2
- [ ] Inverse filtering (IAIF; QCP later), track smoothing and perturbation, time-varying resonator cascade, QC, and label export including source H1–H2.
- [ ] Resynthesised speech and singing corpora.
- [ ] Teacher v2 trained on synthetic + resynthesised data.
- [ ] Iseli–Alwan validation with predicted F/B, by f0 band.

**Accept when:** gold-set metrics improve over v1 beyond bootstrap noise, with no hard-suite regression.

### M3: Self-supervision and candidate rounds
- [ ] Differentiable renderer and reconstruction loss (harmonic-sampled at high f0).
- [ ] Equivariance T1 (on the fly) and T2/T3 (precomputed WORLD pairs).
- [ ] Candidate generator and Viterbi with teacher emissions; gated LP refinement.
- [ ] Rounds of re-track → resynthesise → retrain (at most 3 rounds unless still improving).
- [ ] Ablation table showing the contribution of each loss and data source.
- [ ] (Optional) Active-learning queue and correction tooling.

**Accept when:** each accepted round improves gold metrics beyond noise, and the ablations justify every retained loss.

### M4: Real-time student
- [ ] Streaming VQT and equivalence tests.
- [ ] Causal student with configurable lookahead, trained with distillation and supervised losses.
- [ ] `StreamingAnalyzer` with explicit state; ONNX export and equivalence test.
- [ ] Latency and compute benchmarks on the target devices.

**Accept when:** the student is within an agreed tolerance of the teacher on gold sets *(proposed: detection rate within 3 points, MAE within 10 %)*, and the latency (§10.1) and compute budgets are met.

### M5: Calibration and packaging
- [ ] Per-head temperature scaling on held-out real data; coverage report.
- [ ] VTL calibration against MRI; within-speaker stability report.
- [ ] Causal display smoothing (§8.5).
- [ ] Demo app (§10.4).

**Accept when:** interval coverage is within a few points of nominal and the demo runs live within budget.

### M6 (optional): Physical rig and personal data
- [ ] Printed tracts, driver and sweep measurement; recordings across f0 with vibrato.
- [ ] High-f0 evaluation of the teacher and student.
- [ ] Personal singing set with targeted hand corrections (held-out portion kept as gold).

## 13. Open decisions

| # | Decision | Default *(proposed)* |
|---|---|---|
| 1 | Internal sample rate | 16 kHz (24 kHz if more headroom above F4 is needed) |
| 2 | Hop size | 10 ms |
| 3 | Student lookahead | ablate 0 / 10 / 20 ms |
| 4 | f0 source | PESTO module for v1; joint head later |
| 5 | Reconstruction renderer | cascade + tilt + nuisance; ablate a parallel (FormantNet-style) renderer |
| 6 | VTL outputs | apparent per frame + anatomical aggregate |
| 7 | Slope measure set | Kreiman/Garellek set + dB/octave regression; CPP/HNR optional |
| 8 | Real-time target hardware and runtime | CPU + ONNX Runtime; devices TBD |
| 9 | TIMIT/VTR access | needs an LDC licence; otherwise rely on Hillenbrand and the other sets |
| 10 | Physical rig | defer to M6 unless high-f0 results stall |
| 11 | Accompaniment | out of scope for v1 |
| 12 | Exposed formants | F1–F4, plus a cluster-centre output in the singer's-formant regime? |
| 13 | PESTO weights vs MIR-1K gold (decisions D-008) | Bundled PESTO checkpoints are trained on MIR-1K. Proposed: retrain PESTO (self-supervised, so no labels are needed) on the training pools, which exclude every gold set, and use those weights in §7.1. Alternative: drop MIR-1K as a gold set and rely on MDB-stem-synth (which CREPE has seen) plus PTDB-TUG |
| 14 | M1 acceptance vs Hillenbrand's LPC-derived reference (decisions D-008) | Praat defaults already reach DR(20 %, 200 Hz) ≈ 99 / 93 / 88 % on Hillenbrand F1–F3, partly because the reference is LPC-based. Proposed: keep "beat Praat defaults" but judge it per f0 band and speaker group (children, high f0) with bootstrap intervals, and weight the hard-case synthetic suite and resynthesised sets more heavily |
| 15 | f0 output on non-modal phonation (decisions D-009) | On period-doubled frames autocorrelation trackers report the two-cycle period and neural trackers the cycle rate; creak has no single f0. Proposed: output the cycle rate, and have the voicing/voice-quality heads flag "irregular" frames (PTDB-TUG's EGG categories give labelled examples), so downstream users can tell an f0 estimate from a guess. Score f0 accuracy on modal frames, with modal < 85 Hz as the hard-mode target, and describe non-modal behaviour rather than score it. Also: PESTO breaks down below 70 Hz, so retrained f0 weights (#13) need low male speech and vocal fry in the training pool |

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sim-to-real gap | Resynthesis (§5.2), self-supervision (§5.3), gold-set gating |
| Inverse-filter ripple becomes label noise | Large perturbations, parametric-source variant, QC rejection |
| Equivariance satisfied by features that aren't formants | Always train jointly with supervised losses; watch gold metrics |
| Correlated tracker errors and selection bias | Trackers as candidates only; confidence weighting; hard cases from synthesis |
| High-f0 information limit | Vibrato context, breathiness, priors, honest uncertainty; physical-rig evaluation |
| Back-end measures depend on the recording chain | Fixed setup or per-device calibration; flag uncalibrated sessions |
| Teacher–student gap | Measure every round; tune context and lookahead |
| Data leakage | Manifests plus an automated overlap test |
| Licensing | Licence field per item; restricted corpora never shipped in artefacts |
| ROCm/toolchain issues on gfx1151 | Verify early; CPU fallback for student-scale training |

## 15. Proposed repository layout

```
vocal_analysis/
  synth/      sources.py  tract.py  filters.py  channel.py  sampler.py
  resynth/    inverse_filter.py  tracks.py  labeller.py
  trackers/   praat_sweep.py  fasttrack_select.py  lp_refine.py  viterbi.py
  features/   vqt.py  vqt_stream.py  harmonic_comb.py
  models/     encoder.py  heads.py  teacher.py  student.py  render.py
  losses/     supervised.py  reconstruction.py  equivariance.py  distill.py
  backend/    harmonics.py  corrections.py  slope.py  vtl.py  smoothing.py
  eval/       datasets/  metrics.py  report.py  leakage.py
  rt/         streaming.py  export_onnx.py  bench.py  demo/
configs/
data/manifests/
reports/
tests/
```

## 16. References

Some entries were compiled from memory; verify bibliographic details before citing them externally.

**Formant tracking**
- Lilley, J., & Bunnell, H. T. (2021). Unsupervised training of a DNN-based formant tracker. Interspeech 2021. https://www.isca-archive.org/interspeech_2021/lilley21_interspeech.pdf (code: https://github.com/NemoursResearch/FormantNet)
- Dissen, Y., Goldberger, J., & Keshet, J. (2019). Formant estimation and tracking: A deep learning approach. JASA 145(2), 642–653.
- Shrem, Y., Kreuk, F., & Keshet, J. (2022). Formant estimation and tracking using probabilistic heat-maps. Interspeech 2022. https://cris.iucc.ac.il/en/publications/formant-estimation-and-tracking-using-probabilistic-heat-maps-3/
- Kadiri, S. R., Huang, K., Hagedorn, C., Byrd, D., Alku, P., & Narayanan, S. (2025). Formant tracking by combining deep neural network and linear prediction. IEEE Open Journal of Signal Processing.
- Gowda, D. N., Bollepalli, B., Kadiri, S. R., & Alku, P. (2021). Formant tracking using quasi-closed phase forward-backward linear prediction analysis and deep neural networks. IEEE Access 9, 151631–151640. https://doi.org/10.1109/access.2021.3126280
- Alku, P., Kadiri, S. R., & Gowda, D. (2023). Refining a deep learning-based formant tracker using linear prediction methods. Computer Speech & Language.
- Gowda, D., Airaksinen, M., & Alku, P. (2017). Quasi-closed phase forward-backward linear prediction analysis of speech for accurate formant detection and estimation. JASA 142(3), 1542–1553.
- Mehta, D. D., Rudoy, D., & Wolfe, P. J. (2012). Kalman-based autoregressive moving average modeling and inference for formant and antiformant tracking. JASA 132(3), 1732–1746.
- Schiel, F., & Zitzelberger, T. (2018). Evaluation of automatic formant trackers. LREC 2018.
- Barreda, S. (2021). Fast Track: fast (nearly) automatic formant-tracking using Praat. Linguistics Vanguard.
- Story, B. H., & Bunton, K. (2016). Formant measurement in children's speech based on spectral filtering. Speech Communication 76, 93–111.
- Mikušová, I. (2021). Estimating vocal tract resonances of synthesized high-pitched vowels using CNN. Diploma thesis, TU Wien. https://repositum.tuwien.at/handle/20.500.12708/19264

**Pitch**
- Riou, A., et al. (2025). PESTO: Real-time pitch estimation with self-supervised transposition-equivariant objective. https://arxiv.org/abs/2508.01488 (code: https://github.com/SonyCSLParis/pesto; pip: `pesto-pitch`)
- Riou, A., Lattner, S., Hadjeres, G., & Peeters, G. (2023). PESTO: Pitch estimation with self-supervised transposition-equivariant objective. ISMIR 2023. https://arxiv.org/abs/2309.02265
- Kim, J. W., Salamon, J., Li, P., & Bello, J. P. (2018). CREPE: A convolutional representation for pitch estimation. ICASSP 2018.
- Salamon, J., Bittner, R. M., Bonada, J., Bosch, J. J., Gómez, E., & Bello, J. P. (2017). An analysis/synthesis framework for automatic f0 annotation of multitrack datasets. ISMIR 2017.

**Source, voice quality and corrections**
- Fant, G., Liljencrants, J., & Lin, Q. (1985). A four-parameter model of glottal flow. STL-QPSR 26(4).
- Fant, G. (1995). The LF-model revisited: transformations and frequency domain analysis. STL-QPSR 36(2–3).
- Alku, P. (1992). Glottal wave analysis with pitch synchronous iterative adaptive inverse filtering. Speech Communication.
- Airaksinen, M., Raitio, T., Story, B., & Alku, P. (2014). Quasi closed phase glottal inverse filtering analysis with weighted linear prediction. IEEE/ACM TASLP 22(3).
- Iseli, M., Shue, Y.-L., & Alwan, A. (2007). Age, sex, and vowel dependencies of acoustic measures related to the voice source. JASA 121(4).
- Hawks, J. W., & Miller, J. D. (1995). A formant bandwidth estimation procedure for vowel synthesis. JASA 97(2).
- Shue, Y.-L., Keating, P., Vicenik, C., & Yu, K. (2011). VoiceSauce: A program for voice analysis. ICPhS 2011.
- Kreiman, J., Gerratt, B. R., Garellek, M., Samlan, R., & Zhang, Z. (2014). Toward a unified theory of voice production and perception. Loquens 1(1).
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: a vocoder-based high-quality speech synthesis system for real-time applications. IEICE Trans. Inf. & Syst. E99-D(7).

**Vocal tract and VTL**
- Story, B. H. (2005). A parametric model of the vocal tract area function for vowel and consonant simulation. JASA 117(5).
- Birkholz, P. VocalTractLab. https://www.vocaltractlab.de
- Lammert, A. C., & Narayanan, S. S. (2015). On short-time estimation of vocal tract length from formant frequencies. PLoS ONE 10(7).
- Lim, Y., et al. (2021). A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images. Scientific Data 8.
- Fitch, W. T. (1997). Vocal tract length and formant frequency dispersion correlate with body size in rhesus macaques. JASA 102(2).
- Reby, D., & McComb, K. (2003). Anatomical constraints generate honesty: acoustic cues to age and weight in the roars of red deer stags. Animal Behaviour 65.
- Fujimura, O., & Lindqvist, J. (1971). Sweep-tone measurements of vocal-tract characteristics. JASA 49(2B).
- Joliveau, E., Smith, J., & Wolfe, J. (2004). Tuning of vocal tract resonance by sopranos. Nature 427.
- Sundberg, J. (1987). The Science of the Singing Voice. Northern Illinois University Press.

**Datasets**
- Deng, L., Cui, X., Pruvenok, R., Chen, Y., Momen, S., & Alwan, A. (2006). A database of vocal tract resonance trajectories for research in speech processing. ICASSP 2006.
- Hillenbrand, J., Getty, L. A., Clark, M. J., & Wheeler, K. (1995). Acoustic characteristics of American English vowels. JASA 97(5), 3099–3111.
- Pirker, G., Wohlmayr, M., Petrik, S., & Pernkopf, F. (2011). A pitch tracking corpus with evaluation on multipitch tracking scenario (PTDB-TUG). Interspeech 2011.
- Hsu, C.-L., & Jang, J.-S. R. (2010). On the improvement of singing voice separation for monaural recordings using the MIR-1K dataset. IEEE TASLP.

**Models and tools**
- Gong, Y., Chung, Y.-A., & Glass, J. (2021). AST: Audio Spectrogram Transformer. Interspeech 2021.
- Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP: Differentiable digital signal processing. ICLR 2020.
- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics 71.
