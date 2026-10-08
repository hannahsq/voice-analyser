# Decisions log

Divergences from, and refinements of, [voice-analyser-mk2-plan.md](voice-analyser-mk2-plan.md) (plan §0).

## D-001 (2026-10-07): torch is not a declared dependency

The training stack uses AMD's ROCm wheels for gfx1151 (`torch 2.13.0+rocm10.0.0`, `rocm 10.0.0`, HIP 7.15). If `torch` appeared in `pyproject.toml`, `pip install -e .` could replace that build with a generic wheel. So torch is installed separately from AMD's index, and the package is installed with `pip install -e . --no-deps` plus the explicit dependencies. Everything outside model code (metrics, manifests, loaders) imports without torch.

## D-002 (2026-10-07): GPU verified on gfx1151; MIOpen disabled

Verified on the Radeon 8060S (gfx1151), with ~78 GiB visible to torch:

| Op | Result |
|---|---|
| fp32 matmul | ok, ~3.4 TFLOPS on 4096²; matches CPU |
| Conv2d forward + backward via MIOpen, fp32 and bf16 autocast | ok, matches CPU |
| FFT | ok |
| `nn.GRU` via MIOpen | **fails**: `miopenStatusUnknownError`. MIOpen compiles these kernels at runtime with hipRTC, which can't find `<type_traits>` on this Windows build |
| `nn.GRU` with cuDNN/MIOpen disabled (native kernels), fp32 and bf16 | ok |

~~Decision: `device.disable_miopen: rnn`~~. **Revised the same day: the default is now `all`, MIOpen off everywhere.** Batch norm hits the same compile error (CREPE), and MIOpen convolutions on CREPE's layer shapes crashed the GPU ("unspecified launch failure"). With MIOpen off, PyTorch's native kernels are slower in fp32 but fastest overall in bf16:

| 4× Conv2d(32→32), batch 16×128×200, fwd+bwd | fp32 | bf16 autocast |
|---|---|---|
| MIOpen | 45.6 ms | 67.5 ms |
| native | 93.5 ms | **38.0 ms** |

So train in bf16 with native kernels. `tests/test_device.py` is the regression check.

To re-enable MIOpen later: hipRTC needs C++ standard headers, and none are on this machine (no Visual Studio / Build Tools, and the ROCm wheels don't bundle libc++). The most likely fix is VS 2022 Build Tools with "Desktop development with C++". That's not worth doing now: it only speeds up fp32, and it wouldn't explain the convolution crash. Revisit after a ROCm upgrade.

## D-003 (2026-10-07): speaker ids are corpus-prefixed; TIMIT-derived sets share corpus `timit`

`speaker_id = "<corpus>:<local id>"`, so the same local id in different corpora can't collide by accident. Sets derived from another corpus (VTR-TIMIT → TIMIT) register under the parent corpus's name, so the leakage test sees shared speakers. Speakers who appear in two different corpora under different ids can't be detected automatically; list them as explicit exclusions if that turns up.

## D-004 (2026-10-07): formant detection-rate criterion, AND vs OR (open)

"Within (20 %, 200 Hz)" (plan §6.2) can mean both bounds or either bound. `metrics.detection_rate` supports both and defaults to `mode="and"` (stricter). Check the definition used by the QCP-FB and FormantNet papers before comparing against their reported numbers, and report under that definition.

## D-005 (2026-10-07): gold-set loader formats need verifying against real data

The loaders were written from the documented layouts before any corpus was on disk. Each module marks its assumptions with `ASSUMPTION`. Check them the first time each corpus is loaded:

- ~~Hillenbrand~~: verified 2026-10-07 (1,668 tokens, 139 speakers). `timedata.dat` is now wired in, so each token yields four frames: steady state (the mean of the two judges' times) and 20/50/80 % of the nucleus.
- ~~PTDB-TUG~~: verified (4,718 utterances, 20 speakers). The frame offset is **20 ms**, not the assumed 16 ms; it was measured against a laryngograph f0 track (see `ptdb_tug.py`).
- ~~MIR-1K~~: verified (1,000 clips, 19 singers). The audio is in `Wavfile/`; frames are centred at 20 ms + i·20 ms; voice is on the right channel (energy–voicing correlation 0.82 right vs 0.06 left).
- ~~MDB-stem-synth~~: verified (hop 128 samples at 44.1 kHz; labels end within one hop of the audio). 60 of the 230 stems are singing voice (male/female singer, vocalists, choir) according to the MedleyDB metadata pinned at `537bb7b`, giving 30 artists. The tarball contains macOS AppleDouble `._*` files, which extraction now skips.
- Licence strings are marked `VERIFY` until checked.

## D-006 (2026-10-07): corpus sources (`va download`)

URLs and checksums are pinned in `vocal_analysis/data/download.py`. No Hugging Face mirrors are used (usage tracking).

| Corpus | Source | Why |
|---|---|---|
| Hillenbrand | `santiagobarreda/hillenbrand_et_al_1995` @ `6dd44de`, `h95-alldata.zip` | The original wmich.edu page no longer resolves; this is the original distribution, mirrored with Hillenbrand's permission |
| PTDB-TUG | TU Graz `SPEECH_DATA_ZIPPED.zip` (md5 published by the host) | Official |
| MIR-1K | figshare article 5802891 | The mirlab.org link returns 404; Zenodo 3532216 is restricted |
| MDB-stem-synth | Zenodo 1481172 | Official; CC BY-NC 4.0 |
| TIMIT / VTR-TIMIT | not used | No LDC licence for now (plan §13 #9); Hillenbrand is the primary formant gold set |

Archives are also copied to `Z:\datasets\voice-analyser\` (NAS) as a backup; training reads the local copies under `data/`.

## D-007 (2026-10-07): earlier code as baselines; the MK I Whisper probe is dropped

- **TVLP formant tracker** from `hannahsq/voice-dataset-labeler`: a git submodule at `external/voice-dataset-labeler`, pinned to `0ab5568`. `extractors/tvlp.py` imports only numpy and the standard library, so the baseline adapter loads that single file by path (`importlib.util.spec_from_file_location`). Nothing goes on `sys.path`, so the repo's generic top-level names (`extractors`, `labeller`, `synth`) can't clash with anything. A submodule rather than a copy keeps the exact commit that was scored on record. The usual alternative, a `pkg @ git+https://...@sha` dependency, needs the other repo to be packaged, which this one isn't.
- **MK I Whisper-probe** (`hannahsq/formant-extractor`): not scored. It was a small experiment training an MLP head on frozen Whisper features. Re-scoring it would mean fetching Whisper weights from the Hugging Face hub and rebuilding its evaluation, because its `KFold` split by token put the same speakers in train and validation folds. Plan §7.6 already argues against frozen encoders.
- **"AST-head probe"** in plan §6.5: no such code exists, and the reference probably came from chat memories. Removed from the plan.

## D-008 (2026-10-07): baseline protocol and findings (M0)

`va baselines` writes `reports/baselines.md` and `.json`. Predictions are cached per (baseline, corpus) in `runs/baselines/`, keyed by baseline version and manifest hash; long runs checkpoint and resume.

**Formant baselines (Hillenbrand: 4 frames per token, F4 at steady state only).**
- `praat_default`: Burg with Praat's defaults (5500 Hz ceiling for everyone).
- `praat_sweep`: the plan's §5.4 sweep, 4000–8000 Hz for every talker, with the ceiling chosen by DCT smoothness. It does **worse** than defaults for children (girls F2 DR 70 % vs 83 %). With low ceilings, merged formants give smooth tracks that a smoothness score in Hz rewards; for girls it picked a median 4750 Hz ceiling where the most accurate is about 6000 Hz.
- `praat_fasttrack`: FastTrack's documented protocol, with the ceiling range set by talker size (men 4500–6500, women 5000–7000, children 5500–7500 Hz), its error measure (intercept + 5 cosines, frames with ≥ 4 formants), and selection on F1–F3. It's the strongest baseline. It uses the speaker group plus the oracle vowel span, as FastTrack users would supply them, and both are stated in the report. The ranges come from FastTrack's own documentation, **not** from tuning on Hillenbrand.
- `tvlp`: voice-dataset-labeler @ `0ab5568`.

**Caveat on Hillenbrand as a formant reference.** Its values are LPC peaks with hand checking, so LPC-family trackers agree with it partly by construction (plan §5.4's correlated-error point). Praat defaults already reach DR(20 %, 200 Hz) ≈ 99 / 93 / 88 % on F1–F3. This bears on the M1 acceptance criterion; see plan §13 #14.

**f0 baselines.** PESTO (`mir-1k_g7`, the package default) and CREPE (`full`, torchcrepe, Viterbi), both at 16 kHz with a 10 ms hop and evaluated at reference times by nearest frame (±5 ms). The synthetic step test confirms both report frame-centre times. Voicing thresholds are fixed in advance (PESTO confidence ≥ 0.5; CREPE periodicity ≥ 0.21, from the torchcrepe docs) and never tuned on gold data. CREPE runs in fp32 as published: bf16 was 3.4× faster but moved estimates by a median 8 cents. PESTO is called with a cached model (bit-identical to `pesto.predict`, which reloads the checkpoint on every call).

**Training overlap with gold sets** (marked † in the report): both bundled PESTO checkpoints are trained on MIR-1K; CREPE's training data includes MIR-1K and MDB-stem-synth. PTDB-TUG is the only f0 gold set that neither has seen.

**Plan conflict this exposes:** §7.1 uses pretrained PESTO as the f0 module, but its weights were trained on MIR-1K, which §6.4 treats as gold. Any pipeline f0 number on MIR-1K would then be contaminated. Options are in plan §13 #13.

## D-009 (2026-10-07): PTDB-TUG failure modes; laryngograph phonation subsets; Praat f0 baseline

**Praat autocorrelation added as a traditional f0 baseline** (`praat_ac`: parselmouth `to_pitch_ac`, defaults except a 50–1000 Hz range that covers every gold set, sampled on the 10 ms grid). It isn't trained on anything, so it's a clean reference point. On raw pitch accuracy it's within about 2 points of PESTO on every corpus, it decides voicing much better on speech, and it runs at 0.002× real time on the CPU. The step-timing test now uses a fifth (220→330 Hz) rather than an octave: after an octave step the old period still fits the signal, which delays autocorrelation trackers for reasons unrelated to timestamps.

**What the PTDB-TUG errors are.** PTDB-TUG's reference is RAPT run on the laryngograph (EGG) channel, which we also have. For every voiced reference frame (period T), `eval/phonation.py` correlates the high-passed EGG with itself at lag T (r1) and T/2 (rh) and assigns a category from the EGG alone:

| Category | Rule | Meaning | Share |
|---|---|---|---|
| modal | r1 ≥ 0.75, rh ≤ −0.5 | regular vibration at the reference f0 | 83.0 % |
| ref_error_2x | rh ≥ 0.75 | folds cycle at 2 × the reference: the reference halved | 0.5 % |
| alternating | r1 ≥ 0.75, −0.5 < rh < 0.75 | alternate cycles differ (period doubling); reference and 2 × both defensible | 3.7 % |
| aperiodic | r1 < 0.75, rh < 0.75 | creak / irregular phonation; no single f0 | 9.2 % |
| weak_egg | EGG RMS < 0.1 × the file's median | no usable EGG, so the reference is unreliable | 3.6 % |

The thresholds are round numbers read off the EGG distributions (modal frames: median r1 0.99, rh −0.81). Estimator errors played no part in choosing them, so the categories can't favour any estimator. Synthetic EGG tests in `tests/test_phonation.py` cover each category.

Findings (all three baselines on all of PTDB-TUG; the full tables are in `reports/baselines.md`):
- **Reference errors are rare.** Only 0.5 % of frames show the reference halved, and there the neural trackers are right (PESTO 89 %, CREPE 88 % at 2 × the reference). Praat repeats the halving 59 % of the time, since it shares the autocorrelation family with RAPT.
- **Period doubling splits trackers by method.** On alternating frames, autocorrelation reports the two-cycle period (Praat 87 % at the reference); neural trackers often report the cycle rate (PESTO 31 %, CREPE 33 % at 2 ×). Neither is wrong.
- **Creak is a voicing-decision problem.** On aperiodic frames PESTO judges 22 % voiced, CREPE 78 %, Praat 70 %. PESTO's 0.5 confidence threshold is also strict on speech generally: it judges only 67 % of modal frames voiced (CREPE and Praat 95 %).
- **PESTO fails on low modal voices.** On male modal frames its error grows as f0 falls: median +5 cents at 150–250 Hz, +13 at 85–100, +21 at 70–85, and +1062 below 70 Hz (it breaks down). Praat stays within 0–7 cents across the whole range. CREPE holds up better at low f0 (72 % RPA on modal frames below 85 Hz, against PESTO's 63 % and Praat's 73 %). This is a training-data gap: PESTO's checkpoint was trained on MIR-1K singing, with little below about 80 Hz.
- **An earlier heuristic was dropped.** Flagging reference frames below 0.6 × the speaker median was tried first. Only 13.5 % of the frames it flagged were reference errors (41 % alternating, 28 % weak EGG, 14 % aperiodic), so it was replaced by the EGG categories before any M1 model was scored.

**Evaluation protocol for PTDB-TUG f0 from here on** (`va baselines` implements it):
- Headline RPA over **all** voiced reference frames (unchanged).
- RPA on **modal** frames: the fairest accuracy figure. Also **modal with reference below 85 Hz**: the f0 is well defined but trackers miss it, so this is the hard-mode accuracy target.
- Female/male splits.
- For the non-modal categories, an outcome table (% at the reference / 2 × / ½, % judged voiced) rather than an accuracy, because "the" f0 is undefined or the reference unreliable there.

| PTDB-TUG RPA % | all | modal | modal, < 85 Hz | female modal | male modal |
|---|---|---|---|---|---|
| PESTO | 84.5 | 90.5 | 62.8 | 96.0 | 85.7 |
| Praat AC | 83.9 | 89.9 | 73.4 | 94.0 | 86.4 |
| CREPE | 84.9 | 91.5 | 72.1 | 95.6 | 87.9 |

Open for M1 (plan §13 #15): what the model should output on alternating and aperiodic frames: the cycle rate, the two-cycle period, or "irregular, no single f0" with the voicing/voice-quality heads flagging it.

## D-010 (2026-10-08): M1 generator, back end and teacher v1 design

**Generator (`vocal_analysis/synth/`).**
- *Voiced component by additive synthesis.* Harmonic k at each 1 ms control point has complex amplitude c_k(Rd) · H(k·f0), where c_k is the LF source spectrum (Fant 1995 Rd parameterisation, tabulated on a 0.01 Rd grid) and H is the resonator cascade's exact response. The output is band-limited at any f0 up to 1600 Hz, harmonic amplitudes equal source × |H(k·f0)| exactly (the §5.1 test, to 1e-8 dB), and the source's own harmonic amplitudes are emitted as labels (exact H1–H2 for §8.2 validation). Jitter and shimmer are per glottal period. The LF flow derivative already includes lip radiation, so no separate radiation filter is applied to the voiced part.
- *Noise components* (aspiration modulated by glottal opening with level set by HNR, whisper, frication) run through the same resonators in the time domain (numba). One resonator definition (`synth/filters.py`) serves synthesis, the back end's Iseli–Alwan correction and the tests.
- *Tract: resonator cascade (plan §5.1 fallback) now; lossy-tube sampler later.* F1–F3 targets come from Peterson & Barney (1952) adult-male vowel means scaled by 17.5 cm / L (a population prior from a different study, not gold-set data), mixed with 30 % broad log-uniform coverage. F4–F8 sit near the uniform tube's (2n − 1)c/4L. Bandwidths are Hawks & Miller × log-uniform(0.6, 1.6). Optional nasal and subglottal pole–zero pairs and a singer's-formant cluster are included. All eight oral poles are labelled. Story's area-function model, which needs his published mode shapes, is the next sampler to add.
- *f0 contours are synthetic* (speech-like declination and wander, sung notes with portamento and vibrato, glides, register steps; centre f0 log-uniform with 45 % from 300–1300 Hz). Borrowing PESTO contours from real pools (§5.1) waits for non-gold real audio (M2) *[now M3; milestones renumbered in D-014]*.
- *Channel augmentation* on the fly: synthetic RIRs (per-band T60, early reflections), white/pink/brown/hum/babble-like noise, high/low-pass, parametric EQ, gain, rare clipping, and an emulated phone channel. Codecs wait for ffmpeg.
- *Data:* `va synth train` writes 100k clips (~83 h, seed 1) as int16 shards; `va synth suite` writes the fixed-seed hard-case suite (10 cases × 64 clips: reference, high_f0, f1_near_h1, f1_near_h2, f2_f3_close, nasal, breathy, wide_vibrato, low_snr, phone).

**Back end v1 (`vocal_analysis/backend/`).** The harmonic LS fit models every harmonic below Nyquist (fitting only the first K lets the others leak into the estimates) over a Hann-weighted 3-period window. It's within 0.1 dB on clean frames from 87 to 1180 Hz. Iseli–Alwan correction with the true poles recovers source H1–H2 within 0.5 dB. Finding: correcting with F1–F4 only leaves a bias that grows with f0 (0.85 dB at f0 = 520 Hz), because DC-normalised resonators above F4 still boost H2 relative to H1. `correct(..., upper_from_vtl=True)` adds F5–F8 at the uniform-tube positions implied by the apparent VTL, which brings it back within 0.5 dB. Hawks & Miller coefficients are from VoiceSauce and still marked VERIFY.

**Teacher v1 (`vocal_analysis/models/teacher.py`), ~6.5 M parameters.**
- *Own f0 head instead of PESTO* (plan §13 #4). PESTO's weights were trained on a gold set (MIR-1K, §13 #13) and it fails below 70 Hz (D-009), while synthetic f0 labels are exact. Stage 1 predicts f0 and voicing; the harmonic comb (§7.1) is rendered from its expected f0 (max over harmonics, not sum, so close high harmonics don't merge) and fed to stage 2. During the first 30 % of training the comb uses the true f0 with decaying probability. `--no-comb` is the ablation.
- *Front end:* VQT with 48 bins/octave from 40 Hz to 7.45 kHz (363 bins), windows min(Q·fs/f, 80 ms), as one matrix product on centred frames (so streaming equals batch by construction). Input is per-frame spectral shape (gain-invariant) plus frame level.
- *Encoder:* dilated 2-D convs at full frequency resolution, then two frequency halvings, flatten, 2-layer BiGRU. *Heads:* f0 (320 bins, 20 cents), F1–F4 cascaded heat-maps (368 bins, 20 cents) with a soft mask below the previous formant, log-bandwidth mean and variance, voicing and sonorant logits. Targets are Gaussian with σ = 35 cents; decoding is the local weighted mean within ±80 cents of the mode.
- *Throughput:* with MIOpen off (D-002), PyTorch's fallback convolution is per-sample im2col + GEMM and reaches only ~1 TFLOPS, against 12.7 TFLOPS for bf16 matmul. A one-GEMM channels-last formulation was 2–3× slower still, because of the copies. So full-resolution channels are kept small (24) instead of reducing frequency resolution early.

## D-011 (2026-10-08): Story area-function tract and ffmpeg codecs

**Tract (`synth/tube.py`, plan §5.1 "preferred").** Story's (2005) tiers I–III on 44 tubelets, through a lossy transmission-line model: viscous and thermal boundary-layer losses (Flanagan 1972), yielding walls (Ishizaka, French & Flanagan 1975), glottis closed, lip radiation as Sondhi & Schroeter's (1987) piston load. All loss terms are analytic in s, so the poles are exact complex roots of the denominator, found by Newton's method continued from the lossless tube's real roots as losses are ramped up (retried with finer steps if a root is lost). F = Im s / 2π, B = −Re s / π, the resonator convention of `filters.py`. The tube's poles drive the existing resonator cascade, so the audio, the harmonic-amplitude test and the back end are unchanged and the labels stay exact by construction. Below 4 kHz the cascade stays within 4 dB of the tube's own response for open vowels; the residual comes from frequency-dependent losses.
- *Checks:* the lossless uniform tube gives (2n − 1)c/4L, and two-tube shapes match the analytic resonance condition. The neutral shape gives 632 / 1486 Hz against Story's Fig. 10 (≈ 610 / 1470), the ends of his [ɪʊ] and [ɑu] trajectories match the figure, and the q-plane mesh spans his F1–F2 region.
- *Table IV:* the 2005 printing has the vowel coefficients scrambled. With Table III they give, for example, an /i/ with a 0.03 cm² lip opening. The erratum (Story 2009, JASA 125(2), 1248; `data/downloads/B_H_Story_errata.pdf`) gives the correct values, which are in `STORY_VOWELS`. Through this tube they give a textbook vowel space (/i/ 305 / 2436, /ɑ/ 806 / 1303, /u/ 334 / 1062 Hz), and a test checks its shape. Table III itself needs no correction (the erratum only renames ω to Ω).
- *Sampling:* per clip, the speaker's L comes from the sampler as before. Story's Ω is warped by a smooth random factor (another speaker). The pharynx is scaled relative to the oral cavity by (L / 17.46)^β with β ∈ [0, 0.6], giving shorter tracts a relatively shorter pharynx (Fitch & Giedd 1999). Areas scale with (L / 17.46)² × log-uniform(0.75, 1.33), the cross-section perimeter factor is 1–1.8 (losses), and bandwidths are scaled by log-uniform(1, 2) for glottal losses the closed-glottis tube leaves out. Per target, q1 and q2 are uniform over the Table IV box (Story's Fig. 10), redrawn if the tract narrows below 0.1 cm². With p = 0.15 there is an approximant-like tier II constriction (0.15–0.8 cm², kept open so frames stay sonorant), and with p = 0.3 each, a larynx height change (±0.5 cm) and lip protrusion (−0.3 to +1 cm, tier III). Targets are interpolated as before. Nasal and subglottal pole–zero pairs still come from `tract.sample_extras` (tier IV is not modelled). `length_cm` stays the anatomical L; per-target lengths with tier III changes are in `meta["tube"]`.
- *Mix:* `SynthConfig.p_tube = 0.6`; the rest use the resonator-cascade prior, which also covers formant patterns outside one speaker's two-mode space. A tube clip takes ~0.9 s to generate against ~0.4 s for a cascade clip.

**Codecs (`synth/codecs.py`).** ffmpeg 9.0 (gyan.dev full build via winget): Opus (voip, 8–32 kb/s), AAC (16–48), MP3 (16–64), AMR-WB (6.6–23.85) and AMR-NB (4.75–12.2, at 8 kHz). These are the plan's three plus the two phone codecs. Each encode/decode round trip runs in memory over pipes (~90 ms). The delay of each (codec, bitrate) is calibrated once per process by cross-correlation (AAC 1024 samples, MP3 1105, AMR-WB 95, AMR-NB 80, Opus 0–2) and removed, so labels stay aligned. Tests check the waveform lag on noise to within 1 sample and the energy envelope of a harmonic burst to within 1 ms. Low-rate speech codecs don't keep phase, so a periodic waveform's cross-correlation is not a valid alignment test. `ChannelConfig.p_codec = 0.15`. The codec stage runs last (after gain and clipping) and draws its random numbers last, so every channel drawn without a codec is bit-identical to before. ffmpeg is found via `$VA_FFMPEG`, then PATH, then winget's links folder; training refuses to start with `p_codec > 0` and no ffmpeg.

**Reproducibility.**
- `va synth train` now writes `train_v2` by default. `train_v1` clips regenerate identically with `p_tube = 0`: labels are equal and audio is within int16 quantisation, checked on four clips.
- The ten original `suite_v1` cases are pinned to `p_tube = 0` and `p_codec = 0` and regenerate bit-identically. Two cases are appended: `tube` (p_tube = 1, clean) and `codec` (codec only, no other channel effects).
- `write_set` refuses to add shards to a directory written with different settings. Fields absent from an older index are read as their pre-change values.
- Training runs record their channel config in `config.json`. A resumed run uses the recorded config; runs from before this change resume without codecs, so teacher_v1 is unaffected.

**Fixed validation set.** Teacher v1's synthetic validation drew new crops and channels at every eval: item seeds included a draw from the DataLoader worker's torch RNG, which is reseeded whenever the non-persistent val workers restart. Step-to-step changes were therefore partly a change of test set (F2 MAE 41 → 52 → 44 Hz over steps 16000–18000, at near-zero learning rate). Validation items are now seeded from (seed, item) only (`SynthCrops(fixed=True)`), so evals within a run are paired comparisons. Teacher v1's own log keeps the old behaviour, so its checkpoints should be compared on a fixed set after the run.

## D-012 (2026-10-08): teacher v1's F1 deficit; train_v2 sampling; formants on whispered frames

**Finding (teacher v1 on Hillenbrand, `reports/teacher_v1.md`).** F2–F4 beat Praat defaults; F1 does not (MAE 37 vs 18 Hz). The F1 error is concentrated in women and children: median +26 to +30 Hz and MAE 42–47 Hz, against +11 Hz and 21 Hz for men. Controlled synthetic vowels with known poles (scripts in the session scratchpad, not kept) show:
- *Most of the scatter is intrinsic to high f0.* With 2–3 harmonics under F1, even clean synthetic vowels give F1 MAE 25–29 Hz at f0 215–250 Hz, against 6 Hz at 125 Hz.
- *Spectral tilt (±6 dB/oct), bandwidths (0.6–2.5×) and Rd (0.4–2.2) move the teacher's F1 by ≤ 10 Hz* (≤ 17 Hz at 4× B1), so none of them explains the offset. Praat's F1 on the same vowels reads 8–22 Hz below the true pole and swings by up to ±90 Hz with tilt and source. Hillenbrand's references are also LPC-based, so part of the "+25 Hz" may be the reference reading low. Hillenbrand alone can't separate the two; that needs exact labels on real voices (M2 resynthesis) *[now M3; renumbered in D-014]*.
- *Coverage.* At f0 220–260 Hz, F1 error grows from 22–30 Hz MAE for the tract lengths training pairs with that f0 to 40–69 Hz (medians up to +28 Hz) for 16–18.5 cm tracts. Part of this is physics: a longer tract puts F1 nearer the low harmonics.

**train_v1 coverage, f0 × tract length (voiced frames).** Every cell had data (≥ 0.36 %, about 80k frames), but the f0-coupled length (17.5 − 4·log2(f0 / 110) + N(0, 1), clipped) piled 27 % of all voiced frames into 10–11 cm tracts at f0 > 260 Hz, and 4.6 % into 19–20 cm at f0 < 85 Hz. In training, high f0 mostly meant "child-sized tract". Women's (f0 150–260 Hz, 14–16 cm) and adult singers' (f0 500–1000 Hz, 14–16 cm) regions got 0.8–1.3 % and about 0.55 % per cell.

**Sampling changes (defaults; train_v2).**
- *Two components, one set.* Half the clips draw L uniformly over 10–20 cm whatever the f0 (broad coverage, deliberately including unphysiological corners). The other half follows a gentler coupling: L = 17.5 − 2.5·log2(min(f0, 350) / 110) + N(0, 1.5), redrawn rather than clipped, giving about 15 cm at 220 Hz and about 13.3 cm at 350 Hz and above. This is continuous sampling rather than a coarse grid, so the model can't memorise grid points. On a 1400-clip sample of the new defaults: no cell above 2.5 %, the 10–11 cm row at f0 > 350 Hz at 0.6–1.0 % per cell, women's region 0.9–1.7 %, adult singers 1.1–1.5 %.
- *F1 band.* In cascade clips with centre f0 150–350 Hz, each target has F1 = f0 × log-uniform(1.5, 4.5) with p = 0.3, where women's and children's F1 sits relative to their harmonics.

**Formants on whispered frames: reverts D-010's `sonorant = voiced`.** Plan §4 defines `sonorant_p` as "formant outputs are meaningful", and formants as tract poles, so tract-excited noise qualifies; frication (noise not through the tract) does not. v1 trained formants on voiced frames only. On whispered vowels with exact poles, teacher v1 gives F1–F4 MAE 70 / 273 / 220 / 193 Hz (medians within ±12 Hz, so scatter rather than bias) and bandwidths 1.3–1.65× too wide; voiced vowels at f0 125 Hz give 10 / 10 / 8 / 13 Hz. It also calls whispered vowels voiced (median p 0.70) with hallucinated f0 of 500–1350 Hz, and the comb rendered from that f0 adds error (switching the comb off improves F1 from 86 to 48 Hz, but not F2–F3). A test checks that the time-averaged whispered spectrum equals |H|² × the radiation difference to within 1.5 dB, so the pole labels describe whispered audio. Changes:
- `sonorant = voiced | whispered` (`sonorant_includes_whisper`).
- Fully whispered clips (`p_whisper_clip = 0.06`).
- /h/-like onsets on 25 % of voiced segments: 40–120 ms, rising within 20 ms, about 5–15 dB below the vowel, with the following vowel's poles.
- Whisper gaps between segments as before.

**Hyperparameter: whispered share of formant-labelled frames ≈ 11 %** (measured on the 1400-clip sample). The aim is whispered accuracy close to voiced. Two parts:
- *Distribution:* P(VTL | whispered) matches P(VTL | voiced) by construction, since whisper choices are independent of L; the measured histograms agree within sample noise. This is what prevents bias, and it costs nothing.
- *Amount:* a compute trade-off. Whispered spectra carry no harmonic-sampling problem, so a smaller share than voiced should suffice. If the `whisper` suite case's formant DR trails the voiced `reference` case by more than a few points after teacher v2, raise `p_whisper_clip`.

**Tube pole search fix.** In about 1.2 % of tube clips a heavily damped root (B 275–940 Hz) passes a sharp root in frequency as the losses ramp up. The poles stay distinct in the complex plane, but the old "frequencies stay ascending" check rejected them as lost roots, and generation failed. Losing a root now means two continuations converging to the same complex root, and roots are sorted by frequency. A regression test covers the captured shape. Poles with B > 500 Hz land among F1–F4 in 0.55 % of tube frames; they are kept as labelled, since they are exact poles.

**Suite.** `adult_high_f0` (f0 180–260 Hz, L 14–16.5 cm, cascade, clean) and `whisper` (fully whispered, formants scored on whispered frames) are appended to `suite_v1`, with exact labels where Hillenbrand's are LPC-based. The suite table gains voicing accuracy, and f0 RPA shows "–" for cases with no voiced frames. `V1_SETTINGS` in `sampler.py` records how `train_v1` and the original ten suite cases were generated. The suite pins to it, and `write_set` reads older indexes through it. `write_set` no longer rewrites an index when it writes no shards. Writing the new cases overwrote the original ten indexes' `git_commit`; it was restored by hand, and their content regenerates bit-identically.

## D-013 (2026-10-08): teacher v2 schedule: WSD with branched decays

Teacher v1's fixed-length cosine meant committing to a run length in advance, and its last ~15 % of steps ran at a learning rate too low to help (D-011, fixed-set re-scoring). Teacher v2 uses **WSD**: linear warmup over 1000 steps to 1e-3, then constant. **Decay branches** start from checkpoints of the constant phase and decay over 4000 steps as 1 − sqrt(p) to zero (Hägele et al. 2024, where 1 − sqrt beat linear cooldowns). Each branch resumes the parent's weights, optimiser state, step and channel, and inherits the parent's recorded training config (`va train teacher --branch-from runs/teacher_v2/ckpt_016000.pt` → `runs/teacher_v2_d16k`).
- *Branch points:* 6k, 16k, 26k and 36k, with a constant phase of up to 40k. 6k gives an early comparison with v1, and 16k+4k = 20k matches v1's length. Comparing branches shows what more steps buy, without fixing the length beforehand. Each branch costs 4k steps (about 1.3 h).
- *Why not Schedule-Free AdamW (yet):* it changes the optimiser as well as the schedule, so a v2 that differs from v1 couldn't be attributed to the data. It remains a candidate single-variable experiment once the data is settled.
- *No v1-data control run for now:* WSD plus a cooldown is reported to match tuned cosine at equal compute, so the schedule is the least likely confounder. If v2 disappoints on real data, a v1-data WSD run is the control to run.
- *The comb's true-f0 phase is now absolute* (`comb_truth_steps = 4000`), not 30 % of the run (6000 steps in v1). WSD runs have no fixed length, and this way even the 6k branch gets 2000 full-learning-rate steps on its own comb.
- *Resumed runs and branches reseed the data stream with the step.* Before, a resumed run replayed the batch sequence from step 0.
- Teacher v1's schedule is still available (`schedule = "cosine"`, `comb_truth_steps = None`).

## D-014 (2026-10-08): milestones renumbered; M1's criterion revised; M2's set in advance

**Renumbering.** The synthetic-data iteration that followed teacher v1 (D-011 to D-013) becomes milestone **M2** with **teacher v2** (`runs/teacher_v2`), so run and milestone numbers match. The former M2–M6 move up by one: M3 resynthesis (teacher v3), M4 self-supervision, M5 real-time student, M6 calibration and packaging, M7 physical rig. M3 also gains the training-pool corpora and settling the F1 reference question (D-012). Earlier entries that said "M2" for resynthesis (D-010, D-012) keep their text, with a bracketed note.

**M1's acceptance criterion, revised after its results.** The original criterion, "beats Praat defaults on Hillenbrand (MAE and DR)", was met for F2–F4 and not for F1 (MAE 37 vs 18 Hz, DR 97.3 vs 98.9 %; `reports/teacher_v1.md`). Plan §13 #14 had flagged this bar as hard before v1 was trained: Hillenbrand's references are LPC-based, like Praat. D-012 then showed that the LPC family reads F1 8–22 Hz below the true pole on synthetic vowels, so the comparison favours Praat by construction. M1's purpose was to show that a model trained only on synthetic data learns the task and transfers. The revised criterion: on the synthetic `reference` case, F1–F3 DR ≥ 95 % and f0 RPA ≥ 99 %, and gold-set results reported against all baselines with intervals and the sim-to-real gap by f0 band. **It was chosen after the results were known, so v1 passing it is a description of what v1 achieved, not a test it was at risk of failing.** The original text stays in the plan, struck through, with its outcome. The real-data results stand on their own: trained only on synthetic data, v1 beats Praat on Hillenbrand F2–F4 and every f0 baseline on PTDB-TUG speech.

**M2's criteria, set before any gold-set or decayed-branch result of teacher v2 existed.** The constant phase was at about step 3200. Only the step-1000 synthetic validation had been seen, including its cross-evaluation against v1 on both fixed validation sets, and none of the criteria below uses synthetic validation. Comparing the chosen v2 branch with v1 in the same report:
- Hillenbrand F1 MAE for women and children improves beyond bootstrap noise.
- No regression beyond noise in Hillenbrand F2–F4 or PTDB-TUG f0 RPA (all frames and modal frames).
- The ten original suite cases each lose ≤ 1 point of F1–F3 DR.
- `whisper`: F1–F2 DR ≥ 85 % and voicing accuracy ≥ 95 %. `adult_high_f0`: F1 DR above v1's.

The 85 % whisper threshold is a judgement of "close to voiced" given whisper's smaller share of training data (D-012); it was not derived from any result.

## D-015 (2026-10-08): first training pool: VocalSet

The first real training corpus (plan §5.6) is **VocalSet v1.1** (Wilkins et al. 2018, CC BY 4.0): 10.1 h of 20 professional singers (the directories are `female1`–`female9` and `male1`–`male11`, though the Zenodo text says 9 male and 11 female) singing scales, arpeggios, long tones and excerpts on five vowels, in standard and extended techniques (vibrato, straight, breathy, belt, vocal fry, lip trill, messa di voce and more), plus spoken excerpts. It comes first because singing at high f0 is the model's weakest domain, it is small, and it is on Zenodo (no Hugging Face, D-006).
- *Source:* `VocalSet11.zip` (md5 `0c09396242f946e7111ad7d8fc649b81`, 2.08 GB) from Zenodo record 1442513, the v1.2 record. v1.2's own zip (6.0 GB) holds the same audio three times over (organised by singer, technique and vowel); v1.1 holds it once, by singer.
- *Splits:* speaker-disjoint. `val` is the authors' test singers (`test_singers_technique.txt`), so it matches their published division; everything else is `train`. No VocalSet audio is gold.
- *Cross-corpus speakers:* the leakage test compares corpus-prefixed ids, so it can't detect the same person in two corpora. By hand: VocalSet's singers are professionals recorded at Northwestern University about 2017. The gold sets are Hillenbrand (Western Michigan University, 1990s, adults and children), PTDB-TUG (TU Graz speakers), MIR-1K (amateur singers in Taiwan) and MDB-stem-synth (MedleyDB commercial multitracks). No overlap is known or plausible.
- *Code:* training-pool scanners live in `vocal_analysis/data/pools/` (same `scan()` interface as the gold loaders, with train/val splits). `va manifest` covers both registries.

## D-016 (2026-10-08): resynthesis labeller design (M3)

Plan §5.2, in `vocal_analysis/resynth/` (`va resynth <pool> --teacher CKPT`). Per recording: a trained teacher gives f0, voicing, the sonorant mask and F1–F4 tracks. The tracks are smoothed, the source is recovered by inverse filtering, and then `variants` perturbations (default 2) re-filter that source. The results are cut into 3–6 s clips at quiet frames, QC'd and written as shards that `train.data` reads like the synthetic sets.
- **Tracker: the teacher, not PESTO.** The teacher's f0 head was trained only on synthetic data. The bundled PESTO weights were trained on MIR-1K, a gold set (§13 #13), and using them here would carry gold-set information into training labels. The analysis cache is keyed by the checkpoint's hash, and the set's index records the checkpoint's path, hash and step.
- **Inverse filtering: IAIF up to 250 Hz, formant tracks above it (`method = "auto"`, cross-faded per sample).** On synthetic vowels with known source and tract, IAIF recovers H1–H2 within 0.3–1.3 dB at f0 110–220 Hz. At 330–500 Hz it gives about 25 dB against a true 3.8 dB, because the LPC latches onto harmonics. Track-based inverse filtering uses anti-resonators at the tracked F1–F4 plus uniform-tube poles above them, the exact inverses of the synthesiser's resonators, and is right at every f0 given correct tracks. Both inverse filters have unity gain at DC, like the resonators. QCP is still to do.
- **QC: `tract_left` (β).** The source's harmonic amplitudes below 4 kHz are fitted as LF(Rd) + gain + tilt + β × the tracked tract's dB response. β is the fraction of the tract still in the source. On synthetic vowels:

  | Case | β |
  |---|---|
  | Untouched speech | 0.97–1.23 |
  | Exact inverse filtering | about 0 |
  | IAIF below 250 Hz | −0.19 to 0.06 |
  | IAIF above 300 Hz | −0.3 to −1.2 |

  Aspiration noise (HNR 6–13 dB) and jitter leave β near 0 but raise the fit's RMS. A clip is rejected when its median voiced-frame |β| exceeds 0.25. **That threshold is provisional, to be calibrated on VocalSet.**

  Two failures β can't see:
  - Tracks 10 % off with track-based inverse filtering keep β small but give RMS 4–6 dB. Clean but noisy sources reach 3–4 dB RMS, though, so the RMS is logged per frame (`ripple_db`) and not gated.
  - The fit needs five harmonics below 4 kHz, so f0 above 800 Hz goes unmeasured. Those clips are kept (`accept_unmeasured`) and counted.
- **Perturbation (§5.2 step 4):**
  - α log-uniform(0.8, 1.25) on F1–F4;
  - slow per-formant offsets with SD 3–5 % on a 0.3 s time scale;
  - F5 and up continue from F4 at uniform-tube spacing for the new apparent length;
  - Hawks & Miller bandwidths × log-uniform(0.6, 1.6) per formant;
  - a nasal pole–zero pair with p = 0.15, constant over the recording;
  - ordering at ≥ 150 Hz spacing, and poles above 7.6 kHz dropped, as in the generator.

  Clip RNG is seeded by (seed, recording, variant).
- **Labels.**
  - *Exact:* formants and bandwidths (all eight, as in the synthetic sets).
  - *Tracker estimates:* f0, voicing and sonorant. Re-filtering leaves the original voice's f0 alone, so these are as good as the teacher, and `meta["labels"]` says so. **Open for teacher v3:** whether the f0 and voicing losses on resynthesised clips are down-weighted or masked, since otherwise they distil the teacher into itself. The plan's parametric-source variant (LF fitted per period) would make f0 exact.
  - *Measured:* source_h_db, on the recovered source and on its own scale, so only differences such as H1–H2 mean anything.
  - *Not available:* rd and length_cm are NaN.
- **Non-sonorant stretches keep the original audio**, cross-faded over 20 ms, so consonants stay real. Frames inside a cross-fade get no formant labels. The re-filtered signal is level-matched to the original over sonorant samples.
- **Shards** hold the recordings of `files_per_shard` files and a varying number of clips. `index.json` lists `shard_clips`, and `ShardReader` uses it. An interrupted run resumes; changed settings are refused.
- **Cost:** about 0.3 s CPU per second of audio for the source stage. Each variant is nearly free.
