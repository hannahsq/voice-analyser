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

**Revisited 2026-10-09 (VS Build Tools 18 with the C++ workload, MSVC 14.51, Windows SDK 10.0.26100).** hipRTC reads `INCLUDE`. With it set to the MSVC `include` folder plus the SDK's `ucrt`, MIOpen's GRU and batch norm compile, and GRU, batch norm and plain and dilated Conv2d all match the native kernels (fp32 ≤ 6e-7 relative; bf16 4e-3 for convolutions and 1e-2 for a 50-step GRU). Speed is another matter. On the full teacher training step (bf16, real batch):

| MIOpen used for | ms/step |
|---|---|
| nothing (native, current) | **1055** |
| convolutions | 1572 |
| everything | 1562 |
| convolutions, benchmark mode | 1406 |
| everything, benchmark mode | 1399 |

MIOpen's GRU on its own is 2–3× faster: 27 vs 76 ms (f0 GRU) and 77 vs 159 ms (main GRU), fwd + bwd, batch 32 × 200 frames. So the best policy would be native convolutions with MIOpen around the GRUs, the reverse of the old `rnn` setting, for about 12 % (≈ 68 audio-s/s). Teacher v2 keeps `all` so its runs stay comparable; the GRU-only policy, with `setup_device` finding the headers via `vswhere`, is for teacher v3.

Other measurements from the same session, for the record:
- The data loader is not a bottleneck: 7–10 batches/s against 0.93 needed, and the trainer waits about 1 ms per step for data.
- Evaluation plus checkpointing every 1000 steps costs 0.9 % of wall time.
- The host takes the whole step to queue its kernels (1072 of 1081 ms): the native GRUs step through time, and the native dilated convolutions loop over samples, so a step is tens of thousands of small kernels. The native GRUs are about 26 % of the step.
- HIP graph capture (which would remove the launch overhead without changing kernels) aborts the process with hipBLASLt ("operation not permitted when stream is capturing"). With `TORCH_BLAS_PREFER_HIPBLASLT=0` it works, but rocBLAS makes the whole step 2.5× slower (24 against 60 audio-s/s).
- The per-step `float()` calls in the training loop forced a host-GPU sync each step. Removing them gave +8 % (55.3 → 59.7 audio-s/s, two repeats); pinned memory added nothing.
- Windows Smart App Control switched itself to enforcement on 2026-10-09 and blocked ffmpeg (codec augmentation), crashing a training run; it is now off. If codec calls fail with WinError 4551, check it first.

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

  Aspiration noise (HNR 6–13 dB) and jitter leave β near 0 but raise the fit's RMS. A clip is rejected when its median voiced-frame |β| exceeds 0.5. Below that, training weights the clip (next item).

  Two failures β can't see:
  - Tracks 10 % off with track-based inverse filtering keep β small but give RMS 4–6 dB. Clean but noisy sources reach 3–4 dB RMS, though, so the RMS is logged per frame (`ripple_db`) and not gated.
  - The fit needs five harmonics below 4 kHz, so f0 above 800 Hz goes unmeasured. Those clips are kept (`accept_unmeasured`) and counted.
- **VocalSet trial, and why QC weights rather than rejects.** On 100 training recordings spread across all 15 training singers and 16 techniques (teacher v1 as tracker), clip-median β had 5/25/50/75/95th percentiles of −0.35 / −0.09 / 0.03 / 0.12 / 0.25. A hard cut at 0.25 kept 84 % of clips, but not at random: 50–60 % for two singers (female3 median −0.22, female5) and for breathy, messa di voce, trill and trillo, against 94–100 % for slow piano/forte, belt and vocal fry. Accepted clips centred near 0 in every f0 band (+0.06 below 250 Hz, +0.02 at 250–500, −0.07 above 500), so there was no seam between IAIF and track-based inverse filtering.

  Is the tracker to blame? Teacher v1's own uncertainty didn't separate rejected from accepted clips. The median F1 spread was 48.8 vs 48.0 cents, and every rank correlation with |β| was within ±0.13. Either v1 is confidently wrong there (its uncertainties are uncalibrated until M6), or the cause is one no tracker fixes: sources that aren't LF-like, extra resonances, or measurement noise.

  A hard cut would drop the hardest voice qualities: a mild form of the selection bias §5.4 warns about. So only clearly broken clips are dropped at generation (|β| > 0.5). In training, each resynthesised clip's formant and bandwidth losses are weighted 1 up to |β| = 0.15, falling linearly to 0 at 0.5 (`TRACT_LEFT_WEIGHT`). The weight comes from each clip's stored β, so the curve can change without regenerating.

  Regenerating with each new tracker is the plan's M4 loop, so rejected clips need no separate retry. Comparing per-clip β from teacher v1 and v2 on the same recordings will show how much is the tracker.
- **Training (`va train teacher --resynth DIR --resynth-frac 0.3 --resynth-val DIR`).** `resynth_frac` of training items come from the resynthesised sets, split between them by clip count. On those clips:
  - the f0, voicing and sonorant losses are masked: those frames leave the average, so the terms aren't diluted;
  - formant and bandwidth losses are scaled by the clip weight and still averaged over all labelled frames, so a low-quality clip pulls less.

  During the comb's true-f0 phase, resynthesised clips get their comb from the tracker's f0 label, so their f0 label is defined on every frame, as in the synthetic sets. Held-out resynthesised sets (VocalSet's `val` singers) are scored as `eval_resynth:<name>`, where only the formant metrics are exact. Sets consolidated before this change carry no β and train exactly as before.
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
  - *Tracker estimates:* f0, voicing and sonorant. Re-filtering leaves the original voice's f0 alone, so these are as good as the teacher, and `meta["labels"]` says so. **Decided for teacher v3: the f0 and voicing losses are masked on resynthesised clips**, and only the formant and bandwidth losses apply. Training on the teacher's own f0 and voicing would distil its errors into itself, and they are worst where VocalSet sits: singing, where MIR-1K is v1's weakest f0 set, and voicing, where v1's precision is low. With no ground truth and no self-supervision yet, the f0 heads keep learning from exact synthetic labels only, and v3's gold-set f0 stays a clean measure of synthetic-to-real transfer. Down-weighting or confidence weighting was rejected: there is no clean set to tune the weight on, and confident octave errors pass through. **Later:** the plan's parametric-source variant, an LF source fitted per period along the teacher's f0 track, makes f0 and voicing exact by construction, at the cost of the real source's detail. Mix it in as a separate data source and ablate it.
  - *Measured:* source_h_db, on the recovered source and on its own scale, so only differences such as H1–H2 mean anything.
  - *Not available:* rd and length_cm are NaN.
- **Non-sonorant stretches keep the original audio**, cross-faded over 20 ms, so consonants stay real. Frames inside a cross-fade get no formant labels. The re-filtered signal is level-matched to the original over sonorant samples.
- **Listening check on the first trial, and fixes.** Six excerpts (female1 arpeggios), each heard as the original and two variants.
  - *High-frequency artifacts and clicks, worst at α = 0.85.* Every affected clip had pole tracks crossing 7.6 kHz, so recursive sections switched on and off mid-signal and restarted from the wrong state. They also had 5–13 dB too much energy at 6.5–8 kHz. The clip judged perfect had no such crossings.

    Fix: each recording keeps a fixed set of poles. A track that leaves the band on any frame is left out for the whole recording, in the labels, in the re-filter and in the inverse filter's upper poles (`tracks.whole_tracks`). Afterwards, the 6.5–8 kHz excess was gone (the worst case went from +12.8 to +0.6 dB), and click counts matched the originals.
  - *Lip trills sound gargled.* With the lips vibrating, there is no vowel-like tract to track or resynthesise. VocalSet's `lip_trill` recordings are skipped by `va resynth` (`vocalset.resynth_ok`) and stay in the manifest.
  - *α = 1.22 sounds increasingly like a tenor.* Intended: a 22 % longer tract with soprano f0 is deliberately unphysiological (D-012).
  - *Second round, after those fixes.* Spectrograms with tracks, f0 and level overlaid explained the remaining complaints:
    - **Vowel shifts at note transitions (α = 0.85).** Teacher v1's F2 bumped at transitions and wobbled with vibrato, following the harmonics. Track-based inverse filtering removes the wobble at the old positions and the re-filter adds it back at the new ones, which differ unless α = 1. Fix: track smoothing is now a 9-frame median filter (removes jumps), then a Gaussian with SD 6 frames, which leaves 5–6 Hz wobble at under a fifth of its size. This replaces a 7-frame Savitzky–Golay filter.
    - **A "bird whistle".** Teacher F3 jumped between about 3.0 and 3.9 kHz, so a narrow resonance swept back and forth. The median filter removes it.
    - **Level jumps where vibrato dipped under 250 Hz.** With method `auto`, the inverse filter switched between IAIF and tracks every vibrato cycle, and the two sources differ in level. Fix: one method per phrase, chosen by the phrase's median f0. A phrase is voiced runs merged across gaps under 100 ms.
    - **Not artifacts.** A note that grew louder and changed colour had the perturbed F2 (narrowed from 92 to 71 Hz) landing on its 4th harmonic at f0 ≈ 400 Hz, about 5 dB of real gain from the new tract. In another clip, level differences with the vibrato's rhythm come from the new formants sitting elsewhere among the sweeping harmonics. Both are the new tract's physics, and the labels stay exact.
  - *α = 0.85 sounds like "an intact singer's formant on a chipmunk".* **Open.** It suggests the original singer's-formant cluster (F3–F5 near 3 kHz) survives inverse filtering. At this f0 that inverse filtering uses the tracked F1–F4 plus uniform-tube poles, which can't describe a bunched cluster, and the QC fit stops at 4 kHz. Candidates: a QC term for 2.5–5 kHz, upper poles fitted to the recording rather than a uniform tube, and QCP at high f0.
- **Shards** hold the recordings of `files_per_shard` files and a varying number of clips. `index.json` lists `shard_clips`, and `ShardReader` uses it. An interrupted run resumes; changed settings are refused.
- **Cost:** about 0.3 s CPU per second of audio for the source stage. Each variant is nearly free.

## D-017 (2026-10-08): licences; M4Singer, Common Voice and Zeroth as training pools; M4Singer songs shared with MIR-1K

**Licensing.** The project is personal and research use; it was never meant to be commercial.
- *Code:* MIT (`LICENSE`).
- *Model weights:* carry the most restrictive licence among the corpora they were trained on, with attribution to all of them. The question of whether weights adapt their training data is unsettled, and this is the conservative reading. Weights from synthetic data plus CC BY or CC0 pools are CC BY 4.0, the least restrictive licence those pools allow. Anything trained on M4Singer is CC BY-NC-SA 4.0.
- *Commercial use:* anyone who wants it can retrain on suitably licensed data, since the code is MIT.

`ATTRIBUTION.md` lists every corpus, with its role, licence and citation. Literature stays cited in the plan (§16) and this log. Evaluation-only corpora don't bear on weights.

**New training pools.** Their archives are already on the NAS (`Z:\datasets`), downloaded for another project. Resynthesis needs audio and speaker ids only, so their transcriptions are not needed.
- **M4Singer** (Zhang et al. 2022), `m4singer.zip`: Mandarin pop singing by 20 singers (Soprano/Alto/Tenor/Bass 1–7), 419 songs, 20,861 utterances, each with a MIDI and TextGrid.
  - *Licence:* CC BY-NC-SA 4.0, per `dataset_license.md` in the M4Singer repository. The archive carries no licence file.
  - *Value:* the largest singing pool, about three times VocalSet.
- **Zeroth-Korean**, `zeroth_korean.tar.gz` (10.3 GB): read Korean speech, the first speech pool.
  - *Licence:* CC BY 4.0, from OpenSLR 40. The archive carries no licence file.
  - *Size:* 52.8 h from 115 speakers. The train split is 105 speakers and 51.6 h; the test split is 10 speakers and 1.2 h.
  - *Splits:* speaker-disjoint, and the manifest keeps them (test → `val`).
  - *Privacy:* `AUDIO_INFO` lists each speaker's real name next to the numeric id. Manifests and everything else in the repository use only the numeric id (`zeroth:<id>`) and the recorded sex; names are never copied.
- **Common Voice 27.0 Korean**, `cv-corpus-27.0-2026-09-11-ko.tar.gz`: 7185 MP3 clips, 10.39 h (2.56 h validated), 213 speakers on their own microphones.
  - *Licence:* CC0. The download terms add an agreement not to try to identify the speakers. Nothing here tries to; speaker ids stay opaque `client_id` hashes.
  - *Value:* small, and already lossy-coded. It is most useful for channel diversity and as unlabelled audio for M4's self-supervision.

**Where they live.** Corpora shared with other projects are kept once, in `C:\github\datasets\`, and each repository links to them with directory junctions:
- `data/zeroth_korean` → `datasets\zeroth_korean`;
- `data/cv_ko` → `datasets\cv-corpus-27.0-2026-09-11\ko`;
- `data/m4singer` → `datasets\m4singer`.

The two Korean corpora were moved there from `minimal-pairs/data`, which links back to them under their old names, so that project's paths are unchanged. Manifests store paths relative to the corpus root, so they don't depend on the layout. The archives on `Z:\datasets` remain the backup. The gold sets and VocalSet stay in `data/` as before.

**M4Singer songs that are also in MIR-1K are excluded from training.** The singers are different people, so this is not speaker leakage. But the same songs mean the same melodies, and MIR-1K is an f0 gold set. f0 losses are masked on resynthesised clips, but the shared trunk would still hear those melodies.
- *Method:* matching each MIR-1K singer-song's lyrics (Big5, traditional characters) against each M4Singer song's (`meta.json`, simplified) by character trigrams. A song's best match is about 0.02 for 90 % of MIR-1K singer-songs. Matches of 0.25–0.42 identify 8 shared songs despite the script difference: 成全, 听海, 童话, 当你, 一个像夏天一个像秋天, 水手, 千里之外, 今天你要嫁给我. 情歌王 (0.12–0.17) and 一首简单的歌 (0.18) are weaker matches. 情歌王 is a medley quoting other songs, and both are excluded to be safe.
- *Enforcement:* the M4Singer scanner will mark these 10 songs as excluded, so they never enter a training manifest.

## D-018 (2026-10-08): schedule-free AdamW against the 6k WSD branch, at equal budget

D-013 deferred Schedule-Free AdamW (Defazio et al. 2024) until the data was settled. The data is now settled (train_v2), and the 6k decay branch gives a matched baseline: `runs/teacher_v2_d06k` is AdamW with WSD, 6000 constant steps plus 4000 decay steps.
- **Experiment.** A new run, `runs/teacher_v2_sf`. Data, model, seed, channel, batch, 1000 warmup steps and peak learning rate (1e-3) are all as in teacher v2; only the optimiser differs (`--optimizer schedulefree`). It trains for 10k steps.
- **Comparison with d06k, set before either result exists:** the fixed synthetic validation sets of v1 and v2, the suite (including `whisper` and `adult_high_f0`), and the M2 gold-set measures (Hillenbrand F1–F4 by group, PTDB-TUG f0 RPA), all with bootstrap intervals.
- **Reading the result.** If schedule-free is within noise of d06k or better, later teachers (v3 onward) use it and drop decay branches: any checkpoint is meant to be usable as is, which removes the 4k-step cost of each branch. If it is clearly worse, the result is inconclusive rather than negative, since schedule-free often prefers a higher learning rate than AdamW and only one was tried; WSD stays.
- **Cost.** About 3.3 h of GPU time. The main constant-phase run pauses at its 10k checkpoint, so the two don't share the GPU, and resumes afterwards (resumed runs reseed the data stream, D-013).
- **Implementation.** `TrainConfig.optimizer`. Checkpoints hold schedule-free's averaged (evaluation) weights, so `va eval` and the analysis code load them unchanged. Its learning-rate warmup runs inside the optimiser. The `schedulefree` package (1.4.1, PyPI) is in the `train` extras.
- **Result at 10k (synthetic validation only).** Schedule-free 23.8 / 45.2 / 62.5 / 98.4 Hz F1–F4 MAE, f0 RPA 99.3, voicing 99.0; d06k 24.3 / 45.3 / 64.1 / 101.4, 99.4, 99.2. So formants slightly better, f0 and voicing 0.1–0.2 points behind: a match, without a decay branch. Both runs then continue to 40k (WSD with branches at 16k/26k/36k), compared at equal steps; the suite and gold-set comparisons are made at 20k (d16k vs schedule-free 20k).
- **Running both at once doesn't pay here.** Three processes (two runs plus a branch) exhausted the commit limit (each run ~9 GB plus 12 workers at ~1.9 GB) and reached about 29 audio-s/s in total, against 50–60 for one run alone. Runs, branches and evaluations now go one at a time (orchestrator in the session scratchpad), which also stops on any failed step.

## D-019 (2026-10-09): teacher v2's F1 regression on Hillenbrand: what it is and isn't

**Finding.** d16k (20k steps) reads Hillenbrand F1 worse than d06k and v1 although its synthetic F1 keeps improving: MAE 41 vs 37 Hz, DR 95.1 vs 97.3 %, with the drop in every group (DR men 99.5 → 98.0, women 96.9 → 95.2, boys 94.9 → 92.2, girls 96.4 → 92.3). The sim-to-real F1 gap at f0 150–300 Hz grew from +3.3 (d06k) to +5.5 points.

**Diagnosis** (scripts in the session scratchpad: `f1_drift.py`, `f1_bias.py`, `f1_prior.py`, `resynth_gross.py`):
- *It is an upward bias that grows with training.* Median signed F1 error against Hillenbrand, v1 → d06k → d16k: men +11 → +14 → +17 Hz, women +26 → +21 → +30, boys +29 → +28 → +38, girls +30 → +28 → +41 (Praat FastTrack: +8 to +10). About 5–7 % for women and children, up to 17 % for low-F1 vowels. Frames that moved by more than 20 Hz from d06k to d16k moved up (median +29 Hz), and only 35 % of them towards the reference.
- *Not the references' harmonic snapping.* At f0 ≥ 150 Hz, 44 % of Hillenbrand's F1 references lie within 0.1 harmonic spacing of a harmonic (FastTrack 41 %, uniform 20 %, teacher 22–27 %), so the LPC references do snap. But the teacher's errors are smallest on those frames (39 Hz, against 52–58 elsewhere), and the bias grows for men too, where harmonics are dense and LPC's F1 is reliable.
- *Not the label definition.* On synthetic validation the signed bias is −3 to +1 Hz in every f0 band for all three models, and within ±10 Hz in every (f0, true F1) cell up to f0 260 Hz, so there is no pull towards the training prior on in-distribution data either.
- *Mostly not the glottal source.* On resynthesised held-out VocalSet clips (real sources, exact synthetic-tract labels; 5 val singers, 100 recordings, 500 clips) the median bias is +6 / +9 / +6 Hz (ratio 1.011 / 1.017 / 1.012) and does not grow with training.
- *Remaining candidates:* properties of real vocal tracts the synthesiser lacks (F1 raised and broadened in the open phase through subglottal coupling, wall vibration in the 40 % cascade clips, real F1 bandwidths), or a systematic offset in Hillenbrand's LPC references. These experiments cannot separate the two; a second formant reference made by a different method (VTR-TIMIT) would.

**A second finding: v2 is weaker than v1 on low-pitched singing.** On the resynthesised clips, gross F1 errors (> 20 % off the exact label) at f0 < 150 Hz are 3.7 % for v1 against 10.1 % (d06k) and 7.9 % (d16k). The clips were made by inverse filtering with v1's tracks, so the set was rendered again with d16k as tracker:

| Gross F1 errors (%) | v1-tracked: v1 / d06k / d16k | d16k-tracked: v1 / d06k / d16k |
|---|---|---|
| f0 < 150 | 3.7 / 10.1 / 7.9 | 6.2 / 13.0 / 9.2 |
| f0 150–250 | 8.9 / 12.6 / 11.8 | 7.1 / 12.0 / 10.2 |
| f0 250–400 | 20.2 / 23.5 / 22.2 | 20.1 / 20.4 / 15.9 |
| f0 ≥ 400 | 26.1 / 23.8 / 22.3 | 22.3 / 13.0 / 11.8 |

- The low-f0 weakness survives the swap, so it is real; train_v2's f0 distribution is unchanged from train_v1 (16.6 / 11.1 / 8.2 / 7.1 % of voiced frames below 100 / 100–150 / 150–200 / 200–250 Hz), so coverage is not the cause. D-012's tract-length decoupling (half the clips draw 10–20 cm whatever the f0) is the leading suspect. d16k's low-f0 gross errors include 13–16 % at about twice the label.
- v2 is better than v1 at high f0 on both sets.
- **Tracker circularity is large.** Each set favours the model family that tracked it, most at high f0 (d16k: 22.3 % gross on v1-tracked clips, 11.8 % on its own). Resynthesised sets used for evaluation must therefore be tracked by a model that is not being compared, or be reported under both trackers. The gross errors do not sit at the original (pre-perturbation) tracks (9–12 % of them, against a base rate of about 5 %).

**Closed-phase reference (CMU ARCTIC with EGG).** VTR-TIMIT needs TIMIT's audio (LDC licence), which we don't have and won't obtain informally, so the separating test uses EGG instead: closed-phase covariance LPC (`eval/closed_phase.py`, order 12 at 10 kHz, cycles within ±20 ms, closed phases GCI + 0.3 ms to GOI − 0.3 ms, shifted by the measured speech lag of 0.8–1.1 ms) on every monophthong frame of the three ARCTIC voices that have EGG. 26,663 frames have a closed-phase fit and every estimate. Median signed difference from closed-phase F1 (Hz):

| Voice (f0, closed quotient) | Whole-cycle LPC | Praat Burg (5000/5500 Hz) | v1 | d06k | d16k |
|---|---|---|---|---|---|
| bdl, male (121 Hz, 0.59) | +2 | −6 | +1 | +4 | +5 |
| jmk, male (111 Hz, 0.34) | +3 | −45 | +16 | +26 | +31 |
| slt, female (174 Hz, 0.42) | −10 | −10 | −6 | −7 | −4 |

- *Open-phase coupling is not the cause.* The same estimator over whole cycles is within a few Hz of the closed-phase fit.
- *LPC-based references read F1 low:* Praat Burg by 6–10 Hz for bdl and slt and by 45 Hz for jmk (a breathier voice). Measured against Praat (the Hillenbrand view) the teachers are +6 to +11 Hz for bdl and slt, comparable to their offset on Hillenbrand men, though against the closed phase they are within ±7 Hz. So part of the Hillenbrand F1 "bias" is the reference's.
- *Part is the teacher's.* jmk's bias grows with training (+16 → +26 → +31 Hz; close vowels, closed-phase F1 < 450 Hz: +23 → +40), and at slt's higher f0 the teacher compresses F1 towards the middle (closed-phase F1 < 450 Hz read +48 to +55 Hz high, > 650 Hz read −30 Hz low), matching the Hillenbrand finding that low-F1 vowels read up to 17 % high in women and children. That pattern is regression towards the training prior where harmonics leave F1 ambiguous.
- *Replicated on PTDB-TUG's laryngograph* (`ptdb_egg.py`, `ptdb_summary.py`; 20 speakers, 10 women; no phone labels, so frames are the energy peak ±20 ms of voiced runs ≥ 80 ms, every second utterance; 39,835 frames; speech lag 0.5–0.7 ms). F1 bias against the closed phase, speaker-level median [95 % speaker bootstrap], Hz:

  | | Praat | v1 | d06k | d16k | sf20k |
  |---|---|---|---|---|---|
  | men (114 Hz) | +3 [+1, +8] | +8 [−0, +13] | +17 [+8, +24] | +16 [+8, +20] | +10 [+4, +19] |
  | women (203 Hz) | +5 [−4, +7] | +6 [+3, +10] | +13 [+9, +22] | +18 [+13, +29] | +11 [+9, +23] |

  Median |F1 error| change from v1: d16k +4.8 [+3.1, +6.9] (men), +9.3 [+5.2, +10.9] (women); schedule-free +2.0 / +3.9. Compression in women, by closed-phase F1 < 450 / 450–650 / > 650 Hz: v1 +30 / +7 / −23, d16k +49 / +22 / −32 (Praat +13 / +1 / −12; whole-cycle LPC +13 / −2 / −23). Whole-cycle LPC is within ±7 Hz of the closed phase overall, so open-phase coupling is again ruled out; here Praat does not read low. d16k reads F1 high for all 20 speakers (+4 to +44 Hz). F2 and F3 are within ±18 Hz for every estimator.
- Caveats: ARCTIC has three speakers; closed-phase LPC has its own scatter (9–25 Hz median against whole-cycle), largest for jmk's short closed phases. Evidence about mechanisms, not a new gold standard.

**Consequences.**
- M2's Hillenbrand F1 criterion partly measures LPC's own low reading; a teacher a few Hz above LPC can be closer to the truth. The drift between models is real nonetheless, since it shows against the closed phase too.
- The cause to work on is the training prior, not missing physics: the synthetic F1 distribution conditional on f0 (and tract length, D-012's decoupling), where the teacher falls back on it. Calibrating the synthesiser's distributions against real data (option 1: "learning to simulate") now has a concrete target, with closed-phase F1 on EGG voices as one check.
- Training on resynthesised data (M3) addresses source mismatch, which is not the main cause. It remains useful for the low-f0 singing weakness, provided the tracker is not the model under evaluation.
- Next: a low-f0 ablation of the tract-length sampling; the F1-given-f0 prior of train_v2 against real distributions (Hillenbrand's vowel space by group, closed-phase F1 on ARCTIC).

## D-020 (2026-10-09): tract independent of the source; teacher v2 stops at 20k; decoupling ablation

**Principle.** The long-term use (before/after comparisons where f0 changes and the tract may not, and vice versa) needs formant estimates that do not infer the tract from the pitch. So synthetic tract parameters (length, formant targets, bandwidth scaling) are sampled independently of everything on the source side. Source parameters may covary with each other (low f0 with less tilt, higher closed quotient, less noise): that is realistic and cannot teach the model to read the tract from the source, because in training the tract never covaries with it. A more "realistic" coupled prior would score better on typical voices and on LPC-based references (D-019) and is rejected for that reason. Ambiguity is to be handled by honest uncertainty instead: the heads already output distributions trained with a proper scoring rule; what is missing is checking their calibration on real voices (coverage against closed-phase F1) and reporting it.

**train_v2 is only half decoupled.** `p_length_from_f0 = 0.5` (half the clips draw L from f0) and `p_f1_band = 0.3` (cascade clips at f0 150–350 Hz set F1 = f0 × 1.5–4.5).

**Teacher v2 stops here.** WSD stopped at 17k; d16k (20k steps) and schedule-free at 20k are the candidates. M2 fails on F1 (`runs/m2_paired_20k.txt`): v1 → d16k F1 MAE women +2.3 [+0.3, +4.6], children +6.6 [+4.9, +8.2], men +3.8 [+2.5, +5.1] Hz; F2 −7.4 and F3 −33 Hz MAE better; PTDB-TUG RPA +0.2 (all frames), −0.7 below 85 Hz. Schedule-free at 20k against d16k: F1 MAE women −5.1, children −2.3, F4 −26 Hz (better), but PTDB-TUG RPA −0.6 (all) and −0.9 (modal), beyond noise: D-018 stays undecided (one learning rate tried).

**f0-invariance probe (`eval/invariance.py`).** Story's vowels i, E, ae, A, o, u through the lossy tube at 13, 15.5 and 18 cm (fixed, physical bandwidths), LF source at eight f0 from 80 to 400 Hz; slope of the median F1 estimate against log2 f0 (truth: 0). Before the ablation:

| Estimator | median slope (Hz/octave) | median \|slope\| |
|---|---|---|
| Praat Burg default | +15 | 19 |
| v1 (fully coupled L) | +23 | 51 (u: up to +370, o: up to +162) |
| d06k | +2 | 14 |
| d16k | +8 | 12 |
| schedule-free 20k | +7 | 16 |

D-012's partial decoupling cut the dependence about fourfold; the remainder has the compression shape (close vowels drift up with f0, open vowels down). At the top of the sweep f0 exceeds the F1 of i and u, where some slope is unavoidable.

**Ablation, set before its results.** `train_v3a` = train_v2's settings with `p_length_from_f0 = 0` and `p_f1_band = 0`, 100k clips, seed 2 (`va synth train --synth-set …`). Teacher `runs/teacher_v3a`: d06k's recipe (WSD, 6000 constant + 4000 decay steps, everything else as teacher v2), so d06k is the baseline at equal budget. Measures:
1. f0-invariance probe: median |slope| and the vowel pattern.
2. Closed-phase F1 agreement on CMU ARCTIC and PTDB-TUG (laryngograph): bias by sex, compression (estimate against closed-phase F1 by range).
3. Hillenbrand F1–F4 by group, paired speaker bootstrap against d06k (informative, not decisive: LPC references reward prior pull).
4. Guard-rails: PTDB-TUG f0 RPA, the suite, synthetic validation. *Clarified before the results (2026-10-09):* "synthetic validation" means a swap, with each model on both sets' held-out clips (all 2000 per set, fixed crops and channel) and clip-bootstrap intervals on the paired differences. Each run's own validation loss is not comparable across runs: train_v3a is harder by construction, because f0 no longer predicts the tract. The guard-rail is v3a not being worse than d06k on train_v2's clips; on train_v3a's clips d06k is expected to lose more.

Adopt full decoupling for teacher v3 if the probe's median |slope| falls clearly (by a third or more) and closed-phase F1 agreement is not worse beyond its bootstrap interval; the guard-rails must not regress beyond noise.

## D-021 (2026-10-09, planned): temporal module, crop length, and vocal-tract-length outputs

Planned work, in this order, each step on the winner of the previous one and after the D-020 decision. Each comparison changes one thing.

**1. Conformer instead of the BiGRUs.** The convolutional front end stays; only the temporal modules change. The main encoder's 2-layer BiGRU becomes a small Conformer (about 4 blocks, width 192–256, 4 heads, a time-convolution kernel of 15–31 frames, relative position encoding), with the same parameter count. The f0 branch's BiGRU becomes a dilated 1-D convolution stack. Recipe: d06k's (6000 constant + 4000 decay steps) on the D-020 winner's data. Compared on synthetic validation, the D-020 measures (the f0-invariance probe included, since attention could find a new path from pitch to tract) and step time. Expected: accuracy about equal; training 15–20 % faster per step (the GRUs are about 22 % of a step, and the dilated 2-D convolutions remain the main cost); CPU inference 1.5–1.8× faster (the GRUs are about 45 % of it at 4 threads, and are sequential); and streaming becomes possible with chunked attention, which a BiGRU cannot do. Mamba and RWKV are deferred: they have no fused kernels on Windows ROCm.

**2. Longer and mixed crop lengths.** Training uses 2 s crops (from 2–4 s clips), while evaluation and the intended uses run whole files of tens of seconds with many vowels. Crops of 2, 4 and 8 s (later 16 s), grouped by length with the batch size scaled to keep steps about equal in cost; clips at least as long as the longest crop (`duration_s` up to about (2, 16)). Purpose: remove the train/use length mismatch, and let the model pool tract evidence across many vowels from one speaker. That pooling is tract-to-tract evidence and consistent with D-020. A model that learns "one speaker, one tract" will smooth over a real change within a file, so before/after material is analysed as separate files.

**3. Vocal-tract-length outputs.** The synthesiser knows three related quantities exactly; they are kept distinct:
- *Anatomical VTL* (`length_cm`): constant per speaker. A new head outputs a running per-frame estimate with uncertainty, trained on the clip's label. Its estimate should converge and stay flat as evidence accumulates.
- *Instantaneous acoustic length*: anatomical length with the current larynx height and lip protrusion (per target in `meta["tube"]`). It genuinely varies from vowel to vowel and is the legitimate reason apparent VTL moves.
- *Apparent VTL*: derived from the formants afterwards, not predicted by the model. The formants-to-length mapping is fitted on tube clips, where formants and true lengths are both known, instead of assuming a uniform tube; that also measures the uniform tube's error.

The running anatomical estimate doubles as a diagnostic of the model's beliefs. Drift that follows f0 is the coupling D-020 removes, and the invariance probe is extended to this output; drift that follows the vowel means instantaneous length is being confused with anatomical length. The label must stay a pure tract quantity, sampled independently of the source.
