# Literature notes

Working notes on prior work, kept so the papers in `outline.md` cite it properly and so decisions don't reinvent it. Each entry says what was read: **full text** (read and checked), **abstract** (abstract or listing only: claims unverified), or **code** (the authors' implementation read as a specification). Started 2026-10-28 (Hannah: "read and reference the literature for this one; QCP-FB might not even be state of the art any more").

## Closed-phase and quasi-closed-phase linear prediction

- **Alku (1992)**, *Glottal wave analysis with pitch synchronous iterative adaptive inverse filtering* (IAIF). Separates the glottal source and the tract without an EGG by alternating low- and high-order LP fits. The baseline QCP improves on. *Not yet read.*
- **Alku, Pohjalainen, Vainio, Laukkanen & Story (2013)**, JASA, formant estimation from high-pitched vowels with weighted LP (the AME weighting function). Physical-model synthetic vowels. *Not yet read.*
- **Airaksinen, Raitio, Story & Alku (2014)**, *Quasi closed phase glottal inverse filtering analysis with weighted linear prediction*, IEEE/ACM TASLP 22(3), 596–607, doi:10.1109/TASLP.2013.2294585. Introduces QCP: weighted LP whose weight function emphasises the closed phase and attenuates the main excitation; closures (GCIs) estimated from speech, no EGG; evaluated on LF and physical-model synthetic vowels at low and high f0. *Abstract.*
- **Gowda, Airaksinen & Alku (2017)**, *Quasi-closed phase forward-backward linear prediction analysis of speech for accurate formant detection and estimation*, JASA 142(3), 1542–1550, doi:10.1121/1.5001512. **Full text.**
  - Method: minimise F = Σ w_n (x_n + Σ_k a_k x_{n−k})² + Σ w_n (x_n + Σ_k a_k x_{n+k})², one coefficient set for forward and backward prediction, covariance range (QCP-FBCOV). Normal equations Σ_k d_{i,k} a_k = −d_{i,0} with d_{i,k} = Σ w_n x_{n−i} x_{n−k} + Σ w_n x_{n+i} x_{n+k}.
  - Weights: QCP function with position quotient 0.05, duration quotient 0.7, ramp 7 samples (8 kHz), minimum 10⁻⁵; GCIs from SEDREAMS (Drugman et al. 2012). Order 13 at 8 kHz, 30 ms rectangular frames, pre-emphasis 1 − 0.97 z⁻¹; peaks by convolving the spectrum with a 100 Hz Gaussian-derivative window.
  - Evidence: physical-model vowels (adult male, female, 5-year-old child; f0 100–450 Hz; 10 kHz) and VTR-TIMIT test (192 utterances, detection rate only, since VTR's labels are not ground truth). Forward-backward gives more equations per short closed phase and less line splitting; gains largest for women.
  - Stated limitation: depends on the GCI estimates; robustness to GCI errors under degradation not studied.
- **Gowda, Kadiri, Story & Alku (2020)**, *Time-varying quasi-closed-phase analysis for accurate formant tracking in speech signals* (TVQCP), IEEE/ACM TASLP 28, 1901–1914; arXiv 2308.16540. **Full text (method section).** QCP weights inside time-varying LP (polynomial coefficient trajectories over 200 ms blocks, L1 or L2 error); pitch-adaptive QCP parameters from a grid search on about 65,000 LF vowels (position quotient 0.05, ramp 0.375 ms, duration quotient 0.5–0.9, 0.8 used). Simulated child speech up to f0 600 Hz.
  - **Code**: github.com/njaygowda/ftrack (Apache-2.0, MATLAB; uses the GLOAT toolbox, BSD-style licence, for SEDREAMS and SRH pitch). `qcp_wt`: per cycle, weight 1 from GCI + PQ·T0 for DQ·T0 samples, linear ramps of Nramp samples at each end, the minimum elsewhere. Defaults there: DQ 0.7, PQ 0.05, Nramp 3, minimum 10⁻⁵, order 8, 8 kHz.
- **Gowda, Bollepalli, Kadiri & Alku (2021)**, *Formant tracking using quasi-closed phase forward-backward linear prediction analysis and deep neural networks*, IEEE Access; arXiv 2201.01525. DNN formant estimates refined frame by frame with the nearest QCP-FB spectral peak; reports error reductions of 29 / 48 / 35 % (F1–F3) against Wavesurfer on VTR. *Abstract.*
- **Alku, Kadiri & Gowda (2023)**, *Refining a deep learning-based formant tracker using linear prediction methods*, Computer Speech and Language 81, 101515; arXiv 2308.09051. DeepFormants refined with LP-COV or QCP-FB peaks (no retraining); QCP-FB best on VTR and noisy VTR. *Abstract.*

## Other formant trackers and evaluation

- **Deng et al. (2006)**, VTR-TIMIT formant database (semi-automatic, hand-corrected F1–F3). The field's usual benchmark. **It is built on TIMIT, so we cannot use it without an LDC licence** (AGENTS.md: no unlicensed TIMIT); comparisons with the papers above must be by method on our own references, not by their VTR numbers.
- **Mehta, Rudoy & Wolfe (2012)**, KARMA: Kalman-based ARMA formant and antiformant tracking; arXiv 1107.0076. Notes VTR's labelling errors. *Abstract.*
- **Dissen, Goldberger & Keshet (2019)**, *Formant estimation and tracking: a deep learning approach* (DeepFormants), JASA; code github.com/MLSpeech/DeepFormants. *Not yet read.*
- **Story & Bunton**, spectral filtering for formant measurement in children's speech (PubMed 26855461; POMA 23, 060002). Cepstral smoothing aimed at f0 250–600 Hz. *Not yet read.*
- **Mikusová (2021)**, TU Wien thesis: CNN estimation of tract resonances of synthesised high-pitched vowels. *Listing only.*
- **Meka, Ramesh & Kodukula (2026)**, *Self-supervised source-filter model of speech production* (SoFilNet), arXiv 2610.18120 (13 Oct 2026). Differentiable source-filter network trained by reconstruction only (ordered formant poles, extra poles, zeros); reports the lowest mean formant error on VTR among compared methods, supervised trackers included, and PTDB-TUG f0 as good as the best classical tracker. *Abstract only*: training corpus, per-formant numbers, code and licence not stated there. The closest recent relative of our approach (synthetic source-filter structure, no formant labels); to read in full.

## Direct resonance measurement (high f0)

- **Joliveau, Smith & Wolfe (2004)**, broadband acoustic excitation at the lips during singing: measures tract resonances directly, independent of the voice's harmonics. The only fair reference for sopranos (D-043 addendum: closed-phase F1 fails from about 450 Hz). *Not yet read.*

## What this means here (2026-10-28)

- QCP-FB is implementable from the JASA paper plus the authors' weight code; nothing needs to be guessed. Unlike our closed-phase reference it needs no EGG, so it could give references for Hillenbrand (including children) and other voices without EGG, if it holds up on D-043's grid with GCIs estimated from speech.
- Not clearly state of the art any more: neural trackers refined with QCP-FB (2021, 2023) beat it on VTR, and SoFilNet (2026) claims better still. But a *reference* method needs to be model-driven and independent of our training, which QCP-FB is and the neural trackers are not.
- Our D-043 (closed-phase LPC on physical-model renders over f0, open quotient and closure gap) parallels the synthetic evaluations in Airaksinen et al. (2014) and Gowda et al. (2017); Paper A must cite them and say what D-043 adds (leaky closure, open quotient, the 400–800 Hz breakdown).
