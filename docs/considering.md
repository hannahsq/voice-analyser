# Things to consider

Ideas worth keeping in mind, without a place in the plan. Nothing here is scheduled. An item moves to `roadmap.md` when it is decided, and gets pre-registered in `decisions.md` before it runs. Each entry says where it came from and what would make it worth doing.

## Models

- **MERT as a research comparison** (2026-10-26; arXiv 2306.00107). A self-supervised music encoder: 95M / 330M parameters, 24 kHz input, 75 Hz frames, trained on 5 s segments.
  - Against it:
    - CC BY-NC 4.0 weights, on Hugging Face only, so it needs an explicit exception and can never feed anything released.
    - Training data is 160K hours of internet-mined music with no source list, so MIR-1K and MedleyDB (behind MDB-stem-synth) can't be ruled out; it would be scored on Hillenbrand and PTDB-TUG only.
    - Both of its pretraining teachers are acoustic (EnCodec codes and the CQT), so it likely behaves more like DAC than Parakeet.
  - Worth doing only as a quick frozen-head probe, if a singing-trained encoder becomes a real question.

## Training techniques

- **A single context convention, with sliding windows at inference** (2026-10-26; from MERT, which trains on 5 s and runs long audio as 5 s windows). D-034's context-length checks found Parakeet's layer 5 depends strongly on its input span. If a new target or head cache is built, pick one span for targets, head features and scoring alike, and window long recordings to match. *Up-weighted 2026-10-26*: H5's correction turned out span-independent while layer 5 itself is not, and the VTL head (roadmap Next 4) aggregates over long spans, so the convention matters there.
- **Masked spectral reconstruction as an auxiliary self-supervised objective** (2026-10-26; MERT's CQT teacher). It would apply to pretraining on unlabelled real voices, and our VQT front end makes the target free. Caution: reconstruction keeps fine spectral detail, which is what carried the synthetic F1 shortcut in DAC's heads (D-033). *Down-weighted 2026-10-26*: v4's small but directional error along the F1 read-out (D-034) is the same kind of risk.
- **Pre-LN and attention relaxation for stability at scale** (2026-10-26; MERT needed both to train its 330M model). Relevant only if a student or teacher is scaled up a lot.

## Heads

- **DAC heads on 80 ms means** (2026-10-26). The claim that "acoustic encoders let heads learn the synthetic F1 shortcut" (D-033 part 5, logged as untested) rests on one ASR encoder (Parakeet, 80 ms frames) against one codec (DAC, about 12 ms). Frame rate is confounded with the objective. Training DAC heads on DAC features averaged to 80 ms (six seeds, the existing DAC caches, about 1 h) would separate them. If the overread mostly goes away, frame rate explains much of it, and acoustic-target techniques (codec codes, spectral reconstruction) become less suspect. This matters before relying on the claim when choosing encoders or pretraining objectives.
- **H5 initialised, fine-tuned on a student's 10 ms frames with random offsets** (2026-10-26; Hannah, idle curiosity). Expected to land near the matched-noise control, since the H5 sweep found the students carry no more of H5's correction than noise does. If tried, it needs the same fine-tune on Parakeet plus matched noise as its control. *Down-weighted 2026-10-26*: phase-aligned pooling removed v4's phase mismatch without any training (D-034), and the student line is paused (roadmap).
- **Cascade order between the VTL head and the formant heads** (2026-10-27; Hannah). Today the VTL head reads the formant trunk alone, beside the formant heads. Two directions, each with detached inputs (as the observability head takes formant positions), so neither loss can move the other's outputs:
  - *Formants → VTL*: the VTL head also takes F1–F4's expected positions. The planned D-037 candidate probes this on frozen features first. Risk: the formants' own f0 bias (F1 reads high in high voices) reaching the length estimate, the VTL-vs-f0 guard D-035 failed.
  - *VTL → formants*: the formant heads also take the VTL head's log length and σ, a speaker-size prior. Worth trying only if the VTL head is both confident and accurate on real voices. A prior would help most where the audio says least: F3 / F4, and frames the observability head marks as weakly observed. Risk: a confident but wrong length (or one with f0 drift) dragging every formant with it; formant heads that lean on the prior instead of the evidence.
  - When the joint per-formant head comes back (roadmap, paused), an ablation over the order (none / formants → VTL / VTL → formants), one change at a time, with formant MAE by formant (F3 / F4 especially), closed-phase bias, the invariance probe for both outputs, and the VTL measures.

## Back end and outcomes

- **VTL-normalised formants as an output** (2026-10-27; Hannah, from earlier work where VTL-normalised formants classified vowels better). The formants scaled to a reference length (F × L / L_ref, a shift in log frequency) keep mostly the vowel's shape. Unlike Lobanov or Nearey normalisation, which need a speaker's whole vowel inventory, a VTL estimate works from one recording, which is how outcome recordings arrive. Use the speaker's length from the recording as a whole (or a leave-one-out over its tokens), never the same frame's estimate, which can absorb the vowel itself (D-035's vowel-spread measure).
- **Splitting a formant change into size and shape** (2026-10-27; for the long-term surgical-outcomes aim). A before / after change in formants mixes a change in vocal-tract length (e.g. a raised or smaller larynx) with a change in articulation and resonance strategy. The VTL change and the change in VTL-normalised formants separate them. Worth doing once a VTL estimate passes on real voices, with its uncertainty carried through, since a small length change is easily within noise.

## Evaluation: invariance and equivariance probes

Hannah (2026-10-28): which perturbations should leave an output unchanged, and which should move it by a known amount, is one of the main tests separating a research-grade model from a hobbyist one. Today only the f0 probe exists (synthetic, invariance only). Each probe reports a slope against the imposed change with a shuffle null (equivariance: slope 1; invariance: slope 0), per output (F1–F4, bandwidths, VTL), and whether σ grows instead of the estimate drifting. Where possible each runs twice: on the synthetic probe (exact truth) and on real voices.

- **Counterfactual resynthesis** (the first to build): real voices from a non-gold dev pool, resynthesised unchanged / f0 only / tract length only (all formants scaled by one factor) / both / voice quality only. Equivariance for length and formants under scaling, invariance under f0 and quality. Check first that resynthesis at a scaled length adds no artefacts of its own. Also the check to run on any future consented before/after data before trusting a head on it.
- **Single-formant shift** (resynthesis, one formant moved): the moved formant should follow; the others and VTL should barely move. Measures cross-talk, and how far the cascade leans on priors rather than evidence.
- **Voice quality**: breathiness (aspiration, open quotient / Rd), creak, roughness (jitter, shimmer), falsetto. Post-operative and pathological voices differ here.
- **Vocal effort**: spectral tilt changes with level; formants and length should not.
- **Channel** (no resynthesis needed: degrade real gold-set audio and pair against the clean result, e.g. Hillenbrand MAE): microphone frequency responses, telephone band-limiting (F4 unobservable: σ should grow, not bias), codecs and bitrates (the USC MRI audio is AAC), reverberation, additive noise by SNR (white, babble, café), denoiser artefacts (musical noise, as in the MRI audio).
- **Timing**: speaking rate (time-stretch), crop position and context length, sub-hop offsets, polarity inversion.
- **Time reversal** (frame-level outputs): a frame's spectrum is unchanged by reversal, so a frame-local estimate should be too. A large change shows reliance on temporal or linguistic context.
  - *With a change of speaker* (Hannah): on A-then-B joins, forward against reversed per frame, by distance from the join. Reversal swaps which side of the join gets the other speaker as context, so the difference profile measures leakage across the change and how far it reaches. It is paired on identical frame spectra, so each frame's own estimation error cancels, and smaller leakage is detectable than against references.
  - *Care*: dry audio only (reverberation tails become pre-echo); exact frame mapping (pad to a whole number of hops so frame centres coincide); the analysis windows must be symmetric for the magnitudes to match. Context use within one speaker is legitimate (a speaker-level length *should* pool frames), so reversal alone flags context use; the speaker change is what isolates the harmful kind.
- **Content**: vowel spread of VTL (already measured), across languages (ArtSpeechMRIfr, Zeroth), nasalisation (added pole-zero pairs: F1 should not jump).
