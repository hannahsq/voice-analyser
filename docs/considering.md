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

- **A single context convention, with sliding windows at inference** (2026-10-26; from MERT, which trains on 5 s and runs long audio as 5 s windows). D-034's context-length checks found Parakeet's layer 5 depends strongly on its input span. If a new target or head cache is built, pick one span for targets, head features and scoring alike, and window long recordings to match.
- **Masked spectral reconstruction as an auxiliary self-supervised objective** (2026-10-26; MERT's CQT teacher). It would apply to pretraining on unlabelled real voices, and our VQT front end makes the target free. Caution: reconstruction keeps fine spectral detail, which is what carried the synthetic F1 shortcut in DAC's heads (D-033).
- **Pre-LN and attention relaxation for stability at scale** (2026-10-26; MERT needed both to train its 330M model). Relevant only if a student or teacher is scaled up a lot.

## Heads

- **DAC heads on 80 ms means** (2026-10-26). The claim that "acoustic encoders let heads learn the synthetic F1 shortcut" (D-033 part 5, logged as untested) rests on one ASR encoder (Parakeet, 80 ms frames) against one codec (DAC, about 12 ms). Frame rate is confounded with the objective. Training DAC heads on DAC features averaged to 80 ms (six seeds, the existing DAC caches, about 1 h) would separate them. If the overread mostly goes away, frame rate explains much of it, and acoustic-target techniques (codec codes, spectral reconstruction) become less suspect. This matters before relying on the claim when choosing encoders or pretraining objectives.
- **H5 initialised, fine-tuned on a student's 10 ms frames with random offsets** (2026-10-26; Hannah, idle curiosity). Expected to land near the matched-noise control, since the H5 sweep found the students carry no more of H5's correction than noise does. If tried, it needs the same fine-tune on Parakeet plus matched noise as its control.
