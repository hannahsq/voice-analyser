# Baseline report

Generated 2026-10-28 14:04 UTC by `va baselines` at commit `629ca6b` (with uncommitted changes). Don't edit by hand; rerun the command.

Intervals are 95 % speaker-bootstrap percentile intervals (1000 resamples, whole speakers resampled). Detection rate (DR) counts a frame as correct when its error is within **both** bounds (decisions D-004); missing estimates count as misses. Gross error means more than 30 % off, or missing.

Speakers: tiers dev (D-024).

## Formants: Hillenbrand et al. (1995)

996 tokens from 83 speakers. Each token has 4 reference frames (steady state; 20/50/80 % of the vowel); F4 exists only at steady state. These are hand-checked measurements, and Hillenbrand is never trained on.

- **qcp_fb_published**: QCP-FB as published (Gowda et al. 2017): duration quotient 0.7, 8 kHz, SEDREAMS GCIs (D-046 arm A; not adopted)
- **qcp_fb**: QCP-FB, duration quotient 0.3, 10 kHz, SEDREAMS GCIs (D-046 arm B; narrowly failed its gate, not adopted; report-only)

| Baseline | Formant | n | MAE Hz | MAE % | RMSE Hz | DR (20 %, 200 Hz) % | DR (25 %, 250 Hz) % | DR (30 %, 300 Hz) % | Gross % | Missing % |
|---|---|---|---|---|---|---|---|---|---|---|
| qcp_fb_published | F1 | 3984 | 79 [69, 90] | 10.8 | 137 | 78.9 [75.5, 82.5] | 84.6 | 89.8 | 10.2 | 0.0 |
| qcp_fb_published | F2 | 3962 | 708 [632, 779] | 35.6 | 1003 | 37.9 [32.5, 43.2] | 39.7 | 41.2 | 56.9 | 0.0 |
| qcp_fb_published | F3 | 3863 | 769 [702, 833] | 26.2 | 993 | 32.7 [27.8, 37.7] | 33.1 | 33.7 | 45.1 | 0.0 |
| qcp_fb_published | F4 | 841 | 1004 [899, 1105] | 24.0 | 1217 | 20.3 [14.8, 25.9] | 20.9 | 21.3 | 42.9 | 12.6 |
| qcp_fb | F1 | 3984 | 72 [63, 83] | 9.9 | 137 | 82.7 [79.5, 85.7] | 85.2 | 88.2 | 11.8 | 0.0 |
| qcp_fb | F2 | 3962 | 444 [372, 517] | 21.8 | 758 | 58.7 [52.8, 64.7] | 59.6 | 60.5 | 33.9 | 0.0 |
| qcp_fb | F3 | 3863 | 557 [479, 634] | 18.5 | 822 | 49.4 [43.0, 56.0] | 50.5 | 51.6 | 32.2 | 0.0 |
| qcp_fb | F4 | 841 | 812 [698, 929] | 19.0 | 1139 | 38.5 [33.1, 44.1] | 41.3 | 43.0 | 28.9 | 1.0 |

### DR (20 %, 200 Hz) % by speaker group

| Baseline | men F1 | men F2 | men F3 | women F1 | women F2 | women F3 | boys F1 | boys F2 | boys F3 | girls F1 | girls F2 | girls F3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qcp_fb_published | 98.6 | 65.3 | 58.1 | 76.0 | 33.0 | 27.5 | 66.3 | 15.3 | 12.5 | 56.8 | 15.6 | 12.6 |
| qcp_fb | 98.4 | 89.0 | 81.4 | 80.1 | 51.6 | 40.4 | 74.7 | 40.5 | 30.8 | 62.3 | 28.8 | 20.5 |

n frames per column: men 1296, women 1392, boys 768, girls 528.

### DR (20 %, 200 Hz) % by f0 band (the token's steady-state f0)

| Baseline | <150 F1 | <150 F2 | <150 F3 | 150-300 F1 | 150-300 F2 | 150-300 F3 | 300-500 F1 | 300-500 F2 | 300-500 F3 |
|---|---|---|---|---|---|---|---|---|---|
| qcp_fb_published | 99.5 | 64.2 | 57.7 | 72.3 | 29.3 | 24.5 | 65.0 | 17.5 | 12.5 |
| qcp_fb | 98.8 | 89.9 | 83.6 | 77.4 | 48.2 | 38.3 | 72.5 | 55.0 | 15.0 |

n frames per column: <150 988, 150-300 2956, 300-500 40.

### DR (20 %, 200 Hz) % by frame position

| Baseline | 20/50/80% F1 | 20/50/80% F2 | 20/50/80% F3 | steady F1 | steady F2 | steady F3 |
|---|---|---|---|---|---|---|
| qcp_fb_published | 78.7 | 37.6 | 32.6 | 79.6 | 38.6 | 33.1 |
| qcp_fb | 82.8 | 58.6 | 49.7 | 82.3 | 58.8 | 48.6 |

n frames per column: 20/50/80% 2988, steady 996.

## Provenance

| | |
|---|---|
| Commit | `629ca6b` (dirty) |
| Submodule | `7263f6c9e79442527229c4ae2be79a7c83f1beb2 external/random-cafe (heads/main)` |
| Submodule | `0ab55688d511d18ca9afd52787ddffc871b1d447 external/voice-dataset-labeler (heads/main)` |
| Python | 3.13.15 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| torch | 2.13.0+rocm10.0.0 |
| praat-parselmouth | 0.4.7 |
| pesto-pitch | 2.0.1 |
| torchcrepe | 0.0.24 |
| Manifest `hillenbrand` | `93c302ef20838268` |
| Manifest `mdb_stem_synth` | `f8fcec203d1c79be` |
| Manifest `mir1k` | `3c3856f4161505ce` |
| Manifest `ptdb_tug` | `2403475d28025082` |
| Manifest `vocalset` | `4cea60672f9934ed` |
| Manifest `zeroth_korean` | `bebcb05382d5074e` |
