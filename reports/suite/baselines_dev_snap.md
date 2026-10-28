# Baseline report

Generated 2026-10-28 14:20 UTC by `va baselines` at commit `27b2696`. Don't edit by hand; rerun the command.

Intervals are 95 % speaker-bootstrap percentile intervals (1000 resamples, whole speakers resampled). Detection rate (DR) counts a frame as correct when its error is within **both** bounds (decisions D-004); missing estimates count as misses. Gross error means more than 30 % off, or missing.

Speakers: tiers dev (D-024).

## Formants: Hillenbrand et al. (1995)

996 tokens from 83 speakers. Each token has 4 reference frames (steady state; 20/50/80 % of the vowel); F4 exists only at steady state. These are hand-checked measurements, and Hillenbrand is never trained on.

- **praat_fasttrack**: Praat Burg, FastTrack's per-talker-group ceiling ranges (men 4500-6500, women 5000-7000, children 5500-7500 Hz) and selection error (speaker group + oracle vowel span)
- **fasttrack_qcp_snap**: FastTrack (as praat_fasttrack), each formant snapped to the nearest QCP-FB (arm B) peak within 15 %
- **fasttrack_smooth_qcp_snap**: FastTrack smoothed (median 5 frames, then a zero-phase leaky integrator, alpha 0.15, log Hz), snapped as fasttrack_qcp_snap
- **fasttrack_dct_qcp_snap**: FastTrack's own DCT fit (intercept + 5 cosines) as the seed, snapped as fasttrack_qcp_snap

| Baseline | Formant | n | MAE Hz | MAE % | RMSE Hz | DR (20 %, 200 Hz) % | DR (25 %, 250 Hz) % | DR (30 %, 300 Hz) % | Gross % | Missing % |
|---|---|---|---|---|---|---|---|---|---|---|
| praat_fasttrack | F1 | 3984 | 17 [16, 19] | 2.9 | 31 | 99.4 [99.1, 99.7] | 99.8 | 99.8 | 0.2 | 0.0 |
| praat_fasttrack | F2 | 3962 | 45 [37, 55] | 2.6 | 123 | 97.6 [96.7, 98.4] | 98.2 | 98.5 | 0.9 | 0.0 |
| praat_fasttrack | F3 | 3863 | 53 [46, 62] | 1.8 | 129 | 96.1 [94.9, 97.3] | 97.4 | 97.9 | 0.4 | 0.0 |
| praat_fasttrack | F4 | 841 | 125 [109, 144] | 3.0 | 261 | 86.3 [83.4, 89.0] | 88.7 | 90.4 | 0.8 | 0.0 |
| fasttrack_qcp_snap | F1 | 3984 | 30 [27, 33] | 4.9 | 45 | 98.8 [98.3, 99.2] | 99.4 | 99.7 | 0.3 | 0.0 |
| fasttrack_qcp_snap | F2 | 3962 | 58 [50, 68] | 3.4 | 130 | 97.1 [95.9, 98.2] | 98.1 | 98.4 | 0.9 | 0.0 |
| fasttrack_qcp_snap | F3 | 3863 | 70 [63, 79] | 2.4 | 143 | 94.8 [93.5, 95.9] | 96.4 | 97.3 | 0.5 | 0.0 |
| fasttrack_qcp_snap | F4 | 841 | 139 [120, 160] | 3.3 | 279 | 83.6 [80.0, 86.9] | 88.3 | 91.2 | 1.0 | 0.0 |
| fasttrack_smooth_qcp_snap | F1 | 3984 | 30 [27, 32] | 4.8 | 45 | 98.8 [98.3, 99.3] | 99.4 | 99.7 | 0.3 | 0.0 |
| fasttrack_smooth_qcp_snap | F2 | 3962 | 58 [49, 67] | 3.4 | 129 | 97.1 [95.8, 98.2] | 98.1 | 98.4 | 0.8 | 0.0 |
| fasttrack_smooth_qcp_snap | F3 | 3863 | 70 [63, 78] | 2.4 | 142 | 94.8 [93.6, 96.0] | 96.5 | 97.3 | 0.5 | 0.0 |
| fasttrack_smooth_qcp_snap | F4 | 841 | 136 [117, 157] | 3.2 | 274 | 83.9 [80.4, 87.4] | 88.6 | 91.6 | 0.8 | 0.0 |
| fasttrack_dct_qcp_snap | F1 | 3984 | 30 [27, 32] | 4.8 | 45 | 98.9 [98.4, 99.3] | 99.4 | 99.7 | 0.3 | 0.0 |
| fasttrack_dct_qcp_snap | F2 | 3962 | 58 [49, 67] | 3.4 | 130 | 97.1 [95.9, 98.2] | 98.1 | 98.4 | 0.8 | 0.0 |
| fasttrack_dct_qcp_snap | F3 | 3863 | 71 [63, 79] | 2.4 | 144 | 94.7 [93.5, 95.9] | 96.3 | 97.2 | 0.5 | 0.0 |
| fasttrack_dct_qcp_snap | F4 | 841 | 138 [119, 161] | 3.3 | 277 | 83.1 [79.6, 86.5] | 88.1 | 91.1 | 0.8 | 0.0 |

### DR (20 %, 200 Hz) % by speaker group

| Baseline | men F1 | men F2 | men F3 | women F1 | women F2 | women F3 | boys F1 | boys F2 | boys F3 | girls F1 | girls F2 | girls F3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| praat_fasttrack | 100.0 | 99.2 | 97.4 | 99.6 | 97.2 | 96.0 | 99.5 | 97.5 | 96.9 | 97.5 | 94.9 | 91.9 |
| fasttrack_qcp_snap | 99.8 | 99.3 | 96.9 | 98.6 | 96.1 | 94.3 | 99.1 | 98.0 | 94.8 | 96.6 | 93.1 | 90.7 |
| fasttrack_smooth_qcp_snap | 99.8 | 99.3 | 96.9 | 98.4 | 96.0 | 94.6 | 99.5 | 98.2 | 94.6 | 96.6 | 93.0 | 90.9 |
| fasttrack_dct_qcp_snap | 99.8 | 99.2 | 96.9 | 98.7 | 96.2 | 94.4 | 99.2 | 98.2 | 94.6 | 96.8 | 92.8 | 90.6 |

n frames per column: men 1296, women 1392, boys 768, girls 528.

### DR (20 %, 200 Hz) % by f0 band (the token's steady-state f0)

| Baseline | <150 F1 | <150 F2 | <150 F3 | 150-300 F1 | 150-300 F2 | 150-300 F3 | 300-500 F1 | 300-500 F2 | 300-500 F3 |
|---|---|---|---|---|---|---|---|---|---|
| praat_fasttrack | 100.0 | 99.5 | 97.8 | 99.2 | 97.1 | 95.6 | 100.0 | 90.0 | 90.0 |
| fasttrack_qcp_snap | 99.9 | 99.9 | 97.2 | 98.4 | 96.3 | 94.1 | 97.5 | 90.0 | 82.5 |
| fasttrack_smooth_qcp_snap | 99.9 | 99.9 | 97.2 | 98.5 | 96.3 | 94.2 | 97.5 | 90.0 | 82.5 |
| fasttrack_dct_qcp_snap | 99.9 | 99.8 | 97.2 | 98.6 | 96.3 | 94.1 | 97.5 | 90.0 | 82.5 |

n frames per column: <150 988, 150-300 2956, 300-500 40.

### DR (20 %, 200 Hz) % by frame position

| Baseline | 20/50/80% F1 | 20/50/80% F2 | 20/50/80% F3 | steady F1 | steady F2 | steady F3 |
|---|---|---|---|---|---|---|
| praat_fasttrack | 99.5 | 97.6 | 96.0 | 99.2 | 97.6 | 96.4 |
| fasttrack_qcp_snap | 98.8 | 97.1 | 94.6 | 98.7 | 97.2 | 95.3 |
| fasttrack_smooth_qcp_snap | 98.8 | 97.1 | 94.7 | 98.8 | 97.2 | 95.3 |
| fasttrack_dct_qcp_snap | 98.9 | 97.1 | 94.6 | 98.9 | 97.3 | 95.2 |

n frames per column: 20/50/80% 2988, steady 996.

## Provenance

| | |
|---|---|
| Commit | `27b2696` |
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
