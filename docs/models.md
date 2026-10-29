# Models: names, lineage and checkpoints

Which run each named model is, what made it, and the exact weights. Reasons and results are in `decisions.md`; how the
training sets regenerate is in `synth_sets/README.md`. Checkpoints live in `runs/` (not in git): **sha256** is the
first 16 hex digits of `ckpt_last.pt`'s SHA-256, the same convention the resynthesised sets use to record their
tracker. **Commit** is the one in the run's `config.json` (trunk / decay branch when they differ). Runs before
2026-10-22 are recorded as "dirty" whenever any file in the checkout was, untracked reports included, so that flag
says nothing about the code; from 2026-10-22 it lists uncommitted changes under `vocal_analysis/` and `configs/` only.
Released teachers carry an annotated git tag at their training commit. Versions (2026-10-28): the major number follows the milestone and the training data (v1–v3: M1–M3; v4 is M4's teacher, on re-tracked labels); a minor number marks recipe or head changes within a milestone (teacher v3.1 = roadmap item 5).

Sizes: **full GRU** = front end 24 / 48 / 64 channels, f0 16, embedding 384, BiGRU 256 × 2 (f0 128). **Full
Conformer** = the same front end, Conformer width 128, 2 layers, 4 heads (f0: dilated convs). **S2** = front end
8 / 16 / 32, f0 8, embedding 192 (the fast proxy, D-024).

## Teachers

| Name | Run | Size | Training data | Recipe | Commit | sha256 | Decision | Status |
|---|---|---|---|---|---|---|---|---|
| **teacher v1** (tag `teacher-v1`) | `teacher_v1` | full GRU | train_v1 | AdamW 1e-3, cosine, 20k | `7f89c09` (clean) | `d6a6a64beb52fb5e` | D-010, D-014 | superseded by v2 |
| **teacher v2** (tag `teacher-v2`) | `teacher_v2` → `teacher_v2_d16k` | full GRU | train_v2 | AdamW 1e-3, WSD: constant to 16k, decay 4k | `2a173d0` / `e43dfdd` | `bd90516ef923d592` | D-013, D-020 | superseded by v3; **tracked the resynthesised sets** (VocalSet and Zeroth-Korean `*_d16k`, their `index.json` names it) |
| **teacher v3** (tag `teacher-v3`) | `conformer_v2r2_s1` → `_s1_d16k` | full Conformer | train_v2 + 30 % resynthesised VocalSet and Zeroth-Korean (`v2r2`) | AdamW 1e-3, WSD 16k + 4k, seed 1 | `8e58a13` | `d28ef6f610f5ea04` | D-025 | superseded by v3.1 as the single model |
| **labeller v3** | `conformer_v2r2_s1_d16k` + `conformer_v2r2_s2_d16k` | 2 × full Conformer | as teacher v3 | per-frame mean in log frequency of the two (`ensemble_v3.py`) | `8e58a13`; `316990b` / `4265498` | `d28ef6f610f5ea04` + `d204df7dd521982d` | D-025 | **current** labeller (M4 re-tracking) |

| **teacher v3.1** (tag `teacher-v3.1`) | `d040_5b_s2` → `_s2_b20k` | full Conformer + joint VTL head | as teacher v3 | rA (Muon 2e-3 / Adan 1e-2, wd 0.007; VTL head AdamW 1e-3 / wd 0.01), WSD 16k + 4k, seed 2 | `d9bb3f6` / `4805e5d` | `fa176ad17f13a5f2` | D-031, D-036, D-040 | **current** single model (Hannah chose seed 2, 2026-10-29); sibling seed 1 `d040_5b_s1` → `_s1_b20k` (`62c72e1` / `e66d438`, `22107329f6a0731f`) |

## Other full-size runs that decisions cite

| Run | Size | Data | Recipe | Commit | sha256 | Role |
|---|---|---|---|---|---|---|
| `teacher_v2_d06k` | full GRU | train_v2 | WSD 6k + 4k | `2a173d0` / `2429e99` | `4ba122bfc776a3ed` | teacher v2's early branch (M2 report) |
| `teacher_v2_sf` | full GRU | train_v2 | schedule-free AdamW, 10k | `112673c` | | D-018 comparison |
| `teacher_v2s1_d06k` | full GRU | train_v2 | as v2 d06k, seed 1 | `56ed316` | | seed replicate (D-024) |
| `teacher_v1_d06k` | full GRU | train_v1 | v1 at d06k's schedule | `4e942ab` | | D-024 |
| `teacher_v3a_d06k`, `teacher_v3b_d06k` | full GRU | train_v3a, train_v3b | as v2 d06k | `56ed316` | | **data arms of D-020 / D-022, not teacher v3**: named before the M3 teacher existed |
| `conformer_m3_d06k`, `_d16k` | full Conformer | train_v2 + resynthesised VocalSet | WSD 6k / 16k + 4k | `798bc0a` / `16a6da0` | `f99c41f5e37bcc3d` (d16k) | first M3 Conformer (D-025) |
| `conformer_v2_d16k`, `_s1_d16k`, `_s2_d16k` | full Conformer | train_v2 | WSD 16k + 4k, seeds 0–2 | `24ad7e0` / `f76c9a4`; `2e1a3e4` / `d761787`; `d2d4760` / `316990b` | `85ea44157316d412`, `20e4a5b0f9e5b1bd`, `dccfa24d98751b00` | teacher v3's baselines (D-025, D-026) |
| `d031_rA` → `_b20k`; `d040_5a_s2` → `_s2_b20k` | full Conformer | as teacher v3 | rA, WSD 16k + 4k, seeds 1, 2 | `a4ad27f` / `fc5c71d`; `27b2696` / `d9bb3f6` | `dcb4f2ca4f207d82`, `c8e2cbf2f6c3adb1` | D-040's control 5a (D-031's adopted recipe) |
| `conformer_v2r2_d16k`, `_s2_d16k` | full Conformer | as teacher v3 | WSD 16k + 4k, seeds 0, 2 | `899f8d6` / `8e58a13`; `316990b` / `4265498` | `cf5e780a654708bf`, `d204df7dd521982d` | teacher v3's sibling seeds |

S2 runs (`runs/small/`) are screen models, named in the `decide_*.txt` files that judged them (D-024 onwards); none
is released.

## Lineage

```
train_v1 ──> teacher v1
train_v2 ──> teacher v2 (d16k) ──tracks──> resynthesised VocalSet + Zeroth-Korean (*_d16k)
train_v2 + those sets ──> conformer_v2r2 seeds 0–2 ──> teacher v3 (seed 1), labeller v3 (mean of seeds 1, 2)
those data, rA recipe + joint VTL head ──> d040_5b seeds 1, 2 ──> teacher v3.1 (seed 2)
```

M4 re-tracks with labeller v3; any later resynthesised set records its tracker in its `index.json` the same way.

## Adding a model

When a model is released (a new teacher or labeller): add its row here with the sha256 of `ckpt_last.pt`, tag the
training commit (`git tag -a teacher-vN <commit> -m "<run>, sha256 <16 hex>, D-0NN"`), and score it with
`va eval teacher` (dev speakers by default; `reports/suite/`).
