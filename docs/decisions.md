# Decisions log

Divergences from, and refinements of, [voice-analyser-mk2-plan.md](voice-analyser-mk2-plan.md) (plan §0).

## D-001 (2026-10-07): torch is not a declared dependency

The training stack uses AMD's ROCm wheels for gfx1151 (`torch 2.13.0+rocm10.0.0`, `rocm 10.0.0`, HIP 7.15). If `torch` appeared in `pyproject.toml`, `pip install -e .` could replace that build with a generic wheel. So torch is installed separately from AMD's index, and the package is installed with `pip install -e . --no-deps` plus the explicit dependencies. Everything outside model code (metrics, manifests, loaders) imports without torch.

## D-002 (2026-10-07): GPU verified on gfx1151; MIOpen RNNs disabled

Verified on the Radeon 8060S (gfx1151), with ~78 GiB visible to torch:

| Op | Result |
|---|---|
| fp32 matmul | ok, ~3.4 TFLOPS on 4096²; matches CPU |
| Conv2d forward + backward via MIOpen, fp32 and bf16 autocast | ok, matches CPU |
| FFT | ok |
| `nn.GRU` via MIOpen | **fails**: `miopenStatusUnknownError`. MIOpen compiles these kernels at runtime with hipRTC, which can't find `<type_traits>` on this Windows build |
| `nn.GRU` with cuDNN/MIOpen disabled (native kernels), fp32 and bf16 | ok |

Decision: `device.disable_miopen: rnn` (the default) wraps RNN forward calls in `vocal_analysis.device.rnn_guard()`, keeping MIOpen for convolutions. Use `all` if other MIOpen ops (e.g. batch norm, some activations) hit the same compile error. `tests/test_device.py` is the regression check, so rerun it after any ROCm upgrade. The RNN workaround can go once MIOpen runtime compilation works on Windows.

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
