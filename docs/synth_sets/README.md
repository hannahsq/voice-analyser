# Synthetic training sets: how to regenerate them

Each `<set>.index.json` is the set's own `index.json`: seed, clip count, full `synth_config`, channel config and the
commit that generated it. The sets themselves (about 32 GB each: shards, plus the consolidated `flat/` copy that
training reads) are not kept when disk is short; they regenerate bit-identically from these files.

| Set | Seed | Regenerates bit-identically with | On disk (2026-10-11) |
|---|---|---|---|
| train_v1 | 1 | the generating commit `7f89c09` (not today's sampler) | deleted |
| train_v2 | 2 | today's code (checked 2026-10-11, m3 `6798ba3` onwards) | `flat/` only |
| train_v3a | 2 | today's code | deleted |
| train_v3b | 3 | today's code | `flat/` only |
| train_v2c | 2 | today's code (`p_coupling` 1) | full |

To regenerate (from the m3 worktree, or a worktree at the listed commit; run the script from that directory so its
`vocal_analysis` is imported, not the main checkout's):

```python
import json
from dataclasses import fields
from vocal_analysis.synth.sampler import SynthConfig
from vocal_analysis.synth.shards import write_set
idx = json.load(open("docs/synth_sets/train_v3a.index.json"))
names = {f.name for f in fields(SynthConfig)}
cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in idx["synth_config"].items() if k in names})
write_set("../voice-analyser/data/synth/train_v3a", idx["n_clips"], cfg, seed=idx["seed"], workers=32)
```

Training consolidates the shards into `flat/` on first use (`train.data.consolidate`). Per-clip metadata (speaker,
glottis draws) lives only in the shards; regenerate the set to recover it.
