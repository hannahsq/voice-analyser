"""D-085: render train_long_v3, the long-crop training set. train_v2's config (docs/synth_sets/train_v2.index.json), only
these changed: 8-16 s clips (one anatomy per clip, as every clip); coupling at strength 1 on 42 % of clips (D-080's
coupled share in the mix, as train_v2kb's strength); acoustic labels (the paired heads' second set); and the cascade's
aspiration through the coupled tract (SynthConfig.coupled_aspiration, D-085). Seed 85 (used by no other set).
6000 clips (about 20 h of audio), consolidated to flat arrays.

    python scripts/analysis/d085_render.py [workers]
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT, N, SEED = Path("data/synth/train_long_v3"), 6000, 85


def config():
    import json
    from dataclasses import fields

    from vocal_analysis.synth.sampler import SynthConfig

    idx = json.loads(Path("docs/synth_sets/train_v2.index.json").read_text())
    names = {f.name for f in fields(SynthConfig)}
    cfg = SynthConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in idx["synth_config"].items() if k in names})
    return replace(cfg, duration_s=(8.0, 16.0), p_coupling=0.42, coupling_scale=(1.0, 1.0), acoustic_labels=True,
                   coupled_aspiration=True)


if __name__ == "__main__":
    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import consolidate

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if not (OUT / "flat" / "done").exists():
        write_set(OUT, N, config(), seed=SEED, workers=workers, extra_index={"entry": "D-085"}, log=print)
        consolidate(OUT, log=print)
    print(f"{OUT}: done", flush=True)
