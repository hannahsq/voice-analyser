"""D-085 amendment 1: re-render D-085's two coupled training sets with the coupling strength drawn uniform on [0, 1]
per coupled clip (D-023's draw), in place of strength 1. Everything else, seeds included, is as before, and the
sampler draws the strength from the same rng call either way, so every other parameter of every clip is unchanged.

- train_long_v3u: train_long_v3's config (d085_render.config) and seed 85, 6000 clips of 8-16 s.
- train_v2ku: train_v2kb's config (d078_render.config: every clip a tube clip, coupled) and seed 78, 20000 clips.

The earlier sets (strength 1) are kept as D-080's record. Coupled validation (val_v2ko) stays at strength 1.
Prints each set's clip counts by kind and the drawn strengths' range after rendering.

    python scripts/analysis/d085_render_draw.py [workers]
"""

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DRAW = (0.0, 1.0)


def sets():
    import d078_render
    import d085_render

    return [(Path("data/synth/train_long_v3u"), 6000, 85, replace(d085_render.config(), coupling_scale=DRAW)),
            (Path("data/synth/train_v2ku"), 20000, 78, replace(d078_render.config(), coupling_scale=DRAW))]


def counts(root: Path) -> str:
    kinds, scales = Counter(), []
    for s in sorted(root.glob("shard_*.npz")):
        with np.load(s) as z:
            for m in json.loads(str(z["meta"])):
                t = m["tube"] if isinstance(m["tube"], dict) else None
                coupled = bool(t and t.get("glottis"))
                kinds[(m["tract"], "coupled" if coupled else "uncoupled")] += 1
                if coupled:
                    scales.append(t["coupling_scale"])
    n = sum(kinds.values())
    lines = [f"{root}: {n} clips"] + [f"  {a} {b}: {k} ({100 * k / n:.1f} %)" for (a, b), k in sorted(kinds.items())]
    if scales:
        q = np.quantile(scales, [0, 0.25, 0.5, 0.75, 1])
        lines.append("  strength quantiles 0/25/50/75/100 %: " + " ".join(f"{x:.3f}" for x in q))
    return "\n".join(lines)


if __name__ == "__main__":
    from vocal_analysis.synth.shards import write_set
    from vocal_analysis.train.data import consolidate

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for out, n, seed, cfg in sets():
        if not (out / "flat" / "done").exists():
            write_set(out, n, cfg, seed=seed, workers=workers, extra_index={"entry": "D-085 amendment 1"}, log=print)
            consolidate(out, log=print)
        print(counts(out), flush=True)
        print(f"{out}: done", flush=True)
