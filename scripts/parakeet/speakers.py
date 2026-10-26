"""Speaker id per reference frame for the scoring cache's gold_hillenbrand and gold_ptdb_cp (D-033 cross-fitted seed-pair
selection): the same items in the same order as cache.py, checked against each item's reference-frame count.
Args: scoring cache dir, scratch dir with ptdb_egg_sub.npz."""
import os
import sys
from pathlib import Path

import numpy as np

from vocal_analysis.data.manifest import read_manifest
from vocal_analysis.eval import tiers as TI
from vocal_analysis.eval.datasets import registry

C, S = Path(sys.argv[1]), Path(sys.argv[2])
mods = registry()
TIER = os.environ.get("VA_TIER", "dev")  # as cache.py

spk = []
for row in TI.filter_rows(read_manifest("data/manifests/hillenbrand.csv"), (TIER,)):
    item = mods["hillenbrand"].load(Path("data/hillenbrand"), row)
    ok = np.isfinite(item.times)
    if ok.any():
        spk.append([row.speaker_id] * int(ok.sum()))
ro = np.load(C / "gold_hillenbrand" / "ref_off.npy")
assert [len(x) for x in spk] == list(np.diff(ro)), "Hillenbrand items differ from the cache"
np.save(C / "gold_hillenbrand" / "speaker.npy", np.array(sum(spk, []), dtype=str))

d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
tmap = TI.load()
devm = np.array([TI.tier_of(f"ptdb_tug:{s}", tmap) == TIER for s in d["speaker"]])
spk = []
for s, u in sorted({(str(a), str(b)) for a, b in zip(d["speaker"][devm], d["utt"][devm])}):
    spk.append([f"ptdb_tug:{s}"] * int(((d["speaker"] == s) & (d["utt"] == u)).sum()))
ro = np.load(C / "gold_ptdb_cp" / "ref_off.npy")
assert [len(x) for x in spk] == list(np.diff(ro)), "PTDB-TUG items differ from the cache"
np.save(C / "gold_ptdb_cp" / "speaker.npy", np.array(sum(spk, []), dtype=str))
print("ok", {n: len(set(np.load(C / n / "speaker.npy", allow_pickle=True))) for n in ("gold_hillenbrand", "gold_ptdb_cp")})
