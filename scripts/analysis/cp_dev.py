"""Closed-phase F1 bias per model on PTDB-TUG dev speakers (D-024 tiers; ARCTIC is dev only), with a paired speaker
bootstrap. Args: scratch dir (holding ptdb_egg_sub.npz, arctic_egg_sub.npz), comma-separated models, pairs b-a."""
import os, sys, json, numpy as np
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
S, MODELS, PAIRS = sys.argv[1], sys.argv[2].split(","), [p.split("-") for p in sys.argv[3:]]
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
rng = np.random.default_rng(0)
d = dict(np.load(f"{S}/ptdb_egg_sub.npz", allow_pickle=True))
spk_id = lambda s: s if ":" in s else f"ptdb_tug:{s}"
allowed = set(os.environ.get("VA_TIERS", "dev").split(","))  # semi-private: a use to log in decisions.md
print(f"tiers: {','.join(sorted(allowed))}")
dev = np.array([tiers.get(spk_id(str(s))) in allowed for s in d["speaker"]])
ok = np.isfinite(d["cp"][:, 0]) & np.all([np.isfinite(d[e][:, 0]) for e in MODELS], 0) & dev
for g in np.unique(d["sex"]):
    m = ok & (d["sex"] == g); r = d["cp"][m, 0]; spk = np.unique(d["speaker"][m])
    print(f"PTDB-TUG {'+'.join(sorted(allowed))} {g}: {m.sum()} frames, {len(spk)} speakers; F1 - closed phase, median signed (|err|)")
    for e in MODELS:
        x = d[e][m, 0] - r
        print(f"  {e:20s} {np.median(x):+6.1f} ({np.median(np.abs(x)):5.1f})")
    for b, a in PAIRS:
        per = np.array([np.median(d[b][m & (d['speaker'] == s), 0] - d['cp'][m & (d['speaker'] == s), 0])
                        - np.median(d[a][m & (d['speaker'] == s), 0] - d['cp'][m & (d['speaker'] == s), 0]) for s in spk])
        bs = [np.median(rng.choice(per, len(per))) for _ in range(4000)]
        print(f"  bias {b} - {a}: {np.median(per):+5.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]  per speaker {np.round(per).astype(int).tolist()}")
a = dict(np.load(f"{S}/arctic_egg_sub.npz", allow_pickle=True))
ok = np.isfinite(a["cp"][:, 0]) & np.all([np.isfinite(a[e][:, 0]) for e in MODELS], 0)
for v in np.unique(a["voice"]):
    m = ok & (a["voice"] == v)
    print(f"ARCTIC {v} ({m.sum()} frames): " + "  ".join(f"{e} {np.median(a[e][m, 0] - a['cp'][m, 0]):+.0f}" for e in MODELS))
