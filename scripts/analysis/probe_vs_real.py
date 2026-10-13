"""Do synthetic probe responses predict a model's real closed-phase F1 bias? Across models (seeds, data variants,
architectures, sizes): Spearman correlation of each probe feature with the model's median signed F1 - closed-phase
error on PTDB-TUG dev speakers (frames valid for every model). Args: scratch dir (ptdb_egg_sub.npz), bandwidth probe
output (sensitivity_bw_cmp.py), coupling probe output (coupling_probe3.py), then the model names in probe column order."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
S, BW, CP, MODELS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
CELL = re.compile(r"([+-]\d+) \(\s*([+-]\d+)\)")

# real bias, dev speakers
tiers = json.load(open(Path(__file__).resolve().parents[2] / "vocal_analysis/eval/tiers.json"))["speakers"]
d = dict(np.load(f"{S}/ptdb_egg_sub.npz", allow_pickle=True))
dev = np.array([tiers.get(s if ":" in str(s) else f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
ok = np.isfinite(d["cp"][:, 0]) & np.all([np.isfinite(d[m][:, 0]) for m in MODELS], 0) & dev
real = {}
for sex in ("f", "m"):
    k = ok & (d["sex"] == sex)
    real[sex] = np.array([np.median(d[m][k, 0] - d["cp"][k, 0]) for m in MODELS])

# probe features
feat = {}
group = None
for line in open(BW, encoding="utf-8"):
    if re.match(r"^(men|women) f0 \d+", line):
        group = line.strip()
    cells = CELL.findall(line)
    if group and len(cells) == len(MODELS):
        name = line[:30].strip()
        if name == "baseline":
            feat[f"bw {group}: baseline"] = np.array([float(a) for a, _ in cells])
        else:
            feat[f"bw {group}: {name} (change)"] = np.array([float(b) for _, b in cells])
sex = None
rows = {}
for line in open(CP, encoding="utf-8"):
    m = re.match(r"^(men|women), f0", line)
    if m:
        sex = m.group(1)
    m = re.match(r"^\s*([\d.]+)\s+(\d+)\s+(on|off) \|", line)
    if m and sex:
        cells = CELL.findall(line)[1:]  # Praat first
        if len(cells) == len(MODELS):
            rows[(sex, float(m.group(1)), int(m.group(2)), m.group(3))] = np.array([[float(a), float(b)] for a, b in cells])
for sx in ("men", "women"):
    if (sx, 1.0, 30, "off") not in rows:
        continue
    base = rows[(sx, 1.0, 30, "off")][:, 0]
    feat[f"cpl {sx}: baseline Rd 1.0"] = base
    feat[f"cpl {sx}: coupling, Rd 1.0 (change)"] = rows[(sx, 1.0, 30, "on")][:, 1]
    feat[f"cpl {sx}: coupling, Rd 2.2 (change)"] = rows[(sx, 2.2, 30, "on")][:, 1]
    feat[f"cpl {sx}: breathiness Rd 1.0 -> 2.2"] = rows[(sx, 2.2, 30, "off")][:, 0] - base
    feat[f"cpl {sx}: noise HNR 30 -> 8, Rd 1.0"] = rows[(sx, 1.0, 8, "off")][:, 1]

import os
keep = np.array([bool(re.search(os.environ.get("PROBE_SUBSET", "."), m)) for m in MODELS])  # e.g. conformer
real = {k: v[keep] for k, v in real.items()}
feat = {k: v[keep] for k, v in feat.items()}
MODELS = [m for m, k in zip(MODELS, keep) if k]
print(f"{len(MODELS)} models; real closed-phase F1 bias on PTDB-TUG dev ({ok.sum()} frames valid for all)")
print(f"{'model':22s} {'women':>7s} {'men':>7s}")
for i, m in enumerate(MODELS):
    print(f"{m:22s} {real['f'][i]:+7.1f} {real['m'][i]:+7.1f}")
rng = np.random.default_rng(0)


def rho(x, y):
    r = spearmanr(x, y).statistic
    null = np.array([spearmanr(rng.permutation(x), y).statistic for _ in range(2000)])
    return r, np.mean(np.abs(null) >= abs(r))


print(f"\nSpearman rho of each probe feature with the real bias (permutation p, 2000)\n{'feature':48s} {'women':>15s} {'men':>15s}")
for name, x in feat.items():
    rw, pw = rho(x, real["f"])
    rm, pm = rho(x, real["m"])
    print(f"{name:48s} {rw:+6.2f} (p {pw:.3f}) {rm:+6.2f} (p {pm:.3f})")
