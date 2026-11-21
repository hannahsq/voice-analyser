"""D-095: Saarbrücken's allocation, from the overview metadata and the zips' file names alone (no audio read).

Persons are `SprecherID`s across all five downloaded groups; a person's recordings all follow one tier. Strata:
- group: the first disorder group the person appears in (paresis, functional dysphonia, hypofunctional dysphonia,
  phonasthenia), else healthy;
- sex (`Geschlecht`);
- age band at the person's first recording: under 40, 40-59, 60 and over. Under 18 -> "excluded" (no adult tier).
  Ages are computed in memory and never written; no date leaves this script.
Draw (seeded, per stratum, shuffled then cut systematically with a random offset, so expected shares are exact):
- healthy adults: dev 50 women and 50 men, drawn only from persons with a recording in the zip (the healthy zip holds
  the first 687 of the overview's 869 recordings), stratified by age band; every other healthy adult semi / lockbox
  50 / 50 within (sex, age band, in zip).
- paresis and functional dysphonia: "reserve" one third of the persons recorded once and in one group only (unopened;
  a later decision gives it to training or evaluation without a re-draw); the rest, and every person recorded more than
  once or in two groups, dev / semi / lockbox 40 / 30 / 30 within (group, sex, age band).
- hypofunctional dysphonia, phonasthenia: whole for evaluation, dev / semi / lockbox 40 / 30 / 30.
Writes the saarbruecken entries and counts into vocal_analysis/eval/tiers.json (refuses if any exist) and prints
counts only.

    python scripts/analysis/d095_allocate.py <overview dir> <zip dir>
"""

from __future__ import annotations

import csv
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

SEED = 20261121
CORPUS = "saarbruecken"
GROUPS = (  # (overview csv stem, zip stem, short name), disorder groups in stratum priority order, healthy last
    ("Rekurrensparese", "Rekurrensparese", "paresis"),
    ("Funktionelle_Dysphonie", "Funktionelle Dysphonie", "functional"),
    ("Hypofunktionelle_Dysphonie", "Hypofunktionelle Dysphonie", "hypofunctional"),
    ("Phonasthenie", "Phonasthenie", "phonasthenia"),
    ("healthy", "healthy", "healthy"),
)
RESERVE_GROUPS = ("paresis", "functional")
RESERVE_FRAC = 1 / 3
DISORDER_SPLIT = (("dev", 0.4), ("semi", 0.3), ("lockbox", 0.3))
HEALTHY_DEV = 50  # per sex
TIER_FILE = Path("vocal_analysis/eval/tiers.json")


def _age(birth: str, rec: str) -> int:
    b, r = date.fromisoformat(birth), date.fromisoformat(rec)
    return r.year - b.year - ((r.month, r.day) < (b.month, b.day))


def _band(age: int) -> str:
    return "minor" if age < 18 else "<40" if age < 40 else "40-59" if age < 60 else "60+"


def _cut(members: list[str], shares, rng: random.Random) -> dict[str, str]:
    """Shuffle, then assign by cumulative share at positions (i + offset) / n: exact in expectation, +-1 per stratum."""
    m = members[:]
    rng.shuffle(m)
    off, n, out = rng.random(), len(m), {}
    for i, s in enumerate(m):
        u, acc = (i + off) / n, 0.0
        for tier, p in shares:
            acc += p
            if u < acc or tier == shares[-1][0]:
                out[s] = tier
                break
    return out


def main(ov_dir: Path, zip_dir: Path) -> None:
    recs = defaultdict(list)  # person -> [(group, recording id, sex, age)]
    in_zip = set()
    for stem, zstem, g in GROUPS:
        names = zipfile.ZipFile(zip_dir / f"{zstem}.zip").namelist()
        present = {n.split("/")[0] for n in names if n.split("/")[0].isdigit()}
        with open(ov_dir / f"{stem}.csv", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                p = r["SprecherID"]
                recs[p].append((g, r["AufnahmeID"], r["Geschlecht"], _age(r["Geburtsdatum"], r["AufnahmeDatum"])))
                if r["AufnahmeID"] in present:
                    in_zip.add(p)
    order = [g for *_, g in GROUPS]
    strata = defaultdict(list)
    info = {}
    sex_conflict = 0
    for p, rs in recs.items():
        group = min((r[0] for r in rs), key=order.index)
        sexes = {r[2] for r in rs}
        sex_conflict += len(sexes) > 1
        band = _band(min(r[3] for r in rs))  # age at first recording
        n_groups, n_recs = len({r[0] for r in rs}), len({(r[0], r[1]) for r in rs})
        info[p] = (group, "/".join(sorted(sexes)), band, n_groups == 1 and n_recs == 1, p in in_zip)
    out: dict[str, str] = {}
    for p, (group, sex, band, single, z) in info.items():
        if band == "minor":
            out[p] = "excluded"
        elif group in RESERVE_GROUPS and single:
            strata[("pool", group, sex, band)].append(p)
        elif group == "healthy":
            strata[("healthy", sex, z)].append(p)
        else:
            strata[("eval", group, sex, band)].append(p)
    rng_of = lambda key: random.Random(f"{SEED}:{':'.join(map(str, key))}")  # noqa: E731
    # reserve: one third of the single-recording paresis / functional persons; the rest join their eval stratum
    for key in sorted(k for k in strata if k[0] == "pool"):
        cut = _cut(sorted(strata[key]), (("reserve", RESERVE_FRAC), ("eval", 1 - RESERVE_FRAC)), rng_of(key))
        for p, t in cut.items():
            if t == "reserve":
                out[p] = "reserve"
            else:
                strata[("eval",) + key[1:]].append(p)
    for key in sorted(k for k in strata if k[0] == "eval"):
        out.update(_cut(sorted(strata[key]), DISORDER_SPLIT, rng_of(key)))
    # healthy: dev 50 per sex from in-zip adults, systematic over (age band, random order); the rest semi / lockbox
    for sex in sorted({k[1] for k in strata if k[0] == "healthy"}):
        pool = sorted(strata[("healthy", sex, True)])
        rng = rng_of(("healthy-dev", sex))
        keyed = sorted(pool, key=lambda p: (info[p][2], rng.random()))
        step, off = len(keyed) / HEALTHY_DEV, rng.random()
        dev = {keyed[int((i + off) * step)] for i in range(HEALTHY_DEV)}
        out.update({p: "dev" for p in dev})
        rest = defaultdict(list)
        for z in (True, False):
            for p in strata.get(("healthy", sex, z), []):
                if p not in dev:
                    rest[(sex, info[p][2], z)].append(p)
        for key in sorted(rest):
            out.update(_cut(sorted(rest[key]), (("semi", 0.5), ("lockbox", 0.5)), rng_of(("healthy-rest",) + key)))
    assert set(out) == set(info)

    t = json.loads(TIER_FILE.read_text(encoding="utf-8"))
    if any(s.startswith(CORPUS + ":") for s in t["speakers"]):
        sys.exit("saarbruecken already allocated in tiers.json; refusing to redraw")
    t["speakers"].update({f"{CORPUS}:{p}": tier for p, tier in out.items()})
    t["speakers"] = dict(sorted(t["speakers"].items()))
    t["counts"][CORPUS] = dict(sorted(Counter(out.values()).items()))
    t["counts"] = dict(sorted(t["counts"].items()))
    TIER_FILE.write_text(json.dumps(t, indent=1) + "\n", encoding="utf-8")

    print(f"D-095 allocation, seed {SEED}: {len(out)} persons; sex conflicts across recordings: {sex_conflict}")
    tab = Counter((info[p][0], info[p][1], tier) for p, tier in out.items())
    for g in order:
        for sex in ("w", "m"):
            row = {tier: tab[(g, sex, tier)] for tier in ("dev", "semi", "lockbox", "reserve", "excluded")}
            print(f"  {g:15s} {sex}: " + "  ".join(f"{k} {v}" for k, v in row.items()))
    zc = Counter((info[p][0] == "healthy", p in in_zip, tier) for p, tier in out.items() if info[p][0] == "healthy")
    print("  healthy with a recording in the zip: " + "  ".join(f"{k} {zc[(True, True, k)]}" for k in ("dev", "semi", "lockbox", "excluded"))
          + " | without: " + "  ".join(f"{k} {zc[(True, False, k)]}" for k in ("dev", "semi", "lockbox", "excluded")))
    print(f"written {TIER_FILE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
