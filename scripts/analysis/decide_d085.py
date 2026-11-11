"""D-085 decision (registered 2026-11-08; amendments 1-3): the screening run, base (`d085_base_b20k`) and heads
(`d085_heads_b20k`) arms, one seed each, against each other and against v3.1 (5b's decay branches, `d040_5b_s{1,2}_b20k`,
seed mean). Nothing is adopted here: each question's measures are *called* when a difference clears the registered
threshold (about twice the known seed-to-seed spread), else *reported*:
  - Q1 glottal slope of the closed F1 on aspiration-free coupled validation (D-082's val_v2kn, D-081's measure):
    >= 0.03;
  - synthetic F1-F4 MAE: >= 8 % on any formant or >= 5 % on their mean (at 4 s crops and at 1600 frames, each);
  - Q3 women's PTDB-TUG dev closed-phase F1 |error|: >= 4 Hz;
  - Q2 aVTL MAE on synthetic validation: >= 10 % (at 400 and 1600 frames, each); Hillenbrand length reliability: >= 0.07.
Comparisons: heads - base (Q1-Q3, the heads bundle); base - 5b (Q4, the wider trunk with long context and the coupled
share); heads - 5b reported. Guards (Q3; amendment 3, before any result): D-040's margins (Hillenbrand dev F2 <= +9,
F3 <= +20 Hz; f0 RPA >= -0.1 point), resynthesised validation F1-F3 MAE <= +8 % each, D-051's length guard (w/m in
[0.85, 0.92]; f0 probe excess <= 10 c/oct). Reported: the uncoupled offset between the acoustic and closed heads, the
acoustic head's slope, KL's gap growth (D-074), probe 10, probe 11, the iVTL head, the coupled validation set.

    python decide_d085.py <windowed scratch dir> [--check]

Runs the scoring steps the chain started before this script existed did not (d085_score.py, probe 10 for both arms,
KL's gap growth), each skipped if its output exists. `--check` reads D-090's arms (`d090_f5_s1` as "base",
`d090_f57_s1` as "heads") and CPU scores from scratchpad/d085_check (16 clips), to check the script reads its inputs;
not a D-085 result.

D-098 (the rerun with D-036's fixed length weights) uses this script unchanged with VA_D085_TAG=d098: runs
`d098_{base,heads}_b20k`, scores in cache/d098, reports in reports/analysis/d098, the same rule."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
import decide_d056 as D  # noqa: E402

CHECK = "--check" in sys.argv
TAG = os.environ.get("VA_D085_TAG", "d085")  # D-098: "d098"
O = Path(f"reports/analysis/{TAG}")
REF = ("d040_5b_s1_b20k", "d040_5b_s2_b20k")
BASE, HEADS = ("d090_f5_s1", "d090_f57_s1") if CHECK else (f"{TAG}_base_b20k", f"{TAG}_heads_b20k")
RUNS = (*REF, BASE, HEADS)
SC = Path("scratchpad/d085_check") if CHECK else Path(f"cache/{TAG}")
KL = O / "kl"
LANDMARKS = "label ceiling 0.29 (D-082), frozen v3.1 0.36, full recovery 0.89 (the closed label's own slope)"
T = {"slope": 0.03, "mae_one": 0.08, "mae_mean": 0.05, "cp": 4.0, "avtl": 0.10, "rel": 0.07}


def ensure_scores():
    """The scoring steps added to the chain after it started (each skips if its output exists)."""
    py, log = sys.executable, open(f"runs/{TAG}_post.log", "a")
    env = {**os.environ, "MIOPEN_FIND_MODE": "FAST", "VA_DEVICE": os.environ.get("VA_DEVICE", "cuda")}
    run = lambda *a: subprocess.run([py, "-u", *a], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,  # noqa: E731
                                    env=env, check=True)
    if not all((SC / f"long_{r}.json").exists() and (SC / f"v2kn_{r}.npz").exists() for r in RUNS):
        run("scripts/analysis/d085_score.py", *[f"{r}=runs/{r}/ckpt_last.pt" for r in RUNS], "--out", str(SC))
    for r in (BASE, HEADS):
        if not Path(f"reports/probes/articulation/{r}.json").exists():
            run("-m", "vocal_analysis.eval.probes.run", "score", "articulation", f"runs/{r}/ckpt_last.pt")
    if not (KL / "main_per_file.tsv").exists():
        run("scripts/analysis/d060_kl_gap.py", "main", str(KL), *[f"{r}=runs/{r}/ckpt_last.pt" for r in RUNS])


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --- Q1: the glottal slope ---------------------------------------------------------------------------------------
class Glottal:
    """D-081's glottal slope on D-082's aspiration-free val_v2kn, frames as D-082's `load` (sonorant, finite opening,
    df1, acoustic F1-F4 and closed F1): gap = acoustic - closed F1 (labels); its vowel part fitted from the acoustic
    F1-F4 on even clips; on odd clips the slope of (acoustic F1 - predicted F1) on the gap's rest. Clip bootstrap.
    Checked on the labels (2026-11-11, every frame): closed 0.882 (D-082: 0.887 on its model-length frames), acoustic 0.000."""

    def __init__(self, rows):
        from d081_label_oracle import Slope

        r = np.load("cache/d082/val_render.npz")
        f_off = r["f_off"]
        clip = np.searchsorted(f_off, rows, side="right") - 1
        lab = {k: r[k][rows] for k in ("formants", "formants_acoustic", "glottal_opening", "coupling_df", "sonorant")}
        self.m = (lab["sonorant"].astype(bool) & np.isfinite(lab["glottal_opening"]) & np.isfinite(lab["coupling_df"][:, 0])
                  & np.all(np.isfinite(lab["formants_acoustic"][:, :4]), 1) & np.isfinite(lab["formants"][:, 0]))
        v = {"formants": lab["formants"][self.m], "formants_acoustic": lab["formants_acoustic"][self.m], "clip": clip[self.m]}
        self.S = Slope(v)
        self.a, self.rest, self.clip = self.S.a[self.S.use], self.S.rest, v["clip"][self.S.use]
        self.closed_label = self.S(v["formants"][:, 0])
        crops = np.unique(self.clip)
        self.by = [np.flatnonzero(self.clip == c) for c in crops]
        rng = np.random.default_rng(0)
        self.boot = [np.concatenate([self.by[i] for i in rng.integers(0, len(crops), len(crops))]) for _ in range(1000)]

    def _slope(self, p, idx):
        y, x = (self.a - p)[idx], self.rest[idx]
        k = np.isfinite(y)
        x, y = x[k] - x[k].mean(), y[k] - y[k].mean()
        return float(x @ y / (x @ x))

    def pred(self, f1):
        return f1[self.m][self.S.use]

    def slope(self, f1):
        return self._slope(self.pred(f1), np.arange(len(self.a)))

    def diff(self, f1_new, f1_refs):
        """new - mean(refs), point and 95 % clip-bootstrap interval"""
        pn, pr = self.pred(f1_new), [self.pred(f) for f in f1_refs]
        d = lambda idx: self._slope(pn, idx) - np.mean([self._slope(p, idx) for p in pr])  # noqa: E731
        b = [d(i) for i in self.boot]
        return d(np.arange(len(self.a))), np.percentile(b, [2.5, 97.5])


# --- per-run measures --------------------------------------------------------------------------------------------
def log_eval(r, key="eval"):
    try:
        return D.last(r, key)
    except (IndexError, FileNotFoundError):
        return None


def kl_growth():
    """per model and sex: median over (vowel, f0) of F1 error at gap 0.3 minus at gap 0 (Hz)"""
    p = KL / "main_per_file.tsv"
    if not p.exists():
        return None
    lines = [ln.split("\t") for ln in p.read_text().splitlines()]
    head, rows = lines[0], lines[1:]
    out = {}
    for j, name in enumerate(head[4:], 4):
        for sex in ("Male", "Female"):
            v = {(r[2], r[3], r[1]): float(r[j]) for r in rows if r[0] == sex}
            d = [v[k[0], k[1], "0.3"] - v[k[0], k[1], "0.0"] for k in v if k[2] == "0.0" and (k[0], k[1], "0.3") in v]
            out[name, sex] = float(np.median(d))
    return out


def probe10(r):
    a = jload(f"reports/probes/articulation/{r}.json")
    return None if a is None else {k: a[f"{k}/all/VTL"]["point"] for k in ("larynx", "lips")}


def rel(new, ref):
    return np.asarray(new, float) / np.asarray(ref, float) - 1


def mae_call(rc):
    """rc: relative change per formant -> called?"""
    return bool(np.any(np.abs(rc) >= T["mae_one"]) or abs(np.mean(rc)) >= T["mae_mean"])


def word(called, better):
    return ("CALLED, " + ("better" if better else "worse")) if called else "reported (below threshold)"


def main():
    S = Path([a for a in sys.argv[1:] if a != "--check"][0])
    if not CHECK:
        ensure_scores()
    L = [f"{TAG.upper().replace('D', 'D-')}: the screening run; base and heads arms (one seed each) against each other and against v3.1 (5b's decay "
         f"branches, seed mean)" + ("  [CHECK MODE: D-090's f5 / f57 seed-1 arms as base / heads, 16-clip CPU scores; "
                                    "not a D-085 result]" if CHECK else ""), ""]

    # inputs
    ev = {r: log_eval(r) for r in RUNS}
    lg = {r: jload(SC / f"long_{r}.json") for r in RUNS}
    V = {r: np.load(SC / f"v2kn_{r}.npz") for r in RUNS}
    rows = V[RUNS[0]]["rows"]
    for r in RUNS:
        if not np.array_equal(V[r]["rows"], rows):
            raise SystemExit(f"v2kn rows differ for {r}")
    G = Glottal(rows)
    d = dict(np.load(S / "ptdb_egg_sub.npz", allow_pickle=True))
    dev = np.array([D.tiers.get(f"ptdb_tug:{s}") == "dev" for s in d["speaker"]])
    women = dev & (d["sex"] == "f") & np.isfinite(d["cp"][:, 0])
    vj = jload(O / "vtl_joint.json") or {}
    if CHECK:  # D-090's per-arm files: seed 1 against 5b s1 stands in for each pair
        a, b = jload("reports/analysis/d090/vtl_joint_f5.json"), jload("reports/analysis/d090/vtl_joint_f57.json")
        vj = {"base-5b": a["s1"], "heads-5b": b["s1"], "heads-base": b["s1"]}
    klg = kl_growth()

    def cp_bias(r):
        m = women & np.isfinite(d[r][:, 0])
        return float(np.median(d[r][m, 0] - d["cp"][m, 0]))

    def cp_abs(r, ref):
        m = women & np.isfinite(d[r][:, 0]) & np.isfinite(d[ref][:, 0])
        return float(np.median(np.abs(d[r][m, 0] - d["cp"][m, 0])) - np.median(np.abs(d[ref][m, 0] - d["cp"][m, 0])))

    syn = lambda e: np.array([e[f"F{n}_mae_hz"] for n in range(1, 5)])  # noqa: E731

    # per-run table
    L.append("per run: synthetic F1-F4 MAE Hz at 400 frames (log) | at 1600 / 400 frames on train_long_v3u's held-out clips | "
             "RPA % | aVTL MAE % (400 log; long 1600 / 400) | iVTL MAE % | resynth F1-F3 MAE | glottal slope (closed; acoustic) | "
             "women's cp F1 bias Hz")
    slopes = {}
    for r in RUNS:
        e, l = ev[r], lg[r]
        slopes[r] = G.slope(V[r]["f1"])
        sa = f"; {G.slope(V[r]['f1_acoustic']):+.3f}" if len(V[r]["f1_acoustic"]) else ""
        L.append(f"  {r:18s} {np.round(syn(e), 1)} | {np.round(syn(l['long1600']), 1)} / {np.round(syn(l['long400']), 1)} | "
                 f"{100 * e['f0_rpa']:.2f} | {e['vtl_mae_pct']:.2f} ({l['long1600']['vtl_mae_pct']:.2f} / "
                 f"{l['long400']['vtl_mae_pct']:.2f}) | {e.get('vtl_inst_mae_pct', float('nan')):.2f} | "
                 f"{np.round(D.resynth(r), 1)} | {slopes[r]:+.3f}{sa} | {cp_bias(r):+.1f}")
    L.append(f"  glottal slope landmarks: {LANDMARKS}; this script's closed label {G.closed_label:+.3f} on "
             f"{len(G.a)} odd-clip frames")
    L.append(f"  5b's seeds differ by (s2 - s1): synthetic F1-F4 {np.round(100 * rel(syn(ev[REF[1]]), syn(ev[REF[0]])), 1)} %, "
             f"glottal slope {slopes[REF[1]] - slopes[REF[0]]:+.3f}, aVTL {100 * (ev[REF[1]]['vtl_mae_pct'] / ev[REF[0]]['vtl_mae_pct'] - 1):+.1f} %, "
             f"women's cp |error| {cp_abs(REF[1], REF[0]):+.1f} Hz")
    L.append("")

    # comparisons
    def compare(new, refs, key):
        """new against the mean of refs; returns the called measures and the lines"""
        out, calls, fails = [], [], []
        mean_ref = lambda f: np.mean([f(x) for x in refs], 0)  # noqa: E731
        # Q1
        sd, (lo, hi) = G.diff(V[new]["f1"], [V[x]["f1"] for x in refs])
        c = abs(sd) >= T["slope"]
        out.append(f"  Q1 glottal slope {sd:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {word(c, sd > 0)}")
        calls += [f"Q1 glottal slope {sd:+.3f}"] if c else []
        # synthetic MAE at both lengths
        for tag, get in (("400 frames", lambda x: syn(ev[x])), ("1600 frames", lambda x: syn(lg[x]["long1600"])),
                         ("long clips at 400", lambda x: syn(lg[x]["long400"]))):
            rc = rel(get(new), mean_ref(get))
            c = mae_call(rc)
            big = rc[np.abs(rc) >= T["mae_one"]]  # called by single formants only: their direction, "mixed" if they differ
            w = word(c, rc.mean() < 0)
            if c and abs(rc.mean()) < T["mae_mean"] and len(big) and not (np.all(big < 0) or np.all(big > 0)):
                w = "CALLED, mixed"
            elif c and abs(rc.mean()) < T["mae_mean"]:
                w = word(True, bool(big[0] < 0))
            out.append(f"  synthetic F1-F4 MAE, {tag}: {', '.join(f'{100 * v:+.1f}' for v in rc)} % (mean {100 * rc.mean():+.1f} %)"
                       f" -> {w}")
            calls += [f"synthetic MAE at {tag} {100 * rc.mean():+.1f} %"] if c else []
        # Q2
        for tag, get in (("400 frames", lambda x: ev[x]["vtl_mae_pct"]), ("1600 frames", lambda x: lg[x]["long1600"]["vtl_mae_pct"]),
                         ("long clips at 400", lambda x: lg[x]["long400"]["vtl_mae_pct"])):
            rc = float(rel(get(new), mean_ref(get)))
            c = abs(rc) >= T["avtl"]
            out.append(f"  Q2 aVTL MAE, {tag}: {100 * rc:+.1f} % -> {word(c, rc < 0)}")
            calls += [f"Q2 aVTL MAE at {tag} {100 * rc:+.1f} %"] if c else []
        p = vj.get(key)
        if p is not None:
            c = abs(p["reliability_diff"]) >= T["rel"]
            ci = p["reliability_ci"]
            out.append(f"  Q2 Hillenbrand length reliability {p['reliability_diff']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] -> "
                       f"{word(c, p['reliability_diff'] > 0)}")
            calls += [f"Q2 length reliability {p['reliability_diff']:+.3f}"] if c else []
        else:
            out.append(f"  Q2 Hillenbrand length reliability: no '{key}' in vtl_joint.json")
        p10n, p10r = probe10(new), [probe10(x) for x in refs]
        if p10n and all(p10r):
            out.append("  Q2 probe 10, the aVTL head's slope on the instantaneous length's change (ideal 0): "
                       + ", ".join(f"{k} {p10n[k]:+.3f} (ref {np.mean([q[k] for q in p10r]):+.3f})" for k in p10n))
        # Q3
        cp = float(np.mean([cp_abs(new, x) for x in refs]))
        c = abs(cp) >= T["cp"]
        out.append(f"  Q3 women's closed-phase F1 |error| {cp:+.1f} Hz (bias {cp_bias(new):+.1f}, ref "
                   f"{np.mean([cp_bias(x) for x in refs]):+.1f}) -> {word(c, cp < 0)}")
        calls += [f"Q3 women's cp |error| {cp:+.1f} Hz"] if c else []
        hb = [D.paired(x, new) for x in refs if (D.P / f"{x}_vs_{new}.txt").exists()]
        if hb:
            h = np.mean(hb, 0)
            out.append(f"  Q3 Hillenbrand dev F1 / F2 / F3 MAE {', '.join(f'{v:+.1f}' for v in h)} Hz (paired, reported)")
            fails += [f"Hillenbrand F2 {h[1]:+.1f} Hz"] if h[1] > 9 else []
            fails += [f"Hillenbrand F3 {h[2]:+.1f} Hz"] if h[2] > 20 else []
        else:
            out.append("  Q3 Hillenbrand: no paired file")
        # guards
        rpa = 100 * (ev[new]["f0_rpa"] - np.mean([ev[x]["f0_rpa"] for x in refs]))
        fails += [f"RPA {rpa:+.2f} point"] if rpa < -0.1 else []
        rs = rel(D.resynth(new), mean_ref(D.resynth))
        fails += [f"resynth F{n + 1} {100 * v:+.1f} %" for n, v in enumerate(rs) if v > 0.08]
        lg_ok = None
        if p is not None:
            lg_ok = 0.85 <= p["w/m"] <= 0.92 and p["probe_excess"] <= 10
            fails += [] if lg_ok else [f"length guard (w/m {p['w/m']:.3f}, probe excess {p['probe_excess']:+.1f})"]
        out.append(f"  guards: RPA {rpa:+.2f} point; resynth F1-F3 {', '.join(f'{100 * v:+.1f}' for v in rs)} %; "
                   + (f"w/m {p['w/m']:.3f}, f0 probe excess {p['probe_excess']:+.1f} c/oct; " if p is not None else "")
                   + ("held" if not fails else "FAILED: " + "; ".join(fails)))
        return out, calls, fails

    blocks = {}
    for title, new, refs, key in (("heads - base (Q1-Q3: the heads bundle)", HEADS, (BASE,), "heads-base"),
                                  ("base - 5b (Q4: the wider trunk, long context, coupled share)", BASE, REF, "base-5b"),
                                  ("heads - 5b (reported)", HEADS, REF, "heads-5b")):
        out, calls, fails = compare(new, refs, key)
        L += [title] + out + [""]
        blocks[key] = (calls, fails)

    # reported only
    L.append("reported:")
    eh = ev[HEADS]
    if "Fa1_shift_err_hz" in eh:
        L.append("  heads arm, acoustic - closed heads on uncoupled validation (train_v2, labels equal): "
                 + ", ".join(f"F{n} {eh[f'Fa{n}_shift_err_hz']:+.1f}" for n in range(1, 5)) + " Hz")
        L.append(f"  heads arm, acoustic set MAE at 400 frames: {', '.join(f'{eh[f'Fa{n}_mae_hz']:.1f}' for n in range(1, 5))} Hz; "
                 f"coupled validation (train_v2ku's held-out), acoustic - closed against the labels' shift: "
                 + (", ".join(f"F{n} {log_eval(HEADS, 'eval_extra:train_v2ku')[f'Fa{n}_shift_err_hz']:+.1f}" for n in range(1, 5))
                    if log_eval(HEADS, "eval_extra:train_v2ku") and "Fa1_shift_err_hz" in log_eval(HEADS, "eval_extra:train_v2ku")
                    else "n/a") + " Hz")
    if klg:
        L.append("  KL gap growth (F1 error at gap 0.3 - 0, median over vowel x f0, Hz), men / women: "
                 + "; ".join(f"{r} {klg[r, 'Male']:+.1f} / {klg[r, 'Female']:+.1f}" for r in RUNS if (r, "Male") in klg))
    else:
        L.append("  KL gap growth: not scored")
    for r in RUNS:
        try:
            sc = D.scale(r)
            L.append(f"  probe 11 F1 slope {r}: Hillenbrand {sc['hillenbrand']:.3f}, PTDB-TUG {sc['ptdb_tug']:.3f}")
        except FileNotFoundError:
            L.append(f"  probe 11 {r}: not scored")
    L.append("  one seed per arm: a called difference is a signal to break down later, not a confirmed effect")
    L.append("")

    def summary(key):
        calls, fails = blocks[key]
        return (", ".join(calls) if calls else "nothing called") + (f"; GUARD FAILED: {'; '.join(fails)}" if fails else "; guards held")

    L.append(f"DECISION: heads - base: {summary('heads-base')} | base - 5b: {summary('base-5b')}")
    text = "\n".join(L) + "\n"
    print(text, end="")


if __name__ == "__main__":
    main()
