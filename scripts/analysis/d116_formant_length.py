"""D-116: does the reconstruction arms' loss of men's between-speaker length variance (D-115) appear in the arms' own
formants? Apparent length A from each model's F1-F4 at Hillenbrand dev's reference frames, calibrated as
`vtl_score_joint.py` calibrates labeller v3's A (tube crops' true F1-F4, length not from f0), per file the median over
frames; then D-115's decomposition (`d115_reliability.py report` on these caches). Descriptive, dev only.

    score:  python scripts/analysis/d116_formant_length.py name=ckpt ...      (VA_DEVICE=cuda; caches in cache/d116/)
    report: D115_CACHE=cache/d116 D115_AGG=median python scripts/analysis/d115_reliability.py report
              > reports/analysis/d116/d116.txt
Precondition printed first: labeller v3's A through the same code reproduces vtl_score_joint.py's A (reliability 0.153).
"""
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("cache/d116")


def a_of():
    from vocal_analysis.backend.vtl import apparent_vtl

    W = Path("cache/heads_l3_6")
    v = np.load(W / "vtl.npz")
    son = np.load(W / "sonorant.npy", mmap_mode="r")
    fm = np.load(W / "formants.npy", mmap_mode="r")
    a4, yc = [], []
    for i in np.flatnonzero(v["tube"] & ~v["from_f0"]):
        s = np.asarray(son[i]) > 0.5
        if s.sum() >= 3:
            a4.append(np.nanmedian(apparent_vtl(np.asarray(fm[i])[s])))
            yc.append(np.log2(v["length_cm"][i]))
    cal = np.polyfit(np.log2(a4), yc, 1)
    return lambda F: np.polyval(cal, np.log2(apparent_vtl(F)))


if __name__ == "__main__":
    import torch

    import d115_reliability as D
    from vocal_analysis.device import setup_device
    from vocal_analysis.models.infer import analyse, load_teacher

    pairs = dict(a.split("=", 1) for a in sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(D.CACHE / "files.npz", OUT / "files.npz")
    A = a_of()
    # precondition: labeller v3's A (D-033's cache), as vtl_score_joint.py
    HB = Path("cache/emb_v1/gold_hillenbrand")
    ro = np.load(HB / "ref_off.npy")
    lf = (np.load(HB / "m_tv3_s1_log2F.npy") + np.load(HB / "m_tv3_s2_log2F.npy")) / 2
    a = A(2 ** lf)
    f = np.load(OUT / "files.npz")
    assert len(ro) - 1 == len(f["spk"])
    eA = np.array([np.nanmedian(a[ro[j]:ro[j + 1]]) for j in range(len(ro) - 1)])
    spks = {G: sorted(set(f["spk"][f["group"] == G])) for G in "mw"}
    rel = np.mean([D.parts(eA, f["spk"], f["vowel"], spks[G])["rel"] for G in "mw"])
    print(f"precondition: labeller v3's A reliability {rel:.3f} (vtl_score_joint.py: 0.153)")
    hb = D.hillenbrand_dev()
    dev = setup_device()
    for name, ck in pairs.items():
        out = OUT / f"{name}.npz"
        if out.exists():
            print(f"{name}: cached")
            continue
        model, _ = load_teacher(ck, dev)
        mu, off = [], [0]
        for x, t, *_ in hb:
            r = analyse(model, x, dev)
            idx = np.clip(np.round(t / 0.01).astype(int), 0, len(r["f0"]) - 1)
            mu.append(A(r["formants"][idx][:, :4]))
            off.append(off[-1] + len(idx))
        mu = np.concatenate(mu)
        np.savez(out, mu=mu, sigma=np.ones_like(mu), off=np.array(off))
        print(f"{name}: {len(hb)} files")
        del model
        torch.cuda.empty_cache()
