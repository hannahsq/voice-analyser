r"""D-063: Common Voice 27.0 Korean (validated) as an MFA corpus: <speaker index>/<clip>.wav (16 kHz) + .lab. Speakers
become numeric indices in order of first appearance; `speakers.tsv` keeps index, age band and gender only (no client
ids). Args: CV ko dir, out corpus dir. Then:
    .conda\run-mfa.cmd align <out> <minimal-pairs>\data\ko\korean_mfa_patched.dict korean_mfa <out>_aligned --num_jobs 8
"""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def main():
    cv, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    idx, meta = {}, []
    with open(cv / "validated.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        if r["client_id"] not in idx:
            idx[r["client_id"]] = len(idx)
            meta.append((len(meta), r["age"] or "", r["gender"] or ""))
        s = f"{idx[r['client_id']]:03d}"
        d = out / s
        d.mkdir(exist_ok=True)
        stem = Path(r["path"]).stem
        if not (d / f"{stem}.wav").exists():
            x, sr = sf.read(cv / "clips" / r["path"])
            x = x if x.ndim == 1 else x.mean(1)
            sf.write(d / f"{stem}.wav", resample_poly(x, 16000, sr).astype(np.float32), 16000, subtype="PCM_16")
        (d / f"{stem}.lab").write_text(r["sentence"].strip() + "\n", encoding="utf-8")
    with open(out.parent / f"{out.name}_speakers.tsv", "w", encoding="utf-8") as fh:
        fh.write("speaker\tage\tgender\n")
        for i, a, g in meta:
            fh.write(f"{i:03d}\t{a}\t{g}\n")
    print(f"{len(rows)} clips, {len(idx)} speakers")


if __name__ == "__main__":
    main()
