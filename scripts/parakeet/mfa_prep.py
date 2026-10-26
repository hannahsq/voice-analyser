"""D-034 next student: MFA corpora for isolated-word items (English; training splits, plus the held-out word set's
speakers). Each corpus dir holds <speaker>/<utt>.wav (hard links to the originals) and <utt>.lab (the transcript).
  libritts_train: LibriTTS-R train-clean-100, every utterance of 2-12 s (words with pauses on both sides are rare)
  libritts_train360: LibriTTS-R train-clean-360, every utterance of 2-12 s
  so762_train:    speechocean762 train, every utterance (children and adults)
  libritts_dev:   LibriTTS-R dev-clean, every utterance of 2-12 s (held-out words)
  so762_test:     speechocean762 test, every utterance (held-out words)
Speakers stay as the corpora's numeric ids. Args: datasets root, out root. Then, per corpus:
  .conda\\run-mfa.cmd align <out>/<corpus> english_us_mfa english_mfa <out>/<corpus>_aligned --num_jobs 8"""
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

DS, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def put(corpus, spk, utt, wav, text):
    d = OUT / corpus / spk
    d.mkdir(parents=True, exist_ok=True)
    if not (d / f"{utt}.wav").exists():
        os.link(wav, d / f"{utt}.wav")
    (d / f"{utt}.lab").write_text(text.strip() + "\n", encoding="utf-8")


def libritts(split, corpus, per_spk, seed):
    root = DS / "libritts_r" / "LibriTTS_R" / split
    rng = np.random.default_rng(seed)
    n = 0
    for spk in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        wavs = sorted(spk.glob("*/*.wav"))
        ok = [w for w in wavs if 2.0 <= sf.info(str(w)).duration <= 12.0]
        for w in [ok[i] for i in sorted(rng.permutation(len(ok))[:per_spk])]:
            lab = w.parent / f"{w.stem}.normalized.txt"
            if not lab.exists():  # LibriTTS-R dropped some utterances' transcripts
                continue
            put(corpus, spk.name, w.stem, w, lab.read_text(encoding="utf-8"))
            n += 1
    print(f"{corpus}: {n} utterances")


def so762(split, corpus):
    root = DS / "speechocean762" / "speechocean762"
    text = dict(t for t in (line.split(maxsplit=1) for line in open(root / split / "text", encoding="utf-8")) if len(t) == 2)
    n = 0
    for line in open(root / split / "utt2spk"):
        u, s = line.split()
        if u not in text:  # a few utterances have no transcript line
            continue
        put(corpus, s, u, root / "WAVE" / f"SPEAKER{int(s):04d}" / f"{u}.WAV", text[u])
        n += 1
    print(f"{corpus}: {n} utterances")


libritts("train-clean-100", "libritts_train", 100000, 4101)
so762("train", "so762_train")
libritts("dev-clean", "libritts_dev", 100000, 4102)
libritts("train-clean-360", "libritts_train360", 100000, 4103)
so762("test", "so762_test")
