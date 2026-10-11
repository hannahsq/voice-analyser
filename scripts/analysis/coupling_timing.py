"""Time the coupled synthesis path (D-023) against the uncoupled one on the same clips, on a quiet machine, and
profile where the time goes. Args: number of clips (default 40). Run from this worktree with PYTHONPATH=.
Writes a summary to stdout: CPU seconds per second of audio, and the top functions by cumulative time."""
import cProfile
import pstats
import sys
import time
from dataclasses import replace

import numpy as np

from vocal_analysis.synth.sampler import SynthConfig, generate

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
base = replace(SynthConfig(), p_tube=1.0)  # every clip a tube clip: coupling only applies there
for name, cfg in (("uncoupled", replace(base, p_coupling=0.0)), ("coupled", replace(base, p_coupling=1.0))):
    generate(0, cfg, seed=99)  # warm-up (imports, caches)
    prof = cProfile.Profile()
    t, audio_s = time.process_time(), 0.0
    prof.enable()
    for i in range(N):
        clip = generate(i, cfg, seed=7)
        audio_s += len(clip.audio) / 16000
    prof.disable()
    dt = time.process_time() - t
    print(f"\n{name}: {N} clips, {audio_s:.0f} s of audio, {dt:.1f} CPU s: {dt / audio_s:.3f} CPU s per s of audio", flush=True)
    pstats.Stats(prof).sort_stats("cumulative").print_stats(r"vocal_analysis", 14)
