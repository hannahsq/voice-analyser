"""D-073: learned formant queries as the read-out on teacher v3.1's frozen trunk (D-054's recipe and trunk exactly).

    python scripts/analysis/d073_queries.py <arm> <seed> [steps]   (arm: queries, queries_abs; VA_DEVICE=cuda)
"""

import sys
from dataclasses import replace

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

if __name__ == "__main__":
    import d054_readout as D

    from vocal_analysis.train.teacher import train

    arm, seed = sys.argv[1], int(sys.argv[2])
    assert arm in ("queries", "queries_abs"), arm
    cfg, mcfg = D.configs(arm, seed, int(sys.argv[3]) if len(sys.argv) > 3 else D.STEPS)
    train(replace(cfg, out=f"runs/d073_{arm}_s{seed}"), mcfg)
