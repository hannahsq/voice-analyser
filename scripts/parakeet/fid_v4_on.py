"""D-034, reported only: student v4's held-out fidelity measure (distil_train_v4.py's `fidelity`, run unchanged from that
file) on other students' checkpoints, e.g. v2's, for an apples-to-apples per-phase comparison. A student trained against
another cache's statistics has its output destandardised with those and restandardised with v4's.
Args: v4 cache dir, student's own cache dir (stats.npz), out .jsonl, checkpoints... Run from the main checkout."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from student import load_student  # noqa: E402

C4, CS, OUTF, CKS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
src = (HERE / "distil_train_v4.py").read_text(encoding="utf-8")
g = {"__name__": "fid_v4", "__file__": str(HERE / "distil_train_v4.py")}
sys.argv = [sys.argv[0], C4, tempfile.mkdtemp()]
exec(compile(src[:src.index("torch.manual_seed(SEED)")], "distil_train_v4.py", "exec"), g)  # data loading and fidelity()
dev, MU4, SD4 = g["dev"], g["MU"], g["SD"]
s = np.load(CS / "stats.npz")
MU, SD = (torch.tensor(s[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))


class Restd(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return (self.m(x).float() * SD + MU - MU4) / SD4


done = {json.loads(l)["checkpoint"] for l in OUTF.read_text().splitlines()} if OUTF.exists() else set()
for ck in CKS:
    if ck in done:
        continue
    f = g["fidelity"](Restd(load_student(ck, dev)))
    with open(OUTF, "a") as fo:
        fo.write(json.dumps({"checkpoint": ck, "step": torch.load(ck, map_location="cpu", weights_only=False).get("step"),
                             "fidelity": f}) + "\n")
    print(Path(ck).name, ", ".join(f"{n} {v['r2']:.3f}" for n, v in f.items()), flush=True)
