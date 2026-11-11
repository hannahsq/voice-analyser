"""Parameter counts and feature-map shapes for every architecture in docs/architectures.md (CPU, a few seconds).
Run from the repository root: python scripts/analysis/arch_params.py"""
import sys
from dataclasses import replace

if __name__ == "__main__":
    sys.path.insert(0, "scripts/parakeet")
    import torch
    from vocal_analysis.features import vqt
    from vocal_analysis.models.teacher import Teacher, TeacherConfig, n_params, normalise

    full = TeacherConfig()
    s2 = dict(channels=(8, 16, 32), f0_channels=8, embed=192, gru_hidden=96, f0_gru_hidden=64)
    wide = dict(temporal="conformer", embed=576, conformer_dim=192, vtl_head=True, vtl_inst_head=True)
    cfgs = {
        "full GRU (v1, v2)": full,
        "full Conformer (v3)": replace(full, temporal="conformer"),
        "v3.1 (Conformer + VTL)": replace(full, temporal="conformer", vtl_head=True),
        "S2 GRU": replace(full, **s2),
        "S2 Conformer": replace(full, temporal="conformer", **s2),
        "D-085 base": replace(full, **wide),
        "D-085 heads": replace(full, **wide, n_upper=3, acoustic_heads=True, vtl_link=True),
    }
    for name, c in cfgs.items():
        m = Teacher(c)
        enc = m.enc
        conv, proj = n_params(enc.convs), n_params(enc.proj)
        print(f"{name:24s} total {n_params(m) / 1e6:6.3f} M | f0_net {n_params(m.f0_net) / 1e3:7.1f}k "
              f"(convs {n_params(m.f0_net.convs) / 1e3:.1f}k, flatten proj {n_params(m.f0_net.proj) / 1e3:.1f}k) | "
              f"enc convs {conv / 1e3:7.1f}k, flatten proj {proj / 1e3:7.1f}k, temporal {(n_params(enc) - conv - proj) / 1e3:7.1f}k "
              f"| h width {enc.out_dim}")
        print("    heads (k):", {k: round(n_params(v) / 1e3, 1) for k, v in m.named_children() if k not in ("f0_net", "enc")})
        x = torch.cat([normalise(torch.randn(1, vqt.N_BINS, 20)), torch.zeros(1, 1, vqt.N_BINS, 20)], 1)
        with torch.no_grad():
            _, kept = enc(x if c.comb else x[:, :2], maps=True)
        print("    conv maps (channels, bins):", [tuple(k.shape[1:3]) for k in kept], "| flatten in", enc.proj.in_features)
    v31 = replace(full, temporal="conformer", vtl_head=True)
    for ro in ("mlp", "film", "film_residual", "queries", "queries_abs"):
        print(f"v3.1 + readout {ro:14s} total {n_params(Teacher(replace(v31, readout=ro))) / 1e6:.3f} M")
    for side in ("retrieve", "factorised"):
        m = Teacher(replace(v31, proj_side=side))
        print(f"v3.1 + proj_side {side:12s} total {n_params(m) / 1e6:.3f} M (side path {n_params(m.enc.side) / 1e3:.1f}k)")
    for k in ("obs_head", "opening_head", "fed_standardised", "vtl_inst_head"):
        print(f"v3.1 + {k:18s} total {n_params(Teacher(replace(v31, **{k: True}))) / 1e6:.3f} M")
    print(f"v3.1 + n_upper=3          total {n_params(Teacher(replace(v31, n_upper=3))) / 1e6:.3f} M")
    from student import SIZE, Student
    st = Student(replace(TeacherConfig(), temporal="conformer", **SIZE))
    t = n_params(st.enc) - n_params(st.enc.convs) - n_params(st.enc.proj)
    print(f"student (D-034) total {n_params(st) / 1e6:.3f} M; convs {n_params(st.enc.convs) / 1e3:.1f}k, flatten proj "
          f"{n_params(st.enc.proj) / 1e3:.1f}k, temporal {t / 1e6:.3f} M, output proj {n_params(st.proj) / 1e3:.1f}k")
    from head import ParakeetHead
    print(f"Parakeet layer-5 head (D-033) trainable {n_params(ParakeetHead()) / 1e6:.3f} M")
