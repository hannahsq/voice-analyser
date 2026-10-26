"""Parakeet TDT 0.6B v3 (NVIDIA, CC-BY-4.0) from its local safetensors, with transformers' implementation (PyPI), no
hub access: the files were fetched over plain HTTPS at revision 541d1f9 (model.safetensors sha256 3a202636...)."""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

DIR = Path(__file__).resolve().parents[2] / "external_models" / "parakeet-tdt-0.6b-v3"
SHA256 = "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11"
N_LAYERS = 24
SUBSAMPLING = 8  # encoder frames are 80 ms (mel hop 10 ms)


def feature_extractor():
    from transformers import ParakeetFeatureExtractor

    fe = json.loads((DIR / "processor_config.json").read_text())["feature_extractor"]
    fe.pop("feature_extractor_type", None)
    return ParakeetFeatureExtractor(**fe)


def encoder(device):
    from transformers import ParakeetEncoder, ParakeetEncoderConfig

    cfg = json.loads((DIR / "config.json").read_text())["encoder_config"]
    enc = ParakeetEncoder(ParakeetEncoderConfig(**cfg))
    sd = {k[len("encoder."):]: v for k, v in load_file(DIR / "model.safetensors").items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and all("num_batches" in m or "inv_freq" in m for m in missing), (missing, unexpected)
    return enc.eval().to(device)


def tdt(device):
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    model = ParakeetForTDT(ParakeetTDTConfig(**json.loads((DIR / "config.json").read_text())))
    missing, unexpected = model.load_state_dict(load_file(DIR / "model.safetensors"), strict=False)
    assert not unexpected, unexpected
    return model.eval().to(device), missing
