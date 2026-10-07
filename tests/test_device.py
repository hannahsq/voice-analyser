"""GPU training smoke test, including the MIOpen RNN workaround (D-002)."""

import pytest

torch = pytest.importorskip("torch")

from vocal_analysis.config import DeviceConfig
from vocal_analysis.device import rnn_guard, setup_device

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("bf16", [False, True])
def test_conv_bigru_train_step_on_gpu(bf16):
    dev = setup_device(DeviceConfig(device="cuda"))
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(2, 8, 3, padding=1).to(dev)
    gru = torch.nn.GRU(8 * 16, 32, batch_first=True, bidirectional=True).to(dev)
    opt = torch.optim.AdamW([*conv.parameters(), *gru.parameters()])
    x = torch.randn(4, 2, 16, 40, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        h = conv(x).permute(0, 3, 1, 2).flatten(2)
        with rnn_guard():
            y, _ = gru(h)
        loss = y.float().pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()
