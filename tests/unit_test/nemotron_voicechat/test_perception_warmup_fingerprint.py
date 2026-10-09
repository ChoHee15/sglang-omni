"""CPU warmup qualification rejects model, device, and dtype fingerprint mismatches."""

from dataclasses import replace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.conformer import AudioPerception
from tests.unit_test.nemotron_voicechat.perception_warmup_fakes import (
    run_size,
    spy_encoder,
)


@pytest.mark.parametrize(
    "mismatch", ["model_identity", "resolved_device", "model_dtype"]
)
def test_resource_fingerprint_mismatch_requires_cold_capture_then_requalifies(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: Literal["model_identity", "resolved_device", "model_dtype"],
) -> None:
    forbidden_cuda = Mock(
        side_effect=AssertionError("CPU fingerprint test invoked CUDA")
    )
    for operation in (
        "init",
        "set_device",
        "synchronize",
        "empty_cache",
        "Stream",
        "ExternalStream",
        "CUDAGraph",
        "current_stream",
    ):
        monkeypatch.setattr(torch.cuda, operation, forbidden_cuda)
    initialized_before = torch.cuda.is_initialized()
    encoder, created = spy_encoder(monkeypatch)
    for _ in range(2):
        run_size(encoder, 2)
        encoder.batch_runners[2].close()
        encoder.batch_runners.clear()
    assert [runner.warmup_steps for runner in created] == [3, 1]
    if mismatch == "model_identity":
        replacement_model = Mock(spec=AudioPerception)
        replacement_model.proj = Mock()
        replacement_model.proj.weight.device = encoder.model.proj.weight.device
        replacement_model.proj.weight.dtype = torch.float32
        assert replacement_model is not encoder.model
        encoder.model = replacement_model
    elif mismatch == "resolved_device":
        encoder.model.proj.weight.device = torch.device("cuda:1")
    else:
        # note (caozhuoxi): FP16 weights cannot batch, so only vary the saved CPU dtype tag.
        encoder.warmed_batch_sizes[2] = replace(
            encoder.warmed_batch_sizes[2], model_dtype=str(torch.float16)
        )
    assert encoder.model.proj.weight.dtype == torch.float32
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 3
    assert created[-1].graph_capture_count == 1
    assert created[-1].graph is not None
    encoder.batch_runners[2].close()
    encoder.batch_runners.clear()
    run_size(encoder, 2)
    assert [runner.warmup_steps for runner in created] == [3, 1, 3, 1]
    assert created[-1].graph_capture_count == 1
    encoder.close()
    assert not encoder.warmed_batch_sizes
    assert torch.cuda.is_initialized() == initialized_before
    assert forbidden_cuda.call_count == 0
