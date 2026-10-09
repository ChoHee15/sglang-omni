"""CPU execution and factory contracts for internal capture warmup controls."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import batched_perception, stages
from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    AudioPerception,
    StreamingPerception,
)
from tests.unit_test.nemotron_voicechat import test_streaming_perception

execution_device = test_streaming_perception.execution_device
perception_model = test_streaming_perception.perception_model


@pytest.mark.parametrize("bad_steps", [True, False, 0, -1, 1.5])
@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("execution_device", ["cpu"], indirect=True)
def test_warmup_steps_rejects_invalid_values_before_stream_allocation(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    bad_steps: int | float | bool,
) -> None:
    allocation = Mock(side_effect=AssertionError("allocated before validation"))
    monkeypatch.setattr(batched_perception, "BatchedStreamingPerception", allocation)
    with pytest.raises(ValueError):
        BatchedPerceptionRunner(perception_model, 2, True, warmup_steps=bad_steps)
    assert allocation.call_count == 0


@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("execution_device", ["cpu"], indirect=True)
@torch.inference_mode()
def test_capture_executes_three_one_three_actual_eager_pushes(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    phase = "idle"
    pushes: list[str] = []

    @contextmanager
    def stream_context(stream: Mock) -> Iterator[None]:
        nonlocal phase
        phase = "warmup"
        try:
            yield
        finally:
            phase = "idle"

    @contextmanager
    def graph_context(graph: Mock, *, stream: Mock) -> Iterator[None]:
        nonlocal phase
        phase = "capture-body"
        try:
            yield
        finally:
            phase = "idle"

    resource = Mock()
    resource.capture_stream = Mock()
    resource.state = Mock()
    resource.close = Mock()
    monkeypatch.setattr(batched_perception, "CapturedStep", Mock(return_value=resource))
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=Mock()))
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Mock(return_value=Mock()))
    observed = []
    for warmup_steps in (3, 1, 3):
        runner = BatchedPerceptionRunner(
            perception_model, 2, True, warmup_steps=warmup_steps
        )
        original_push = runner.stream.push

        def push(samples: torch.Tensor) -> torch.Tensor:
            pushes.append(phase)
            return original_push(samples)

        monkeypatch.setattr(runner.stream, "push", push)
        pushes.clear()
        runner.capture()
        observed.append(pushes.count("warmup"))
        assert pushes.count("capture-body") == 1
        assert runner.graph_capture_count == 1
    assert observed == [3, 1, 3]
    singleton = BatchedPerceptionRunner(perception_model, 1, True, warmup_steps=1)
    assert singleton.stream.perception is perception_model
    assert batched_perception.GRAPH_WARMUP_STEPS == 3


def test_non_fp32_factory_keeps_effective_singleton_and_no_batch_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock(spec=AudioPerception)
    model.proj = Mock()
    model.proj.weight.dtype = torch.float16
    model.proj.weight.device = torch.device("cpu")
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=model))
    monkeypatch.setattr(stages, "load_module", Mock())
    monkeypatch.setattr(
        stages, "GraphPerception", Mock(return_value=Mock(spec=StreamingPerception))
    )
    scheduler = stages.create_perception_executor(
        "tiny", device="cpu", dtype="float16", max_batch_size=8
    )
    assert scheduler.max_batch_size == 1
    assert scheduler.encoder.model.proj.weight.dtype == torch.float16
    assert not scheduler.encoder.batch_runners
    assert not scheduler.encoder.warmed_batch_sizes
