"""Public acoustic-output contracts for independent batched recordings."""

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
    GraphPerception,
    StreamingPerception,
)
from tests.unit_test.nemotron_voicechat import test_streaming_perception

execution_device = test_streaming_perception.execution_device
perception_model = test_streaming_perception.perception_model


def reference_features(model: AudioPerception, waveform: torch.Tensor) -> torch.Tensor:
    stream = StreamingPerception(model)
    if waveform.numel() == 0:
        return stream.flush().clone()
    else:
        return model(waveform.unsqueeze(0), stream=stream)[0].clone()


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "frame_counts",
    [(0,), (0, 0), (1,), (0, 7), (1, 2), (2, 7, 0, 20), (0, 1, 2, 7, 20, 1, 2, 7)],
)
@torch.inference_mode()
def test_batch_matches_independent_recordings(
    perception_model: AudioPerception,
    frame_counts: tuple[int, ...],
    enable_cuda_graph: bool,
) -> None:
    device = perception_model.proj.weight.device
    waveforms = [
        torch.randn(count * SAMPLES_PER_FRAME, device=device) / 100
        for count in frame_counts
    ]
    expected = [
        reference_features(perception_model, waveform) for waveform in waveforms
    ]
    runner = BatchedPerceptionRunner(
        perception_model, len(waveforms), enable_cuda_graph
    )
    actual = runner.encode(waveforms)
    assert len(actual) == len(expected)
    for count, features, reference in zip(frame_counts, actual, expected):
        assert features.shape == (count + 1, perception_model.proj.out_features)
        assert torch.isfinite(features).all()
        torch.testing.assert_close(features, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@torch.inference_mode()
def test_batch_row_permutation_and_waveform_isolation(
    perception_model: AudioPerception,
    enable_cuda_graph: bool,
) -> None:
    device = perception_model.proj.weight.device
    waveforms = [
        torch.randn(count * SAMPLES_PER_FRAME, device=device) / 100
        for count in (1, 20, 0, 7)
    ]
    runner = BatchedPerceptionRunner(perception_model, 4, enable_cuda_graph)
    original = runner.encode(waveforms)
    saved = [features.clone() for features in original]
    permutation = [2, 0, 3, 1]
    permuted = runner.encode([waveforms[index] for index in permutation])
    for position, original_index in enumerate(permutation):
        torch.testing.assert_close(
            permuted[position], saved[original_index], rtol=1e-4, atol=1e-5
        )
    changed_waveforms = [waveform.clone() for waveform in waveforms]
    changed_waveforms[1].normal_()
    changed = runner.encode(changed_waveforms)
    for index in (0, 2, 3):
        torch.testing.assert_close(changed[index], saved[index], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(changed[1], saved[1], rtol=1e-4, atol=1e-5)
    for features, reference in zip(original, saved):
        torch.testing.assert_close(features, reference, rtol=0, atol=0)


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@torch.inference_mode()
def test_reused_batch_changes_active_rows_and_preserves_outputs(
    perception_model: AudioPerception,
    enable_cuda_graph: bool,
) -> None:
    device = perception_model.proj.weight.device
    runner = BatchedPerceptionRunner(perception_model, 2, enable_cuda_graph)
    previous_outputs: list[torch.Tensor] = []
    previous_references: list[torch.Tensor] = []
    for frame_counts in ((20, 1), (0, 7), (2, 0), (1, 20)):
        waveforms = [
            torch.randn(count * SAMPLES_PER_FRAME, device=device) / 100
            for count in frame_counts
        ]
        actual = runner.encode(waveforms)
        expected = [
            reference_features(perception_model, waveform) for waveform in waveforms
        ]
        for features, reference in zip(actual, expected):
            torch.testing.assert_close(features, reference, rtol=1e-4, atol=1e-5)
        for features, reference in zip(previous_outputs, previous_references):
            torch.testing.assert_close(features, reference, rtol=0, atol=0)
        previous_outputs.extend(actual)
        previous_references.extend(features.clone() for features in actual)


@torch.inference_mode()
def test_singleton_nonempty_retains_original_graph_exactness(
    perception_model: AudioPerception,
) -> None:
    device = perception_model.proj.weight.device
    runner = BatchedPerceptionRunner(perception_model, 1, True)
    reference_stream = GraphPerception(perception_model)
    for frame_count in (1, 20, 2):
        waveform = torch.randn(frame_count * SAMPLES_PER_FRAME, device=device) / 100
        expected = perception_model(waveform.unsqueeze(0), stream=reference_stream)[0]
        actual = runner.encode([waveform])[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@torch.inference_mode()
def test_attention_context_boundary_and_rollover(
    perception_model: AudioPerception,
    enable_cuda_graph: bool,
) -> None:
    device = perception_model.proj.weight.device
    waveforms = [
        torch.randn(count * SAMPLES_PER_FRAME, device=device) / 100
        for count in (69, 70, 71, 72)
    ]
    runner = BatchedPerceptionRunner(perception_model, 4, enable_cuda_graph)
    actual = runner.encode(waveforms)
    for waveform, features in zip(waveforms, actual):
        expected = reference_features(perception_model, waveform)
        torch.testing.assert_close(features, expected, rtol=1e-4, atol=1e-5)


@torch.inference_mode()
def test_known_graph_capture_failure_keeps_fresh_eager_outputs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    perception_model: AudioPerception,
) -> None:
    if perception_model.proj.weight.device.type != "cuda":
        pytest.skip("CUDA capture fallback requires a GPU")
    else:
        pass
    device = perception_model.proj.weight.device
    waveforms = [
        torch.randn(count * SAMPLES_PER_FRAME, device=device) / 100 for count in (2, 7)
    ]
    expected = [
        reference_features(perception_model, waveform) for waveform in waveforms
    ]
    runner = BatchedPerceptionRunner(perception_model, 2, True)
    monkeypatch.setattr(
        torch.cuda,
        "CUDAGraph",
        Mock(side_effect=RuntimeError("CUDA graph capture not supported")),
    )
    for iteration in range(2):
        actual = runner.encode(waveforms)
        for features, reference in zip(actual, expected):
            torch.testing.assert_close(features, reference, rtol=1e-4, atol=1e-5)
    assert runner.graph_capture_count == 0
    assert (
        len([record for record in caplog.records if "using eager" in record.message])
        == 1
    )
