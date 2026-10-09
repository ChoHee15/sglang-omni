"""Output, residency, and failure contracts for the acoustic graph cache."""

import asyncio
import gc
import weakref
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
    BatchedStreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    SAMPLES_PER_FRAME,
    AudioPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import GraphExecutionState
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.perception_scheduler import (
    PerceptionScheduler,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.nemotron_voicechat import test_streaming_perception

execution_device = test_streaming_perception.execution_device
perception_model = test_streaming_perception.perception_model


def cache_executor(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    capacity: int | None = None,
    enable_cuda_graph: bool = True,
    max_batch_size: int = 5,
) -> PerceptionScheduler:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    return stages.create_perception_executor(
        "test",
        device=perception_model.proj.weight.device.type,
        dtype="float32",
        max_batch_size=max_batch_size,
        max_cached_batch_runners=capacity,
        enable_cuda_graph=enable_cuda_graph,
    )


def payloads(frame_counts: tuple[int, ...], cohort_index: int) -> list[StagePayload]:
    return [
        StagePayload(
            f"cohort-{cohort_index}-row-{row}",
            OmniRequest(None),
            {
                **NemotronVoiceChatState(
                    waveform=torch.randn(frame_count * SAMPLES_PER_FRAME) / 100,
                    num_frames=frame_count,
                ).to_dict(),
                "caller_marker": f"marker-{cohort_index}-{row}",
            },
        )
        for row, frame_count in enumerate(frame_counts)
    ]


@torch.inference_mode()
def encode_and_compare(
    executor: PerceptionScheduler,
    frame_counts: tuple[int, ...],
    cohort_index: int,
) -> list[StagePayload]:
    requests = payloads(frame_counts, cohort_index)
    expected = [
        executor.encoder.model(
            NemotronVoiceChatState.from_dict(request.data).waveform.to(
                executor.encoder.model.proj.weight.device
            )[None]
        )[0].cpu()
        for request in requests
    ]
    results = executor.encoder.encode_validated(
        [executor.encoder.validate(request) for request in requests]
    )
    assert [result.request_id for result in results] == [
        request.request_id for request in requests
    ]
    for row, (result, reference) in enumerate(zip(results, expected)):
        frames = NemotronVoiceChatState.from_dict(result.data).acoustic_frames
        assert frames.shape == (frame_counts[row] + 1, 8)
        assert result.data["caller_marker"] == f"marker-{cohort_index}-{row}"
        torch.testing.assert_close(frames.cpu(), reference, rtol=1e-4, atol=1e-5)
    return results


def require_cuda(perception_model: AudioPerception) -> None:
    if perception_model.proj.weight.device.type != "cuda":
        pytest.skip("CUDA graph cache requires a GPU")
    else:
        pass


@torch.inference_mode()
def test_recurring_sizes_preserve_mixed_outputs_and_singleton_ownership(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    executor = cache_executor(monkeypatch, perception_model, capacity=2)
    saved_outputs: list[torch.Tensor] = []
    saved_references: list[torch.Tensor] = []
    for cohort_index, frame_counts in enumerate(
        ((0, 7), (1, 0, 20, 2, 7), (3,), (20, 1), (0, 2, 1, 7, 20), (1,))
    ):
        results = encode_and_compare(executor, frame_counts, cohort_index)
        for frames, reference in zip(saved_outputs, saved_references):
            torch.testing.assert_close(frames, reference, rtol=0, atol=0)
        frames = [
            NemotronVoiceChatState.from_dict(result.data).acoustic_frames
            for result in results
        ]
        saved_outputs.extend(frames)
        saved_references.extend(frame.clone() for frame in frames)
        assert len(executor.encoder.batch_runners) <= 2
    assert executor.encoder.graph_capture_count == 2


@pytest.mark.parametrize(
    "capacity, expected_captures", [(1, (1, 2, 3, 4)), (2, (1, 2, 2, 2))]
)
@torch.inference_mode()
def test_capacity_controls_recapture_for_alternating_sizes(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    capacity: int,
    expected_captures: tuple[int, ...],
) -> None:
    require_cuda(perception_model)
    executor = cache_executor(monkeypatch, perception_model, capacity)
    for cohort_index, (batch_size, capture_count) in enumerate(
        zip((2, 5, 2, 5), expected_captures)
    ):
        encode_and_compare(executor, (2,) * batch_size, cohort_index)
        assert executor.encoder.graph_capture_count == capture_count
        assert len(executor.encoder.batch_runners) <= capacity


@torch.inference_mode()
def test_lru_hit_changes_which_size_recaptures_and_releases_evicted_runner(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    executor = cache_executor(monkeypatch, perception_model, capacity=2)
    runner_references: dict[int, weakref.ReferenceType[BatchedPerceptionRunner]] = {}
    allocated_runners: list[weakref.ReferenceType[BatchedPerceptionRunner]] = []

    def construct_runner(
        model: AudioPerception,
        batch_size: int,
        enable_cuda_graph: bool,
        graph_execution_state: GraphExecutionState | None = None,
    ) -> BatchedPerceptionRunner:
        assert sum(reference() is not None for reference in allocated_runners) < 2
        runner = BatchedPerceptionRunner(
            model, batch_size, enable_cuda_graph, graph_execution_state
        )
        allocated_runners.append(weakref.ref(runner))
        return runner

    monkeypatch.setattr(
        "sglang_omni.models.nemotron_voicechat.perception_encoder.BatchedPerceptionRunner",
        construct_runner,
    )
    for cohort_index, (batch_size, capture_count) in enumerate(
        zip((2, 3, 2, 4, 3), (1, 2, 2, 3, 4))
    ):
        encode_and_compare(executor, (1,) * batch_size, cohort_index)
        assert executor.encoder.graph_capture_count == capture_count
        if cohort_index < 2:
            runner_references[batch_size] = weakref.ref(
                executor.encoder.batch_runners[batch_size]
            )
        elif cohort_index == 3:
            gc.collect()
            assert runner_references[3]() is None
            assert runner_references[2]() is not None
        else:
            pass
        assert len(executor.encoder.batch_runners) <= 2
    gc.collect()
    assert runner_references[2]() is None


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@torch.inference_mode()
def test_eager_policy_retains_only_one_runner(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    enable_cuda_graph: bool,
) -> None:
    if enable_cuda_graph and perception_model.proj.weight.device.type == "cuda":
        pytest.skip("Graph-enabled CUDA has the bounded multi-entry policy")
    else:
        pass
    executor = cache_executor(
        monkeypatch, perception_model, capacity=3, enable_cuda_graph=enable_cuda_graph
    )
    for cohort_index, batch_size in enumerate((2, 3, 2)):
        encode_and_compare(executor, (0, 2) + (1,) * (batch_size - 2), cohort_index)
        assert len(executor.encoder.batch_runners) == 1
    assert executor.encoder.graph_capture_count == 0


@torch.inference_mode()
def test_singleton_default_allocates_no_batch_resources(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    executor = cache_executor(monkeypatch, perception_model, max_batch_size=1)
    assert executor.encoder.max_cached_batch_runners == 1
    constructor = Mock(side_effect=AssertionError("singleton allocated a batch runner"))
    monkeypatch.setattr(
        "sglang_omni.models.nemotron_voicechat.perception_encoder.BatchedPerceptionRunner",
        constructor,
    )
    for cohort_index, frame_count in enumerate((0, 2, 7)):
        encode_and_compare(executor, (frame_count,), cohort_index)
    assert constructor.call_count == 0
    assert executor.encoder.graph_capture_count == 0
    assert len(executor.encoder.batch_runners) == 0


@torch.inference_mode()
def test_capture_fallback_invalidates_all_warmed_graphs_and_never_replays_them(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    executor = cache_executor(monkeypatch, perception_model, capacity=3)
    for cohort_index, batch_size in enumerate((2, 3)):
        encode_and_compare(executor, (2,) * batch_size, cohort_index)
    old_runner_references = [
        weakref.ref(runner) for runner in executor.encoder.batch_runners.values()
    ]
    replay = Mock(side_effect=AssertionError("stale graph replayed after fallback"))
    for runner in executor.encoder.batch_runners.values():
        assert runner.graph is not None
        monkeypatch.setattr(runner.graph, "replay", replay)
    runner = None
    capture = Mock(side_effect=RuntimeError("CUDA graph capture not supported"))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", capture)
    for cohort_index, batch_size in enumerate((4, 2, 3, 4), start=2):
        encode_and_compare(executor, (0, 7) + (1,) * (batch_size - 2), cohort_index)
        assert not executor.encoder.enable_cuda_graph
        assert len(executor.encoder.batch_runners) == 1
    assert capture.call_count == 1
    assert replay.call_count == 0
    assert executor.encoder.graph_capture_count == 2
    assert (
        len([record for record in caplog.records if "using eager" in record.message])
        == 1
    )
    # note (caozhuoxi): MonkeyPatch owns patched graph objects, not their runners.
    gc.collect()
    assert all(reference() is None for reference in old_runner_references)


@torch.inference_mode()
def test_fallback_then_ordinary_error_still_latches_eager_policy(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    executor = cache_executor(monkeypatch, perception_model, capacity=3)
    for cohort_index, batch_size in enumerate((2, 3)):
        encode_and_compare(executor, (2,) * batch_size, cohort_index)
    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(
            torch.cuda,
            "CUDAGraph",
            Mock(side_effect=RuntimeError("CUDA graph capture not supported")),
        )
        failure_patch.setattr(
            BatchedStreamingPerception,
            "flush",
            Mock(side_effect=RuntimeError("ordinary flush failure")),
        )
        with pytest.raises(RuntimeError, match="ordinary flush failure"):
            encode_and_compare(executor, (2,) * 4, 2)
    assert not executor.encoder.enable_cuda_graph
    assert len(executor.encoder.batch_runners) == 1
    for cohort_index, batch_size in enumerate((2, 3), start=3):
        encode_and_compare(executor, (2,) * batch_size, cohort_index)
    assert executor.encoder.graph_capture_count == 2


@torch.inference_mode()
def test_fatal_error_precedes_fallback_cleanup_and_stops_later_cohorts(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    executor = cache_executor(monkeypatch, perception_model, capacity=3)
    for cohort_index, batch_size in enumerate((2, 3)):
        encode_and_compare(executor, (2,) * batch_size, cohort_index)
    selected_runner = executor.encoder.batch_runners[2]
    selected_runner.enable_cuda_graph = False
    failure = Mock(side_effect=RuntimeError("CUDA error: illegal memory access"))
    reset = Mock(side_effect=AssertionError("reset after fatal CUDA failure"))
    cleanup = Mock(side_effect=AssertionError("cache cleanup after fatal CUDA failure"))
    monkeypatch.setattr(selected_runner, "encode", failure)
    monkeypatch.setattr(selected_runner.stream, "reset", reset)
    monkeypatch.setattr(executor.encoder, "disable_batch_graphs", cleanup)
    incoming = [
        IncomingMessage(request.request_id, "new_request", request)
        for request in payloads((2, 2), 2)
    ]
    loop = asyncio.new_event_loop()
    try:
        executor.run_batch(incoming, loop)
        executor.run_batch(incoming, loop)
    finally:
        loop.close()
    assert executor.fatal_error is not None
    assert failure.call_count == 1
    assert reset.call_count == 0
    assert cleanup.call_count == 0
    assert len(executor.encoder.batch_runners) == 2
    assert executor.encoder.enable_cuda_graph
    assert executor.outbox.empty()
