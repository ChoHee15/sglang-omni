"""Long-lived Perception outputs and holders survive stream churn and fatal errors."""

import gc
import logging
import weakref
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import cuda_graph
from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    AudioPerception,
    GraphPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import (
    CapturedStep,
    GraphExecutionState,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from tests.unit_test.nemotron_voicechat import (
    test_perception_capture_resources,
    test_streaming_perception,
)
from tests.unit_test.nemotron_voicechat.test_perception_graph_cache import (
    cache_executor,
    encode_and_compare,
    payloads,
    require_cuda,
)

native_runtime = test_perception_capture_resources.native_runtime
execution_device = test_streaming_perception.execution_device
perception_model = test_streaming_perception.perception_model
CHURN_CAPTURE_COUNT = 100
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("capacity", [1, 2, 3])
@pytest.mark.parametrize("perception_model", [2], indirect=True)
@torch.inference_mode()
def test_native_capture_lifetime_survives_pooled_stream_rollover(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    capacity: int,
) -> None:
    require_cuda(perception_model)
    gc.collect()
    runtime = cuda_graph.cuda_runtime
    assert runtime is not None
    native_create = runtime.cudaStreamCreateWithFlags
    native_destroy = runtime.cudaStreamDestroy
    live_handles: set[int] = set()
    handle_events: list[tuple[str, int]] = []
    maximum_live_handles = 0

    def create_stream(flags: int) -> tuple[int, int]:
        nonlocal maximum_live_handles
        result = native_create(flags)
        assert result[0] == runtime.cudaError_t.cudaSuccess
        stream_handle = int(result[1])
        assert stream_handle not in live_handles
        live_handles.add(stream_handle)
        handle_events.append(("create", stream_handle))
        maximum_live_handles = max(maximum_live_handles, len(live_handles))
        assert len(live_handles) <= 1 + capacity
        return tuple(int(value) for value in result)

    def destroy_stream(stream_handle: int) -> tuple[int]:
        assert int(stream_handle) in live_handles
        result = native_destroy(stream_handle)
        assert result[0] == runtime.cudaError_t.cudaSuccess
        live_handles.remove(int(stream_handle))
        handle_events.append(("destroy", int(stream_handle)))
        return tuple(int(value) for value in result)

    monkeypatch.setattr(runtime, "cudaStreamCreateWithFlags", create_stream)
    monkeypatch.setattr(runtime, "cudaStreamDestroy", destroy_stream)
    executor = cache_executor(monkeypatch, perception_model, capacity)
    retained_frames: list[torch.Tensor] = []
    retained_references: list[torch.Tensor] = []
    unrelated_matrix = torch.randn((16, 16), device=perception_model.proj.weight.device)
    try:
        for cohort_index in range(CHURN_CAPTURE_COUNT):
            batch_size = (2, 3, 4, 5)[cohort_index % 4]
            frame_counts = (0, 3) + (1,) * (batch_size - 2)
            results = encode_and_compare(executor, frame_counts, 2 * cohort_index)
            assert executor.encoder.graph_capture_count == cohort_index + 1
            singleton_results = encode_and_compare(executor, (2,), 2 * cohort_index + 1)
            for frames, reference in zip(retained_frames, retained_references):
                torch.testing.assert_close(frames, reference, rtol=0, atol=0)
            if cohort_index % 25 == 0:
                frames = NemotronVoiceChatState.from_dict(
                    results[1].data
                ).acoustic_frames
                singleton_frames = NemotronVoiceChatState.from_dict(
                    singleton_results[0].data
                ).acoustic_frames
                retained_frames.extend((frames, singleton_frames))
                retained_references.extend((frames.clone(), singleton_frames.clone()))
            else:
                pass
            pooled_stream = torch.cuda.Stream(
                device=perception_model.proj.weight.device
            )
            pooled_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(pooled_stream):
                torch.mm(unrelated_matrix, unrelated_matrix)
            torch.cuda.current_stream().wait_stream(pooled_stream)
            unrelated_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(unrelated_graph, stream=pooled_stream):
                unrelated_output = torch.mm(unrelated_matrix, unrelated_matrix)
            unrelated_graph.replay()
            torch.cuda.current_stream().wait_stream(pooled_stream)
            unrelated_output.cpu()
            unrelated_output = None
            unrelated_graph.reset()
            unrelated_graph = None
        assert executor.encoder.graph_capture_count >= 97
        assert maximum_live_handles == 1 + capacity
        assert (
            len([event for event, handle in handle_events if event == "create"])
            == 1 + CHURN_CAPTURE_COUNT
        )
    finally:
        executor.encoder.close()
    assert live_handles == set()
    logger.info(
        f"capacity={capacity} batch_captures={executor.encoder.graph_capture_count} native_creates={len(handle_events) // 2} native_destroys={len(handle_events) // 2} max_live_handles={maximum_live_handles} outstanding_after_close={len(live_handles)} pooled_graphs={CHURN_CAPTURE_COUNT}"
    )
    assert (
        len([event for event, handle in handle_events if event == "destroy"])
        == 1 + CHURN_CAPTURE_COUNT
    )
    executor.encoder.close()
    assert live_handles == set()
    logger.info(
        f"capacity={capacity} batch_captures={executor.encoder.graph_capture_count} native_creates={len(handle_events) // 2} native_destroys={len(handle_events) // 2} max_live_handles={maximum_live_handles} outstanding_after_close={len(live_handles)} pooled_graphs={CHURN_CAPTURE_COUNT}"
    )


@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("execution_device", ["cpu"], indirect=True)
def test_encoder_fatal_abandons_singleton_and_every_cached_holder_without_dropping_refs(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
    perception_model: AudioPerception,
) -> None:
    executor = cache_executor(monkeypatch, perception_model, capacity=3)
    encoder = executor.encoder
    batch_runners = [
        BatchedPerceptionRunner(
            perception_model, batch_size, True, encoder.graph_execution_state
        )
        for batch_size in (2, 3)
    ]
    encoder.batch_runners.update(
        (runner.batch_size, runner) for runner in batch_runners
    )
    holders = [encoder.stream, *batch_runners]
    captured_steps: list[CapturedStep] = []
    retained_graphs: list[Mock] = []
    retained_outputs: list[torch.Tensor] = []
    for holder in holders:
        owner = CapturedStep(torch.device("cuda:1"), encoder.graph_execution_state)
        owner.state.graph = Mock()
        owner.state.output = torch.ones(1)
        holder.captured_step = owner
        captured_steps.append(owner)
        retained_graphs.append(owner.graph)
        retained_outputs.append(owner.output)
    requests = [encoder.validate(request) for request in payloads((2, 2), 0)]
    encode = Mock(side_effect=RuntimeError("CUDA device-side assert"))
    guard = Mock(side_effect=AssertionError("CUDA API after fatal encode"))
    monkeypatch.setattr(encoder, "encode_requests", encode)
    monkeypatch.setattr(torch.cuda, "device", guard)
    with pytest.raises(RuntimeError, match="device-side assert"):
        encoder.encode_validated(requests)
    assert encoder.graph_execution_state.is_abandoned
    for owner, graph, output in zip(captured_steps, retained_graphs, retained_outputs):
        assert owner.state.execution_state.is_abandoned
        assert owner.graph is graph
        assert owner.output is output
        owner.close()
        owner.finalizer()
        assert graph.reset.call_count == 0
    encoder.close()
    with pytest.raises(RuntimeError, match="closed or abandoned"):
        encoder.encode_validated(requests)
    with pytest.raises(RuntimeError, match="closed or abandoned"):
        encoder.validate(payloads((2,), 1)[0])
    assert encode.call_count == 1
    assert guard.call_count == 0
    assert native_runtime.cudaStreamDestroy.call_count == 0
    assert len(encoder.batch_runners) == 2


@pytest.mark.parametrize("perception_model", [2], indirect=True)
@torch.inference_mode()
def test_factory_gc_releases_native_graph_resources_and_retains_owned_outputs(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    gc.collect()
    runtime = cuda_graph.cuda_runtime
    assert runtime is not None
    create = runtime.cudaStreamCreateWithFlags
    destroy = runtime.cudaStreamDestroy
    created_handles: list[int] = []
    destroyed_handles: list[int] = []

    def create_stream(flags: int) -> tuple[int, int]:
        status, stream_handle = create(flags)
        assert int(status) == 0
        created_handles.append(int(stream_handle))
        return (int(status), int(stream_handle))

    def destroy_stream(stream_handle: int) -> tuple[int]:
        (status,) = destroy(stream_handle)
        assert int(status) == 0
        destroyed_handles.append(int(stream_handle))
        return (int(status),)

    monkeypatch.setattr(runtime, "cudaStreamCreateWithFlags", create_stream)
    monkeypatch.setattr(runtime, "cudaStreamDestroy", destroy_stream)
    executor = cache_executor(monkeypatch, perception_model)
    results = encode_and_compare(executor, (2,), 0)
    results.extend(encode_and_compare(executor, (0, 3), 1))
    saved_frames = [
        NemotronVoiceChatState.from_dict(result.data).acoustic_frames
        for result in results
    ]
    references = [frames.clone() for frames in saved_frames]
    singleton_reference = weakref.ref(executor.encoder.stream.captured_step)
    batch_reference = weakref.ref(executor.encoder.batch_runners[2].captured_step)
    assert len(created_handles) == 2
    executor = None
    gc.collect()
    assert singleton_reference() is None
    assert batch_reference() is None
    assert sorted(created_handles) == sorted(destroyed_handles)
    for frames, reference in zip(saved_frames, references):
        torch.testing.assert_close(frames, reference, rtol=0, atol=0)


@pytest.mark.parametrize("perception_model", [2], indirect=True)
@torch.inference_mode()
def test_singleton_runner_wrapper_close_releases_native_handle_and_preserves_outputs(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
) -> None:
    require_cuda(perception_model)
    runtime = cuda_graph.cuda_runtime
    assert runtime is not None
    create = runtime.cudaStreamCreateWithFlags
    destroy = runtime.cudaStreamDestroy
    live_handles: set[int] = set()
    destroyed_handles: list[int] = []

    def create_stream(flags: int) -> tuple[int, int]:
        status, stream_handle = create(flags)
        assert int(status) == 0
        live_handles.add(int(stream_handle))
        return (int(status), int(stream_handle))

    def destroy_stream(stream_handle: int) -> tuple[int]:
        assert int(stream_handle) in live_handles
        (status,) = destroy(stream_handle)
        assert int(status) == 0
        live_handles.remove(int(stream_handle))
        destroyed_handles.append(int(stream_handle))
        return (int(status),)

    monkeypatch.setattr(runtime, "cudaStreamCreateWithFlags", create_stream)
    monkeypatch.setattr(runtime, "cudaStreamDestroy", destroy_stream)
    shared_state = GraphExecutionState()
    runner = BatchedPerceptionRunner(perception_model, 1, True, shared_state)
    assert isinstance(runner.stream, GraphPerception)
    assert runner.stream.graph_execution_state is runner.graph_execution_state
    assert runner.graph_execution_state is shared_state
    waveform = (
        torch.randn(
            2 * test_streaming_perception.SAMPLES_PER_FRAME,
            device=perception_model.proj.weight.device,
        )
        / 100
    )
    outputs = runner.encode([waveform])
    published_frames = outputs[0].cpu()
    reference = published_frames.clone()
    assert len(live_handles) == 1
    runner.close()
    runner.close()
    assert live_handles == set()
    assert len(destroyed_handles) == 1
    torch.testing.assert_close(published_frames, reference, rtol=0, atol=0)
    torch.testing.assert_close(outputs[0].cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("execution_device", ["cpu"], indirect=True)
@pytest.mark.parametrize("abandon_method", ["runner", "stream", "shared-state"])
def test_singleton_runner_wrapper_shares_abandon_state_and_skips_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
    perception_model: AudioPerception,
    abandon_method: str,
) -> None:
    shared_state = GraphExecutionState()
    runner = BatchedPerceptionRunner(perception_model, 1, True, shared_state)
    assert isinstance(runner.stream, GraphPerception)
    assert runner.stream.graph_execution_state is shared_state
    owner = CapturedStep(torch.device("cuda:1"), shared_state)
    owner.state.graph = Mock()
    owner.state.output = torch.ones(1)
    runner.stream.captured_step = owner
    graph = owner.graph
    output = owner.output
    if abandon_method == "runner":
        runner.abandon()
    elif abandon_method == "stream":
        runner.stream.abandon()
    else:
        shared_state.is_abandoned = True
    guard = Mock(side_effect=AssertionError("CUDA API after singleton wrapper abandon"))
    monkeypatch.setattr(torch.cuda, "device", guard)
    runner.close()
    runner.close()
    owner.finalizer()
    assert shared_state.is_abandoned
    assert owner.graph is graph
    assert owner.output is output
    assert graph.reset.call_count == 0
    assert native_runtime.cudaStreamDestroy.call_count == 0
    assert guard.call_count == 0
