"""Real CUDA acceptance cases for cold and qualified batch graph recapture."""

import logging
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
    BatchedStreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    AudioPerception,
    GraphPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.perception_scheduler import (
    PerceptionScheduler,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.nemotron_voicechat import test_streaming_perception

execution_device = test_streaming_perception.execution_device
perception_model = test_streaming_perception.perception_model
logger = logging.getLogger(__name__)


def executor(
    monkeypatch: pytest.MonkeyPatch, model: AudioPerception, capacity: int
) -> PerceptionScheduler:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=model))
    monkeypatch.setattr(stages, "load_module", Mock())
    return stages.create_perception_executor(
        "tiny",
        device="cuda",
        dtype="float32",
        max_batch_size=12,
        max_cached_batch_runners=capacity,
    )


def check_cohort(
    scheduler: PerceptionScheduler, frame_counts: tuple[int, ...], cohort_index: int
) -> list[torch.Tensor]:
    model = scheduler.encoder.model
    payloads = [
        StagePayload(
            f"{cohort_index}-{row}",
            OmniRequest(None),
            NemotronVoiceChatState(
                waveform=torch.randn(frame_count * 1280) / 100, num_frames=frame_count
            ).to_dict(),
        )
        for row, frame_count in enumerate(frame_counts)
    ]
    references = [
        model(
            NemotronVoiceChatState.from_dict(payload.data).waveform.to(
                model.proj.weight.device
            )[None],
            stream=StreamingPerception(model),
        )[0].cpu()
        for payload in payloads
    ]
    results = scheduler.encoder.encode_validated(
        [scheduler.encoder.validate(payload) for payload in payloads]
    )
    assert [result.request_id for result in results] == [
        payload.request_id for payload in payloads
    ]
    actual = [
        NemotronVoiceChatState.from_dict(result.data).acoustic_frames
        for result in results
    ]
    for frames, reference in zip(actual, references):
        torch.testing.assert_close(frames.cpu(), reference, rtol=1e-4, atol=1e-5)
    return actual


def observe_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int, int, int]]:
    observations: list[tuple[int, int, int, int]] = []
    pushes = 0
    original_push = BatchedStreamingPerception.push
    original_capture = BatchedPerceptionRunner.capture_step

    def push(stream: BatchedStreamingPerception, samples: torch.Tensor) -> torch.Tensor:
        nonlocal pushes
        pushes += 1
        return original_push(stream, samples)

    def capture(runner: BatchedPerceptionRunner) -> None:
        previous_pushes = pushes
        previous_count = runner.graph_capture_count
        original_capture(runner)
        if runner.graph is not None and runner.graph_capture_count > previous_count:
            assert runner.captured_step is not None
            assert runner.captured_step.state.raw_stream_handle is not None
            observed_warm_pushes = pushes - previous_pushes - 1
            assert observed_warm_pushes == runner.warmup_steps
            observations.append(
                (
                    runner.batch_size,
                    observed_warm_pushes,
                    runner.captured_step.state.raw_stream_handle,
                    id(runner),
                )
            )
        else:
            raise AssertionError("CUDA capture fell back during warmup acceptance")

    monkeypatch.setattr(BatchedStreamingPerception, "push", push)
    monkeypatch.setattr(BatchedPerceptionRunner, "capture_step", capture)
    return observations


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA acceptance is root-owned"
)
@pytest.mark.parametrize("execution_device", ["cuda"], indirect=True)
@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("batch_size", range(2, 9))
@pytest.mark.parametrize("cudnn_tf32", [False, True])
@pytest.mark.parametrize("singleton_first", [False, True])
@torch.inference_mode()
def test_each_batch_cold_three_then_new_runner_recap_one(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    batch_size: int,
    cudnn_tf32: bool,
    singleton_first: bool,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", cudnn_tf32)
    observed = observe_captures(monkeypatch)
    scheduler = executor(monkeypatch, perception_model, 1)
    singleton_pushes: list[int] = []
    original_singleton_push = StreamingPerception.push

    def singleton_push(
        stream: StreamingPerception, samples: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(stream, GraphPerception):
            singleton_pushes.append(1)
        else:
            pass
        return original_singleton_push(stream, samples)

    monkeypatch.setattr(StreamingPerception, "push", singleton_push)
    saved_outputs: list[torch.Tensor] = []
    saved_copies: list[torch.Tensor] = []
    try:
        if singleton_first:
            saved_outputs.extend(check_cohort(scheduler, (2,), -1))
        else:
            pass
        counts = (0, 7) + (1,) * (batch_size - 2)
        saved_outputs.extend(check_cohort(scheduler, counts, 0))
        saved_copies.extend(frames.clone() for frames in saved_outputs)
        first_runner = scheduler.encoder.batch_runners[batch_size]
        first_resource = first_runner.captured_step
        assert first_resource is not None
        check_cohort(scheduler, (2,) * (3 if batch_size == 2 else 2), 1)
        assert first_resource.state.is_closed
        check_cohort(scheduler, tuple(reversed(counts)), 2)
        assert scheduler.encoder.batch_runners[batch_size] is not first_runner
        assert [
            warm for size, warm, handle, runner_id in observed if size == batch_size
        ] == [3, 1]
        assert scheduler.encoder.graph_capture_count == 3
        if not singleton_first:
            check_cohort(scheduler, (2,), 3)
        else:
            pass
        assert isinstance(scheduler.encoder.stream, GraphPerception)
        assert scheduler.encoder.stream.graph is not None
        assert len(singleton_pushes) == 4
        for frames, saved in zip(saved_outputs, saved_copies):
            torch.testing.assert_close(frames, saved, rtol=0, atol=0)
        logger.info(
            f"B={batch_size} cuDNN_TF32={cudnn_tf32} matmul_TF32=False B1first={singleton_first} actual_capture_warmups={observed} fallback=0"
        )
    finally:
        scheduler.encoder.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA acceptance is root-owned"
)
@pytest.mark.parametrize("execution_device", ["cuda"], indirect=True)
@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("capacity", [1, 2, 3])
@torch.inference_mode()
def test_hundred_forced_captures_all_sizes_with_owned_outputs(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    capacity: int,
) -> None:
    observed = observe_captures(monkeypatch)
    scheduler = executor(monkeypatch, perception_model, capacity)
    retained: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    unrelated_input = torch.ones((16, 16), device="cuda")
    try:
        for cohort_index in range(105):
            batch_size = 2 + cohort_index % 7
            frames = check_cohort(
                scheduler, (0, 7) + (1,) * (batch_size - 2), cohort_index
            )
            check_cohort(scheduler, (2,), 1000 + cohort_index)
            if cohort_index % 25 == 0:
                retained.extend(frames)
                references.extend(frame.clone() for frame in frames)
            else:
                pass
            for owned, reference in zip(retained, references):
                torch.testing.assert_close(owned, reference, rtol=0, atol=0)
            assert scheduler.encoder.graph_capture_count == cohort_index + 1
            assert len(scheduler.encoder.warmed_batch_sizes) <= 8
            pooled_stream = torch.cuda.Stream()
            pooled_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(pooled_stream):
                torch.mm(unrelated_input, unrelated_input)
            torch.cuda.current_stream().wait_stream(pooled_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=pooled_stream):
                output = torch.mm(unrelated_input, unrelated_input)
            graph.replay()
            output.cpu()
            output = None
            graph.reset()
            graph = None
        assert len(observed) == 105
        assert [warm for size, warm, handle, runner_id in observed[:7]] == [3] * 7
        assert all(warm == 1 for size, warm, handle, runner_id in observed[7:])
        logger.info(
            f"capacity={capacity} captures={len(observed)} fallback=0 warmups={observed}"
        )
    finally:
        scheduler.encoder.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA acceptance is root-owned"
)
@pytest.mark.parametrize("execution_device", ["cuda"], indirect=True)
@pytest.mark.parametrize("perception_model", [2], indirect=True)
@pytest.mark.parametrize("sizes", [(2, 3, 4), (2, 3, 4, 5, 6)])
@pytest.mark.parametrize("capacity", [1, 2, 3])
@torch.inference_mode()
def test_three_and_five_size_cycles_replay_or_recapture_without_output_changes(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: AudioPerception,
    sizes: tuple[int, ...],
    capacity: int,
) -> None:
    observed = observe_captures(monkeypatch)
    scheduler = executor(monkeypatch, perception_model, capacity)
    try:
        for cohort_index in range(35 * len(sizes)):
            batch_size = sizes[cohort_index % len(sizes)]
            check_cohort(scheduler, (0, 7) + (1,) * (batch_size - 2), cohort_index)
        expected_captures = len(sizes) if len(sizes) <= capacity else 35 * len(sizes)
        assert (
            len(observed) == scheduler.encoder.graph_capture_count == expected_captures
        )
        assert [warm for size, warm, handle, runner_id in observed[: len(sizes)]] == [
            3
        ] * len(sizes)
        assert all(
            warm == 1 for size, warm, handle, runner_id in observed[len(sizes) :]
        )
        logger.info(
            f"sizes={sizes} capacity={capacity} captures={len(observed)} fallback=0 warmups={observed}"
        )
    finally:
        scheduler.encoder.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA acceptance is root-owned"
)
@pytest.mark.parametrize("execution_device", ["cuda"], indirect=True)
@pytest.mark.parametrize("perception_model", [2], indirect=True)
@torch.inference_mode()
def test_real_fallback_clears_qualification_and_keeps_eager_outputs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    perception_model: AudioPerception,
) -> None:
    scheduler = executor(monkeypatch, perception_model, 3)
    retained = check_cohort(scheduler, (0, 7), 0)
    references = [frames.clone() for frames in retained]
    check_cohort(scheduler, (1, 2, 3), 1)
    assert len(scheduler.encoder.warmed_batch_sizes) == 2
    failure = Mock(side_effect=RuntimeError("CUDA graph capture not supported"))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", failure)
    try:
        for cohort_index, batch_size in enumerate((4, 2, 3, 4), start=2):
            check_cohort(scheduler, (0, 7) + (1,) * (batch_size - 2), cohort_index)
            assert not scheduler.encoder.warmed_batch_sizes
        assert not scheduler.encoder.enable_cuda_graph
        assert failure.call_count == 1
        assert scheduler.encoder.graph_capture_count == 2
        assert (
            len(
                [record for record in caplog.records if "using eager" in record.message]
            )
            == 1
        )
        for frames, saved in zip(retained, references):
            torch.testing.assert_close(frames, saved, rtol=0, atol=0)
    finally:
        scheduler.encoder.close()
