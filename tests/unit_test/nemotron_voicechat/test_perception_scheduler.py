"""Admission and publication contracts for offline Perception cohorts."""

import asyncio
import threading
import time
from collections import deque
from queue import Empty
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import stages
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.perception_encoder import (
    ValidatedPerceptionRequest,
)
from sglang_omni.models.nemotron_voicechat.perception_scheduler import (
    PerceptionScheduler,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingStageControlPlane
from tests.unit_test.nemotron_voicechat import test_streaming_perception
from tests.unit_test.pipeline.helpers import make_stage

execution_device = test_streaming_perception.execution_device
perception_model = test_streaming_perception.perception_model


def message(request_id: str, frame_count: int = 4) -> IncomingMessage:
    waveform = torch.full((frame_count * 1280,), float(len(request_id)) / 100)
    payload = StagePayload(
        request_id,
        OmniRequest(None),
        NemotronVoiceChatState(waveform=waveform, num_frames=frame_count).to_dict(),
    )
    return IncomingMessage(request_id, "new_request", payload)


class RecordingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.failure: RuntimeError | None = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.should_block = False

    def validate(self, payload: StagePayload) -> ValidatedPerceptionRequest:
        state = NemotronVoiceChatState.from_dict(payload.data)
        waveform = state.waveform
        if waveform is None or waveform.numel() != state.num_frames * 1280:
            raise ValueError("waveform frame count mismatch")
        else:
            return ValidatedPerceptionRequest(
                payload=payload, state=state, waveform=waveform
            )

    def encode_validated(
        self, requests: list[ValidatedPerceptionRequest]
    ) -> list[StagePayload]:
        self.calls.append([request.payload.request_id for request in requests])
        self.started.set()
        if self.should_block:
            assert self.release.wait(timeout=5)
        else:
            pass
        if self.failure is not None:
            raise self.failure
        else:
            pass
        results: list[StagePayload] = []
        for request in requests:
            request.state.acoustic_frames = torch.full(
                (request.state.num_frames + 1, 8), float(request.waveform.mean())
            )
            request.payload.data = request.state.to_dict()
            results.append(request.payload)
        return results

    def __call__(self, payload: StagePayload) -> StagePayload:
        return self.encode_validated([self.validate(payload)])[0]


def scheduler(
    encoder: RecordingEncoder,
    batch_size: int = 4,
    wait_ms: float = 0,
    frame_budget: int = 4096,
    padding_ratio: float = 1.25,
) -> PerceptionScheduler:
    return PerceptionScheduler(
        encoder,
        max_batch_size=batch_size,
        max_batch_wait_ms=wait_ms,
        max_batch_frames=frame_budget,
        max_padding_ratio=padding_ratio,
    )


def outcomes(executor: PerceptionScheduler) -> list[OutgoingMessage]:
    messages: list[OutgoingMessage] = []
    while True:
        try:
            messages.append(executor.outbox.get_nowait())
        except Empty:
            return messages


def run_cohort(executor: PerceptionScheduler, cohort: list[IncomingMessage]) -> None:
    loop = asyncio.new_event_loop()
    try:
        executor.run_batch(cohort, loop)
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("frame_counts", "budget", "ratio", "accepted"),
    [
        ((4, 5, 7), 4096, 1.25, ["anchor", "candidate-0"]),
        ((4, 4, 4), 10, 1.25, ["anchor", "candidate-0"]),
        ((4, 5, 7), 4096, 1.5, ["anchor", "candidate-0", "candidate-1"]),
        ((3, 4, 3), 4096, 1.25, ["anchor", "candidate-1"]),
        ((0, 1, 0), 4096, 1.25, ["anchor", "candidate-1"]),
    ],
)
def test_collector_public_length_and_work_boundaries(
    frame_counts: tuple[int, ...], budget: int, ratio: float, accepted: list[str]
) -> None:
    executor = scheduler(RecordingEncoder(), frame_budget=budget, padding_ratio=ratio)
    anchor = message("anchor", frame_counts[0])
    for index, frame_count in enumerate(frame_counts[1:]):
        executor.enqueue(message(f"candidate-{index}", frame_count))
    cohort = executor.collect_batch(anchor)
    assert [request.request_id for request in cohort] == accepted
    remaining: list[str] = []
    while executor.pending_messages or not executor.inbox.empty():
        remaining.append(executor.next_message().request_id)
    assert set(accepted + remaining) == {"anchor", "candidate-0", "candidate-1"}


def test_pending_prefix_then_inbox_restores_fifo_and_barrier() -> None:
    executor = scheduler(RecordingEncoder())
    executor.pending_messages = deque(
        [
            message("pending-long", 20),
            message("pending-match"),
            message("pending-tail", 20),
        ]
    )
    malformed = message("malformed")
    malformed.data.data["num_frames"] = True
    executor.enqueue(message("inbox-match"))
    executor.enqueue(malformed)
    executor.enqueue(message("unvisited"))
    cohort = executor.collect_batch(message("anchor"))
    assert [request.request_id for request in cohort] == [
        "anchor",
        "pending-match",
        "inbox-match",
    ]
    assert [executor.next_message().request_id for index in range(4)] == [
        "pending-long",
        "pending-tail",
        "malformed",
        "unvisited",
    ]


def test_control_barrier_preserves_control_and_unvisited_requests() -> None:
    executor = scheduler(RecordingEncoder())
    executor.enqueue(message("peer"))
    executor.enqueue(IncomingMessage("control", "stream_done"))
    executor.enqueue(message("later"))
    assert [
        request.request_id for request in executor.collect_batch(message("anchor"))
    ] == ["anchor", "peer"]
    assert executor.next_message().request_id == "control"
    assert executor.next_message().request_id == "later"


def test_scan_bound_consumes_cancelled_and_incompatible_candidates() -> None:
    executor = scheduler(RecordingEncoder(), batch_size=2)
    for index in range(20):
        executor.enqueue(message(f"long-{index}", 20))
    assert len(executor.collect_batch(message("anchor"))) == 1
    assert executor.inbox.qsize() == 12
    assert [executor.next_message().request_id for index in range(20)] == [
        f"long-{index}" for index in range(20)
    ]


@pytest.mark.parametrize("anchor_frames", [0, 4, 9000])
def test_no_idle_wait_or_consumption_for_singleton_exemptions(
    anchor_frames: int,
) -> None:
    executor = scheduler(RecordingEncoder(), wait_ms=1000, frame_budget=10)
    anchor = message("anchor", anchor_frames)
    started = time.monotonic()
    assert executor.collect_batch(anchor) == [anchor]
    assert time.monotonic() - started < 0.1


def test_deadline_starts_only_after_second_compatible_arrival() -> None:
    executor = scheduler(RecordingEncoder(), wait_ms=25)
    executor.enqueue(message("peer"))
    started = time.monotonic()
    assert len(executor.collect_batch(message("anchor"))) == 2
    elapsed = time.monotonic() - started
    assert 0.018 <= elapsed < 0.15


def test_malformed_request_isolation_and_original_ids() -> None:
    encoder = RecordingEncoder()
    executor = scheduler(encoder)
    invalid = message("invalid")
    invalid.data.data["num_frames"] = 1
    run_cohort(executor, [message("valid-a"), invalid, message("valid-b")])
    published = outcomes(executor)
    assert [(result.request_id, result.type) for result in published] == [
        ("valid-a", "result"),
        ("invalid", "error"),
        ("valid-b", "result"),
    ]
    assert encoder.calls == [["valid-a", "valid-b"]]


@pytest.mark.parametrize("cancel_phase", ["before", "during"])
def test_cancellation_never_publishes_and_survivors_keep_ids(cancel_phase: str) -> None:
    encoder = RecordingEncoder()
    executor = scheduler(encoder)
    cohort = [message("cancel"), message("live")]
    if cancel_phase == "before":
        executor.abort("cancel")
        run_cohort(executor, cohort)
    else:
        encoder.should_block = True
        worker = threading.Thread(target=run_cohort, args=(executor, cohort))
        worker.start()
        assert encoder.started.wait(timeout=2)
        executor.abort("cancel")
        encoder.release.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert [(result.request_id, result.type) for result in outcomes(executor)] == [
        ("live", "result")
    ]


def test_common_inference_failure_is_never_success() -> None:
    encoder = RecordingEncoder()
    encoder.failure = RuntimeError("ordinary inference failure")
    executor = scheduler(encoder)
    run_cohort(executor, [message("first"), message("second")])
    assert [(result.request_id, result.type) for result in outcomes(executor)] == [
        ("first", "error"),
        ("second", "error"),
    ]


def test_fatal_cuda_error_escapes_start_and_stops_later_work() -> None:
    encoder = RecordingEncoder()
    encoder.failure = RuntimeError(
        "CUDA error: an illegal memory access was encountered"
    )
    executor = scheduler(encoder, batch_size=2)
    executor.enqueue(message("first"))
    executor.enqueue(message("second"))
    executor.enqueue(message("later"))
    with pytest.raises(RuntimeError, match="illegal memory access"):
        executor.start()
    assert not executor.running
    assert encoder.calls == [["first", "second"]]
    assert outcomes(executor) == []


@torch.inference_mode()
def test_real_factory_inbox_encodes_compatible_requests_together(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: torch.nn.Module,
    execution_device: torch.device,
) -> None:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    executor = stages.create_perception_executor(
        "test", device=execution_device.type, dtype="float32", max_batch_size=4
    )
    incoming = [message("original-a"), message("original-b", 5)]
    expected = [
        perception_model(
            NemotronVoiceChatState.from_dict(request.data.data)
            .waveform.to(execution_device)
            .unsqueeze(0)
        )[0].cpu()
        for request in incoming
    ]
    for request in incoming:
        executor.enqueue(request)
    worker = threading.Thread(target=executor.start)
    worker.start()
    try:
        published = [executor.outbox.get(timeout=15) for index in range(2)]
    finally:
        executor.stop()
        worker.join(timeout=2)
    assert [result.request_id for result in published] == ["original-a", "original-b"]
    assert all(result.type == "result" for result in published)
    assert executor.encoder.last_effective_batch_size == 2
    for result, reference in zip(published, expected):
        actual = NemotronVoiceChatState.from_dict(
            result.data.data
        ).acoustic_frames.cpu()
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


def test_stage_fatal_failure_terminates_all_active_original_ids() -> None:
    async def execute() -> None:
        encoder = RecordingEncoder()
        encoder.failure = RuntimeError("CUDA error: device-side assert triggered")
        executor = scheduler(encoder, batch_size=2)
        control_plane = RecordingStageControlPlane()
        stage = make_stage(scheduler=executor, control_plane=control_plane)
        stage.active_requests.update(("first", "second", "queued"))
        for request_id in ("first", "second", "queued"):
            executor.enqueue(message(request_id))
        await stage.start()
        try:
            for index in range(100):
                if stage.scheduler_crash_error is not None:
                    break
                else:
                    await asyncio.sleep(0.01)
            assert stage.scheduler_crash_error is not None
            assert not stage.running
            assert {
                completion.request_id for completion in control_plane.completions
            } == {"first", "second", "queued"}
            assert all(
                not completion.success for completion in control_plane.completions
            )
            assert encoder.calls == [["first", "second"]]
            assert outcomes(executor) == []
        finally:
            await stage.stop()

    asyncio.run(execute())


def test_stage_filters_result_published_after_abort_check() -> None:
    async def execute() -> None:
        executor = scheduler(RecordingEncoder())
        control_plane = RecordingStageControlPlane()
        stage = make_stage(scheduler=executor, control_plane=control_plane)
        stage.active_requests.add("cancelled")
        stage.on_abort("cancelled")
        executor.outbox.put(
            OutgoingMessage("cancelled", "result", message("cancelled").data)
        )
        await stage.drain_outbox()
        assert control_plane.sent_to_stage == []
        assert control_plane.completions == []

    asyncio.run(execute())


@pytest.mark.parametrize("bad_samples", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_waveform_has_own_error_and_valid_peer_runs(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: torch.nn.Module,
    execution_device: torch.device,
    bad_samples: float,
) -> None:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    executor = stages.create_perception_executor(
        "test", device=execution_device.type, dtype="float32", max_batch_size=4
    )
    invalid = message("invalid")
    invalid.data.data = NemotronVoiceChatState(
        waveform=torch.full((5120,), bad_samples), num_frames=4
    ).to_dict()
    run_cohort(executor, [invalid, message("valid")])
    assert [(result.request_id, result.type) for result in outcomes(executor)] == [
        ("invalid", "error"),
        ("valid", "result"),
    ]


@pytest.mark.parametrize("cancel_all", [False, True])
def test_observed_cancellation_is_consumed_without_inference(cancel_all: bool) -> None:
    encoder = RecordingEncoder()
    executor = scheduler(encoder)
    cohort = [message("first"), message("second")]
    executor.abort("first")
    if cancel_all:
        executor.abort("second")
    else:
        pass
    run_cohort(executor, cohort)
    if cancel_all:
        assert encoder.calls == []
        assert outcomes(executor) == []
    else:
        assert encoder.calls == [["second"]]
        assert [result.request_id for result in outcomes(executor)] == ["second"]


def test_collector_exception_restores_every_consumed_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = scheduler(RecordingEncoder())
    executor.pending_messages.append(message("deferred", 20))
    peer = message("accepted")
    monkeypatch.setattr(
        executor.inbox,
        "get_nowait",
        Mock(side_effect=[peer, RuntimeError("queue failure")]),
    )
    with pytest.raises(RuntimeError, match="queue failure"):
        executor.collect_batch(message("anchor"))
    assert [executor.next_message().request_id for index in range(2)] == [
        "deferred",
        "accepted",
    ]


def test_cancelled_candidates_count_toward_scan_limit() -> None:
    executor = scheduler(RecordingEncoder(), batch_size=2)
    for index in range(12):
        request_id = f"cancel-{index}"
        executor.enqueue(message(request_id))
        executor.abort(request_id)
    assert len(executor.collect_batch(message("anchor"))) == 1
    assert executor.inbox.qsize() == 4
    assert executor.pending_messages == deque()


@torch.inference_mode()
def test_serialization_failure_stages_errors_before_any_publication(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: torch.nn.Module,
    execution_device: torch.device,
) -> None:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    executor = stages.create_perception_executor(
        "test", device=execution_device.type, dtype="float32", max_batch_size=4
    )
    cohort = [message("first", 4), message("second", 5)]
    original_serialize = NemotronVoiceChatState.to_dict

    def serialize(state: NemotronVoiceChatState) -> dict:
        if state.acoustic_frames is not None and state.num_frames == 5:
            raise RuntimeError("serialization failed")
        else:
            return original_serialize(state)

    monkeypatch.setattr(NemotronVoiceChatState, "to_dict", serialize)
    run_cohort(executor, cohort)
    assert [(result.request_id, result.type) for result in outcomes(executor)] == [
        ("first", "error"),
        ("second", "error"),
    ]


@torch.inference_mode()
def test_legacy_state_gets_fresh_features_and_preserves_unknown_metadata(
    monkeypatch: pytest.MonkeyPatch,
    perception_model: torch.nn.Module,
    execution_device: torch.device,
) -> None:
    monkeypatch.setattr(stages, "perception_config", Mock(return_value={}))
    monkeypatch.setattr(stages, "AudioPerception", Mock(return_value=perception_model))
    monkeypatch.setattr(stages, "load_module", Mock())
    executor = stages.create_perception_executor(
        "test", device=execution_device.type, dtype="float32", max_batch_size=4
    )
    waveform = torch.randn(5120) / 100
    payload = StagePayload(
        "legacy",
        OmniRequest(None),
        {
            "waveform": waveform,
            "acoustic_frames": torch.full((5, 8), 999.0),
            "num_frames": 4,
            "custom_metadata": "preserved",
        },
    )
    actual = executor.fn(payload)
    state = NemotronVoiceChatState.from_dict(actual.data)
    expected = perception_model(waveform.to(execution_device).unsqueeze(0))[0]
    torch.testing.assert_close(
        state.acoustic_frames.to(execution_device), expected, rtol=0, atol=0
    )
    assert actual.data["custom_metadata"] == "preserved"
