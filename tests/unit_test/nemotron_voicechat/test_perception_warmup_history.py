"""Bounded warmup qualification follows capture evidence and execution settings."""

import asyncio
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.conformer import StreamingPerception
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.perception_encoder import PerceptionEncoder
from sglang_omni.models.nemotron_voicechat.perception_scheduler import (
    PerceptionScheduler,
)
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.nemotron_voicechat.perception_warmup_fakes import (
    CaptureOutcome,
    requests,
    run_size,
    spy_encoder,
)


def test_fingerprint_change_a_b_a_forces_cold_after_old_graph_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    precision = Mock(return_value="highest")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", precision)
    for setting in ("highest", "high", "highest"):
        precision.return_value = setting
        run_size(encoder, 2)
        encoder.batch_runners[2].close()
        encoder.batch_runners.clear()
    assert [runner.warmup_steps for runner in created] == [3, 3, 3]
    assert all(runner.close_count == 1 for runner in created)


def test_successful_recapture_uses_one_and_new_encoder_is_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    for batch_size in (2, 3, 2):
        run_size(encoder, batch_size)
    assert [runner.warmup_steps for runner in created] == [3, 3, 1]
    other = PerceptionEncoder(encoder.model, Mock(spec=StreamingPerception), True, 1)
    run_size(other, 2)
    assert created[-1].warmup_steps == 3


def test_history_forgets_lru_shapes_after_more_than_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    for batch_size in range(2, 11):
        run_size(encoder, batch_size)
        assert len(encoder.warmed_batch_sizes) <= 8
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 3
    run_size(encoder, 10)
    assert created[-1].warmup_steps == 1


@pytest.mark.parametrize("outcome", ["empty-graph", "no-delta", "ordinary"])
def test_without_real_successful_capture_does_not_qualify(
    monkeypatch: pytest.MonkeyPatch, outcome: CaptureOutcome
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    run_size(encoder, 2, 0)
    created[-1].outcome = outcome
    if outcome == "ordinary":
        with pytest.raises(RuntimeError, match="ordinary inference failure"):
            run_size(encoder, 2)
    else:
        run_size(encoder, 2)
    assert not encoder.warmed_batch_sizes
    run_size(encoder, 3)
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 3


def test_zero_retained_first_real_capture_qualifies_and_refreshes_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    precision = Mock(return_value="highest")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", precision)
    run_size(encoder, 2, 0)
    assert not encoder.warmed_batch_sizes
    precision.return_value = "high"
    run_size(encoder, 2)
    run_size(encoder, 3)
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 1


def test_real_capture_qualification_survives_followup_ordinary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    run_size(encoder, 2, 0)
    created[-1].outcome = "capture-error"
    with pytest.raises(RuntimeError, match="after successful capture"):
        run_size(encoder, 2)
    run_size(encoder, 3)
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 1


def test_fallback_clears_history_and_fatal_adds_no_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch, capacity=2)
    run_size(encoder, 2)
    run_size(encoder, 3, 0)
    created[-1].outcome = "fallback"
    run_size(encoder, 3)
    assert not encoder.warmed_batch_sizes
    assert not encoder.enable_cuda_graph
    fatal_encoder, fatal_created = spy_encoder(monkeypatch)
    run_size(fatal_encoder, 2, 0)
    fatal_created[-1].outcome = "fatal"
    close_count = fatal_created[-1].close_count
    with pytest.raises(RuntimeError, match="illegal memory access"):
        run_size(fatal_encoder, 2)
    fatal_encoder.close()
    assert fatal_created[-1].close_count == close_count
    assert not fatal_encoder.warmed_batch_sizes


@pytest.mark.parametrize("device, enabled", [("cpu", True), ("cuda:0", False)])
def test_cpu_and_graph_disabled_never_create_warm_qualification(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    enabled: bool,
) -> None:
    encoder, created = spy_encoder(
        monkeypatch, device=device, enable_cuda_graph=enabled
    )
    for batch_size in (2, 3, 2):
        run_size(encoder, batch_size)
    assert not encoder.warmed_batch_sizes
    assert [runner.warmup_steps for runner in created] == [3, 3, 3]


@pytest.mark.parametrize(
    "setting", ["matmul_tf32", "cudnn_tf32", "benchmark", "deterministic"]
)
def test_each_execution_flag_a_b_a_requires_three_after_release(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    if setting == "matmul_tf32":
        target = torch.backends.cuda.matmul
        field = "allow_tf32"
    elif setting == "cudnn_tf32":
        target = torch.backends.cudnn
        field = "allow_tf32"
    else:
        target = torch.backends.cudnn
        field = setting
    for value in (False, True, False):
        monkeypatch.setattr(target, field, value)
        run_size(encoder, 2)
        encoder.batch_runners[2].close()
        encoder.batch_runners.clear()
    assert [runner.warmup_steps for runner in created] == [3, 3, 3]


def test_hot_graph_and_zero_input_skip_fingerprint_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    precision = Mock(return_value="highest")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", precision)
    run_size(encoder, 2, 0)
    assert precision.call_count == 0
    run_size(encoder, 2)
    assert precision.call_count == 1
    run_size(encoder, 2, 3)
    assert precision.call_count == 1
    assert created[0].graph_capture_count == 1


def test_zero_retained_stale_history_is_rechecked_before_first_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    precision = Mock(return_value="highest")
    monkeypatch.setattr(torch, "get_float32_matmul_precision", precision)
    run_size(encoder, 2)
    run_size(encoder, 3)
    run_size(encoder, 2, 0)
    precision.return_value = "high"
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 3
    run_size(encoder, 3)
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 1


def test_capture_marker_survives_ordinary_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    validated = requests(encoder, 2)
    with monkeypatch.context() as failure_patch:
        failure_patch.setattr(
            NemotronVoiceChatState,
            "to_dict",
            Mock(side_effect=RuntimeError("ordinary serialization failure")),
        )
        with pytest.raises(RuntimeError, match="ordinary serialization failure"):
            encoder.encode_validated(validated)
    run_size(encoder, 3)
    run_size(encoder, 2)
    assert created[-1].warmup_steps == 1


def test_fatal_after_actual_capture_does_not_add_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    run_size(encoder, 2, 0)
    created[-1].outcome = "fatal-capture"
    with pytest.raises(RuntimeError, match="illegal memory access"):
        run_size(encoder, 2)
    assert created[-1].graph_capture_count == 1
    assert created[-1].graph is not None
    assert not encoder.warmed_batch_sizes
    encoder.close()
    assert created[-1].close_count == 0


def test_successful_recapture_promotes_history_before_forgetting_other_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    for batch_size in range(2, 10):
        run_size(encoder, batch_size)
    for batch_size in (2, 10, 2, 3):
        run_size(encoder, batch_size)
    assert [runner.warmup_steps for runner in created[-4:]] == [1, 3, 1, 3]
    assert len(encoder.warmed_batch_sizes) == 8


def test_scheduler_history_uses_actual_valid_survivor_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder, created = spy_encoder(monkeypatch)
    scheduler = PerceptionScheduler(
        encoder,
        max_batch_size=8,
        max_batch_wait_ms=0.0,
        max_batch_frames=4096,
        max_padding_ratio=1.25,
    )
    incoming = [
        IncomingMessage(request.payload.request_id, "new_request", request.payload)
        for request in requests(encoder, 3)
    ]
    incoming[1].data.data["num_frames"] = True
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_batch(incoming, loop)
    finally:
        loop.close()
    assert created[-1].batch_size == 2
    assert 2 in encoder.warmed_batch_sizes and 3 not in encoder.warmed_batch_sizes
    results = [scheduler.outbox.get_nowait() for index in range(3)]
    assert [(result.request_id, result.type) for result in results] == [
        ("row-0", "result"),
        ("row-1", "error"),
        ("row-2", "result"),
    ]
