"""Native capture release ordering, status checks, and fatal boundaries."""

import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import cuda_graph
from sglang_omni.models.nemotron_voicechat.cuda_graph import (
    CapturedStep,
    GraphExecutionState,
    check_cuda_status,
    is_fatal_cuda_error,
)


@pytest.fixture
def native_runtime(monkeypatch: pytest.MonkeyPatch) -> Mock:
    runtime = Mock()
    runtime.cudaStreamNonBlocking = 1
    runtime.cudaStreamCreateWithFlags.return_value = (0, 123)
    runtime.cudaStreamDestroy.return_value = (0,)
    monkeypatch.setattr(cuda_graph, "cuda_runtime", runtime)
    monkeypatch.setattr(
        torch.cuda, "device", Mock(side_effect=lambda device: contextmanager_device())
    )
    monkeypatch.setattr(torch.cuda, "ExternalStream", Mock(return_value=Mock()))
    return runtime


@contextmanager
def contextmanager_device() -> Iterator[None]:
    yield


@pytest.mark.parametrize("release_method", ["explicit", "finalizer"])
def test_release_drops_output_before_graph_reset_before_native_destroy(
    native_runtime: Mock,
    release_method: str,
) -> None:
    events: list[str] = []
    owner = CapturedStep(torch.device("cuda:1"), GraphExecutionState())
    output = torch.ones(1)
    output_reference = weakref.ref(
        output, lambda reference: events.append("output-dropped")
    )
    owner.state.output = output
    output = None
    graph = Mock()
    graph_reference = weakref.ref(graph)

    def reset_graph() -> None:
        assert output_reference() is None
        events.append("graph-reset")

    def destroy_stream(stream_handle: int) -> tuple[int]:
        assert stream_handle == 123
        gc.collect()
        assert graph_reference() is None
        events.append("stream-destroy")
        return (0,)

    graph.reset.side_effect = reset_graph
    owner.state.graph = graph
    graph = None
    native_runtime.cudaStreamDestroy.side_effect = destroy_stream
    owner_reference = weakref.ref(owner)
    finalizer = owner.finalizer
    if release_method == "explicit":
        owner.close()
        owner.close()
        assert not finalizer.alive
    else:
        pass
    owner = None
    assert owner_reference() is None
    assert not finalizer.alive
    assert events == ["output-dropped", "graph-reset", "stream-destroy"]
    assert native_runtime.cudaStreamDestroy.call_count == 1
    gc.collect()
    assert native_runtime.cudaStreamDestroy.call_count == 1


def test_creation_and_release_use_resolved_device_guard(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
) -> None:
    guarded_device: torch.device | None = None
    visited_devices: list[torch.device] = []
    resolved_device = torch.device("cuda:1")

    @contextmanager
    def device_guard(device: torch.device) -> Iterator[None]:
        nonlocal guarded_device
        assert guarded_device is None
        guarded_device = device
        visited_devices.append(device)
        try:
            yield
        finally:
            guarded_device = None

    def create_stream(flags: int) -> tuple[int, int]:
        assert flags == 1
        assert guarded_device == resolved_device
        return (0, 123)

    def borrow_stream(stream_handle: int, device: torch.device) -> Mock:
        assert stream_handle == 123
        assert device == resolved_device
        assert guarded_device == resolved_device
        return Mock()

    def destroy_stream(stream_handle: int) -> tuple[int]:
        assert stream_handle == 123
        assert guarded_device == resolved_device
        return (0,)

    monkeypatch.setattr(torch.cuda, "device", device_guard)
    monkeypatch.setattr(torch.cuda, "ExternalStream", borrow_stream)
    native_runtime.cudaStreamCreateWithFlags.side_effect = create_stream
    native_runtime.cudaStreamDestroy.side_effect = destroy_stream
    owner = CapturedStep(resolved_device, GraphExecutionState())
    owner.close()
    assert visited_devices == [resolved_device, resolved_device]


def test_ordinary_external_stream_creation_error_releases_native_handle(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
) -> None:
    shared_state = GraphExecutionState()
    monkeypatch.setattr(
        torch.cuda,
        "ExternalStream",
        Mock(side_effect=RuntimeError("external wrapper failure")),
    )
    with pytest.raises(RuntimeError, match="external wrapper failure"):
        CapturedStep(torch.device("cuda:1"), shared_state)
    gc.collect()
    assert native_runtime.cudaStreamDestroy.call_count == 1
    assert not shared_state.is_abandoned


@pytest.mark.parametrize("status", [1, 2, 400])
def test_binding_create_status_error_is_checked_before_borrow_or_destroy(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
    status: int,
) -> None:
    native_runtime.cudaStreamCreateWithFlags.return_value = (status, 0)
    borrow = Mock()
    monkeypatch.setattr(torch.cuda, "ExternalStream", borrow)
    with pytest.raises(RuntimeError, match="runtime status"):
        CapturedStep(torch.device("cuda:1"), GraphExecutionState())
    assert borrow.call_count == 0
    assert native_runtime.cudaStreamDestroy.call_count == 0


@pytest.mark.parametrize(
    "status_name",
    [
        "cudaErrorIllegalAddress",
        "cudaErrorAssert",
        "cudaErrorMisalignedAddress",
        "cudaErrorHardwareStackError",
        "cudaErrorIllegalInstruction",
        "cudaErrorInvalidPc",
        "cudaErrorLaunchFailure",
    ],
)
def test_actual_binding_fatal_enums_normalize_to_fatal_exception(
    status_name: str,
) -> None:
    runtime = cuda_graph.cuda_runtime
    if runtime is None:
        pytest.skip("NVIDIA runtime binding enums are unavailable")
    else:
        pass
    status = {
        "cudaErrorIllegalAddress": runtime.cudaError_t.cudaErrorIllegalAddress,
        "cudaErrorAssert": runtime.cudaError_t.cudaErrorAssert,
        "cudaErrorMisalignedAddress": runtime.cudaError_t.cudaErrorMisalignedAddress,
        "cudaErrorHardwareStackError": runtime.cudaError_t.cudaErrorHardwareStackError,
        "cudaErrorIllegalInstruction": runtime.cudaError_t.cudaErrorIllegalInstruction,
        "cudaErrorInvalidPc": runtime.cudaError_t.cudaErrorInvalidPc,
        "cudaErrorLaunchFailure": runtime.cudaError_t.cudaErrorLaunchFailure,
    }[status_name]
    with pytest.raises(RuntimeError) as failure:
        check_cuda_status(status)
    assert is_fatal_cuda_error(failure.value)


@pytest.mark.parametrize("status", [700, 710, 714, 715, 716, 718, 719])
def test_fatal_creation_status_abandons_shared_execution_without_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
    status: int,
) -> None:
    shared_state = GraphExecutionState()
    first = CapturedStep(torch.device("cuda:1"), shared_state)
    first.state.graph = Mock()
    first.state.output = torch.ones(1)
    saved_graph = first.graph
    saved_output = first.output
    native_runtime.cudaStreamCreateWithFlags.return_value = (status, 0)
    guard = Mock(side_effect=lambda device: contextmanager_device())
    monkeypatch.setattr(torch.cuda, "device", guard)
    with pytest.raises(RuntimeError):
        CapturedStep(torch.device("cuda:1"), shared_state)
    assert shared_state.is_abandoned
    guard.reset_mock()
    first.close()
    first.finalizer()
    assert guard.call_count == 0
    assert native_runtime.cudaStreamDestroy.call_count == 0
    assert first.graph is saved_graph
    assert first.output is saved_output
    assert saved_graph.reset.call_count == 0


@pytest.mark.parametrize(
    ("fatal_boundary", "fatal_status"),
    [
        ("graph-reset", 700),
        *[("stream-destroy", status) for status in (700, 710, 714, 715, 716, 718, 719)],
    ],
)
def test_fatal_cleanup_stops_followup_holders_and_finalizers(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
    fatal_boundary: str,
    fatal_status: int,
) -> None:
    shared_state = GraphExecutionState()
    first = CapturedStep(torch.device("cuda:1"), shared_state)
    second = CapturedStep(torch.device("cuda:1"), shared_state)
    first.state.graph = Mock()
    second.state.graph = Mock()
    second.state.output = torch.ones(1)
    second_graph = second.graph
    second_output = second.output
    if fatal_boundary == "graph-reset":
        first.state.graph.reset.side_effect = RuntimeError("CUDA illegal memory access")
        expected_destroy_calls = 0
    else:
        native_runtime.cudaStreamDestroy.return_value = (fatal_status,)
        expected_destroy_calls = 1
    device_guard = MagicMock()
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=device_guard))
    with pytest.raises(RuntimeError):
        first.close()
    assert device_guard.__exit__.call_count == 0
    assert shared_state.is_abandoned
    guard = Mock(side_effect=AssertionError("device guard after fatal cleanup"))
    monkeypatch.setattr(torch.cuda, "device", guard)
    second.close()
    first.close()
    first.finalizer()
    second.finalizer()
    assert guard.call_count == 0
    assert native_runtime.cudaStreamDestroy.call_count == expected_destroy_calls
    assert second.graph is second_graph
    assert second.output is second_output
    assert second_graph.reset.call_count == 0


def test_binding_destroy_status_error_is_not_reported_as_success(
    native_runtime: Mock,
) -> None:
    owner = CapturedStep(torch.device("cuda:1"), GraphExecutionState())
    native_runtime.cudaStreamDestroy.return_value = (400,)
    with pytest.raises(RuntimeError, match="runtime status 400"):
        owner.close()
    assert not owner.state.is_closed
    assert owner.state.raw_stream_handle == 123
    native_runtime.cudaStreamDestroy.return_value = (0,)
    owner.close()
    assert owner.state.is_closed
    assert native_runtime.cudaStreamDestroy.call_count == 2


def test_fatal_capture_wait_preserves_resources_and_stops_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    native_runtime: Mock,
) -> None:
    owner = CapturedStep(torch.device("cuda:1"), GraphExecutionState())
    owner.state.graph = Mock()
    owner.state.output = torch.ones(1)
    saved_graph = owner.graph
    saved_output = owner.output
    current_stream = Mock()
    current_stream.wait_stream.side_effect = RuntimeError("CUDA misaligned address")
    with pytest.raises(RuntimeError, match="misaligned address"):
        owner.wait_for_capture(current_stream)
    monkeypatch.setattr(
        torch.cuda,
        "device",
        Mock(side_effect=AssertionError("device guard after fatal wait")),
    )
    owner.close()
    owner.finalizer()
    assert owner.state.execution_state.is_abandoned
    assert owner.graph is saved_graph
    assert owner.output is saved_output
    assert saved_graph.reset.call_count == 0
    assert native_runtime.cudaStreamDestroy.call_count == 0


@pytest.mark.parametrize(
    "failure_message", ["CUDNN_STATUS_EXECUTION_FAILED", "CUDA out of memory"]
)
def test_ordinary_wait_error_does_not_abandon_or_block_normal_release(
    native_runtime: Mock,
    failure_message: str,
) -> None:
    shared_state = GraphExecutionState()
    owner = CapturedStep(torch.device("cuda:1"), shared_state)
    current_stream = Mock()
    current_stream.wait_stream.side_effect = RuntimeError(failure_message)
    with pytest.raises(RuntimeError, match=failure_message):
        owner.wait_for_capture(current_stream)
    assert not shared_state.is_abandoned
    owner.close()
    assert owner.state.is_closed
    assert native_runtime.cudaStreamDestroy.call_count == 1
