# SPDX-License-Identifier: Apache-2.0
"""CUDA graph capture for fixed-shape VoiceChat kernels."""

import logging
import weakref
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

import torch

try:
    from cuda.bindings import runtime as cuda_runtime
except ImportError:
    cuda_runtime = None
else:
    pass

logger = logging.getLogger(__name__)

CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_ASSERT = 710
CUDA_ERROR_HARDWARE_STACK = 714
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_MISALIGNED_ADDRESS = 716
CUDA_ERROR_INVALID_PROGRAM_COUNTER = 718
CUDA_ERROR_LAUNCH_FAILURE = 719


def is_fatal_cuda_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "device-side assert",
            "illegal memory access",
            "hardware stack error",
            "illegal instruction",
            "misaligned address",
            "invalid program counter",
            "unspecified launch failure",
        )
    )


def check_cuda_status(status: int) -> None:
    if status == 0:
        return
    elif status == CUDA_ERROR_ILLEGAL_ADDRESS:
        raise RuntimeError("CUDA illegal memory access")
    elif status == CUDA_ERROR_ASSERT:
        raise RuntimeError("CUDA device-side assert")
    elif status == CUDA_ERROR_HARDWARE_STACK:
        raise RuntimeError("CUDA hardware stack error")
    elif status == CUDA_ERROR_ILLEGAL_INSTRUCTION:
        raise RuntimeError("CUDA illegal instruction")
    elif status == CUDA_ERROR_MISALIGNED_ADDRESS:
        raise RuntimeError("CUDA misaligned address")
    elif status == CUDA_ERROR_INVALID_PROGRAM_COUNTER:
        raise RuntimeError("CUDA invalid program counter")
    elif status == CUDA_ERROR_LAUNCH_FAILURE:
        raise RuntimeError("CUDA unspecified launch failure")
    else:
        raise RuntimeError(f"CUDA capture stream runtime status {status}")


@dataclass(kw_only=True)
class GraphExecutionState:
    is_abandoned: bool = False


class CaptureDeviceGuard:
    """Restore the previous device only while CUDA execution remains usable."""

    def __init__(
        self, device: torch.device, execution_state: GraphExecutionState
    ) -> None:
        self.device_guard: torch.cuda.device = torch.cuda.device(device)
        self.execution_state: GraphExecutionState = execution_state

    def __enter__(self) -> None:
        try:
            self.device_guard.__enter__()
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.execution_state.is_abandoned = True
            else:
                pass
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if isinstance(error, Exception) and is_fatal_cuda_error(error):
            self.execution_state.is_abandoned = True
        else:
            pass
        if self.execution_state.is_abandoned:
            return
        else:
            try:
                self.device_guard.__exit__(exception_type, error, traceback)
            except Exception as restore_error:
                if is_fatal_cuda_error(restore_error):
                    self.execution_state.is_abandoned = True
                else:
                    pass
                raise


@dataclass(kw_only=True)
class CapturedStepState:
    device: torch.device
    execution_state: GraphExecutionState
    raw_stream_handle: int | None = None
    capture_stream: torch.cuda.Stream | None = None
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None
    is_closed: bool = False

    def abandon(self) -> None:
        self.execution_state.is_abandoned = True
        logger.debug(f"Capture stream abandoned handle={self.raw_stream_handle}")

    def close(self) -> None:
        if self.execution_state.is_abandoned or self.is_closed:
            return
        else:
            pass
        try:
            with CaptureDeviceGuard(self.device, self.execution_state):
                self.output = None
                if self.graph is not None:
                    self.graph.reset()
                    self.graph = None
                else:
                    pass
                self.capture_stream = None
                if self.raw_stream_handle is not None:
                    assert cuda_runtime is not None
                    (status,) = cuda_runtime.cudaStreamDestroy(self.raw_stream_handle)
                    check_cuda_status(int(status))
                    logger.debug(
                        f"Capture stream destroyed handle={self.raw_stream_handle}"
                    )
                    self.raw_stream_handle = None
                else:
                    pass
                self.is_closed = True
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise


class CapturedStep:
    """Own a captured output, graph, and independent native capture stream."""

    def __init__(
        self,
        device: torch.device,
        execution_state: GraphExecutionState,
        *,
        allow_pooled_fallback: bool = False,
    ) -> None:
        self.state: CapturedStepState = CapturedStepState(
            device=device, execution_state=execution_state
        )
        self.finalizer: weakref.finalize = weakref.finalize(self, self.state.close)
        self.finalizer.atexit = False
        if execution_state.is_abandoned:
            raise RuntimeError("Perception CUDA execution was abandoned")
        else:
            pass
        try:
            with CaptureDeviceGuard(device, execution_state):
                if torch.version.hip is not None and allow_pooled_fallback:
                    self.state.capture_stream = torch.cuda.Stream(device=device)
                elif cuda_runtime is None or torch.version.hip is not None:
                    raise RuntimeError(
                        "CUDA graph capture not supported without NVIDIA runtime bindings"
                    )
                else:
                    status, raw_stream_handle = cuda_runtime.cudaStreamCreateWithFlags(
                        cuda_runtime.cudaStreamNonBlocking
                    )
                    check_cuda_status(int(status))
                    self.state.raw_stream_handle = int(raw_stream_handle)
                    logger.debug(
                        f"Capture stream created handle={self.state.raw_stream_handle}"
                    )
                    self.state.capture_stream = torch.cuda.ExternalStream(
                        self.state.raw_stream_handle, device=device
                    )
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                self.close()
            raise

    @property
    def capture_stream(self) -> torch.cuda.Stream:
        assert self.state.capture_stream is not None
        return self.state.capture_stream

    @property
    def graph(self) -> torch.cuda.CUDAGraph | None:
        return self.state.graph

    @property
    def output(self) -> torch.Tensor | None:
        return self.state.output

    def wait_for_capture(self, current_stream: torch.cuda.Stream) -> None:
        if self.state.execution_state.is_abandoned:
            raise RuntimeError("Perception CUDA execution was abandoned")
        else:
            pass
        try:
            current_stream.wait_stream(self.capture_stream)
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise

    def close(self) -> None:
        self.state.close()
        if self.state.is_closed:
            self.finalizer.detach()
        else:
            pass

    def abandon(self) -> None:
        self.state.abandon()


class GraphForward(Protocol):
    def __call__(self) -> torch.Tensor: ...


@torch.inference_mode()
def capture_cuda_graph(
    forward: GraphForward,
    device: torch.device,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = forward()
    return graph, output
