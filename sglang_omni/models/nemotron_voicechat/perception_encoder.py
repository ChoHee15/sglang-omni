"""Offline Perception encoding with bounded exact-size CUDA graph reuse."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields

import torch

from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    GRAPH_WARMUP_STEPS,
    SAMPLES_PER_FRAME,
    AudioPerception,
    GraphPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import (
    GraphExecutionState,
    is_fatal_cuda_error,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.proto.request import StagePayload

MAX_WARMUP_HISTORY_BATCH_SIZES = 8
RECAPTURE_WARMUP_STEPS = 1


@dataclass(kw_only=True, frozen=True)
class CaptureFingerprint:
    encoder_identity: int
    model_identity: int
    resolved_device: str
    model_dtype: str
    matmul_precision: str
    matmul_tf32: bool
    cudnn_tf32: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool


@dataclass(kw_only=True)
class ValidatedPerceptionRequest:
    payload: StagePayload
    state: NemotronVoiceChatState
    waveform: torch.Tensor


class PerceptionEncoder:
    """Own a singleton stream and bounded exact-size batch runners."""

    def __init__(
        self,
        model: AudioPerception,
        stream: StreamingPerception,
        enable_cuda_graph: bool,
        max_cached_batch_runners: int,
    ) -> None:
        if (
            isinstance(max_cached_batch_runners, bool)
            or not isinstance(max_cached_batch_runners, int)
            or max_cached_batch_runners < 1
        ):
            raise ValueError("max_cached_batch_runners must be a positive integer")
        else:
            pass
        self.model: AudioPerception = model
        self.stream: StreamingPerception = stream
        self.enable_cuda_graph: bool = enable_cuda_graph
        self.max_cached_batch_runners: int = max_cached_batch_runners
        self.batch_runners: OrderedDict[int, BatchedPerceptionRunner] = OrderedDict()
        self.warmed_batch_sizes: OrderedDict[int, CaptureFingerprint] = OrderedDict()
        self.graph_capture_count: int = 0
        self.last_effective_batch_size: int = 0
        self.graph_execution_state: GraphExecutionState = (
            stream.graph_execution_state
            if isinstance(stream, GraphPerception)
            else GraphExecutionState()
        )
        self.is_closed: bool = False

    def abandon(self) -> None:
        self.graph_execution_state.is_abandoned = True

    def close(self) -> None:
        self.warmed_batch_sizes.clear()
        if self.graph_execution_state.is_abandoned or self.is_closed:
            return
        else:
            pass
        try:
            for runner in self.batch_runners.values():
                runner.close()
            if isinstance(self.stream, GraphPerception):
                self.stream.close()
            else:
                pass
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise
        else:
            self.batch_runners.clear()
            self.is_closed = True

    def disable_batch_graphs(self, runner: BatchedPerceptionRunner) -> None:
        self.enable_cuda_graph = False
        self.warmed_batch_sizes.clear()
        for cached_runner in self.batch_runners.values():
            if cached_runner is not runner:
                cached_runner.close()
            else:
                pass
        self.batch_runners.clear()
        self.batch_runners[runner.batch_size] = runner

    def remember_successful_capture(
        self,
        runner: BatchedPerceptionRunner,
        previous_capture_count: int,
        fingerprint: CaptureFingerprint | None,
    ) -> None:
        if (
            fingerprint is None
            or runner.graph_capture_count - previous_capture_count != 1
            or runner.graph is None
            or not runner.enable_cuda_graph
            or self.graph_execution_state.is_abandoned
        ):
            return
        else:
            pass
        if runner.batch_size not in self.warmed_batch_sizes:
            while len(self.warmed_batch_sizes) >= MAX_WARMUP_HISTORY_BATCH_SIZES:
                self.warmed_batch_sizes.popitem(last=False)
        else:
            pass
        self.warmed_batch_sizes[runner.batch_size] = fingerprint
        self.warmed_batch_sizes.move_to_end(runner.batch_size)

    def validate(self, payload: StagePayload) -> ValidatedPerceptionRequest:
        if self.graph_execution_state.is_abandoned or self.is_closed:
            raise RuntimeError("Perception execution is closed or abandoned")
        else:
            pass
        try:
            return self.validate_request(payload)
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise

    def validate_request(self, payload: StagePayload) -> ValidatedPerceptionRequest:
        if not isinstance(payload.data, dict):
            raise ValueError("Perception payload must be a dictionary")
        else:
            pass
        frame_count = payload.data.get("num_frames")
        if (
            not isinstance(frame_count, int)
            or isinstance(frame_count, bool)
            or frame_count < 0
        ):
            raise ValueError("num_frames must be a nonnegative integer")
        else:
            pass
        state = NemotronVoiceChatState.from_dict(payload.data)
        waveform = state.waveform
        if not isinstance(waveform, torch.Tensor) or not waveform.is_floating_point():
            raise ValueError("Perception waveform must be a floating tensor")
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = waveform[0]
        elif waveform.ndim == 1:
            pass
        else:
            raise ValueError(
                "Perception waveform must have shape [samples] or [1, samples]"
            )
        if waveform.numel() != frame_count * SAMPLES_PER_FRAME:
            raise ValueError(
                "Perception waveform sample count must equal num_frames * 1280"
            )
        else:
            pass
        if not torch.isfinite(waveform).all().item():
            raise ValueError("Perception waveform must contain only finite samples")
        else:
            return ValidatedPerceptionRequest(
                payload=payload, state=state, waveform=waveform
            )

    @torch.inference_mode()
    def encode_validated(
        self, requests: list[ValidatedPerceptionRequest]
    ) -> list[StagePayload]:
        if self.graph_execution_state.is_abandoned or self.is_closed:
            raise RuntimeError("Perception execution is closed or abandoned")
        else:
            pass
        try:
            return self.encode_requests(requests)
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise

    def encode_requests(
        self, requests: list[ValidatedPerceptionRequest]
    ) -> list[StagePayload]:
        self.last_effective_batch_size = len(requests)
        if len(requests) == 1:
            request = requests[0]
            frames = self.model(
                request.waveform[None].to(
                    device=self.model.proj.weight.device,
                    dtype=self.model.proj.weight.dtype,
                ),
                stream=self.stream,
            )
            outputs = [frames[0]]
        else:
            batch_size = len(requests)
            if batch_size in self.batch_runners:
                self.batch_runners.move_to_end(batch_size)
                runner = self.batch_runners[batch_size]
            else:
                cache_capacity = (
                    self.max_cached_batch_runners
                    if self.enable_cuda_graph
                    and self.model.proj.weight.device.type == "cuda"
                    else 1
                )
                while len(self.batch_runners) >= cache_capacity:
                    next(iter(self.batch_runners.values())).close()
                    self.batch_runners.popitem(last=False)
                runner = BatchedPerceptionRunner(
                    self.model,
                    batch_size,
                    self.enable_cuda_graph,
                    self.graph_execution_state,
                )
                self.batch_runners[batch_size] = runner
            capture_fingerprint: CaptureFingerprint | None = None
            if (
                runner.graph is None
                and runner.enable_cuda_graph
                and any(request.state.num_frames > 0 for request in requests)
            ):
                capture_fingerprint = CaptureFingerprint(
                    encoder_identity=id(self),
                    model_identity=id(self.model),
                    resolved_device=str(self.model.proj.weight.device),
                    model_dtype=str(self.model.proj.weight.dtype),
                    matmul_precision=torch.get_float32_matmul_precision(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                )
                previous_fingerprint = self.warmed_batch_sizes.get(batch_size)
                if previous_fingerprint == capture_fingerprint:
                    runner.warmup_steps = RECAPTURE_WARMUP_STEPS
                else:
                    runner.warmup_steps = GRAPH_WARMUP_STEPS
                    if previous_fingerprint is not None:
                        self.warmed_batch_sizes.pop(batch_size)
                    else:
                        pass
            else:
                pass
            previous_capture_count = runner.graph_capture_count
            try:
                outputs = runner.encode([request.waveform for request in requests])
            except Exception as error:
                if is_fatal_cuda_error(error):
                    raise
                elif not runner.enable_cuda_graph:
                    self.disable_batch_graphs(runner)
                else:
                    self.remember_successful_capture(
                        runner, previous_capture_count, capture_fingerprint
                    )
                raise
            else:
                if not runner.enable_cuda_graph:
                    self.disable_batch_graphs(runner)
                else:
                    self.remember_successful_capture(
                        runner, previous_capture_count, capture_fingerprint
                    )
            finally:
                self.graph_capture_count += (
                    runner.graph_capture_count - previous_capture_count
                )
        if len(outputs) != len(requests):
            raise ValueError("Perception result count differs from request count")
        else:
            pass
        state_field_names = {field.name for field in fields(NemotronVoiceChatState)}
        serialized_payloads: list[StagePayload] = []
        for request, frames in zip(requests, outputs):
            if frames.ndim != 2 or frames.shape[0] != request.state.num_frames + 1:
                raise ValueError(
                    "Perception output must contain num_frames + 1 acoustic rows"
                )
            else:
                pass
            request.state.acoustic_frames = frames
            serialized_payloads.append(
                StagePayload(
                    request_id=request.payload.request_id,
                    request=request.payload.request,
                    data={
                        **{
                            name: value
                            for name, value in request.payload.data.items()
                            if name not in state_field_names
                        },
                        **request.state.to_dict(),
                    },
                )
            )
        for request, serialized_payload in zip(requests, serialized_payloads):
            request.payload.data = serialized_payload.data
        return [request.payload for request in requests]

    def __call__(self, payload: StagePayload) -> StagePayload:
        return self.encode_validated([self.validate(payload)])[0]
