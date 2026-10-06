"""Length-aware offline Perception scheduling and execution."""

from __future__ import annotations

import asyncio
import queue
import time
from dataclasses import dataclass, fields
from typing import Protocol

import torch

from sglang_omni.models.nemotron_voicechat.batched_perception import (
    BatchedPerceptionRunner,
    is_fatal_cuda_error,
)
from sglang_omni.models.nemotron_voicechat.conformer import (
    AudioPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

SAMPLES_PER_FRAME = 1_280
SCAN_MULTIPLIER = 4


@dataclass(kw_only=True)
class ValidatedPerceptionRequest:
    payload: StagePayload
    state: NemotronVoiceChatState
    waveform: torch.Tensor


class PerceptionCompute(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...

    def validate(self, payload: StagePayload) -> ValidatedPerceptionRequest: ...

    def encode_validated(
        self, requests: list[ValidatedPerceptionRequest]
    ) -> list[StagePayload]: ...


def message_frame_count(message: IncomingMessage) -> int | None:
    if message.type != "new_request" or not isinstance(message.data, StagePayload):
        return None
    elif not isinstance(message.data.data, dict):
        return None
    else:
        frame_count = message.data.data.get("num_frames")
        return (
            frame_count
            if isinstance(frame_count, int)
            and not isinstance(frame_count, bool)
            and frame_count >= 0
            else None
        )


class PerceptionEncoder:
    """Own one singleton stream and at most one exact-size batch runner."""

    def __init__(
        self,
        model: AudioPerception,
        stream: StreamingPerception,
        enable_cuda_graph: bool,
    ) -> None:
        self.model = model
        self.stream = stream
        self.enable_cuda_graph = enable_cuda_graph
        self.batch_runner: BatchedPerceptionRunner | None = None
        self.graph_capture_count = 0
        self.last_effective_batch_size = 0

    def validate(self, payload: StagePayload) -> ValidatedPerceptionRequest:
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
            if self.batch_runner is None or self.batch_runner.batch_size != len(
                requests
            ):
                self.batch_runner = None
                self.batch_runner = BatchedPerceptionRunner(
                    self.model, len(requests), self.enable_cuda_graph
                )
            else:
                pass
            previous_capture_count = self.batch_runner.graph_capture_count
            try:
                outputs = self.batch_runner.encode(
                    [request.waveform for request in requests]
                )
            finally:
                if not self.batch_runner.enable_cuda_graph:
                    self.enable_cuda_graph = False
                else:
                    pass
                self.graph_capture_count += (
                    self.batch_runner.graph_capture_count - previous_capture_count
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


class PerceptionScheduler(SimpleScheduler[StagePayload, StagePayload]):
    """Bounded FIFO lookahead with independent request outcomes."""

    def __init__(
        self,
        encoder: PerceptionCompute,
        *,
        max_batch_size: int,
        max_batch_wait_ms: float,
        max_batch_frames: int,
        max_padding_ratio: float,
    ) -> None:
        super().__init__(
            encoder,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_wait_when_idle=False,
        )
        self.encoder = encoder
        self.max_batch_frames = max_batch_frames
        self.max_padding_ratio = max_padding_ratio
        self.fatal_error: Exception | None = None
        self.last_batch_size = 0
        self.completed_batches = 0

    def collect_batch(self, first_msg: IncomingMessage) -> list[IncomingMessage]:
        batch = [first_msg]
        anchor_frames = message_frame_count(first_msg)
        if (
            self.max_batch_size == 1
            or anchor_frames is None
            or anchor_frames + 1 > self.max_batch_frames
        ):
            return batch
        else:
            pass
        bucket = max(anchor_frames, 1).bit_length() - 1
        maximum_frames = anchor_frames + 1
        total_frames = maximum_frames
        pending_count = len(self.pending_messages)
        consumed: list[IncomingMessage] = []
        deferred: list[IncomingMessage] = []
        cancelled_ids: set[str] = set()
        deadline: float | None = None
        try:
            for scan_index in range(SCAN_MULTIPLIER * self.max_batch_size):
                if len(batch) == self.max_batch_size:
                    break
                elif scan_index < pending_count:
                    candidate = self.pending_messages.popleft()
                else:
                    try:
                        candidate = self.inbox.get_nowait()
                    except queue.Empty:
                        remaining_seconds = (
                            0.0 if deadline is None else deadline - time.monotonic()
                        )
                        if remaining_seconds <= 0:
                            break
                        else:
                            try:
                                candidate = self.inbox.get(timeout=remaining_seconds)
                            except queue.Empty:
                                break
                consumed.append(candidate)
                if self.consume_if_aborted(candidate.request_id):
                    cancelled_ids.add(candidate.request_id)
                    continue
                else:
                    pass
                candidate_frames = message_frame_count(candidate)
                if candidate_frames is None:
                    deferred.append(candidate)
                    break
                else:
                    pass
                proposed_maximum = max(maximum_frames, candidate_frames + 1)
                proposed_total = total_frames + candidate_frames + 1
                proposed_padded = (len(batch) + 1) * proposed_maximum
                if (
                    max(candidate_frames, 1).bit_length() - 1 != bucket
                    or proposed_padded > self.max_batch_frames
                    or proposed_padded / proposed_total > self.max_padding_ratio
                ):
                    deferred.append(candidate)
                else:
                    batch.append(candidate)
                    maximum_frames = proposed_maximum
                    total_frames = proposed_total
                    if deadline is None and self.max_batch_wait_s > 0:
                        deadline = time.monotonic() + self.max_batch_wait_s
                    else:
                        pass
                if deadline is not None and time.monotonic() >= deadline:
                    break
                else:
                    pass
        except Exception:
            self.pending_messages.extendleft(
                reversed(
                    [
                        message
                        for message in consumed
                        if message.request_id not in cancelled_ids
                    ]
                )
            )
            raise
        else:
            self.pending_messages.extendleft(reversed(deferred))
            return batch

    def run_batch(
        self, batch: list[IncomingMessage], loop: asyncio.AbstractEventLoop
    ) -> None:
        if self.fatal_error is not None:
            return
        else:
            pass
        outcomes: dict[str, OutgoingMessage] = {}
        requests: list[ValidatedPerceptionRequest] = []
        live_messages: list[IncomingMessage] = []
        cancelled_ids: set[str] = set()
        for message in batch:
            if self.consume_if_aborted(message.request_id):
                cancelled_ids.add(message.request_id)
                continue
            else:
                pass
            try:
                if (
                    not isinstance(message.data, StagePayload)
                    or message.data.request_id != message.request_id
                ):
                    raise ValueError(
                        "Perception message must carry its matching StagePayload"
                    )
                else:
                    requests.append(self.encoder.validate(message.data))
                    live_messages.append(message)
            except Exception as error:
                if is_fatal_cuda_error(error):
                    self.fatal_error = error
                    self.running = False
                    return
                else:
                    pass
                outcomes[message.request_id] = OutgoingMessage(
                    request_id=message.request_id, type="error", data=error
                )
        cancelled_ids.update(
            message.request_id
            for message in live_messages
            if self.consume_if_aborted(message.request_id)
        )
        requests = [
            request
            for request in requests
            if request.payload.request_id not in cancelled_ids
        ]
        live_messages = [
            message
            for message in live_messages
            if message.request_id not in cancelled_ids
        ]
        self.last_batch_size = len(requests)
        if requests:
            try:
                results = self.encoder.encode_validated(requests)
                if len(results) != len(live_messages) or any(
                    result.request_id != message.request_id
                    for message, result in zip(live_messages, results)
                ):
                    raise ValueError(
                        "Perception results do not match original request IDs"
                    )
                else:
                    pass
                for message, result in zip(live_messages, results):
                    outcomes[message.request_id] = OutgoingMessage(
                        request_id=message.request_id, type="result", data=result
                    )
            except Exception as error:
                if is_fatal_cuda_error(error):
                    self.fatal_error = error
                    self.running = False
                    return
                else:
                    for message in live_messages:
                        outcomes[message.request_id] = OutgoingMessage(
                            request_id=message.request_id, type="error", data=error
                        )
        else:
            pass
        self.completed_batches += 1
        for message in batch:
            if message.request_id in cancelled_ids or self.consume_if_aborted(
                message.request_id
            ):
                continue
            else:
                self.outbox.put(outcomes[message.request_id])

    def start(self) -> None:
        if self.fatal_error is not None:
            raise self.fatal_error
        else:
            pass
        super().start()
        if self.fatal_error is not None:
            raise self.fatal_error
        else:
            pass
