"""Length-aware offline Perception scheduling and execution."""

from __future__ import annotations

import asyncio
import queue
import time
from typing import Protocol

from sglang_omni.models.nemotron_voicechat.cuda_graph import is_fatal_cuda_error
from sglang_omni.models.nemotron_voicechat.perception_encoder import (
    ValidatedPerceptionRequest,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

SCAN_MULTIPLIER = 4


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
