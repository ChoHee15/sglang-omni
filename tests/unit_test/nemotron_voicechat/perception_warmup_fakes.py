"""CPU spies for graph capture evidence without allocating CUDA resources."""

from typing import Literal
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron_voicechat import perception_encoder
from sglang_omni.models.nemotron_voicechat.conformer import (
    AudioPerception,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import GraphExecutionState
from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.models.nemotron_voicechat.perception_encoder import (
    PerceptionEncoder,
    ValidatedPerceptionRequest,
)
from sglang_omni.proto.request import OmniRequest, StagePayload

CaptureOutcome = Literal[
    "capture",
    "capture-error",
    "fatal-capture",
    "empty-graph",
    "no-delta",
    "fallback",
    "fatal",
    "ordinary",
]


class ObservedRunner:
    def __init__(
        self,
        model: AudioPerception,
        batch_size: int,
        enable_cuda_graph: bool,
        graph_execution_state: GraphExecutionState | None = None,
        *,
        warmup_steps: int = 3,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.enable_cuda_graph = (
            enable_cuda_graph and model.proj.weight.device.type == "cuda"
        )
        self.graph_capture_count = 0
        self.graph: Mock | None = None
        self.close_count = 0
        self.outcome: CaptureOutcome = "capture"

    def close(self) -> None:
        self.close_count += 1
        self.graph = None

    def encode(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.outcome == "fatal":
            raise RuntimeError("CUDA illegal memory access")
        elif self.outcome == "ordinary":
            raise RuntimeError("ordinary inference failure")
        elif self.outcome == "fallback":
            self.enable_cuda_graph = False
            self.graph = None
        elif (
            self.enable_cuda_graph
            and self.graph is None
            and any(waveform.numel() for waveform in waveforms)
        ):
            self.graph = None if self.outcome == "empty-graph" else Mock()
            self.graph_capture_count += 0 if self.outcome == "no-delta" else 1
            if self.outcome == "fatal-capture":
                raise RuntimeError("CUDA illegal memory access")
            elif self.outcome == "capture-error":
                raise RuntimeError("ordinary error after successful capture")
            else:
                pass
        else:
            pass
        return [
            torch.zeros((waveform.numel() // 1280 + 1, 8)) for waveform in waveforms
        ]


def spy_encoder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_cuda_graph: bool = True,
    device: str = "cuda:0",
    capacity: int = 1,
) -> tuple[PerceptionEncoder, list[ObservedRunner]]:
    model = Mock(spec=AudioPerception)
    model.proj = Mock()
    model.proj.weight.device = torch.device(device)
    model.proj.weight.dtype = torch.float32
    created: list[ObservedRunner] = []

    def create_runner(
        model: AudioPerception,
        batch_size: int,
        enable_cuda_graph: bool,
        graph_execution_state: GraphExecutionState | None = None,
        *,
        warmup_steps: int = 3,
    ) -> ObservedRunner:
        runner = ObservedRunner(
            model,
            batch_size,
            enable_cuda_graph,
            graph_execution_state,
            warmup_steps=warmup_steps,
        )
        created.append(runner)
        return runner

    monkeypatch.setattr(perception_encoder, "BatchedPerceptionRunner", create_runner)
    return (
        PerceptionEncoder(
            model, Mock(spec=StreamingPerception), enable_cuda_graph, capacity
        ),
        created,
    )


def requests(
    encoder: PerceptionEncoder, batch_size: int, frame_count: int = 2
) -> list[ValidatedPerceptionRequest]:
    return [
        encoder.validate(
            StagePayload(
                f"row-{index}",
                OmniRequest(None),
                NemotronVoiceChatState(
                    waveform=torch.zeros(frame_count * 1280), num_frames=frame_count
                ).to_dict(),
            )
        )
        for index in range(batch_size)
    ]


def run_size(encoder: PerceptionEncoder, batch_size: int, frame_count: int = 2) -> None:
    results = encoder.encode_validated(requests(encoder, batch_size, frame_count))
    assert [result.request_id for result in results] == [
        f"row-{index}" for index in range(batch_size)
    ]
    assert all(
        NemotronVoiceChatState.from_dict(result.data).acoustic_frames.shape
        == (frame_count + 1, 8)
        for result in results
    )
