# SPDX-License-Identifier: Apache-2.0
"""Fixed-cohort streaming audio encoding with independent causal histories."""

import logging
import math

import torch
from einops import einsum, rearrange

from sglang_omni.models.nemotron_voicechat.conformer import (
    GRAPH_WARMUP_STEPS,
    SAMPLES_PER_FRAME,
    AudioPerception,
    GraphPerception,
    RelPositionMultiHeadAttention,
    StreamingPerception,
)
from sglang_omni.models.nemotron_voicechat.cuda_graph import (
    CapturedStep,
    GraphExecutionState,
    is_fatal_cuda_error,
)
from sglang_omni.models.nemotron_voicechat.preprocess import (
    DEFAULT_LOG_ZERO_GUARD,
    DEFAULT_PREEMPHASIS,
)

logger = logging.getLogger(__name__)


class BatchedStreamingPerception(StreamingPerception):
    """Framewise encoding that freezes every finished request's history."""

    def __init__(self, perception: AudioPerception, batch_size: int) -> None:
        assert batch_size > 1
        super().__init__(perception, batch_size=batch_size)
        self.active_rows = torch.ones(batch_size, device=self.device, dtype=torch.bool)

    def write_state(self, buffer: torch.Tensor, candidate: torch.Tensor) -> None:
        row_mask = self.active_rows.reshape(
            self.batch_size, *((1,) * (buffer.ndim - 1))
        )
        buffer.copy_(torch.where(row_mask, candidate, buffer))

    def advance_frame_count(self) -> None:
        self.cached_frame_count.add_(self.active_rows).clamp_(max=self.max_keys)

    def featurize(self, samples: torch.Tensor) -> torch.Tensor:
        preprocessor = self.perception.preprocessor
        previous_samples = torch.cat((self.preemphasis_carry, samples[:, :-1]), dim=1)
        emphasized_samples = samples - DEFAULT_PREEMPHASIS * previous_samples
        window_samples = torch.cat((self.sample_buffer, emphasized_samples), dim=1)
        self.write_state(
            self.sample_buffer, window_samples[:, -preprocessor.left_padding :]
        )
        self.write_state(self.preemphasis_carry, samples[:, -1:])
        if self.device.type == "cuda" and torch.backends.cudnn.allow_tf32:
            # note (caozhuoxi): TF32 amplifies shape-dependent log-mel rounding.
            return torch.cat(
                [self.log_mel_features(row) for row in window_samples], dim=0
            )
        else:
            pass
        spectrum = torch.stft(
            window_samples,
            n_fft=preprocessor.n_fft,
            hop_length=preprocessor.hop_length,
            win_length=preprocessor.win_length,
            return_complex=True,
            center=False,
            window=preprocessor.featurizer.window,
        )
        mel = einsum(
            preprocessor.featurizer.fb[0],
            spectrum.abs().square(),
            "m f, b f t -> b t m",
        )
        return torch.log(mel + DEFAULT_LOG_ZERO_GUARD)

    def pointwise_convolve(
        self, conv: torch.nn.Conv1d | torch.nn.Conv2d, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(conv, torch.nn.Conv2d):
            assert conv.kernel_size == (1, 1) and conv.stride == (1, 1)
            assert conv.padding == (0, 0) and conv.dilation == (1, 1)
            assert conv.groups == 1
            if self.device.type == "cuda" and torch.backends.cudnn.allow_tf32:
                return torch.cat([conv(row[None]) for row in hidden_states], dim=0)
            else:
                return conv(hidden_states)
        else:
            pass
        assert conv.kernel_size == (1,) and conv.stride == (1,)
        assert conv.padding == (0,) and conv.dilation == (1,) and conv.groups == 1
        # note (caozhuoxi): cuDNN selects reduced-precision 1x1 kernels for small batches.
        projected = torch.nn.functional.linear(
            hidden_states.transpose(1, 2), conv.weight[:, :, 0], conv.bias
        )
        return projected.transpose(1, 2)

    def attend(
        self,
        index: int,
        attention: RelPositionMultiHeadAttention,
        hidden_11D: torch.Tensor,
    ) -> torch.Tensor:
        query = rearrange(
            attention.linear_q(hidden_11D), "b 1 (h s) -> b h s", h=self.num_heads
        )
        current_key = rearrange(
            attention.linear_k(hidden_11D), "b 1 (h s) -> b 1 h s", h=self.num_heads
        )
        current_value = rearrange(
            attention.linear_v(hidden_11D), "b 1 (h s) -> b 1 h s", h=self.num_heads
        )
        keys = torch.cat((self.key_caches[index], current_key), dim=1)[:, 1:]
        values = torch.cat((self.value_caches[index], current_value), dim=1)[:, 1:]
        self.write_state(self.key_caches[index], keys)
        self.write_state(self.value_caches[index], values)
        content_scores = einsum(
            query + attention.pos_bias_u, keys, "b h s, b k h s -> b k h"
        )
        position_scores = einsum(
            query + attention.pos_bias_v,
            self.posproj_LPHS[index],
            "b h s, k h s -> b k h",
        )
        visible_key_counts = (self.cached_frame_count + 1).clamp(max=self.max_keys)
        padding_mask = self.key_positions_K[None, :] < (
            self.max_keys - visible_key_counts[:, None]
        )
        scores = (content_scores + position_scores) / math.sqrt(self.head_size)
        weights = scores.masked_fill(padding_mask[:, :, None], float("-inf")).softmax(
            dim=1
        )
        attended = einsum(weights, values, "b k h, b k h s -> b h s")
        return attention.linear_out(rearrange(attended, "b h s -> b 1 (h s)"))

    @torch.inference_mode()
    def push(self, samples: torch.Tensor) -> torch.Tensor:
        assert not self.flushed
        assert samples.shape == (self.batch_size, SAMPLES_PER_FRAME)
        return self.encode_step(self.featurize(samples))[:, 0]

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        assert not self.flushed
        self.flushed = True
        self.active_rows.fill_(True)
        mel_bins = self.perception.preprocessor.featurizer.fb.shape[1]
        zero_mel = torch.zeros(
            self.batch_size, 1, mel_bins, device=self.device, dtype=self.dtype
        )
        return self.encode_step(zero_mel)[:, 0]


class BatchedPerceptionRunner:
    """Own one exact-size eager stream and optional frame CUDA graph."""

    def __init__(
        self,
        model: AudioPerception,
        batch_size: int,
        enable_cuda_graph: bool,
        graph_execution_state: GraphExecutionState | None = None,
        *,
        warmup_steps: int = GRAPH_WARMUP_STEPS,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        else:
            pass
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or warmup_steps < 1
        ):
            raise ValueError("warmup_steps must be a positive integer")
        else:
            pass
        self.warmup_steps: int = warmup_steps
        self.model = model
        self.batch_size = batch_size
        self.device = model.proj.weight.device
        self.dtype = model.proj.weight.dtype
        if batch_size > 1 and self.dtype != torch.float32:
            raise ValueError("Batched Perception requires float32 weights")
        else:
            pass
        self.stream: StreamingPerception | BatchedStreamingPerception
        if batch_size == 1:
            self.stream = (
                GraphPerception(model)
                if enable_cuda_graph
                else StreamingPerception(model)
            )
        else:
            self.stream = BatchedStreamingPerception(model, batch_size)
        self.enable_cuda_graph = (
            enable_cuda_graph and self.device.type == "cuda" and batch_size > 1
        )
        self.graph_execution_state: GraphExecutionState = (
            GraphExecutionState()
            if graph_execution_state is None
            else graph_execution_state
        )
        if isinstance(self.stream, GraphPerception):
            self.stream.graph_execution_state = self.graph_execution_state
        else:
            pass
        self.captured_step: CapturedStep | None = None
        self.graph_capture_count = 0
        self.step_samples = torch.empty(
            batch_size, SAMPLES_PER_FRAME, device=self.device, dtype=self.dtype
        )

    @property
    def graph(self) -> torch.cuda.CUDAGraph | None:
        return None if self.captured_step is None else self.captured_step.graph

    @property
    def graph_output(self) -> torch.Tensor | None:
        return None if self.captured_step is None else self.captured_step.output

    def close(self) -> None:
        if self.captured_step is not None:
            self.captured_step.close()
        else:
            pass
        if isinstance(self.stream, GraphPerception):
            self.stream.close()
        else:
            pass

    def abandon(self) -> None:
        self.graph_execution_state.is_abandoned = True
        if self.captured_step is not None:
            self.captured_step.abandon()
        else:
            pass

    @torch.inference_mode()
    def capture(self) -> None:
        try:
            self.capture_step()
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise

    def capture_step(self) -> None:
        assert isinstance(self.stream, BatchedStreamingPerception)
        current_stream = torch.cuda.current_stream(self.device)
        try:
            self.step_samples.zero_()
            self.stream.active_rows.fill_(True)
            self.captured_step = CapturedStep(self.device, self.graph_execution_state)
            capture_stream = self.captured_step.capture_stream
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                for _ in range(self.warmup_steps):
                    self.stream.push(self.step_samples)
            self.captured_step.wait_for_capture(current_stream)
            self.stream.reset()
            self.captured_step.state.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self.captured_step.state.graph, stream=capture_stream
            ):
                self.captured_step.state.output = self.stream.push(self.step_samples)
            self.stream.reset()
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
                raise
            else:
                if self.captured_step is not None:
                    self.captured_step.wait_for_capture(current_stream)
                    self.close()
                else:
                    pass
                self.stream.reset()
                if isinstance(error, RuntimeError) and any(
                    marker in str(error).lower()
                    for marker in ("capture", "cuda graph", "not supported", "cufft")
                ):
                    self.enable_cuda_graph = False
                    logger.warning(
                        f"Perception CUDA graph capture unavailable; using eager: {error}"
                    )
                else:
                    raise
        else:
            self.graph_capture_count += 1

    @torch.inference_mode()
    def encode(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        try:
            return self.encode_waveforms(waveforms)
        except Exception as error:
            if is_fatal_cuda_error(error):
                self.abandon()
            else:
                pass
            raise

    def encode_waveforms(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.graph_execution_state.is_abandoned:
            raise RuntimeError("Perception CUDA execution was abandoned")
        else:
            pass
        if len(waveforms) != self.batch_size:
            raise ValueError("waveform count must match runner batch_size")
        else:
            pass
        for waveform in waveforms:
            if waveform.ndim != 1 or not waveform.is_floating_point():
                raise ValueError("waveforms must be floating one-dimensional tensors")
            elif waveform.numel() % SAMPLES_PER_FRAME != 0:
                raise ValueError("waveform sample length must be divisible by 1280")
            else:
                pass
        self.stream.reset()
        if self.batch_size == 1:
            return [self.model(waveforms[0][None], stream=self.stream)[0]]
        else:
            pass
        assert isinstance(self.stream, BatchedStreamingPerception)
        frame_counts = [waveform.numel() // SAMPLES_PER_FRAME for waveform in waveforms]
        maximum_frames = max(frame_counts)
        if self.enable_cuda_graph and self.graph is None and maximum_frames > 0:
            self.capture()
        else:
            pass
        padded_waveforms = torch.zeros(
            self.batch_size,
            maximum_frames * SAMPLES_PER_FRAME,
            device=self.device,
            dtype=self.dtype,
        )
        acoustic_frames = torch.empty(
            maximum_frames + 1,
            self.batch_size,
            self.model.proj.out_features,
            device=self.device,
            dtype=self.dtype,
        )
        for row, waveform in enumerate(waveforms):
            padded_waveforms[row, : waveform.numel()].copy_(waveform)
        device_frame_counts = torch.tensor(frame_counts, device=self.device)
        for frame in range(maximum_frames):
            self.stream.active_rows.copy_(device_frame_counts > frame)
            self.step_samples.copy_(
                padded_waveforms[
                    :, frame * SAMPLES_PER_FRAME : (frame + 1) * SAMPLES_PER_FRAME
                ]
            )
            if self.graph is None:
                output = self.stream.push(self.step_samples)
            else:
                self.graph.replay()
                assert self.graph_output is not None
                output = self.graph_output
            acoustic_frames[frame].copy_(output)
        flush_output = self.stream.flush()
        for row, frame_count in enumerate(frame_counts):
            acoustic_frames[frame_count, row].copy_(flush_output[row])
        acoustic_outputs = [
            acoustic_frames[: frame_count + 1, row].clone()
            for row, frame_count in enumerate(frame_counts)
        ]
        self.stream.reset()
        return acoustic_outputs
