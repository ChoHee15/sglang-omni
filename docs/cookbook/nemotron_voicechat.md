# NemotronLabs VoiceChat

[NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) is an 11B end-to-end full-duplex speech-to-speech model. It is frame-locked at 12.5 Hz: every 80 ms of caller audio advances a Fast Conformer perception encoder, a Nemotron-H thinker that emits one text token and one function token per frame, an EAR-TTS talker that turns each text token into 31 RVQ codes, and an RVQ-VAE codec that renders the codes to 22.05 kHz audio. The model decides when to speak; there is no external VAD.

SGLang-Omni supports offline recordings and live duplex conversations over `/v1/realtime`.

## Prerequisites

The checkpoint is a single 44 GB fp32 `model.safetensors` with a NeMo-style `config.json`. Download it with:

```bash
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B
```

The checkpoint ships neither an HF tokenizer nor the backbone `config.json`; both come from the sibling repo named in its config (`model.stt.model.pretrained_llm`, currently `nvidia/NVIDIA-Nemotron-Nano-9B-v2`), about 17 MB of files and no weights. In offline environments, download those files too:

```bash
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 config.json tokenizer.json tokenizer_config.json special_tokens_map.json
```

If they are missing and the Hub is unreachable, start-up fails with an error naming the repo and this command.

The four stages share one GPU. Measured on one H200 (143.8 GB) with the offline example, the run peaks at about 129 GB: the talker engine reserves `mem_fraction_static=0.35` (about 47 GB, most of it KV pool) and the thinker takes what remains (about 73 GB, 17.7 GB of it bf16 weights). Lower `--talker.engine.mem_fraction_static` if the GPU is smaller or shared; the perception encoder and codec need only a few GB each.

## Running the offline example

```bash
python examples/run_nemotron_voicechat.py \
  --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
  --audio /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav \
  --out reply.wav
```

The example builds `NemotronVoiceChatPipelineConfig`, starts the four stages with `MultiProcessPipelineRunner`, sends one request and writes the reply as 16-bit PCM at 22.05 kHz. Pick the GPU with `CUDA_VISIBLE_DEVICES`. Start-up takes about a minute once the checkpoint is in the page cache; the 41 s `turn_taking.wav` sample renders in roughly real time.

Two input conventions matter:

- The recording is resampled to 16 kHz and **channel 0 is used** (a two-party recording carries the agent on channel 1); it is zero-padded to a multiple of 1280 samples (80 ms).
- Leave trailing silence after the caller's last words. The model answers only while it hears silence, so a clip that ends immediately after the question yields a short or empty reply.

The reply text is the thinker's spoken tokens only; frames where the model is listening carry a marker token and are dropped from the text.

## Offline Perception batching

Cross-request Perception encoding is opt-in. The default remains
`max_batch_size=1`; set it back to 1 to restore singleton admission and
the existing singleton encoder. Configure the Perception stage through the
existing dotted factory options, for example:

```bash
--perception.factory.max_batch_size 4 \
--perception.factory.max_batch_wait_ms 0.0 \
--perception.factory.max_batch_frames 4096 \
--perception.factory.max_padding_ratio 1.25
```

These options also work in the stage's YAML `factory` mapping or as
keyword arguments to `create_perception_executor`. Batch size and frame
budget must be positive integers; wait must be finite and nonnegative;
padding ratio must be finite and at least 1. The frame budget counts padded
80 ms steps, including each request's final flush. Oversized recordings
retain their singleton path rather than gaining a new duration limit.

Cohorts anchor the oldest queued request and admit lengths from the same
power-of-two bucket. Nearby lengths on opposite sides of a bucket boundary
can therefore run separately. Bounded lookahead preserves deferred FIFO
order. A positive wait starts only after a compatible second request joins;
a lone request never waits to fill its batch. The default wait is zero.

Batching supports the checkpoint's default float32 precision. A selected
non-float32 dtype is preserved and explicitly falls back to effective batch
size 1 with a warning. `enable_cuda_graph=false` selects eager execution;
CPU execution is eager. Each executor retains its singleton stream and at
most one exact-size batched runner, so changing cohort size may recapture
the CUDA graph. Perception batching does not change the thinker or talker's
singleton limits, or the native realtime scheduling path.

On one B200 with the official checkpoint, float32 weights and CUDA Graphs,
a trial with `max_batch_size=8` and the other options above produced the
following Perception-stage throughput changes against the original
graph-enabled singleton. Each case used five measured rounds after warmup,
with singleton controls before and after the candidate runs.

| Concurrent requests | 7-frame audio | 72-frame audio | 512-frame audio |
|---|---|---|---|
| 4 | +75.7% | +12.8% | +27.7% |
| 8 | +147.1% | +92.8% | +130.9% |

At concurrency 1, throughput changed by less than 0.5%. In a separate
identical arrival trace with 240 short requests per concurrency, short-request
p95 latency fell from about 117 ms to 54 ms at concurrency 4 and from 234 ms
to 66 ms at concurrency 8. These are stage measurements, not end-to-end
speech throughput gains: the thinker and talker still admit one running
request each.

Results depend on length compatibility and graph reuse. Mixed-length cases
at concurrency 4 and 8 gained less than 1%; one sparse-arrival case at
concurrency 8 lost 8.1% throughput as cohort sizes changed and graphs were
recaptured. Eager execution remained correct but was slower than the
graph-enabled singleton at low concurrency. Keep the default at 1, and
measure the actual arrival pattern, capture churn, and allocated and reserved
memory before enabling batching in a deployment.

## Realtime conversation

```bash
python examples/run_nemotron_voicechat_duplex.py \
  --model-path /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
  --serve --port 8097
```

Open http://localhost:8097, click **Start**, and allow microphone access.
If the server runs on a remote GPU host, forward port 8097 to your local machine first.
Headphones help avoid acoustic feedback. The example uses the shared realtime
playground and supports one session on one GPU, with a four-minute browser limit.
It uses the checkpoint's default prompt and voice.

## Offline request parameters

| Parameter | Effect |
|---|---|
| `temperature`, `top_p`, `top_k` | **Ignored.** The thinker always decodes greedily so that the tokens it streams to the talker are the tokens it committed to; a non-zero `temperature` logs a warning. |
| `max_new_tokens` | Ignored; the frame count is fixed by the input length (one token per 80 ms). |

Audio randomness is governed by the checkpoint's own settings, read from `config.json` (`inference_noise_scale`, `inference_top_p_or_k`): the talker samples its codes with Gumbel-max component selection and Gaussian noise inside the MoG head, so two runs on the same input produce different code sequences and slightly different audio even though the text is identical. There is no `seed` control yet.

## Known limitations

- The thinker and talker each handle one running request at a time (`max_running_requests=1`), even when offline Perception batching is enabled; the offline API has no barge-in interface.
- Single speaker (`Aria`, the checkpoint's baked-in prompt latents).
- The `function_head` tool-call channel is decoded and carried through but nothing acts on it.
- Classifier-free guidance is off (`guidance_scale=0`) although the checkpoint config enables it at 0.2.
- Talker output is not bit-identical to NeMo's offline script even with deterministic sampling on both sides (about 82 % of quantizer cells and 63 of 150 frames agree on a 12 s fixture; transcripts and amplitude envelopes match). Known causes: the backbone and KV cache run in bfloat16 because SGLang's attention backends and `sgl_kernel`'s Gemma RMSNorm have no fp32 path, while NeMo runs the whole talker in fp32; NeMo's offline script enables classifier-free guidance by default; and NeMo's offline recipe walks the system-prompt region before frame 0 (realtime convention here; see `prompt_region_steps`). The thinker is frame-exact with NeMo.
- NeMo decodes the perception encoder's flush row as a 151st frame; this pipeline computes it but does not decode it, so replies are one frame (80 ms) shorter.

## Tests

`tests/unit_test/nemotron_voicechat/` runs on CPU without weights: request frame counts, streaming codec equivalence with whole-utterance decoding, and checkpoint-shim isolation.
