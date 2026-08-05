# Changelog

## Unreleased

### Formats

- **GGUF** is read in place, as a third store backend. `Q4_0`, `Q5_0` and `Q8_0`
  are byte-for-byte the engine's own formats, so those weights need no
  conversion; `Q4_K` and `Q6_K` have their own readers. The config, vocabulary
  and chat template are all recovered from GGUF metadata.
- `Q5` and `Q6` block formats, at 5.5 and 6.5 bits per weight, filling the gap
  between Q4 and Q8. Q5 is laid out as GGUF's `Q5_0`.
- `pack --hot-experts` gives the experts a trace leans on a finer format than the
  ones it barely touches.
- `moe info` names the container it read — `safetensors`, `packed` or `gguf` —
  instead of calling every single-file checkpoint packed.

### Speed

- `--draft N` speculates ahead with a lookup drafter — no second model — and
  verifies in one batched step. Lossless: bit-identical greedily, and the
  standard rejection correction under temperature.
- Expert prefetch, on by default: before a decode step the kernel is advised to
  fetch the experts the previous token used, at every layer at once.
- `--pin trace.jsonl` keeps a workload's hot experts resident.
- `serve --slots N` keeps several prefix caches, so alternating conversations do
  not evict each other.
- `run --prompts file.txt` answers a file of prompts in one load, reusing the
  cache between them.

### Correctness

- Four rope-scaling schemes read from the config — linear, NTK, Llama 3 and YaRN
  — where before any factor was applied as linear interpolation. Frequencies are
  resolved once at load rather than per token per head.
- Sliding-window attention, in all four config conventions, including
  checkpoints that alternate local and global layers.
- Chat templates are rendered from the checkpoint's own Jinja `chat_template`,
  falling back to control-token detection only when the template is beyond the
  engine's subset.
- Three faults fixed in parsers that read untrusted files: a `.moe` header length
  that panicked when sliced, a vocabulary id trusted as an allocation size, and
  an added token with empty content that made encoding loop forever.

### New commands

- `moe chat` — a conversation in the terminal, reusing the cache between turns.
- `moe route` — routing entropy, load balance and coverage, per layer, plus the
  SVG that `scripts/routeviz.py` used to draw. Takes a model and a prompt, or
  traces from disk.
- `moe eval` — perplexity, bits per token and bits per byte on held-out text,
  with `--vs` to score a second model on the same tokens.
- `moe embed` — pooled hidden states as a vector, with `--vs` for cosine
  similarity.

### New capabilities

- Constrained decoding: `--json` and `--schema`, and `response_format` over HTTP.
  A pushdown automaton over bytes masks any token that could not continue a valid
  document.
- `pack --keep-experts` prunes a checkpoint to the experts a traced workload
  reaches, renumbering the survivors and narrowing the router to match.
- Router interventions for asking what an expert is for: `--disable-expert`,
  `--force-expert`, `--router-temp`, `--top-k-experts`.
- `/v1/embeddings` and `/metrics`; plus `logprobs`, `n` and `echo` on the
  completion endpoints.

### Packaging

- A `Dockerfile`, a `flake.nix` and a Homebrew formula.
- CI gained a throughput floor, a container build, and a nightly job that builds
  and briefly runs the `fuzz/` targets.

## 0.1.0

First release.

### Engine

- CPU inference for sparse mixture-of-experts decoders, in one binary with no
  BLAS, no GPU and no Python at runtime.
- Architecture read from the checkpoint rather than configured: grouped-query or
  latent attention, qk-norm in either convention, softmax or sigmoid gating,
  group-limited routing, shared experts, and dense-versus-routed decided per
  layer.
- Weights memory mapped and read in place as `F32`, `F16` or `BF16`, or as 4- and
  8-bit blocks after packing. Per-expert and fused `[experts, 2*inter, hidden]`
  layouts both addressed as row ranges.
- One forward path for prefill and decode, with hand-written AVX2/FMA and NEON
  kernels and a scalar fallback.

### Commands

- `moe run` — generate from a local path, a packed file, a URL or a Hugging Face
  repo id, downloading and caching as needed.
- `moe serve` — OpenAI-compatible HTTP: chat and text completions, streaming or
  not, plus `/v1/models` and `/health`. Keeps its KV cache between requests, so a
  prompt extending the last one prefills only what was added.
- `moe pack` — re-quantise a checkpoint into one self-contained file with the
  tokenizer embedded.
- `moe pull`, `moe info`, `moe bench`, `moe tokenize`.
- `--trace` records every routing decision; `scripts/routeviz.py` renders one
  trace as a heatmap or two as their difference.

### Platforms

Linux, macOS and Windows, x86-64 and arm64. Prebuilt binaries, an install script
for each platform, or `cargo install moe-ontap`.
