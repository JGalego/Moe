# Changelog

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
