<div align="center">

<img src="assets/logo.svg" alt="moe" width="360">

### Every expert on tap.

CPU inference for sparse mixture-of-experts models.<br>
One binary. Linux, macOS, Windows. No GPU, no BLAS, no Python.

[![ci](https://img.shields.io/github/actions/workflow/status/JGalego/Moe/ci.yml?branch=main&style=flat-square&label=ci&color=17b3a3)](https://github.com/JGalego/Moe/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/JGalego/Moe?style=flat-square&color=e0973a&label=release)](https://github.com/JGalego/Moe/releases/latest)
[![platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-8b95a6?style=flat-square)](#install)
[![rust](https://img.shields.io/badge/rust-1.75%2B-e0973a?style=flat-square)](https://www.rust-lang.org)
[![gpu](https://img.shields.io/badge/GPU-not%20required-17b3a3?style=flat-square)](#why-another-engine)
[![license](https://img.shields.io/badge/license-MIT-8b95a6?style=flat-square)](LICENSE)

</div>

---

**Better call Moe!** Point it at a Hugging Face repo and it downloads, caches, and
works out what the model is by reading the weights — no config file, no conversion
step, no `--model-type`. About 3,100 lines of Rust.

```console
$ moe run mistralai/Mixtral-8x7B-v0.1 -p "A sparse model is one where" -n 64
```

That is the whole setup. The weights are fetched once with a progress bar, cached,
and reused instantly on every run after that.

```console
$ moe info ~/models/mixtral
MixtralForCausalLM
  layers 32  hidden 4096  vocab 32000
  attention  GQA(32 q / 8 kv heads, head_dim=128)
  ffn        8 experts, top-2, softmax gate

weights    995 tensors, 86.99 GB
  dense    2.99 GB
  experts  84.00 GB (97%)
  formats  BF16 86.99 GB
kv cache   256.00 MB per 1k tokens
source     /home/you/models/mixtral (safetensors)
```

*(That is the layout `moe info` prints. The figures are computed from
Mixtral-8x7B's published shapes rather than measured on a run — 97% of the
checkpoint is expert weights, which is the whole point.)*

## Why another engine

A sparse MoE checkpoint is mostly weights that any given token never touches.
Routing picks a handful of experts per layer, so the arithmetic per token stays
small even when the file is enormous — what actually hurts on a CPU is moving
bytes. Moe is built around that one observation:

- **Copy-free.** Weights are memory mapped and read in place, in whatever format
  they already have. Loading a checkpoint is an `mmap` call, not a deserialise
  pass, so start-up is instant and the resident set is whatever the OS decides to
  keep — a model larger than RAM streams from disk with no special mode.
- **Only what routing selects.** A routed layer touches the experts it chose and
  nothing else. Expert weights are addressed as row ranges inside the checkpoint,
  including the fused `[experts, 2*inter, hidden]` stacks that current exports
  use, so selecting an expert costs an offset calculation.
- **One code path for prefill and decode.** A step takes `t` tokens; decode is
  `t == 1`. Each weight row is dequantised once and dotted against every token in
  the batch, so prefill reads each weight once rather than once per token, and
  the MoE layer runs each selected expert once over all the tokens that chose it.
- **Detected, not configured.** Nothing about the architecture is hard-coded.
  Latent attention, qk-norm, sigmoid gating, group-limited routing, shared
  experts, dense-prefix layers — each is inferred from the tensors that exist.

## Install

A single binary with no runtime dependencies. Linux, macOS and Windows, x86-64
and arm64.

```console
# macOS / Linux
$ curl -fsSL https://raw.githubusercontent.com/JGalego/Moe/main/install.sh | sh

# Windows (PowerShell)
> irm https://raw.githubusercontent.com/JGalego/Moe/main/install.ps1 | iex
```

Or from source, with a Rust toolchain (1.75+):

```console
$ cargo install --git https://github.com/JGalego/Moe    # or: cargo build --release
$ cargo test                                            # full suite, no downloads, ~2s
```

Building from source compiles a TLS stack, which wants a C compiler. If you do
not have one — or do not want downloads at all — `--no-default-features` drops
both, leaving an engine that takes local paths. For the best kernels on your own
machine, build with `RUSTFLAGS="-C target-cpu=native"`.

## Use

`<model>` is whatever you have:

```console
$ moe run mistralai/Mixtral-8x7B-v0.1 -p "The capital of France is"   # Hub repo
$ moe run mistralai/Mixtral-8x7B-v0.1@refs/pr/7 -p ...                # a revision
$ moe run ~/models/mixtral -p ... --temp 0.7                          # local directory
$ moe run ./mixtral.moe -p ...                                        # packed file
$ moe run https://example.com/mixtral.moe -p ...                      # direct download
```

Remote models are downloaded once into a cache — `%LOCALAPPDATA%\moe` on
Windows, `~/Library/Caches/moe` on macOS, `$XDG_CACHE_HOME/moe` or `~/.cache/moe`
elsewhere — and reused after that. `MOE_CACHE` moves it, `HF_TOKEN` reaches gated
repos, `--offline` refuses to download, and `moe pull <model>` fetches without
running anything. From a repo, only the files inference reads are downloaded:
`config.json`, `tokenizer.json` and the weights, skipping the `.bin` duplicates,
demos and conversions that repos accumulate. If the repo publishes a packed
`.moe`, that is taken on its own instead.

Then pack it, if you like. Packing re-quantises every weight, drops tensors inference
never reads, and embeds `tokenizer.json`, giving one self-contained file:

```console
$ moe pack mistralai/Mixtral-8x7B-v0.1 --quant q8 --expert-quant q4
$ moe run ./Mixtral-8x7B-v0.1.moe -p "Explain routing in one sentence." --stats
```

`--quant` covers the dense trunk, `--expert-quant` the routed experts. Since the
experts are the bulk of the file and the least sensitive part of it, `--quant q8
--expert-quant q4` is a good default: Q8 costs 8.5 bits per weight and Q4 costs
4.5, so a bf16 checkpoint shrinks 1.88x and 3.56x respectively. Norms, biases and
router gates always stay f32 — they are tiny, and quantisation noise there feeds
straight into routing decisions.

Other commands:

```console
$ moe pull  <model>                    # download into the cache, print the path
$ moe info  <model>                    # architecture, footprint, kv cache size
$ moe bench <model> -n 64              # prefill and decode throughput
$ moe tokenize <model> -p "hello"      # token ids, for debugging
$ moe --help                           # every flag
```

Generation reads `--prompt`, `--prompt-file` or `--ids` (raw token ids, no
tokenizer needed). Sampling is greedy by default; `--temp`, `--top-p`, `--top-k`,
`--repeat-penalty` and `--seed` are there when you want them.

## Supported models

The engine handles the sparse-decoder family: RMSNorm, rotary embeddings,
grouped-query **or** latent attention, SwiGLU experts with top-k routing.

| Family | What it needs | Status |
| --- | --- | --- |
| Mixtral | GQA, softmax routing, per-expert or fused weights | verified on a checkpoint |
| Qwen2-MoE / Qwen3-MoE | qk-norm, shared expert with its own gate | expected — same tensors, untested |
| OLMoE | GQA, qk-norm, unnormalised top-k | expected — same tensors, untested |
| DeepSeek-V2 / V3 | latent attention, sigmoid gate, group-limited routing, shared experts, dense prefix | mechanisms covered by tests, no checkpoint run |
| Dense Llama-style | no experts at all | works, it is the same path minus routing |

"Verified" and "expected" mean exactly what they say: see
[Validation](#validation) for what was actually run. Anything in this family that
uses standard Hugging Face tensor names should load; if it does not, `moe info`
reports the first tensor it could not find, which is usually the whole diagnosis.

Not supported today: MXFP4 checkpoints, attention sinks and sliding-window
attention (so GPT-OSS will not load), YaRN and other non-linear rope scaling, NFC
normalisation in the tokenizer, and encoder-decoder models. See
[docs/MODELS.md](docs/MODELS.md).

## Validation

Correctness is not asserted, it is checked against implementations written
independently of this one.

- **Forward pass.** `scripts/oracle.py` builds tiny random checkpoints and runs a
  reference forward pass in pure Python, written from the model definitions
  rather than from this code. Two fixtures cover the interesting mechanisms:
  grouped-query attention with qk-norm and softmax routing, and latent attention
  with sigmoid gating, group-limited routing, a shared expert and a dense first
  layer. The engine has to reproduce the reference logits at every position,
  under incremental decode, batched prefill and split prefill alike.
- **Quantisation.** The same fixtures are packed and re-checked, so a packed
  model must keep both its logits (within the format's resolution) and its
  argmax.
- **Tokenizer.** `scripts/tokcheck.py` compares `moe tokenize` against Hugging
  Face's `tokenizers` on 326 cases — awkward fixed ones plus generated whitespace,
  digit, code and mixed-script strings. Both families currently match exactly:
  326/326 on a byte-level BPE vocabulary and 326/326 on a metaspace one.

```console
$ cargo test                                       # 21 tests, no downloads
$ python3 scripts/oracle.py tests/fixtures         # regenerate fixtures
$ pip install tokenizers
$ python3 scripts/tokcheck.py ~/models/qwen3-moe/tokenizer.json
```

## How it works

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks through the whole thing —
the storage layer, the quantisation formats, the forward pass, and where the time
goes. The short version:

| File | Lines | Role |
| --- | --- | --- |
| `src/quant.rs` | 411 | block formats, dequantise + matmul kernels, AVX2 and NEON |
| `src/store.rs` | 317 | mmap safetensors and `.moe`, tensor views, packing |
| `src/spec.rs` | 205 | architecture detection from config + tensor shapes |
| `src/model.rs` | 629 | weight binding and the forward pass |
| `src/tokenizer.rs` | 591 | `tokenizer.json` BPE, both pre-tokenizer families |
| `src/fetch.rs` | 361 | resolving paths, URLs and Hub repos; the download cache |
| `src/sample.rs` | 127 | temperature, top-k, top-p, repetition penalty |
| `src/main.rs` | 402 | CLI |

## Limitations

Worth knowing before you file an issue:

- Attention is exact and quadratic in context; there is no flash-attention
  kernel and no paged KV cache. The KV cache is f32 and preallocated for `--ctx`.
- Activations are always f32. Only weights are quantised.
- Hand-written SIMD covers x86-64 (AVX2/FMA, detected at runtime) and arm64
  (NEON). Anywhere else the scalar kernels rely on the autovectoriser, which is
  decent but not the same thing.
- Downloads are resumed at file granularity, not byte ranges: an interrupted
  file restarts, though completed ones are kept.
- Throughput numbers are not quoted here because they depend entirely on your
  machine's memory bandwidth. Run `moe bench` and you will have real ones.

## License

MIT. See [LICENSE](LICENSE).
