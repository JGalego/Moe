<div align="center">

<img src="assets/logo.svg" alt="moe" width="330">

### Every expert on tap.

CPU inference for sparse mixture-of-experts models.<br>
One binary. Linux, macOS, Windows. No GPU, no BLAS, no Python.

[![ci](https://img.shields.io/github/actions/workflow/status/JGalego/Moe/ci.yml?branch=main&style=flat-square&label=ci&color=17b3a3)](https://github.com/JGalego/Moe/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/JGalego/Moe?style=flat-square&color=e0973a&label=release)](https://github.com/JGalego/Moe/releases/latest)
[![platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-8b95a6?style=flat-square)](#install)
[![rust](https://img.shields.io/badge/rust-1.85%2B-e0973a?style=flat-square)](https://www.rust-lang.org)
[![gpu](https://img.shields.io/badge/GPU-not%20required-17b3a3?style=flat-square)](#why-another-engine)
[![license](https://img.shields.io/badge/license-MIT-8b95a6?style=flat-square)](LICENSE)

</div>

---

**Better call Moe!** Point it at a Hugging Face repo and it downloads, caches, and
works out what the model is by reading the weights — no config file, no conversion
step, no `--model-type`. About 4,000 lines of Rust.

![moe run](assets/run.gif)

The weights are fetched once with a progress bar, cached, and reused instantly on
every run after that.

![moe info](assets/info.gif)

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

Or from source, with a Rust toolchain (1.85+):

```console
$ cargo install ontap                                  # the crate is `ontap`, the binary `moe`
$ cargo install --git https://github.com/JGalego/Moe    # or straight from source
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
demos and conversions that repos accumulate. A repo publishing a packed `.moe`
yields just that one file instead.

Then pack it, if you like. Packing re-quantises every weight, drops tensors
inference never reads, and embeds `tokenizer.json`, giving one self-contained
file:

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
$ moe serve <model> --port 8080        # OpenAI-compatible HTTP server
$ moe tokenize <model> -p "hello"      # token ids, for debugging
$ moe --help                           # every flag
```

Generation reads `--prompt`, `--prompt-file` or `--ids` (raw token ids, no
tokenizer needed). Sampling is greedy by default; `--temp`, `--top-p`, `--top-k`,
`--repeat-penalty` and `--seed` are there when you want them.

## Serving

```console
$ moe serve allenai/OLMoE-1B-7B-0924 --port 8080
$ curl localhost:8080/v1/completions -H 'content-type: application/json' \
    -d '{"prompt":"The capital of France is","max_tokens":8}'
```

`/v1/chat/completions`, `/v1/completions` (both with `"stream": true`),
`/v1/models` and `/health` — enough of the OpenAI API that existing clients work
without knowing what they are talking to. `prompt` also accepts an array of token
ids, so a client with no tokenizer can still drive the model.

**It serves one request at a time, on purpose.** At batch size one the engine is
memory-bandwidth-bound and already uses every core, so concurrent generations
would halve each other's throughput rather than add any. Requests queue on a
single session; past `--max-queue` they get a 503 instead of piling up.

That single session is also what makes it quick: it keeps its KV cache between
requests, so a prompt that extends the last one only prefills what was added.
Measured on OLMoE with a 125-token prompt, second turn: **1.1s with the cache
against 5.2s without**, and the gap widens as a conversation grows. `--no-prefix-cache`
turns it off.

Chat needs a prompt format, and the one a checkpoint wants is read from the
control tokens in its vocabulary — `<|im_start|>` means chatml,
`<|start_header_id|>` llama3, `[/INST]` mistral — so an instruct model is served
correctly the moment you point at it. `--chat-format` names one directly, and
`/v1/completions` needs none.

It binds `127.0.0.1` by default, so it is yours until you say otherwise.

## Seeing the routing

`--trace` records which experts every token chose, and at what weight, as JSONL.
`scripts/routeviz.py` renders one trace as a heatmap, or two as their difference:

```console
$ moe run <model> -p "import numpy as np ..."   --trace code.jsonl
$ moe run <model> -p "The French Revolution ..." --trace prose.jsonl
$ python3 scripts/routeviz.py code.jsonl prose.jsonl -o routing.svg
```

![expert routing, code vs prose](assets/routing.svg)

Each cell is one expert in one layer; amber means the code prompt picked it more
often, teal the prose prompt, grey means neither. The strongly coloured cells are
the point — this checkpoint really does send code and prose to different experts,
and that is why a sparse model can be big without being slow. Point it at your own
two prompts and see where they diverge.

The script also prints the same numbers as text, so the picture is never the only
way to read the trace.

## Supported models

The engine handles the sparse-decoder family: RMSNorm, rotary embeddings,
grouped-query **or** latent attention, SwiGLU experts with top-k routing.

| Family | What it uses |
| --- | --- |
| OLMoE | GQA, whole-projection qk-norm, unnormalised top-k |
| Mixtral | GQA, softmax routing, per-expert or fused expert weights |
| Qwen2-MoE / Qwen3-MoE | per-head qk-norm, shared expert with its own gate |
| DeepSeek-V2 / V3 | latent attention, sigmoid gate, group-limited routing, shared experts, dense prefix |
| Dense Llama-style | the same path, minus routing |

Anything in this family using standard Hugging Face tensor names loads as it is;
`moe info` reports what it detected, so a checkpoint tells you what it is before
you generate a token. [docs/MODELS.md](docs/MODELS.md) lists every tensor name and
config key the engine reads.

## Validation

Correctness is checked against implementations written independently of this one.

- **Forward pass.** `scripts/oracle.py` builds tiny checkpoints and runs a
  reference forward pass in pure Python, written from the model definitions
  rather than from this code. Fixtures cover grouped-query attention with both
  qk-norm conventions, and latent attention with sigmoid gating, group-limited
  routing, a shared expert and a dense first layer. The engine reproduces the
  reference logits at every position, under incremental decode, batched prefill
  and split prefill alike.
- **Quantisation.** The same fixtures are packed and re-checked, so a packed
  model keeps both its logits and its argmax.
- **KV cache.** Rewinding the cache and continuing produces the same logits as
  never having cached — the property the server's prefix reuse rests on.
- **Tokenizer.** `scripts/tokcheck.py` compares `moe tokenize` against Hugging
  Face's `tokenizers`, in both directions, on awkward fixed cases plus generated
  whitespace, digit, code and mixed-script strings. All three vocabularies tried
  match exactly, encode and decode: **226/226 each** on OLMoE's and Qwen's
  byte-level BPE and on a metaspace one.
- **A real model.** OLMoE-1B-7B runs end to end — the recordings above are that
  model, unedited.

```console
$ cargo test                                       # 33 tests, no downloads, ~2s
$ python3 scripts/oracle.py tests/fixtures         # regenerate fixtures
$ pip install tokenizers
$ python3 scripts/tokcheck.py ~/models/qwen3-moe/tokenizer.json
```

Every commit runs the full suite on Linux, macOS and Windows, plus clippy, both
feature sets, and a pinned-MSRV build.

## How it works

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks through the whole thing —
the storage layer, the quantisation formats, the forward pass, and where the time
goes. The short version:

| File | Lines | Role |
| --- | --- | --- |
| `src/quant.rs` | 411 | block formats, dequantise + matmul kernels, AVX2 and NEON |
| `src/model.rs` | 688 | weight binding, the forward pass, the routing trace |
| `src/store.rs` | 317 | mmap safetensors and `.moe`, tensor views, packing |
| `src/spec.rs` | 205 | architecture detection from config + tensor shapes |
| `src/tokenizer.rs` | 696 | `tokenizer.json` BPE, both pre-tokenizer families |
| `src/fetch.rs` | 364 | resolving paths, URLs and Hub repos; the download cache |
| `src/serve.rs` | 455 | the OpenAI API, sessions, prefix reuse, chat formats |
| `src/http.rs` | 169 | a small bounded HTTP/1.1 server with SSE |
| `src/sample.rs` | 127 | temperature, top-k, top-p, repetition penalty |
| `src/main.rs` | 492 | CLI |

## License

MIT. See [LICENSE](LICENSE).
