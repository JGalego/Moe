# Moe

**Better call Moe!** — a small CPU inference engine for sparse mixture-of-experts
language models.

One binary, ~2,700 lines of Rust, three dependencies, no BLAS, no GPU, no Python
at runtime. Point it at a Hugging Face checkpoint and it works out what the model
is by reading the weights.

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

$ moe run ~/models/mixtral -p "A sparse model is one where" -n 64
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

Needs a Rust toolchain (1.75+). Nothing else.

```console
$ cargo build --release        # target/release/moe
$ cargo test                   # full suite, no checkpoint needed, ~2s
```

For the best kernels on your own machine, build with
`RUSTFLAGS="-C target-cpu=native" cargo build --release`.

## Use

Run straight from a Hugging Face download:

```console
$ moe run ~/models/mixtral -p "The capital of France is" -n 40 --temp 0.7
```

Or pack it first. Packing re-quantises every weight, drops tensors inference
never reads, and embeds `tokenizer.json`, giving one self-contained file:

```console
$ moe pack ~/models/mixtral -o mixtral.moe --quant q8 --expert-quant q4
$ moe run mixtral.moe -p "Explain routing in one sentence." -n 80 --stats
```

`--quant` covers the dense trunk, `--expert-quant` the routed experts. Since the
experts are the bulk of the file and the least sensitive part of it, `--quant q8
--expert-quant q4` is a good default: Q8 costs 8.5 bits per weight and Q4 costs
4.5, so a bf16 checkpoint shrinks 1.88x and 3.56x respectively. Norms, biases and
router gates always stay f32 — they are tiny, and quantisation noise there feeds
straight into routing decisions.

Other commands:

```console
$ moe info  model.moe                  # architecture, footprint, kv cache size
$ moe bench model.moe -n 64            # prefill and decode throughput
$ moe tokenize model.moe -p "hello"    # token ids, for debugging
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
| `src/quant.rs` | 380 | block formats, dequantise + matmul kernels, AVX2 dot |
| `src/store.rs` | 317 | mmap safetensors and `.moe`, tensor views, packing |
| `src/spec.rs` | 205 | architecture detection from config + tensor shapes |
| `src/model.rs` | 629 | weight binding and the forward pass |
| `src/tokenizer.rs` | 591 | `tokenizer.json` BPE, both pre-tokenizer families |
| `src/sample.rs` | 127 | temperature, top-k, top-p, repetition penalty |
| `src/main.rs` | 372 | CLI |

## Limitations

Worth knowing before you file an issue:

- Attention is exact and quadratic in context; there is no flash-attention
  kernel and no paged KV cache. The KV cache is f32 and preallocated for `--ctx`.
- Activations are always f32. Only weights are quantised.
- Hand-written SIMD covers x86-64 AVX2/FMA; elsewhere the scalar kernels rely on
  the autovectoriser, which is decent but not the same thing.
- Throughput numbers are not quoted here because they depend entirely on your
  machine's memory bandwidth. Run `moe bench` and you will have real ones.

## License

MIT. See [LICENSE](LICENSE).
