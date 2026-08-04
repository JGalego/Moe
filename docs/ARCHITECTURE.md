# How Moe works

Each module owns one stage of getting from a model spec to a token.

```
main.rs      CLI: run, pull, pack, info, bench, tokenize
  fetch.rs   where the model comes from -> paths, URLs, Hub repos, the cache
  store.rs   where weights live         -> mmap safetensors / .moe, tensor views
  spec.rs    what the model is          -> shape detection from config + tensors
  model.rs   the forward pass           -> attention, routing, experts
  quant.rs   the arithmetic             -> block formats, dequant + matmul kernels
  tokenizer.rs  text <-> ids
  sample.rs  ids <- logits
```

## Resolution

Every command takes a model spec and turns it into a local path before anything
else happens. A spec that exists on disk is used as-is; otherwise `owner/name` is
a Hub repo (with `@revision`, or `hf:` to force the reading), and an `http(s)`
URL is a download — except a Hub *model page*, which is recognised and treated as
the repo it names.

Repo snapshots take only what inference reads: `config.json`, `tokenizer.json`
and top-level `*.safetensors`, or a published `.moe` on its own if the repo has
one. Subdirectories are skipped, which is what keeps ONNX exports, GGUF
conversions and `.bin` duplicates out of the download. Each file is written to a
`.part` and renamed on completion, and a file already present at the expected
size is skipped, so an interrupted download costs one file rather than the lot.

Downloads honour `HTTPS_PROXY` and `SSL_CERT_FILE`, so a machine behind a
corporate proxy needs no rebuild, and `HF_TOKEN` for gated repos. The whole
module sits behind the `fetch` feature: `--no-default-features` builds an engine
with no TLS stack that takes local paths only.

## Storage

`Store` is one lookup API over two backends. A Hugging Face directory is a set of
`*.safetensors` shards: each has an 8-byte length, a JSON header naming every
tensor with its dtype, shape and byte range, then the payload. A `.moe` file is
the same idea with a different header — `MOEF`, a version, a JSON blob holding
the original `config.json`, an embedded `tokenizer.json` and the tensor index,
then the payload.

Both are memory mapped, and the maps are deliberately leaked. They live as long
as the process anyway, and leaking them means every weight view can be
`&'static`, which in turn means views are `Copy`, `Send` and `Sync` with no
lifetime plumbing and no reference counting on the hot path. Loading a model is
an `mmap` plus a header parse. The pages themselves arrive when something reads
them, which is what lets a checkpoint bigger than RAM work with no streaming mode
to enable: the kernel pages in what routing touches and evicts what it does not.
`--warm N` faults in the first N GB up front when you would rather pay that cost
before the first token.

A tensor record is `(slabs, rows, cols)`. `slabs` is 1 for an ordinary matrix and
the expert count for a fused `[experts, rows, cols]` stack; either way the
payload is `slabs * rows` contiguous rows of `cols` values. `Store::view` takes a
slab index and a row range, so one expert inside a fused stack is an offset
calculation, and no unpacking step exists to be skipped.

### Packing

`moe pack` walks the index, picks a target format per tensor, and rewrites the
file. Tensors inference never reads (`rotary_emb`, `inv_freq`) are dropped. Norms,
biases, router gates and anything 1-D stay f32. Everything else goes to
`--quant`, except routed experts which go to `--expert-quant`. A tensor whose
width is not a multiple of the 32-value block falls back to f32 rather than
silently changing shape.

Packing is optional. Running from a raw Hugging Face directory works and gives
identical results; packing exists to make the file smaller and its layout more
convenient, and to fold the tokenizer in.

## Quantisation and kernels

Weights are stored as `F32`, `F16`, `BF16`, or block-quantised:

| Format | Layout per 32 values | Bits/weight |
| --- | --- | --- |
| `Q8` | f16 scale + 32 int8 | 8.5 |
| `Q4` | f16 scale + 16 packed byte pairs | 4.5 |

Both are symmetric: `Q8` stores `round(v / d)` with `d = max|v| / 127`, `Q4`
stores `round(v / d) + 8` in a nibble with `d = max|v| / 7`, low nibbles holding
the first half of the block and high nibbles the second.

The kernel is deliberately uniform. Instead of a specialised inner loop per
format, `matmul` dequantises one weight row into a small f32 scratch buffer and
then dots it against every activation vector in the batch:

```
for each row r (in parallel bands):
    dequantise row r  ->  buf[cols]
    for each token t:  out[t][r] = dot(buf, x[t])
```

Adding a format means adding a `dequant` arm, not a kernel. The dequantisation
cost is amortised across the batch, so prefill reads each weight once instead of
once per token. And there is exactly one hot loop to optimise — `dot` — which has
hand-written AVX2/FMA (detected at runtime) and NEON paths, plus a scalar
fallback shaped to autovectorise. The scratch buffer is `cols` f32, which stays
in L1 for realistic hidden sizes.

Rows are split into bands across the rayon pool; results accumulate row-major and
are transposed once at the end, an `O(rows * t)` shuffle next to `O(rows * cols *
t)` of arithmetic.

## Architecture detection

`Spec::derive` reads the handful of values that genuinely cannot be recovered
from the weights — head count, rms epsilon, rope theta, top-k, gating flags — and
infers everything else:

| Property | How it is found |
| --- | --- |
| hidden size, vocabulary | embedding tensor shape |
| head dim, kv heads | `q_proj` / `k_proj` row counts |
| latent attention | `kv_a_proj_with_mqa` exists |
| qk-norm | `self_attn.q_norm.weight` exists |
| expert count | fused stack's leading dimension, or how many per-expert tensors exist |
| routed vs dense layer | whether *that layer* carries experts |
| shared expert | `mlp.shared_expert(s).*` exists |
| shared expert gate | `mlp.shared_expert_gate.weight` exists |
| sigmoid gating | `scoring_func`, or an `e_score_correction_bias` |
| tied embeddings | config flag, or no `lm_head.weight` |
| layer name prefix | scanned from the tensor names |

Detecting per-layer rather than globally is what makes dense-prefix models
(`first_k_dense_replace`) and every-Nth-layer sparsity (`decoder_sparse_step`,
`mlp_only_layers`) work without a single line about any of them.

## The forward pass

One function, `Model::forward(tokens, state)`, runs `t` tokens starting at
`state.pos` and returns the last token's logits. Decode is `t == 1`; prefill is a
chunk. There is no separate incremental path, so there is nothing for the two to
disagree about — a property the test suite checks directly.

Per layer: RMSNorm, attention, residual, RMSNorm, feed-forward, residual.

### Attention

Both front-ends produce the same thing — per-head queries, keys and values —
after which the causal softmax attention is shared.

**Grouped-query.** `q`, `k`, `v` projections, optional biases, optional qk-norm,
then rotary embedding. The norm's width picks the convention: a weight as wide as
one head normalises each head separately, one as wide as the whole projection
normalises it in a single pass. Keys and values are appended to the cache. Query
head `h` reads kv head `h / (heads/kv_heads)`.

**Latent.** Queries optionally pass through a low-rank bottleneck
(`q_a` → norm → `q_b`). Keys and values come from one compressed vector: `kv_a`
produces a latent part plus a rotary part shared across heads, the latent part is
normed and expanded by `kv_b` into per-head no-rope keys and values, and each
head's key is that no-rope half concatenated with the shared rotary half. The
cache holds the expanded per-head keys and values, which trades memory for a
simpler and faster inner loop.

Rotary embedding is the half-split form Hugging Face checkpoints are exported in:
element `i` pairs with `i + dim/2`. Attention runs one task per `(token, head)`.

### Routing

Router logits go through softmax or sigmoid. Selection uses the scores plus an
optional correction bias; group-limited routing scores each group by its top two
experts, keeps the strongest groups and masks the rest. The top-k weights are
then taken from the *scores* (not the biased ranks), optionally renormalised to
sum to one, and scaled by `routed_scaling_factor`.

Then the routing table is inverted. Rather than looping over tokens and gathering
experts, the layer loops over the experts that were selected at all, gathers the
tokens that chose each one, and runs that expert once over all of them in
parallel. An expert's weights are touched once per step regardless of how many
tokens picked it, and unselected experts are never read — the reason a step's
cost tracks `top_k`, not the model's size. Shared experts, when present, run over
every token and are added on top, with a sigmoid gate if the checkpoint has one.

## Tokenizer

`tokenizer.json` is read directly. Two pre-tokenizer families cover this model
family: byte-level (GPT-2 style, using the printable-codepoint bijection) and
metaspace (sentencepiece style, with `<0xXX>` byte fallback).

Byte-level checkpoints declare a `Split` regex, and every one of them uses the
same alternation with small variations. Rather than embed a regex engine, Moe
reads that regex and extracts the variations as flags — whether a word may absorb
any non-alphanumeric character or only a space, whether digits are single, up to
three, or whole runs, whether newlines bind to symbols, whether contractions are
case-insensitive — then applies the alternation directly, in order. The result is
checked against Hugging Face's own tokenizer (see `scripts/tokcheck.py`).

Merging uses a heap over live neighbour pairs, ordered by rank then position,
which matters because metaspace checkpoints declare no pre-tokenizer at all and
hand BPE the entire prompt as one word — where a naive quadratic scan would be
painful.

Decoding works in bytes, not strings, because one character can span several
tokens. `Stream` holds back an incomplete tail so streaming output never prints a
broken glyph.

## Where the time goes

Per decoded token, a routed layer does:

- attention projections: `O(hidden * (q + 2*kv + out))`
- attention itself: `O(heads * pos * head_dim)`, growing with context
- routing: `O(hidden * experts)`, negligible
- experts: `O(top_k * 3 * hidden * inter)`, the bulk

All of it is memory-bound at batch size one: every weight byte is read once and
used for a couple of flops. Decode speed is therefore roughly *bytes of selected
weights ÷ memory bandwidth*, which is why the format you pack to matters more
than almost anything else, and why prefill — which reuses each row across the
whole chunk — is so much faster per token than decode.

`moe bench` measures both. `moe run --stats` additionally reports how many expert
activations happened, how many repeated the previous token's choice, and how many
expert bytes were touched.

## Tracing the routing

`--trace` turns on `State::trace`, and every routed layer appends its `(position,
layer, [(expert, weight)])` decision as it makes it. The cost is one allocation
per token per routed layer, so it is off unless asked for.

The CLI writes it as JSONL: a header line naming the model, its layer and expert
counts and its top-k, then one self-contained object per token per routed layer.
Dense layers write nothing, so a layer absent from the file is a layer without
experts. `scripts/routeviz.py` turns one file into a heatmap and two into their
difference, and prints the same numbers as text.

Weights in the trace are the ones actually applied: after the top-k cut, after
renormalisation, and after `routed_scaling_factor` — so they sum to that factor
rather than to 1.
