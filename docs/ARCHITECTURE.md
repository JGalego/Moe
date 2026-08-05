# How Moe works

Each module owns one stage of getting from a model spec to a token.

```
main.rs        CLI: run, chat, serve, pull, pack, info, bench, route, eval, embed
  fetch.rs     where the model comes from -> paths, URLs, Hub repos, the cache
  serve.rs     the OpenAI API             -> the prefix-cache pool, metrics
  http.rs      the socket                 -> a bounded HTTP/1.1 server with SSE
  chat.rs      a prompt from a turn       -> the Jinja subset templates need
  store.rs     where weights live        -> mmap safetensors / .moe / GGUF, views
  gguf.rs      the GGUF container        -> metadata, names, dimensions
  spec.rs      what the model is         -> shape detection from config + tensors
  model.rs     the forward pass          -> attention, routing, experts, pooling
  quant.rs     the arithmetic            -> block formats, dequant + matmul kernels
  generate.rs  the decode loop           -> speculation, verification, stopping
  draft.rs     guessing ahead            -> lookup drafting, no second model
  grammar.rs   constraining output       -> the JSON automaton and the logit mask
  tokenizer.rs text <-> ids
  sample.rs    ids <- logits, and verifying a draft against them
  route.rs     what the routing did      -> statistics and the SVG
  eval.rs      how surprised it was      -> perplexity, bits per byte
```

Three of those are downstream of one observation. Because a routed layer touches
only what it selected, the routing is *visible* — so `route.rs` reports it,
`model.rs` lets it be overridden, and `store.rs` can prune a checkpoint down to
the part a workload reaches. Nothing else in the engine has that shape.

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

`Store` is one lookup API over three backends. A Hugging Face directory is a set
of `*.safetensors` shards: each has an 8-byte length, a JSON header naming every
tensor with its dtype, shape and byte range, then the payload. A `.moe` file is
the same idea with a different header — `MOEF`, a version, a JSON blob holding
the original `config.json`, an embedded `tokenizer.json`, the chat template and
the tensor index, then the payload. A **GGUF** file is the same idea again, with
typed binary metadata instead of JSON.

GGUF needs three translations and no conversion. Tensor names are rewritten onto
the Hugging Face ones detection reads (`blk.3.ffn_gate_exps.weight` becomes
`model.layers.3.mlp.experts.gate_proj`); dimensions are reversed, because GGUF
stores the fastest-varying first; and the config, vocabulary and chat template are
synthesised from metadata keys. The *weights* are untouched, because GGUF's
`Q4_0`, `Q5_0` and `Q8_0` blocks are byte-for-byte the engine's Q4, Q5 and Q8.
`Q4_K` and `Q6_K` are 256-value super-block formats with per-sub-block scales, and
get their own readers; they are read-only, since writing them well needs the
search llama.cpp does at quantisation time.

Every length and offset in all three headers is checked before use, and each
tensor is validated to lie inside the map when the index is built. These are the
only parsers that read bytes the user did not write, and on a memory-mapped file a
bounds mistake is worse than a crash. `tests/malformed.rs` sweeps them with
mutations and truncations on every commit; `fuzz/` goes deeper.

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

Three expert layouts exist in the wild and which one a checkpoint uses is visible
in its tensors rather than its config: one fused `[E, 2*inter, hidden]` stack
holding gate and up together, a separate 3-D stack per projection (what GGUF
writes), or a tensor per expert. All three address an expert as a row range, so
the forward pass does not care which it got.

### Packing

`moe pack` walks the index, picks a target format per tensor, and rewrites the
file. Tensors inference never reads (`rotary_emb`, `inv_freq`) are dropped. Norms,
biases, router gates and anything 1-D stay f32. Everything else goes to
`--quant`, except routed experts which go to `--expert-quant`. A tensor whose
width is not a multiple of the 32-value block falls back to f32 rather than
silently changing shape.

Two refinements ride on the same walk. `--hot-experts` reads a trace and gives the
experts a workload leans on a finer format than the rest, which is a better trade
than one rate for all of them because routing is skewed. And `--keep-experts`
prunes: kept experts are renumbered contiguously, the router is narrowed to
exactly the rows that survive — in the same order — the per-expert gate bias is
narrowed by *column* instead, fused stacks keep whole slabs, and the config is
rewritten so a loader detects the count it actually has. What comes out is a
smaller model, not a checkpoint with holes in it.

Packing is optional. Running from a raw Hugging Face directory or a GGUF works and
gives identical results; packing exists to make the file smaller and its layout
more convenient, and to fold the tokenizer and chat template in.

## Quantisation and kernels

Weights are stored as `F32`, `F16`, `BF16`, or block-quantised:

| Format | Layout per 32 values | Bits/weight |
| --- | --- | --- |
| `Q8` | f16 scale + 32 int8 | 8.5 |
| `Q6` | f16 scale + 16 byte pairs + a 2-bit plane | 6.5 |
| `Q5` | f16 scale + a 1-bit plane + 16 byte pairs | 5.5 |
| `Q4` | f16 scale + 16 packed byte pairs | 4.5 |

All four are symmetric, and all four share one layout so the same indexing
reasoning holds for each: value `i` in the low nibble of byte `i`, value `i + 16`
in the high nibble. `Q8` stores `round(v / d)` with `d = max|v| / 127`; `Q4`
stores `round(v / d) + 8` in a nibble with `d = max|v| / 7`; `Q5` and `Q6` add the
remaining bits in a trailing plane — one bit each packed eight to a byte, or two
bits packed four to a byte. `Q5`'s plane sits *before* its nibbles, which is
GGUF's `Q5_0` layout, so such a tensor is read with no conversion.

GGUF's two K-quant formats are read as well. They are 256-value super-blocks with
a scale per 32-value sub-block rather than one for the whole block, which is what
buys their accuracy, and they are read-only: writing them well needs the search
llama.cpp does at quantisation time, so `Dt::parse` does not accept their names
and the CLI cannot ask for one.

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

## Decoding

`generate.rs` holds the decode loop, and one invariant explains its shape:
`pending` is a token that has been committed and handed to the caller but is not
yet in the KV cache. Every round forwards it — together with any draft riding
behind it — and ends with the next committed token in its place.

With no draft, that round is an ordinary single-token decode. So speculation is
not a second code path: `--draft N` asks `draft.rs` for a guess at the next N
tokens, the round forwards `[pending, ...draft]` in one step, and each row of the
result judges the draft token after it. Whatever prefix the model agrees with is
kept and the rest of the cache is rewound — which is the same `State::truncate`
the server's prefix reuse uses, so both rest on the same tested property.

Verification lives in `sample.rs`, next to the sampler whose distribution it has
to match. Greedily, a guess is accepted exactly when it is the argmax. With
temperature, it is accepted with probability `p(guess)` and a rejection draws from
the target with the guess removed and renormalised — the standard correction for a
drafter that proposes one token with certainty, which every lookup drafter is.
That is why speculation is lossless rather than merely close, and why the
correction is checked empirically rather than argued for.

A constraint hooks the same rows. `grammar.rs` compiles a JSON Schema into a
pushdown automaton over *bytes* — bytes, because a token is an arbitrary byte
string that straddles any boundary — and masks every candidate whose bytes could
not continue a valid document. Its state is `Copy` and shallow, so testing a
candidate is a stack copy of a few dozen bytes; over a 50k vocabulary that is a
small fraction of the forward pass it rides along with. Constraints and
speculation compose with no special case: a drafted token that would break the
grammar has probability zero and is rejected like any other bad guess.

## Serving

`moe serve` puts a pool of `State` behind one mutex and lets connections queue on
it. Serialising is not a placeholder for batching: at batch size one the engine is
memory-bandwidth-bound and rayon already has every core, so a second concurrent
generation would take throughput from the first rather than add any. Real
concurrency needs continuous batching — shared prefill, per-sequence KV pools, a
scheduler — which is a different engine.

Queueing is what buys the prefix cache. Each slot remembers the token ids that
produced its cache; a new request takes the slot sharing the longest prefix with
it, calls `State::truncate` to move that cursor back, and prefills only the
remainder. Conversations grow by appending, so each turn prefills the message that
was added rather than the whole transcript.

The pool exists because one slot is not enough: with a single cache, two clients
taking turns evict each other on every request and the feature that made a second
turn cheap does nothing at all. `--slots N` holds several; a request that matches
nothing takes the coldest rather than evicting a conversation that may come back.
The cost is exact and worth stating rather than burying — one full KV cache per
slot — so the default is 1 and the memory is opted into.

Truncation is cheap because the KV cache is append-only per position: nothing
needs clearing, since whatever is forwarded next overwrites it. What does get
cleared is the per-layer previous-selection used for the reuse statistic,
because the token before the cursor has changed.

A wrong cache hit would produce *silently wrong* output rather than an error, so
the comparison is on exact token ids rather than a hash, `--no-prefix-cache`
turns it off, and `truncating_the_cache_matches_a_fresh_state` asserts that
rewinding and continuing gives the same logits as never having cached. That
assertion is on logits deliberately: the difference a stale cache makes is around
1e-3, which changes no argmax on the fixtures and would sail past any test that
compared generated tokens.

Chat formats come from the checkpoint too, but by *declaration* rather than
inference: instruct models ship their format as a Jinja `chat_template` in
`tokenizer_config.json`, and `chat.rs` renders it. That is a subset of Jinja — the
tags, operators, tests and filters chat templates actually use — and anything
outside it is a parse error rather than a silent misrendering.

When a template cannot be rendered, the older path takes over: a vocabulary with
`<|im_start|>` is chatml, `<|start_header_id|>` is llama3, `[/INST]` is mistral,
each expressed as a per-role prefix and suffix plus the opening of the assistant's
turn. Falling back to a working guess beats failing, and both beat rendering a
prompt wrongly. A checkpoint matching neither gets an error, not a guess.

## Reading the routing

Everything in `route.rs` is a statistic over the flat `(position, layer, experts)`
list a trace is. Three of them are worth the name. Normalised **entropy** says
whether a router is using its capacity — 1.0 is an even spread, 0.0 is every token
choosing the same expert. **Peak over uniform** says how lopsided the busiest
expert is. **Coverage** says what fraction of `(layer, expert)` pairs a prompt
touched at all, which is the number that says whether pruning would pay.

The SVG is the same numbers. Colour follows the job: one hue with monotone
lightness for magnitude, two hues either side of a neutral grey for polarity, both
interpolated in OKLab so the steps are perceptually even. Those two properties are
asserted in a test rather than eyeballed — a sequential ramp must rise
monotonically in lightness, and a diverging midpoint must be the least chromatic
step, or zero reads as a value.

`Routing` in `model.rs` is the other direction: refuse an expert and the weights
renormalise over what remains, force one and it displaces the weakest pick, raise
the router temperature and the choice flattens. It is empty by default and checked
only when not, so an ordinary run pays nothing for it existing.

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
