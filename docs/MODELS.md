# Models

Moe does not have a per-architecture code path. It has one sparse-decoder
implementation and a detection step that reads the checkpoint. A model works if
its tensors are in that shape and use recognisable names.

## What the engine implements

- RMSNorm; SwiGLU feed-forward
- Rotary embeddings, half-split layout, with four context-extension schemes:
  linear interpolation, NTK-aware base raising, Llama 3's piecewise-by-wavelength
  variant, and YaRN — including its attention-scale correction and DeepSeek's
  two-term `mscale` form
- Sliding-window attention, per layer, in all four config conventions, so
  checkpoints alternating local and global layers work
- Grouped-query attention, with optional projection biases and optional qk-norm,
  applied per head or across the whole projection depending on how wide the
  checkpoint's norm weight is
- Multi-head latent attention: low-rank queries, compressed KV, decoupled rotary
  keys
- Top-k routing with softmax or sigmoid scores, an optional correction bias,
  optional weight renormalisation, a routed scaling factor, and group-limited
  selection
- Shared experts, with or without their own sigmoid gate
- Per-layer dense/routed choice, so dense prefixes and every-Nth-layer sparsity
  come for free
- Tied or separate output embeddings

## Tensor names

Layer tensors are `<prefix><layer>.<suffix>`, where the prefix is detected
(usually `model.layers.`). Alternatives are tried in order.

| Role | Names |
| --- | --- |
| embedding | `model.embed_tokens.weight`, `embed_tokens.weight`, `transformer.wte.weight` |
| final norm | `model.norm.weight`, `norm.weight`, `model.final_layernorm.weight` |
| output | `lm_head.weight`, `output.weight`, else tied to the embedding |
| layer norms | `input_layernorm.weight`, `post_attention_layernorm.weight` |
| attention | `self_attn.{q,k,v,o}_proj.weight` (+ `.bias`), `self_attn.{q,k}_norm.weight` |
| latent attention | `self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,kv_a_layernorm,kv_b_proj,o_proj}` |
| router | `mlp.gate.weight`, `block_sparse_moe.gate.weight` |
| router bias | `mlp.gate.e_score_correction_bias` |
| experts (per expert) | `mlp.experts.<e>.{gate,up,down}_proj.weight`, `block_sparse_moe.experts.<e>.{w1,w3,w2}.weight` |
| experts (fused pair) | `mlp.experts.gate_up_proj` `[E, 2*inter, hidden]`, `mlp.experts.down_proj` `[E, hidden, inter]` |
| experts (stacked) | `mlp.experts.{gate,up,down}_proj` `[E, rows, cols]` — one 3-D stack per projection, which is what GGUF writes |
| shared expert | `mlp.shared_expert.*`, `mlp.shared_experts.*` |
| shared gate | `mlp.shared_expert_gate.weight` |
| dense ffn | `mlp.{gate,up,down}_proj.weight`, `mlp.{w1,w3,w2}.weight` |

In the fused-pair layout the first half of `gate_up_proj`'s rows is the gate
projection and the second half is the up projection, matching a `chunk(2)` on the
output of a single linear. All three layouts address an expert as a row range, so
the forward pass does not care which one it got, and the expert count is taken
from the tensors rather than the config — a checkpoint whose config disagrees with
its weights still loads.

### GGUF names

A GGUF file's tensors are renamed onto the table above before anything else looks
at them, so detection, packing and pruning all work unchanged. `token_embd` is the
embedding, `output_norm` the final norm, `output` the head; per layer,
`attn_norm`, `attn_{q,k,v,output}`, `attn_{q,k}_norm`, `ffn_norm`, `ffn_gate_inp`
(the router), `ffn_{gate,up,down}_exps` (the stacked experts),
`ffn_{gate,up,down}_shexp` (a shared expert) and `ffn_{gate,up,down}` (a dense
layer). Dimensions are reversed on the way in, because GGUF stores the
fastest-varying first. Tensors with no counterpart — rope frequency tables — are
skipped, since the engine computes them.

## Config keys

Only these are read; everything else is inferred from tensor shapes. Aliases are
tried in order, and each has a fallback.

`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`,
`rms_norm_eps`, `max_position_embeddings`, `tie_word_embeddings`,
`num_experts_per_tok`, `norm_topk_prob`, `scoring_func`, `routed_scaling_factor`,
`n_group`, `topk_group`, `bos_token_id`, `eos_token_id`.

Rope: `rope_theta` (also under `rope_parameters`), and from `rope_scaling` (or
`rope_parameters`) the keys `rope_type` / `type`, `factor`,
`original_max_position_embeddings`, `low_freq_factor`, `high_freq_factor`,
`beta_fast`, `beta_slow`, `attention_factor`, `mscale`, `mscale_all_dim`.

Windows: `sliding_window` or `attention_window_size`, plus whichever of
`layer_types`, `sliding_window_pattern`, or `use_sliding_window` with
`max_window_layers` the checkpoint uses to say which layers are local.

A missing `config.json` is not fatal, but `num_attention_heads` has no sensible
default, so it is the one value that must be present. A GGUF file has no
`config.json` at all; one is synthesised from its metadata keys.

`tokenizer_config.json` is read for one thing: the Jinja `chat_template`, which is
also looked for as a `chat_template.jinja` file beside the weights. Both are
carried into a packed `.moe`.

## Status

| Family | Notes |
| --- | --- |
| OLMoE | whole-projection qk-norm, `norm_topk_prob: false` — run end to end on the 7B checkpoint |
| Mixtral | fused or per-expert weights, both handled |
| Qwen2-MoE | shared expert with a sigmoid gate |
| Qwen3-MoE | per-head qk-norm, fused experts |
| DeepSeek-V2 / V3 | latent attention, sigmoid gating, group-limited routing, shared experts, dense prefix |
| Dense Llama-style | no routed layers; the same path without routing |
| Any of the above as GGUF | `Q4_0`, `Q5_0` and `Q8_0` read in place; `Q4_K` and `Q6_K` via their own readers |

If a checkpoint in this family does not load, `moe info` names the first tensor
it could not resolve, which is nearly always the answer.

## Roadmap

The next architectures on the list, in rough order:

- **MXFP4 weights and attention sinks**, which together with the sliding windows
  already implemented bring GPT-OSS in. Its fused expert tensors are transposed
  and interleaved relative to the layouts above, so it needs a fourth branch as
  well.
- **FP8 checkpoints**, which the safetensors reader will pick up once the dtype
  is understood.
- **NFC normalisation** in the tokenizer, for text that arrives decomposed.
- **The remaining GGUF quantisations** — `Q2_K` through `Q5_K`, and the IQ
  formats. Each is a `dequant` arm rather than a new kernel.
- **A SentencePiece unigram tokenizer**, which a GGUF `llama`-model vocabulary
  needs and which the BPE reader currently declines rather than approximating.
- **Encoder-decoder, multimodal and hybrid Mamba/SSM stacks**, each of which is a
  new block type rather than a new name.

## Adding one

If a checkpoint is in this family but does not load, in rough order of
likelihood: a tensor name not in the table above (add an alternative in
`Model::load_with`, or in `gguf::rename` for a GGUF file), a fused layout with a
different split or transpose (add a branch to the `expert` closure), or a
genuinely new mechanism (`model.rs`). For
anything in the last category, add a fixture to `scripts/oracle.py` first — a
reference implementation written from the model definition is what makes the
result trustworthy.
