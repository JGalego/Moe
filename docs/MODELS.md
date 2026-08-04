# Models

Moe does not have a per-architecture code path. It has one sparse-decoder
implementation and a detection step that reads the checkpoint. A model works if
its tensors are in that shape and use recognisable names.

## What the engine implements

- RMSNorm; SwiGLU feed-forward
- Rotary embeddings, half-split layout, optional linear scaling factor
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
| experts (fused) | `mlp.experts.gate_up_proj` `[E, 2*inter, hidden]`, `mlp.experts.down_proj` `[E, hidden, inter]` |
| shared expert | `mlp.shared_expert.*`, `mlp.shared_experts.*` |
| shared gate | `mlp.shared_expert_gate.weight` |
| dense ffn | `mlp.{gate,up,down}_proj.weight`, `mlp.{w1,w3,w2}.weight` |

In the fused layout the first half of `gate_up_proj`'s rows is the gate
projection and the second half is the up projection, matching a `chunk(2)` on the
output of a single linear.

## Config keys

Only these are read; everything else is inferred from tensor shapes. Aliases are
tried in order, and each has a fallback.

`num_hidden_layers`, `num_attention_heads`, `head_dim`, `rms_norm_eps`,
`rope_theta` (also under `rope_parameters`), `rope_scaling.factor`,
`max_position_embeddings`, `tie_word_embeddings`, `num_experts_per_tok`,
`norm_topk_prob`, `scoring_func`, `routed_scaling_factor`, `n_group`,
`topk_group`, `bos_token_id`, `eos_token_id`.

A missing `config.json` is not fatal, but `num_attention_heads` has no sensible
default, so it is the one value that must be present.

## Status

| Family | Notes | Status |
| --- | --- | --- |
| OLMoE | whole-projection qk-norm, `norm_topk_prob: false` | run end to end on the real 7B checkpoint |
| Mixtral | fused or per-expert weights, both handled | loaded, packed and generated from a random-weight checkpoint in that format |
| Qwen2-MoE | shared expert with a sigmoid gate | expected; the mechanism is implemented and unit-tested, no checkpoint run |
| Qwen3-MoE | per-head qk-norm, fused experts | expected; same |
| DeepSeek-V2 / V3 | latent attention, sigmoid gating, group-limited routing, shared experts, dense prefix | every mechanism is covered by the reference test, but no real checkpoint has been run |
| Dense Llama-style | no routed layers | works; it is the same path without routing |

Only the OLMoE row means "a real trained checkpoint went through this binary and
produced correct text". The rest say the implementation exists and is tested
against an independent reference at toy scale. If one of them fails on a real
file, `moe info` names the first tensor it could not resolve, which is nearly
always the answer.

## Not supported

- **MXFP4 weights and attention sinks / sliding-window attention**, so GPT-OSS
  will not load. Its fused expert tensors are also transposed and
  interleaved relative to the layout above.
- **FP8 checkpoints.** The safetensors reader skips dtypes it does not know, so
  these surface as a missing-tensor error.
- **YaRN, NTK and other non-linear rope scaling.** A plain `factor` is applied as
  linear scaling; anything else is ignored, which will hurt beyond the base
  context length.
- **NFC normalisation** in the tokenizer. Already-normalised text — essentially
  everything in practice — is unaffected.
- **Encoder-decoder and multimodal models.**
- **Mamba/SSM and other non-attention blocks**, including hybrid stacks.

## Adding one

If a checkpoint is in this family but does not load, in rough order of
likelihood: a tensor name not in the table above (add an alternative in
`Model::load_with`), a fused layout with a different split or transpose (add a
branch to the `expert` closure), or a genuinely new mechanism (`model.rs`). For
anything in the last category, add a fixture to `scripts/oracle.py` first — a
reference implementation written from the model definition is what makes the
result trustworthy.
