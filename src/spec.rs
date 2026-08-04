//! Model description.
//!
//! Most of a sparse decoder's shape is already implied by the checkpoint: hidden
//! size, vocabulary, expert width, whether attention is latent, whether a layer
//! is dense or routed. So [`Spec`] reads only the handful of values that cannot
//! be recovered from tensor shapes, and everything else is detected from the
//! weights themselves, which is why new checkpoints usually need no code.

use crate::store::Store;
use serde_json::Value;

/// Multi-head latent attention dimensions (compressed KV, decoupled RoPE).
#[derive(Clone, Copy, Debug)]
pub struct Mla {
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_head: usize,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub arch: String,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f32,
    pub rope_scale: f32,
    pub max_ctx: usize,
    pub tie: bool,
    pub qk_norm: bool,
    pub mla: Option<Mla>,
    /// Routed experts per MoE layer (0 for a fully dense checkpoint).
    pub experts: usize,
    pub top_k: usize,
    pub norm_topk: bool,
    /// Sigmoid gating (`scoring_func: sigmoid`) instead of softmax.
    pub sigmoid: bool,
    pub routed_scale: f32,
    pub n_group: usize,
    pub topk_group: usize,
    /// Prefix shared by per-layer tensor names, e.g. `model.layers.`.
    pub prefix: String,
    pub eos: Vec<u32>,
    pub bos: Option<u32>,
}

fn num(cfg: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| cfg.get(*k).and_then(|v| v.as_f64()))
}

fn flag(cfg: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter().find_map(|k| cfg.get(*k).and_then(|v| v.as_bool()))
}

fn ids(cfg: &Value, key: &str) -> Vec<u32> {
    match cfg.get(key) {
        Some(Value::Number(n)) => n.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_u64().map(|v| v as u32)).collect(),
        _ => Vec::new(),
    }
}

/// Detect the shared per-layer tensor prefix (`model.layers.`, `layers.`, ...).
fn detect_prefix(store: &Store) -> String {
    for n in store.names() {
        if let Some(i) = n.find(".0.") {
            let head = &n[..i + 1];
            if head.ends_with("layers.") || head.ends_with("h.") || head.ends_with("blocks.") {
                return head.to_string();
            }
        }
    }
    "model.layers.".to_string()
}

impl Spec {
    /// Derive a spec from a store's config plus the shapes actually present.
    pub fn derive(store: &Store) -> Result<Spec, String> {
        let cfg = &store.config;
        let prefix = detect_prefix(store);
        let l0 = |suffix: &str| format!("{prefix}0.{suffix}");

        let embed = store
            .any(&["model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight"])
            .ok_or("no embedding tensor (looked for model.embed_tokens.weight)")?;
        let (vocab, hidden) = (embed.rows, embed.cols);

        let layers = num(cfg, &["num_hidden_layers", "n_layers", "n_layer"]).map(|v| v as usize).unwrap_or_else(|| {
            (0..).take_while(|l| store.names().any(|n| n.starts_with(&format!("{prefix}{l}.")))).count()
        });
        let heads =
            num(cfg, &["num_attention_heads", "n_heads", "n_head"]).ok_or("config lacks num_attention_heads")? as usize;

        let mla = store.get(&l0("self_attn.kv_a_proj_with_mqa.weight")).map(|kva| {
            let qk_rope = num(cfg, &["qk_rope_head_dim"]).unwrap_or(64.0) as usize;
            Mla {
                q_lora: num(cfg, &["q_lora_rank"]).unwrap_or(0.0) as usize,
                kv_lora: num(cfg, &["kv_lora_rank"]).unwrap_or((kva.rows - qk_rope) as f64) as usize,
                qk_nope: num(cfg, &["qk_nope_head_dim"]).unwrap_or(128.0) as usize,
                qk_rope,
                v_head: num(cfg, &["v_head_dim"]).unwrap_or(128.0) as usize,
            }
        });

        let (head_dim, kv_heads) = match &mla {
            Some(m) => (m.qk_rope, heads),
            None => {
                let q = store.get(&l0("self_attn.q_proj.weight")).ok_or("no self_attn.q_proj.weight in layer 0")?;
                let hd = num(cfg, &["head_dim"]).map(|v| v as usize).unwrap_or(q.rows / heads);
                let kv = store.get(&l0("self_attn.k_proj.weight")).map(|k| k.rows / hd).unwrap_or(heads);
                (hd, kv.max(1))
            }
        };

        // Experts may be one tensor each or a single fused `[E, rows, cols]`
        // stack; either way the count comes from the checkpoint, not the config.
        let fused = (0..layers).find_map(|l| store.shape(&format!("{prefix}{l}.mlp.experts.gate_up_proj")));
        let experts = match fused {
            Some((e, _, _)) => e,
            None => (0..)
                .take_while(|e| {
                    (0..layers.min(2)).any(|l| {
                        let p = format!("{prefix}{l}.");
                        store.has(&format!("{p}mlp.experts.{e}.down_proj.weight"))
                            || store.has(&format!("{p}block_sparse_moe.experts.{e}.w2.weight"))
                    })
                })
                .count(),
        };

        let scoring = cfg.get("scoring_func").and_then(|v| v.as_str()).unwrap_or("softmax");
        let sigmoid = scoring == "sigmoid" || store.has(&l0("mlp.gate.e_score_correction_bias"));

        Ok(Spec {
            arch: cfg["architectures"][0]
                .as_str()
                .or_else(|| cfg["model_type"].as_str())
                .unwrap_or("unknown")
                .to_string(),
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            vocab,
            eps: num(cfg, &["rms_norm_eps", "layer_norm_eps"]).unwrap_or(1e-6) as f32,
            // Recent configs moved rope settings under `rope_parameters`.
            theta: num(cfg, &["rope_theta"])
                .or_else(|| cfg["rope_parameters"]["rope_theta"].as_f64())
                .unwrap_or(10000.0) as f32,
            rope_scale: cfg["rope_scaling"]["factor"]
                .as_f64()
                .or_else(|| cfg["rope_parameters"]["factor"].as_f64())
                .unwrap_or(1.0) as f32,
            max_ctx: num(cfg, &["max_position_embeddings"]).unwrap_or(8192.0) as usize,
            tie: flag(cfg, &["tie_word_embeddings"]).unwrap_or(false) || !store.has("lm_head.weight"),
            qk_norm: store.has(&l0("self_attn.q_norm.weight")),
            mla,
            experts,
            top_k: num(cfg, &["num_experts_per_tok", "moe_top_k", "top_k"]).unwrap_or(2.0) as usize,
            norm_topk: flag(cfg, &["norm_topk_prob", "norm_expert_prob"]).unwrap_or(true),
            sigmoid,
            routed_scale: num(cfg, &["routed_scaling_factor"]).unwrap_or(1.0) as f32,
            n_group: num(cfg, &["n_group"]).unwrap_or(1.0) as usize,
            topk_group: num(cfg, &["topk_group"]).unwrap_or(1.0) as usize,
            prefix,
            eos: ids(cfg, "eos_token_id"),
            bos: ids(cfg, "bos_token_id").first().copied(),
        })
    }

    pub fn summary(&self) -> String {
        let attn = match &self.mla {
            Some(m) => format!("MLA(kv_lora={}, rope={}, nope={}, v={})", m.kv_lora, m.qk_rope, m.qk_nope, m.v_head),
            None => format!("GQA({} q / {} kv heads, head_dim={})", self.heads, self.kv_heads, self.head_dim),
        };
        let moe = if self.experts == 0 {
            "dense".to_string()
        } else {
            format!(
                "{} experts, top-{}{}{}",
                self.experts,
                self.top_k,
                if self.sigmoid { ", sigmoid gate" } else { ", softmax gate" },
                if self.n_group > 1 { format!(", {} groups", self.n_group) } else { String::new() }
            )
        };
        format!(
            "{}\n  layers {}  hidden {}  vocab {}\n  attention  {}{}\n  ffn        {}",
            self.arch,
            self.layers,
            self.hidden,
            self.vocab,
            attn,
            if self.qk_norm { " + qk-norm" } else { "" },
            moe
        )
    }
}
