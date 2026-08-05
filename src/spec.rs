//! Model description.
//!
//! Most of a sparse decoder's shape is already implied by the checkpoint: hidden
//! size, vocabulary, expert width, whether attention is latent, whether a layer
//! is dense or routed. So [`Spec`] reads only the handful of values that cannot
//! be recovered from tensor shapes, and everything else is detected from the
//! weights themselves, which is why new checkpoints usually need no code.

use crate::store::Store;
use serde_json::Value;

/// How a checkpoint stretches rotary embeddings past the context it was trained
/// on. The choice matters: applied wrongly, a model looks fine for a few hundred
/// tokens and then degrades silently, which is worse than refusing to load.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RopeKind {
    /// Trained context only.
    #[default]
    None,
    /// Position interpolation: divide positions by the factor, squeezing the
    /// whole rotation into the trained range.
    Linear,
    /// NTK-aware: raise the base instead, which spreads the loss of resolution
    /// across frequencies rather than concentrating it in the high ones.
    Dynamic,
    /// YaRN: interpolate per frequency between those two, with a smooth ramp and
    /// a compensating factor on the attention scale.
    Yarn,
    /// Llama 3's piecewise scheme: scale the low frequencies, leave the high ones
    /// alone, ramp between.
    Llama3,
}

/// Everything the rotary embedding needs, resolved once at load.
#[derive(Clone, Copy, Debug)]
pub struct Rope {
    pub kind: RopeKind,
    pub theta: f32,
    pub factor: f32,
    /// Context the checkpoint was trained for, which is what the scaling is
    /// relative to.
    pub original_ctx: usize,
    /// Llama 3: which wavelengths count as low and high.
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    /// YaRN: the frequency band over which the ramp runs.
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// YaRN: extra multiplier on the attention scale.
    pub attn_factor: f32,
    /// DeepSeek's YaRN variant reports the two mscale terms separately.
    pub mscale: f32,
    pub mscale_all_dim: f32,
}

impl Default for Rope {
    fn default() -> Rope {
        Rope {
            kind: RopeKind::None,
            theta: 10000.0,
            factor: 1.0,
            original_ctx: 4096,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
            mscale: 0.0,
            mscale_all_dim: 0.0,
        }
    }
}

/// YaRN's length-dependent correction to the attention scale.
fn yarn_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

impl Rope {
    /// Per-dimension inverse frequencies for a head of `head_dim` values.
    ///
    /// Computed once rather than per token per head: the unscaled form needs a
    /// `powf` per dimension, which the forward pass has no business repeating
    /// millions of times.
    pub fn inv_freqs(&self, head_dim: usize) -> Vec<f32> {
        let half = head_dim / 2;
        let dim = head_dim as f32;
        // The base frequencies, before any stretching.
        let base = |theta: f32| -> Vec<f32> { (0..half).map(|i| theta.powf(-2.0 * i as f32 / dim)).collect() };
        let s = self.factor.max(1e-6);
        match self.kind {
            RopeKind::None => base(self.theta),
            RopeKind::Linear => base(self.theta).into_iter().map(|f| f / s).collect(),
            // Scaling the base by s^(dim/(dim-2)) is the NTK-aware equivalent of
            // interpolating positions, without crushing the high frequencies.
            RopeKind::Dynamic => {
                let theta = self.theta * s.powf(dim / (dim - 2.0).max(1.0));
                base(theta)
            }
            RopeKind::Llama3 => {
                let l = self.original_ctx as f32;
                let (lo_wave, hi_wave) = (l / self.low_freq_factor, l / self.high_freq_factor);
                base(self.theta)
                    .into_iter()
                    .map(|f| {
                        let wave = 2.0 * std::f32::consts::PI / f;
                        if wave > lo_wave {
                            // Long wavelengths carry position over distances the
                            // model never saw; those are the ones to compress.
                            f / s
                        } else if wave < hi_wave {
                            f
                        } else {
                            let t = ((l / wave) - self.low_freq_factor)
                                / (self.high_freq_factor - self.low_freq_factor).max(1e-6);
                            (1.0 - t) * f / s + t * f
                        }
                    })
                    .collect()
            }
            RopeKind::Yarn => {
                let l = self.original_ctx as f32;
                let ln_base = self.theta.ln().max(1e-6);
                // The dimensions whose wavelength brackets the trained context:
                // below `low` extrapolate, above `high` interpolate, ramp between.
                let bound = |beta: f32| dim * (l / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * ln_base);
                let (low, high) = (bound(self.beta_fast).floor(), bound(self.beta_slow).ceil());
                let span = (high - low).max(1e-3);
                base(self.theta)
                    .into_iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let ramp = ((i as f32 - low) / span).clamp(0.0, 1.0);
                        // ramp 1 = fully extrapolated (unscaled), 0 = interpolated.
                        (1.0 - ramp) * (f / s) + ramp * f
                    })
                    .collect()
            }
        }
    }

    /// Multiplier on the attention scale. YaRN compensates for the entropy the
    /// stretched rotation adds; everything else leaves the scale alone.
    pub fn attn_scale(&self) -> f32 {
        if self.kind != RopeKind::Yarn {
            return 1.0;
        }
        // DeepSeek reports two mscale terms and uses their ratio; the original
        // formulation is the single-term case.
        if self.mscale > 0.0 || self.mscale_all_dim > 0.0 {
            let num = yarn_mscale(self.factor, self.mscale);
            let den = yarn_mscale(self.factor, self.mscale_all_dim);
            return num / den.max(1e-6) * self.attn_factor;
        }
        yarn_mscale(self.factor, 1.0) * self.attn_factor
    }

    pub fn summary(&self) -> String {
        match self.kind {
            RopeKind::None => format!("theta {:.0}", self.theta),
            k => format!("theta {:.0}, {k:?} x{:.3} from {}", self.theta, self.factor, self.original_ctx),
        }
    }
}

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
    pub rope: Rope,
    /// Attention window per layer, or `None` where a layer sees everything.
    /// Modern checkpoints alternate local and global layers, and which is which
    /// is declared rather than inferable from the weights.
    pub windows: Vec<Option<u32>>,
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

        // Experts may be one tensor each, a single fused `[E, 2*inter, hidden]`
        // stack, or one 3-D stack per projection as GGUF writes them. Either way
        // the count comes from the checkpoint rather than the config, so a
        // checkpoint whose config disagrees with its weights still loads.
        let stacked = (0..layers).find_map(|l| {
            ["mlp.experts.gate_up_proj", "mlp.experts.gate_proj", "mlp.experts.down_proj"]
                .iter()
                .find_map(|n| store.shape(&format!("{prefix}{l}.{n}")))
                .filter(|(e, _, _)| *e > 1)
        });
        let experts = match stacked {
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
        let max_ctx = num(cfg, &["max_position_embeddings"]).unwrap_or(8192.0) as usize;

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
            rope: rope_of(cfg, max_ctx),
            windows: windows_of(cfg, layers),
            max_ctx,
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
        // Only mention windows when some layer actually has one.
        let local = self.windows.iter().filter(|w| w.is_some()).count();
        let window = match self.windows.iter().flatten().max() {
            Some(w) if local == self.layers => format!("\n  window     {w} on every layer"),
            Some(w) => format!("\n  window     {w} on {local} of {} layers", self.layers),
            None => String::new(),
        };
        format!(
            "{}\n  layers {}  hidden {}  vocab {}\n  attention  {}{}\n  rope       {}{}\n  ffn        {}",
            self.arch,
            self.layers,
            self.hidden,
            self.vocab,
            attn,
            if self.qk_norm { " + qk-norm" } else { "" },
            self.rope.summary(),
            window,
            moe
        )
    }
}

/// Read the rope settings, wherever this generation of configs keeps them.
fn rope_of(cfg: &Value, max_ctx: usize) -> Rope {
    // Recent configs moved everything under `rope_parameters`; older ones split
    // `rope_theta` from a `rope_scaling` sub-object.
    let scaling = if cfg["rope_scaling"].is_object() { &cfg["rope_scaling"] } else { &cfg["rope_parameters"] };
    let theta = num(cfg, &["rope_theta"]).or_else(|| cfg["rope_parameters"]["rope_theta"].as_f64()).unwrap_or(10000.0);
    let mut r = Rope { theta: theta as f32, ..Rope::default() };
    r.original_ctx = scaling["original_max_position_embeddings"].as_u64().map(|v| v as usize).unwrap_or(max_ctx);

    let named = scaling["rope_type"].as_str().or_else(|| scaling["type"].as_str()).unwrap_or("");
    let factor = scaling["factor"].as_f64().unwrap_or(1.0) as f32;
    r.kind = match named {
        "linear" => RopeKind::Linear,
        "dynamic" | "ntk" => RopeKind::Dynamic,
        "yarn" => RopeKind::Yarn,
        "llama3" => RopeKind::Llama3,
        // A factor with no type named is the original position-interpolation
        // paper, which is what `linear` means.
        _ if factor > 1.0 => RopeKind::Linear,
        _ => RopeKind::None,
    };
    // A factor of 1 stretches nothing, whatever it is called.
    if factor <= 1.0 {
        r.kind = RopeKind::None;
    }
    r.factor = factor;
    let f = |k: &str, d: f32| scaling[k].as_f64().map(|v| v as f32).unwrap_or(d);
    r.low_freq_factor = f("low_freq_factor", 1.0);
    r.high_freq_factor = f("high_freq_factor", 4.0);
    r.beta_fast = f("beta_fast", 32.0);
    r.beta_slow = f("beta_slow", 1.0);
    r.attn_factor = f("attention_factor", 1.0);
    r.mscale = f("mscale", 0.0);
    r.mscale_all_dim = f("mscale_all_dim", 0.0);
    r
}

/// Which layers attend locally, and how far back.
///
/// Four conventions exist and they disagree, so all four are read: an explicit
/// `layer_types` list, a `sliding_window_pattern` where every Nth layer is
/// global, Qwen's `use_sliding_window` with a `max_window_layers` cutoff, and a
/// bare `sliding_window` meaning every layer.
fn windows_of(cfg: &Value, layers: usize) -> Vec<Option<u32>> {
    let w = num(cfg, &["sliding_window", "attention_window_size"]).map(|v| v as u32).filter(|w| *w > 0);
    let Some(w) = w else { return vec![None; layers] };

    if let Some(types) = cfg["layer_types"].as_array() {
        return (0..layers)
            .map(|l| match types.get(l).and_then(|t| t.as_str()) {
                Some("sliding_attention") | Some("local_attention") => Some(w),
                _ => None,
            })
            .collect();
    }
    if let Some(pattern) = cfg["sliding_window_pattern"].as_u64().filter(|p| *p > 1) {
        // Every `pattern`-th layer sees everything; the rest are local.
        return (0..layers).map(|l| if (l + 1) % pattern as usize == 0 { None } else { Some(w) }).collect();
    }
    if cfg["use_sliding_window"].as_bool() == Some(false) {
        return vec![None; layers];
    }
    // Qwen applies the window only from `max_window_layers` upward.
    let from = cfg["max_window_layers"].as_u64().map(|v| v as usize).unwrap_or(0);
    (0..layers).map(|l| if l >= from { Some(w) } else { None }).collect()
}

#[cfg(test)]
mod detection {
    use super::*;

    fn rope(cfg: serde_json::Value) -> Rope {
        rope_of(&cfg, 4096)
    }

    #[test]
    fn a_config_with_no_scaling_scales_nothing() {
        let r = rope(serde_json::json!({"rope_theta": 500000.0}));
        assert_eq!(r.kind, RopeKind::None);
        assert!((r.theta - 500000.0).abs() < 1.0);
        // The unscaled frequencies are the plain geometric series.
        let inv = r.inv_freqs(8);
        assert_eq!(inv.len(), 4);
        assert!((inv[0] - 1.0).abs() < 1e-6, "the first frequency is always 1");
        assert!(inv.windows(2).all(|w| w[1] < w[0]), "frequencies must decrease");
        assert!((r.attn_scale() - 1.0).abs() < 1e-6);
    }

    /// Every named scheme must be recognised, and a factor of 1 must disable all
    /// of them however it is spelled.
    #[test]
    fn every_scheme_is_recognised() {
        for (name, kind) in [
            ("linear", RopeKind::Linear),
            ("dynamic", RopeKind::Dynamic),
            ("ntk", RopeKind::Dynamic),
            ("yarn", RopeKind::Yarn),
            ("llama3", RopeKind::Llama3),
        ] {
            let r = rope(serde_json::json!({"rope_scaling": {"rope_type": name, "factor": 4.0}}));
            assert_eq!(r.kind, kind, "{name}");
            assert!((r.factor - 4.0).abs() < 1e-6);
            // Older configs spell the key `type`.
            let old = rope(serde_json::json!({"rope_scaling": {"type": name, "factor": 4.0}}));
            assert_eq!(old.kind, kind, "{name} under `type`");
            // A factor of 1 is not a stretch.
            let flat = rope(serde_json::json!({"rope_scaling": {"rope_type": name, "factor": 1.0}}));
            assert_eq!(flat.kind, RopeKind::None, "{name} with factor 1");
        }
        // A bare factor is the original interpolation paper.
        assert_eq!(rope(serde_json::json!({"rope_scaling": {"factor": 2.0}})).kind, RopeKind::Linear);
    }

    /// Linear scaling divides every frequency equally; that is what makes it the
    /// scheme that costs the most high-frequency resolution.
    #[test]
    fn linear_divides_every_frequency() {
        let base = rope(serde_json::json!({"rope_theta": 10000.0})).inv_freqs(16);
        let lin =
            rope(serde_json::json!({"rope_theta": 10000.0, "rope_scaling": {"rope_type": "linear", "factor": 4.0}}))
                .inv_freqs(16);
        for (b, l) in base.iter().zip(&lin) {
            assert!((l - b / 4.0).abs() < 1e-9, "{b} -> {l}");
        }
    }

    /// NTK raises the base instead, so the *highest* frequency is left almost
    /// untouched while the low ones absorb the stretch.
    #[test]
    fn ntk_spares_the_high_frequencies() {
        let base = rope(serde_json::json!({"rope_theta": 10000.0})).inv_freqs(64);
        let ntk =
            rope(serde_json::json!({"rope_theta": 10000.0, "rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}))
                .inv_freqs(64);
        let lin =
            rope(serde_json::json!({"rope_theta": 10000.0, "rope_scaling": {"rope_type": "linear", "factor": 8.0}}))
                .inv_freqs(64);
        // Dimension 0 is untouched by NTK and divided by 8 by linear.
        assert!((ntk[0] - base[0]).abs() < 1e-9);
        assert!((lin[0] - base[0] / 8.0).abs() < 1e-9);
        // The lowest frequency ends up compressed at least as hard as linear.
        let last = ntk.len() - 1;
        assert!(ntk[last] <= lin[last] * 1.05, "ntk {} vs linear {}", ntk[last], lin[last]);
    }

    /// Llama 3 leaves short wavelengths alone, divides long ones, and ramps in
    /// between — so the result is bracketed by unscaled and fully scaled.
    #[test]
    fn llama3_is_piecewise_between_the_two_extremes() {
        let cfg = serde_json::json!({"rope_theta": 500000.0, "rope_scaling":
            {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
             "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}});
        let r = rope(cfg);
        let base = rope(serde_json::json!({"rope_theta": 500000.0})).inv_freqs(128);
        let got = r.inv_freqs(128);
        assert_eq!(got.len(), base.len());
        for (i, (b, g)) in base.iter().zip(&got).enumerate() {
            assert!(*g <= b + 1e-9, "dim {i} was scaled up");
            assert!(*g >= b / 8.0 - 1e-9, "dim {i} was scaled past the factor");
        }
        // The highest frequency is short-wavelength and must be untouched.
        assert!((got[0] - base[0]).abs() < 1e-9);
        // The lowest is long-wavelength and must be fully divided.
        let last = got.len() - 1;
        assert!((got[last] - base[last] / 8.0).abs() < 1e-9);
    }

    /// YaRN interpolates per frequency and corrects the attention scale, which
    /// no other scheme does.
    #[test]
    fn yarn_ramps_and_rescales_attention() {
        let r = rope(serde_json::json!({"rope_theta": 10000.0, "rope_scaling":
            {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}}));
        let base = rope(serde_json::json!({"rope_theta": 10000.0})).inv_freqs(64);
        let got = r.inv_freqs(64);
        for (i, (b, g)) in base.iter().zip(&got).enumerate() {
            assert!(*g <= b + 1e-9 && *g >= b / 4.0 - 1e-9, "dim {i}: {g} outside [{}, {b}]", b / 4.0);
        }
        // 0.1 * ln(4) + 1
        let want = 0.1 * 4.0f32.ln() + 1.0;
        assert!((r.attn_scale() - want).abs() < 1e-6, "{} vs {want}", r.attn_scale());
        // Only YaRN touches the scale.
        assert!(
            (rope(serde_json::json!({"rope_scaling": {"rope_type": "linear", "factor": 4.0}})).attn_scale() - 1.0)
                .abs()
                < 1e-9
        );
    }

    /// DeepSeek states the two mscale terms separately and uses their ratio.
    #[test]
    fn deepseeks_yarn_uses_the_mscale_ratio() {
        let r = rope(serde_json::json!({"rope_scaling":
            {"rope_type": "yarn", "factor": 40.0, "mscale": 1.0, "mscale_all_dim": 1.0}}));
        // Equal terms cancel exactly.
        assert!((r.attn_scale() - 1.0).abs() < 1e-6, "{}", r.attn_scale());
        let r2 = rope(serde_json::json!({"rope_scaling":
            {"rope_type": "yarn", "factor": 40.0, "mscale": 1.0, "mscale_all_dim": 0.0}}));
        assert!(r2.attn_scale() > 1.0);
    }

    #[test]
    fn a_config_without_a_window_attends_globally() {
        let w = windows_of(&serde_json::json!({}), 4);
        assert_eq!(w, vec![None; 4]);
        // A zero window is not a window.
        assert_eq!(windows_of(&serde_json::json!({"sliding_window": 0}), 3), vec![None; 3]);
    }

    #[test]
    fn each_window_convention_is_read() {
        // A bare window applies everywhere.
        assert_eq!(windows_of(&serde_json::json!({"sliding_window": 128}), 3), vec![Some(128); 3]);

        // An explicit list wins.
        let listed = serde_json::json!({"sliding_window": 128,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention"]});
        assert_eq!(windows_of(&listed, 3), vec![Some(128), None, Some(128)]);

        // A pattern makes every Nth layer global.
        let pattern = serde_json::json!({"sliding_window": 64, "sliding_window_pattern": 4});
        assert_eq!(windows_of(&pattern, 4), vec![Some(64), Some(64), Some(64), None]);

        // Qwen switches it on from a given depth.
        let qwen = serde_json::json!({"sliding_window": 32, "use_sliding_window": true, "max_window_layers": 2});
        assert_eq!(windows_of(&qwen, 4), vec![None, None, Some(32), Some(32)]);

        // ...and can decline it outright while still declaring the size.
        let off = serde_json::json!({"sliding_window": 32, "use_sliding_window": false});
        assert_eq!(windows_of(&off, 2), vec![None, None]);
    }
}
