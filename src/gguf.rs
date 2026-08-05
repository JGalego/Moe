//! Reading GGUF, the container llama.cpp quantises into.
//!
//! An enormous number of already-quantised mixture-of-experts checkpoints exist
//! only as GGUF, and the format suits this engine exactly: a header of metadata
//! and tensor offsets, then tensor data laid out contiguously, which is a memory
//! map and an offset calculation — the same thing the rest of the storage layer
//! already does.
//!
//! Three translations are needed. Tensor *names* differ, so they are mapped onto
//! the Hugging Face ones the engine detects architecture from. Tensor *dimensions*
//! are stored fastest-varying-first, so they reverse. And the *config* lives in
//! typed metadata keys rather than a `config.json`, so one is synthesised — along
//! with a `tokenizer.json`, since GGUF carries its vocabulary too and a file that
//! needs nothing beside it should stay that way.
//!
//! Quantisation formats mostly need no translation at all: GGUF's `Q4_0`, `Q5_0`
//! and `Q8_0` are byte-for-byte the engine's own Q4, Q5 and Q8, so those weights
//! are read in place. `Q4_K` and `Q6_K` are super-block formats and get their own
//! readers. Anything else is refused by name rather than misread.

use crate::quant::Dt;
use serde_json::{json, Map, Value};

const MAGIC: &[u8; 4] = b"GGUF";

/// A metadata value, in the handful of shapes that matter here.
#[derive(Debug, Clone, PartialEq)]
pub enum Meta {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strs(Vec<String>),
}

impl Meta {
    fn as_i64(&self) -> Option<i64> {
        match self {
            Meta::Int(v) => Some(*v),
            Meta::Float(v) => Some(*v as i64),
            Meta::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Meta::Float(v) => Some(*v),
            Meta::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Meta::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor's shape and where its bytes start, relative to the data section.
#[derive(Debug, Clone)]
pub struct Info {
    pub name: String,
    /// Logical shape, outermost first — already reversed out of GGUF's order.
    pub shape: Vec<usize>,
    pub dt: Dt,
    pub offset: usize,
}

/// A parsed GGUF header.
#[derive(Debug)]
pub struct Gguf {
    pub meta: Map<String, Value>,
    raw: std::collections::BTreeMap<String, Meta>,
    pub tensors: Vec<Info>,
    /// Where tensor data begins in the file.
    pub data: usize,
}

/// A cursor that cannot read past the end of the buffer.
struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).ok_or("offset overflow")?;
        let out = self.b.get(self.at..end).ok_or_else(|| format!("truncated at byte {}", self.at))?;
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        // A length field is the one place a corrupt file can ask for the world.
        if n > self.b.len() {
            return Err(format!("a string claims {n} bytes in a {}-byte file", self.b.len()));
        }
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    /// One typed value. `depth` stops an array of arrays from recursing.
    fn value(&mut self, kind: u32, depth: usize) -> Result<Meta, String> {
        Ok(match kind {
            0 => Meta::Int(self.take(1)?[0] as i64),
            1 => Meta::Int(self.take(1)?[0] as i8 as i64),
            2 => Meta::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Meta::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Meta::Int(self.u32()? as i64),
            5 => Meta::Int(self.u32()? as i32 as i64),
            6 => Meta::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Meta::Bool(self.take(1)?[0] != 0),
            8 => Meta::Str(self.string()?),
            9 => {
                if depth > 0 {
                    return Err("nested arrays are not supported".into());
                }
                let elem = self.u32()?;
                let n = self.u64()? as usize;
                // Every element is at least one byte, so this bounds the count.
                if n > self.b.len() {
                    return Err(format!("an array claims {n} elements in a {}-byte file", self.b.len()));
                }
                let mut ints = Vec::new();
                let mut floats = Vec::new();
                let mut strs = Vec::new();
                for _ in 0..n {
                    match self.value(elem, depth + 1)? {
                        Meta::Str(s) => strs.push(s),
                        Meta::Float(f) => floats.push(f),
                        v => ints.push(v.as_i64().unwrap_or(0)),
                    }
                }
                if !strs.is_empty() {
                    Meta::Strs(strs)
                } else if !floats.is_empty() {
                    Meta::Floats(floats)
                } else {
                    Meta::Ints(ints)
                }
            }
            10 => Meta::Int(self.u64()? as i64),
            11 => Meta::Int(self.u64()? as i64),
            12 => Meta::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(format!("unknown metadata type {other}")),
        })
    }
}

/// Map a ggml type number onto a storage format the engine can read.
fn dtype(t: u32) -> Result<Dt, String> {
    Ok(match t {
        0 => Dt::F32,
        1 => Dt::F16,
        // Q4_0, Q5_0 and Q8_0 are byte-identical to the engine's own formats.
        2 => Dt::Q4,
        6 => Dt::Q5,
        8 => Dt::Q8,
        12 => Dt::Q4K,
        14 => Dt::Q6K,
        30 => Dt::BF16,
        other => {
            let name = match other {
                3 => "Q4_1",
                7 => "Q5_1",
                9 => "Q8_1",
                10 => "Q2_K",
                11 => "Q3_K",
                13 => "Q5_K",
                15 => "Q8_K",
                16..=23 => "an IQ format",
                24..=29 => "an integer format",
                31 => "TQ1_0",
                32 => "TQ2_0",
                _ => "an unrecognised format",
            };
            return Err(format!(
                "ggml type {other} ({name}) is not supported; \
                 the readable formats are F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K and Q6_K"
            ));
        }
    })
}

/// Whether a buffer starts with GGUF's magic.
pub fn is_gguf(b: &[u8]) -> bool {
    b.len() >= 4 && &b[..4] == MAGIC
}

impl Gguf {
    pub fn parse(b: &[u8]) -> Result<Gguf, String> {
        if !is_gguf(b) {
            return Err("not a GGUF file".into());
        }
        let mut c = Cursor { b, at: 4 };
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("GGUF v{version}; this build reads v2 and v3"));
        }
        let n_tensors = c.u64()? as usize;
        let n_meta = c.u64()? as usize;
        // Both counts index into the file, so neither can exceed its length.
        if n_tensors > b.len() || n_meta > b.len() {
            return Err("header counts exceed the file size".into());
        }

        let mut raw = std::collections::BTreeMap::new();
        for _ in 0..n_meta {
            let key = c.string()?;
            let kind = c.u32()?;
            let value = c.value(kind, 0)?;
            raw.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let dims = c.u32()? as usize;
            if dims == 0 || dims > 4 {
                return Err(format!("{name} has {dims} dimensions"));
            }
            let mut shape: Vec<usize> = Vec::with_capacity(dims);
            for _ in 0..dims {
                shape.push(c.u64()? as usize);
            }
            // GGUF stores the fastest-varying dimension first; the engine thinks
            // outermost-first, so this reverses.
            shape.reverse();
            let dt = dtype(c.u32()?)?;
            let offset = c.u64()? as usize;
            tensors.push(Info { name, shape, dt, offset });
        }

        // Tensor data starts at the next alignment boundary after the header.
        let align = raw.get("general.alignment").and_then(|v| v.as_i64()).unwrap_or(32).max(1) as usize;
        let data = c.at.next_multiple_of(align);
        if data > b.len() {
            return Err("the header runs past the end of the file".into());
        }
        let meta = raw
            .iter()
            .map(|(k, v)| {
                let j = match v {
                    Meta::Int(i) => json!(i),
                    Meta::Float(f) => json!(f),
                    Meta::Bool(b) => json!(b),
                    Meta::Str(s) => json!(s),
                    Meta::Ints(v) => json!(v),
                    Meta::Floats(v) => json!(v),
                    // Vocabularies are huge; summarise rather than duplicate.
                    Meta::Strs(v) => json!(v.len()),
                };
                (k.clone(), j)
            })
            .collect();
        Ok(Gguf { meta, raw, tensors, data })
    }

    fn get(&self, key: &str) -> Option<&Meta> {
        self.raw.get(key)
    }

    pub fn arch(&self) -> &str {
        self.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("llama")
    }

    /// A key under the architecture's own namespace, as GGUF scopes them.
    fn arch_key(&self, suffix: &str) -> Option<&Meta> {
        self.get(&format!("{}.{suffix}", self.arch()))
    }

    /// Synthesise the `config.json` the engine's detection reads.
    ///
    /// Only the values that cannot be recovered from tensor shapes are needed —
    /// which is the same short list `Spec::derive` asks a Hugging Face config for.
    pub fn config(&self) -> Value {
        let mut c = Map::new();
        let int = |m: &Gguf, k: &str| m.arch_key(k).and_then(|v| v.as_i64());
        let float = |m: &Gguf, k: &str| m.arch_key(k).and_then(|v| v.as_f64());
        fn set(c: &mut Map<String, Value>, k: &str, v: Option<Value>) {
            if let Some(v) = v {
                c.insert(k.into(), v);
            }
        }
        set(&mut c, "model_type", Some(json!(self.arch())));
        set(&mut c, "architectures", Some(json!([format!("{}ForCausalLM", self.arch())])));
        set(&mut c, "num_hidden_layers", int(self, "block_count").map(|v| json!(v)));
        set(&mut c, "num_attention_heads", int(self, "attention.head_count").map(|v| json!(v)));
        set(&mut c, "num_key_value_heads", int(self, "attention.head_count_kv").map(|v| json!(v)));
        set(&mut c, "head_dim", int(self, "attention.key_length").map(|v| json!(v)));
        set(&mut c, "max_position_embeddings", int(self, "context_length").map(|v| json!(v)));
        set(&mut c, "num_experts", int(self, "expert_count").map(|v| json!(v)));
        set(&mut c, "num_experts_per_tok", int(self, "expert_used_count").map(|v| json!(v)));
        set(&mut c, "routed_scaling_factor", float(self, "expert_weights_scale").map(|v| json!(v)));
        set(&mut c, "n_group", int(self, "expert_group_count").map(|v| json!(v)));
        set(&mut c, "topk_group", int(self, "expert_group_used_count").map(|v| json!(v)));
        // 1 is softmax, 2 is sigmoid, in llama.cpp's numbering.
        if let Some(g) = int(self, "expert_gating_func") {
            set(&mut c, "scoring_func", Some(json!(if g == 2 { "sigmoid" } else { "softmax" })));
        }
        // Whether the top-k routing weights are renormalised after selection.
        //
        // Some checkpoints state it; most do not, because llama.cpp folds it into
        // the architecture rather than the metadata. Getting it wrong produces
        // fluent nonsense rather than an error — the experts are right and their
        // mixture is not — so an architecture default is not optional here.
        let norm = self
            .arch_key("expert_weights_norm")
            .and_then(|v| match v {
                Meta::Bool(b) => Some(*b),
                other => other.as_i64().map(|i| i != 0),
            })
            .unwrap_or_else(|| !matches!(self.arch(), "olmoe" | "qwen2moe" | "grok"));
        set(&mut c, "norm_topk_prob", Some(json!(norm)));
        set(&mut c, "rms_norm_eps", float(self, "attention.layer_norm_rms_epsilon").map(|v| json!(v)));
        set(&mut c, "rope_theta", float(self, "rope.freq_base").map(|v| json!(v)));
        set(&mut c, "sliding_window", int(self, "attention.sliding_window").map(|v| json!(v)));
        // Rope scaling, in GGUF's flat spelling.
        if let Some(f) = float(self, "rope.scaling.factor").filter(|f| *f > 1.0) {
            let kind = self.arch_key("rope.scaling.type").and_then(|v| v.as_str()).unwrap_or("linear").to_string();
            let mut sc = Map::new();
            sc.insert("rope_type".into(), json!(kind));
            sc.insert("factor".into(), json!(f));
            if let Some(orig) = int(self, "rope.scaling.original_context_length") {
                sc.insert("original_max_position_embeddings".into(), json!(orig));
            }
            c.insert("rope_scaling".into(), Value::Object(sc));
        }
        // GGUF names an expert-count key even for dense models; drop a zero so
        // detection does not go looking for experts that are not there.
        if c.get("num_experts").and_then(|v| v.as_i64()) == Some(0) {
            c.remove("num_experts");
            c.remove("num_experts_per_tok");
        }
        let ids = |m: &Gguf, k: &str| m.get(&format!("tokenizer.ggml.{k}")).and_then(|v| v.as_i64());
        // GGUF names a beginning-of-sequence token *and* says whether to prepend
        // it. Honouring the id but not the flag silently prefixes every prompt
        // with a token the checkpoint did not want — which does not fail, it just
        // scores and generates as a different model would.
        let add_bos = self.get("tokenizer.ggml.add_bos_token").and_then(|v| match v {
            Meta::Bool(b) => Some(*b),
            other => other.as_i64().map(|i| i != 0),
        });
        if add_bos != Some(false) {
            set(&mut c, "bos_token_id", ids(self, "bos_token_id").map(|v| json!(v)));
        }
        set(&mut c, "eos_token_id", ids(self, "eos_token_id").map(|v| json!(v)));
        Value::Object(c)
    }

    /// Build a `tokenizer.json` from the vocabulary GGUF carries.
    ///
    /// Returns `None` when the file has no vocabulary, or uses one this engine
    /// cannot express — a SentencePiece model without merges, for instance, which
    /// the BPE reader would silently mis-tokenise.
    pub fn tokenizer(&self) -> Option<Value> {
        let Some(Meta::Strs(tokens)) = self.get("tokenizer.ggml.tokens") else { return None };
        let merges = match self.get("tokenizer.ggml.merges") {
            Some(Meta::Strs(m)) => m.clone(),
            _ => Vec::new(),
        };
        let model = self.get("tokenizer.ggml.model").and_then(|v| v.as_str()).unwrap_or("gpt2");
        if merges.is_empty() && model != "gpt2" {
            return None;
        }
        let mut vocab = Map::new();
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.clone(), json!(i));
        }
        // Token types: 3 is a control token, which is what `is_special` means.
        let types = match self.get("tokenizer.ggml.token_type") {
            Some(Meta::Ints(v)) => v.clone(),
            _ => Vec::new(),
        };
        let added: Vec<Value> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| types.get(*i).copied() == Some(3))
            .map(|(i, t)| json!({"id": i, "content": t, "special": true}))
            .collect();
        // A `llama`-model vocabulary is metaspace; `gpt2` is byte-level. The
        // tokenizer reads that off the pre-tokenizer description, so say it.
        let (pre, decoder) = if model == "llama" || model == "spm" {
            (json!({"type": "Metaspace", "replacement": "\u{2581}"}), json!({"type": "Metaspace"}))
        } else {
            (json!({"type": "ByteLevel"}), json!({"type": "ByteLevel"}))
        };
        Some(json!({
            "model": {"type": "BPE", "vocab": Value::Object(vocab), "merges": merges},
            "added_tokens": added,
            "pre_tokenizer": pre,
            "decoder": decoder,
        }))
    }

    /// The chat template GGUF stores as a plain metadata string.
    pub fn chat_template(&self) -> Option<String> {
        self.get("tokenizer.chat_template").and_then(|v| v.as_str()).map(String::from)
    }
}

/// Rewrite a GGUF tensor name into the Hugging Face one the engine looks for.
///
/// Returns `None` for tensors with no counterpart — rope frequency tables and the
/// like, which the engine computes rather than reads.
pub fn rename(name: &str) -> Option<String> {
    // The model-level tensors.
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, suffix) = rest.split_once('.')?;
    layer.parse::<u32>().ok()?;
    // Longest first, so `attn_q_norm` is not matched as `attn_q`.
    const MAP: [(&str, &str); 22] = [
        ("attn_norm.weight", "input_layernorm.weight"),
        ("attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("attn_k_norm.weight", "self_attn.k_norm.weight"),
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        ("attn_q.bias", "self_attn.q_proj.bias"),
        ("attn_k.bias", "self_attn.k_proj.bias"),
        ("attn_v.bias", "self_attn.v_proj.bias"),
        ("ffn_norm.weight", "post_attention_layernorm.weight"),
        // The router, and its per-expert bias.
        ("ffn_gate_inp.weight", "mlp.gate.weight"),
        ("exp_probs_b.bias", "mlp.gate.e_score_correction_bias"),
        // Fused expert stacks, one per projection rather than a combined pair.
        ("ffn_gate_exps.weight", "mlp.experts.gate_proj"),
        ("ffn_up_exps.weight", "mlp.experts.up_proj"),
        ("ffn_down_exps.weight", "mlp.experts.down_proj"),
        // A shared expert every token also runs, and the sigmoid gate that
        // scales it. Without the gate the shared expert runs at full strength,
        // which is wrong rather than merely different.
        ("ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"),
        ("ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"),
        ("ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"),
        ("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"),
        // Dense layers, including the dense prefix of a sparse model.
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
    ];
    for (from, to) in MAP {
        if suffix == from {
            return Some(format!("model.layers.{layer}.{to}"));
        }
    }
    // `ffn_down` has to come after `ffn_down_exps` and `ffn_down_shexp`.
    if suffix == "ffn_down.weight" {
        return Some(format!("model.layers.{layer}.mlp.down_proj.weight"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal GGUF in memory, so the reader is tested against bytes
    /// rather than against a file someone has to download.
    struct Builder {
        meta: Vec<u8>,
        count: u64,
        tensors: Vec<u8>,
        n_tensors: u64,
        data: Vec<u8>,
    }

    fn wstr(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    impl Builder {
        fn new() -> Builder {
            Builder { meta: Vec::new(), count: 0, tensors: Vec::new(), n_tensors: 0, data: Vec::new() }
        }

        fn u32_kv(&mut self, key: &str, v: u32) -> &mut Self {
            wstr(&mut self.meta, key);
            self.meta.extend(4u32.to_le_bytes());
            self.meta.extend(v.to_le_bytes());
            self.count += 1;
            self
        }

        fn f32_kv(&mut self, key: &str, v: f32) -> &mut Self {
            wstr(&mut self.meta, key);
            self.meta.extend(6u32.to_le_bytes());
            self.meta.extend(v.to_le_bytes());
            self.count += 1;
            self
        }

        fn str_kv(&mut self, key: &str, v: &str) -> &mut Self {
            wstr(&mut self.meta, key);
            self.meta.extend(8u32.to_le_bytes());
            wstr(&mut self.meta, v);
            self.count += 1;
            self
        }

        fn strs_kv(&mut self, key: &str, v: &[&str]) -> &mut Self {
            wstr(&mut self.meta, key);
            self.meta.extend(9u32.to_le_bytes());
            self.meta.extend(8u32.to_le_bytes());
            self.meta.extend((v.len() as u64).to_le_bytes());
            for s in v {
                wstr(&mut self.meta, s);
            }
            self.count += 1;
            self
        }

        fn ints_kv(&mut self, key: &str, v: &[i32]) -> &mut Self {
            wstr(&mut self.meta, key);
            self.meta.extend(9u32.to_le_bytes());
            self.meta.extend(5u32.to_le_bytes());
            self.meta.extend((v.len() as u64).to_le_bytes());
            for i in v {
                self.meta.extend(i.to_le_bytes());
            }
            self.count += 1;
            self
        }

        /// `shape` is outermost-first; it is written reversed, as GGUF does.
        fn tensor(&mut self, name: &str, shape: &[usize], ggml: u32, bytes: &[u8]) -> &mut Self {
            wstr(&mut self.tensors, name);
            self.tensors.extend((shape.len() as u32).to_le_bytes());
            for d in shape.iter().rev() {
                self.tensors.extend((*d as u64).to_le_bytes());
            }
            self.tensors.extend(ggml.to_le_bytes());
            self.tensors.extend((self.data.len() as u64).to_le_bytes());
            self.data.extend(bytes);
            // Keep every tensor 32-byte aligned, as writers do.
            while self.data.len() % 32 != 0 {
                self.data.push(0);
            }
            self.n_tensors += 1;
            self
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(MAGIC);
            out.extend(3u32.to_le_bytes());
            out.extend(self.n_tensors.to_le_bytes());
            out.extend(self.count.to_le_bytes());
            out.extend(&self.meta);
            out.extend(&self.tensors);
            while out.len() % 32 != 0 {
                out.push(0);
            }
            out.extend(&self.data);
            out
        }
    }

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn a_minimal_file_parses() {
        let mut b = Builder::new();
        b.str_kv("general.architecture", "llama")
            .u32_kv("llama.block_count", 2)
            .u32_kv("llama.attention.head_count", 4)
            .u32_kv("llama.attention.head_count_kv", 2)
            .f32_kv("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .f32_kv("llama.rope.freq_base", 10000.0)
            .tensor("token_embd.weight", &[3, 4], 0, &f32s(&[1.0; 12]));
        let raw = b.build();
        let g = Gguf::parse(&raw).expect("parse");

        assert_eq!(g.arch(), "llama");
        assert_eq!(g.tensors.len(), 1);
        let t = &g.tensors[0];
        // Written reversed, read back outermost-first.
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.dt, Dt::F32);
        assert_eq!(t.offset, 0);
        assert_eq!(g.data % 32, 0, "the data section must be aligned");

        let cfg = g.config();
        assert_eq!(cfg["num_hidden_layers"], 2);
        assert_eq!(cfg["num_attention_heads"], 4);
        assert_eq!(cfg["num_key_value_heads"], 2);
        assert!((cfg["rope_theta"].as_f64().unwrap() - 10000.0).abs() < 1.0);
        assert!(cfg["num_experts"].is_null(), "a dense model must declare no experts");
    }

    #[test]
    fn a_sparse_config_comes_through() {
        let mut b = Builder::new();
        b.str_kv("general.architecture", "qwen2moe")
            .u32_kv("qwen2moe.block_count", 1)
            .u32_kv("qwen2moe.attention.head_count", 2)
            .u32_kv("qwen2moe.expert_count", 8)
            .u32_kv("qwen2moe.expert_used_count", 2)
            .u32_kv("qwen2moe.attention.sliding_window", 128)
            .f32_kv("qwen2moe.rope.scaling.factor", 4.0)
            .str_kv("qwen2moe.rope.scaling.type", "yarn")
            .u32_kv("qwen2moe.rope.scaling.original_context_length", 4096);
        let g = Gguf::parse(&b.build()).unwrap();
        let cfg = g.config();
        assert_eq!(cfg["num_experts"], 8);
        assert_eq!(cfg["num_experts_per_tok"], 2);
        assert_eq!(cfg["sliding_window"], 128);
        assert_eq!(cfg["rope_scaling"]["rope_type"], "yarn");
        assert_eq!(cfg["rope_scaling"]["factor"], 4.0);
        assert_eq!(cfg["rope_scaling"]["original_max_position_embeddings"], 4096);
        // And the spec must read it back.
        assert_eq!(cfg["architectures"][0], "qwen2moeForCausalLM");
    }

    /// Routing details that GGUF states differently, or does not state at all.
    ///
    /// Getting these wrong does not fail: the experts are right and their mixture
    /// is not, so the model emits fluent nonsense. That makes them exactly the
    /// keys worth pinning down.
    #[test]
    fn routing_details_survive_the_translation() {
        // Stated explicitly, in the form deepseek-style checkpoints use.
        let mut b = Builder::new();
        b.str_kv("general.architecture", "deepseek2")
            .u32_kv("deepseek2.expert_count", 64)
            .u32_kv("deepseek2.expert_used_count", 6)
            .u32_kv("deepseek2.expert_gating_func", 2)
            .f32_kv("deepseek2.expert_weights_scale", 2.5)
            .u32_kv("deepseek2.expert_group_count", 8)
            .u32_kv("deepseek2.expert_group_used_count", 3);
        let cfg = Gguf::parse(&b.build()).unwrap().config();
        assert_eq!(cfg["scoring_func"], "sigmoid");
        assert_eq!(cfg["routed_scaling_factor"], 2.5);
        assert_eq!(cfg["n_group"], 8);
        assert_eq!(cfg["topk_group"], 3);
        // Absent, so it takes the general default.
        assert_eq!(cfg["norm_topk_prob"], true);

        // OLMoE and Qwen1.5-MoE do not renormalise, and no GGUF of either says
        // so — llama.cpp folds it into the architecture. The default has to know
        // the same table, because the mistake is silent.
        for arch in ["olmoe", "qwen2moe", "grok"] {
            let mut o = Builder::new();
            o.str_kv("general.architecture", arch).u32_kv(&format!("{arch}.expert_count"), 64);
            assert_eq!(Gguf::parse(&o.build()).unwrap().config()["norm_topk_prob"], false, "{arch}");
        }
        // Everything else does, Qwen3-MoE included — it changed between versions.
        for arch in ["llama", "qwen3moe", "deepseek2"] {
            let mut o = Builder::new();
            o.str_kv("general.architecture", arch).u32_kv(&format!("{arch}.expert_count"), 64);
            assert_eq!(Gguf::parse(&o.build()).unwrap().config()["norm_topk_prob"], true, "{arch}");
        }

        // An explicit key always wins over the architecture default.
        let mut e = Builder::new();
        e.str_kv("general.architecture", "olmoe")
            .u32_kv("olmoe.expert_count", 64)
            .u32_kv("olmoe.expert_weights_norm", 1);
        assert_eq!(Gguf::parse(&e.build()).unwrap().config()["norm_topk_prob"], true);

        // A gating function of 1 is softmax, the common case.
        let mut g = Builder::new();
        g.str_kv("general.architecture", "qwen2moe").u32_kv("qwen2moe.expert_gating_func", 1);
        assert_eq!(Gguf::parse(&g.build()).unwrap().config()["scoring_func"], "softmax");
    }

    #[test]
    fn a_vocabulary_becomes_a_tokenizer() {
        let mut b = Builder::new();
        b.str_kv("general.architecture", "llama")
            .str_kv("tokenizer.ggml.model", "gpt2")
            .strs_kv("tokenizer.ggml.tokens", &["a", "b", "ab", "<|end|>"])
            .strs_kv("tokenizer.ggml.merges", &["a b"])
            .ints_kv("tokenizer.ggml.token_type", &[1, 1, 1, 3])
            .u32_kv("tokenizer.ggml.eos_token_id", 3);
        let g = Gguf::parse(&b.build()).unwrap();
        let j = g.tokenizer().expect("a tokenizer");
        assert_eq!(j["model"]["vocab"]["ab"], 2);
        assert_eq!(j["model"]["merges"][0], "a b");
        // The one control token is marked as added and special.
        assert_eq!(j["added_tokens"][0]["id"], 3);
        assert_eq!(j["added_tokens"][0]["special"], true);
        assert_eq!(j["pre_tokenizer"]["type"], "ByteLevel");
        // ...and it must actually load.
        let tok = crate::Tokenizer::from_json(&j).expect("load");
        assert!(tok.is_special(3));
        assert!(!tok.is_special(0));
        assert_eq!(g.config()["eos_token_id"], 3);

        // A SentencePiece vocabulary with no merges cannot be expressed, and
        // saying so beats mis-tokenising.
        let mut spm = Builder::new();
        spm.str_kv("general.architecture", "llama")
            .str_kv("tokenizer.ggml.model", "llama")
            .strs_kv("tokenizer.ggml.tokens", &["x"]);
        assert!(Gguf::parse(&spm.build()).unwrap().tokenizer().is_none());
    }

    #[test]
    fn names_map_onto_the_engines_own() {
        for (from, to) in [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"),
            ("blk.7.attn_q.weight", "model.layers.7.self_attn.q_proj.weight"),
            ("blk.7.attn_q_norm.weight", "model.layers.7.self_attn.q_norm.weight"),
            ("blk.7.attn_output.weight", "model.layers.7.self_attn.o_proj.weight"),
            ("blk.1.ffn_norm.weight", "model.layers.1.post_attention_layernorm.weight"),
            ("blk.1.ffn_gate_inp.weight", "model.layers.1.mlp.gate.weight"),
            ("blk.1.ffn_gate_exps.weight", "model.layers.1.mlp.experts.gate_proj"),
            ("blk.1.ffn_down_exps.weight", "model.layers.1.mlp.experts.down_proj"),
            ("blk.1.ffn_down.weight", "model.layers.1.mlp.down_proj.weight"),
            ("blk.1.ffn_gate_shexp.weight", "model.layers.1.mlp.shared_expert.gate_proj.weight"),
            // The shared expert's gate. Its name starts with the router's, so a
            // prefix match here would silently make one of them the other.
            ("blk.1.ffn_gate_inp_shexp.weight", "model.layers.1.mlp.shared_expert_gate.weight"),
        ] {
            assert_eq!(rename(from).as_deref(), Some(to), "{from}");
        }
        // Tensors the engine computes rather than reads have no counterpart.
        assert_eq!(rename("rope_freqs.weight"), None);
        assert_eq!(rename("blk.0.something_else.weight"), None);
        assert_eq!(rename("blk.x.attn_q.weight"), None);
    }

    /// A corrupt or hostile header must be refused, never trusted into an
    /// out-of-bounds read. This is the only code that parses a downloaded file.
    #[test]
    fn malformed_headers_are_refused() {
        let good = {
            let mut b = Builder::new();
            b.str_kv("general.architecture", "llama").tensor("token_embd.weight", &[2, 2], 0, &f32s(&[0.0; 4]));
            b.build()
        };
        assert!(Gguf::parse(&good).is_ok());

        assert!(Gguf::parse(b"").is_err());
        assert!(Gguf::parse(b"NOPE").is_err());
        // Every truncation must be an error, not a panic.
        for cut in 0..good.len() {
            let _ = Gguf::parse(&good[..cut]);
        }
        // A wrong version is refused by name.
        let mut bad_version = good.clone();
        bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(Gguf::parse(&bad_version).unwrap_err().contains("v99"));
        // An absurd tensor count cannot be trusted.
        let mut huge = good.clone();
        huge[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Gguf::parse(&huge).is_err());
        // Nor an absurd metadata count.
        let mut huge_meta = good.clone();
        huge_meta[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Gguf::parse(&huge_meta).is_err());
    }

    #[test]
    fn unsupported_quantisations_are_named() {
        let mut b = Builder::new();
        // 13 is Q5_K, which the engine does not read.
        b.str_kv("general.architecture", "llama").tensor("token_embd.weight", &[2, 256], 13, &[0u8; 512]);
        let e = Gguf::parse(&b.build()).unwrap_err();
        assert!(e.contains("Q5_K"), "{e}");
        assert!(e.contains("Q4_K"), "the error should say what is readable: {e}");
    }
}
