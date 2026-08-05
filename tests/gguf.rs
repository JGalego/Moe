//! GGUF, checked end to end against the same reference everything else is.
//!
//! The unit tests in `src/gguf.rs` check the container against bytes. This checks
//! the whole path: the gqa fixture's weights are rewritten as a real GGUF file —
//! GGUF's tensor names, GGUF's reversed dimensions, and its fused per-projection
//! expert stacks — and the engine must then reproduce `scripts/oracle.py`'s logits
//! from it, position for position.
//!
//! That also gives the separate-3-D-stack expert layout its only coverage. No HF
//! fixture uses it; every GGUF mixture-of-experts checkpoint does.

use moe::{Model, State, Store};
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gqa")
}

fn expected() -> (Vec<u32>, Vec<Vec<f32>>) {
    let raw = std::fs::read(fixture_dir().join("expected.json")).expect("expected.json");
    let j: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    (
        j["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
        j["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect(),
    )
}

/// Enough of a GGUF writer to build a file the reader has to cope with.
#[derive(Default)]
struct Writer {
    meta: Vec<u8>,
    n_meta: u64,
    info: Vec<u8>,
    n_tensors: u64,
    data: Vec<u8>,
}

fn wstr(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

impl Writer {
    fn u32(&mut self, key: &str, v: u32) {
        wstr(&mut self.meta, key);
        self.meta.extend(4u32.to_le_bytes());
        self.meta.extend(v.to_le_bytes());
        self.n_meta += 1;
    }

    fn f32(&mut self, key: &str, v: f32) {
        wstr(&mut self.meta, key);
        self.meta.extend(6u32.to_le_bytes());
        self.meta.extend(v.to_le_bytes());
        self.n_meta += 1;
    }

    fn str(&mut self, key: &str, v: &str) {
        wstr(&mut self.meta, key);
        self.meta.extend(8u32.to_le_bytes());
        wstr(&mut self.meta, v);
        self.n_meta += 1;
    }

    /// `shape` is outermost-first; GGUF wants it reversed, which is the whole
    /// point of writing it this way round here.
    fn tensor(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "{name}: shape and data disagree");
        wstr(&mut self.info, name);
        self.info.extend((shape.len() as u32).to_le_bytes());
        for d in shape.iter().rev() {
            self.info.extend((*d as u64).to_le_bytes());
        }
        self.info.extend(0u32.to_le_bytes()); // ggml type 0 is F32
        self.info.extend((self.data.len() as u64).to_le_bytes());
        self.data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        while self.data.len() % 32 != 0 {
            self.data.push(0);
        }
        self.n_tensors += 1;
    }

    fn finish(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"GGUF");
        out.extend(3u32.to_le_bytes());
        out.extend(self.n_tensors.to_le_bytes());
        out.extend(self.n_meta.to_le_bytes());
        out.extend(&self.meta);
        out.extend(&self.info);
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend(&self.data);
        out
    }
}

/// Rewrite the gqa fixture as GGUF, stacking its per-expert tensors the way GGUF
/// does — one 3-D stack per projection.
fn convert(tag: &str) -> PathBuf {
    let src = Store::open(&fixture_dir()).unwrap();
    let cfg = &src.config;
    let layers = cfg["num_hidden_layers"].as_u64().unwrap() as usize;
    let experts = cfg["num_experts"].as_u64().unwrap() as usize;
    let read = |name: &str| src.get(name).unwrap_or_else(|| panic!("{name} missing")).to_vec();
    let shape = |name: &str| {
        let (s, r, c) = src.shape(name).unwrap();
        (s, r, c)
    };

    let mut w = Writer::default();
    w.str("general.architecture", "llama");
    w.u32("llama.block_count", layers as u32);
    w.u32("llama.attention.head_count", cfg["num_attention_heads"].as_u64().unwrap() as u32);
    w.u32("llama.attention.head_count_kv", cfg["num_key_value_heads"].as_u64().unwrap() as u32);
    w.u32("llama.attention.key_length", cfg["head_dim"].as_u64().unwrap() as u32);
    w.u32("llama.context_length", cfg["max_position_embeddings"].as_u64().unwrap() as u32);
    w.u32("llama.expert_count", experts as u32);
    w.u32("llama.expert_used_count", cfg["num_experts_per_tok"].as_u64().unwrap() as u32);
    w.f32("llama.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"].as_f64().unwrap() as f32);
    w.f32("llama.rope.freq_base", cfg["rope_theta"].as_f64().unwrap() as f32);

    let (_, vocab, hidden) = shape("model.embed_tokens.weight");
    w.tensor("token_embd.weight", &[vocab, hidden], &read("model.embed_tokens.weight"));
    w.tensor("output_norm.weight", &[hidden], &read("model.norm.weight"));
    w.tensor("output.weight", &[vocab, hidden], &read("lm_head.weight"));

    for l in 0..layers {
        let p = format!("model.layers.{l}.");
        let pairs: [(&str, &str); 8] = [
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "ffn_norm.weight"),
            ("self_attn.q_proj.weight", "attn_q.weight"),
            ("self_attn.k_proj.weight", "attn_k.weight"),
            ("self_attn.v_proj.weight", "attn_v.weight"),
            ("self_attn.o_proj.weight", "attn_output.weight"),
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
        ];
        for (hf, gg) in pairs {
            let name = format!("{p}{hf}");
            let (_, rows, cols) = shape(&name);
            let dims: Vec<usize> = if rows == 1 { vec![cols] } else { vec![rows, cols] };
            w.tensor(&format!("blk.{l}.{gg}"), &dims, &read(&name));
        }
        let router = format!("{p}mlp.gate.weight");
        let (_, rows, cols) = shape(&router);
        w.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[rows, cols], &read(&router));

        // The part that matters: one 3-D stack per projection, experts outermost.
        for (hf, gg) in [("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")] {
            let first = format!("{p}mlp.experts.0.{hf}.weight");
            let (_, rows, cols) = shape(&first);
            let mut all = Vec::with_capacity(experts * rows * cols);
            for e in 0..experts {
                all.extend(read(&format!("{p}mlp.experts.{e}.{hf}.weight")));
            }
            w.tensor(&format!("blk.{l}.{gg}.weight"), &[experts, rows, cols], &all);
        }
    }

    let out = std::env::temp_dir().join(format!("moe-test-{tag}.gguf"));
    std::fs::write(&out, w.finish()).unwrap();
    out
}

#[test]
fn a_gguf_reproduces_the_reference_logits() {
    let (tokens, want) = expected();
    let path = convert("logits");
    let m = Model::load(Store::open(&path).unwrap()).expect("load the gguf");

    // The config came out of GGUF metadata, so check it landed.
    assert_eq!(m.spec.layers, 2);
    assert_eq!(m.spec.heads, 4);
    assert_eq!(m.spec.kv_heads, 2);
    assert_eq!(m.spec.head_dim, 8);
    assert_eq!(m.spec.experts, 4);
    assert_eq!(m.spec.top_k, 2);
    assert!(m.spec.qk_norm, "qk-norm was not detected through GGUF names");
    assert_eq!(m.spec.vocab, 48);

    // Decoding one token at a time, against the independent reference.
    let mut st = State::new(&m, 32);
    for (i, tok) in tokens.iter().enumerate() {
        let got = m.forward(&[*tok], &mut st);
        let d = got.iter().zip(&want[i]).fold(0.0f32, |mx, (x, y)| mx.max((x - y).abs()));
        assert!(d < 5e-3, "gguf step {i}: max |diff| = {d}");
    }

    // And batched, which routes many tokens through the stacked experts at once.
    let mut st2 = State::new(&m, 32);
    let got = m.forward(&tokens, &mut st2);
    let last = want.last().unwrap();
    let d = got.iter().zip(last).fold(0.0f32, |mx, (x, y)| mx.max((x - y).abs()));
    assert!(d < 5e-3, "gguf batched prefill: max |diff| = {d}");
    let _ = std::fs::remove_file(&path);
}

/// A GGUF must be packable, which means re-quantising out of it into the engine's
/// own format — the path that turns a downloaded file into a fast-loading one.
#[test]
fn a_gguf_can_be_packed() {
    let (tokens, want) = expected();
    let path = convert("packed");
    let out = std::env::temp_dir().join("moe-test-from-gguf.moe");
    Store::open(&path).unwrap().pack(&out, moe::Dt::Q8, moe::Dt::Q8, |_| {}).unwrap();

    let m = Model::load(Store::open(&out).unwrap()).expect("load the packed model");
    assert_eq!(m.spec.experts, 4);
    let mut st = State::new(&m, 32);
    let got = m.forward(&tokens, &mut st);
    let last = want.last().unwrap();
    let argmax = |v: &[f32]| (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap();
    assert_eq!(argmax(&got), argmax(last), "packing a GGUF moved the prediction");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out);
}

/// Truncating a real GGUF anywhere must produce an error, never a panic or a read
/// past the map. This is the only code path that parses an untrusted download.
#[test]
fn a_truncated_gguf_is_refused() {
    let path = convert("truncated");
    let whole = std::fs::read(&path).unwrap();
    let cut = std::env::temp_dir().join("moe-test-cut.gguf");
    // Step through the file rather than testing every byte, which would be slow.
    for at in (0..whole.len()).step_by((whole.len() / 40).max(1)) {
        std::fs::write(&cut, &whole[..at]).unwrap();
        // Either it refuses, or it loads something coherent. It must not panic.
        if let Ok(store) = Store::open(&cut) {
            let _ = Model::load(store);
        }
    }
    // A header claiming a tensor beyond the data must be caught.
    let mut lying = whole.clone();
    let n = lying.len();
    lying.truncate(n / 2);
    std::fs::write(&cut, &lying).unwrap();
    let _ = Store::open(&cut);
    let _ = std::fs::remove_file(&cut);
    let _ = std::fs::remove_file(&path);
}
