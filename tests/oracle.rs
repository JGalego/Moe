//! End-to-end checks against reference logits produced by
//! `scripts/oracle.py`, an independent implementation of the same model
//! definitions. Regenerate the fixtures with:
//!
//!     python3 scripts/oracle.py tests/fixtures

use moe::{Dt, Model, State, Store};
use std::path::{Path, PathBuf};

struct Fixture {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
    dir: PathBuf,
}

fn fixture(name: &str) -> Fixture {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    let raw = std::fs::read(dir.join("expected.json"))
        .unwrap_or_else(|e| panic!("{}: {e} — run python3 scripts/oracle.py tests/fixtures", dir.display()));
    let j: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    Fixture {
        tokens: j["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
        logits: j["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect())
            .collect(),
        dir,
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|a, b| v[*a].total_cmp(&v[*b])).unwrap()
}

fn load(path: &Path) -> Model {
    Model::load(Store::open(path).unwrap()).unwrap()
}

/// Feeding tokens one at a time must reproduce the reference at every position.
fn check_incremental(name: &str) {
    let f = fixture(name);
    let m = load(&f.dir);
    let mut st = State::new(&m, 32);
    for (i, tok) in f.tokens.iter().enumerate() {
        let got = m.forward(&[*tok], &mut st);
        let d = max_abs_diff(&got, &f.logits[i]);
        assert!(d < 5e-3, "{name} step {i}: max |diff| = {d}");
    }
    assert_eq!(st.pos, f.tokens.len());
}

/// Prefilling the whole prompt in one batched step must agree with decoding it
/// token by token — same maths, different loop order.
fn check_batched(name: &str) {
    let f = fixture(name);
    let m = load(&f.dir);
    let mut st = State::new(&m, 32);
    let got = m.forward(&f.tokens, &mut st);
    let last = f.logits.last().unwrap();
    let d = max_abs_diff(&got, last);
    assert!(d < 5e-3, "{name} batched prefill: max |diff| = {d}");
    assert_eq!(argmax(&got), argmax(last));

    // A split prefill must land in the same place as a single one.
    let mut st2 = State::new(&m, 32);
    let (a, b) = f.tokens.split_at(f.tokens.len() / 2);
    m.forward(a, &mut st2);
    let got2 = m.forward(b, &mut st2);
    assert!(max_abs_diff(&got, &got2) < 1e-4, "{name}: chunked prefill diverged");
}

/// Packing re-quantises every weight; the model must still make the same
/// prediction, with error bounded by the format's resolution.
fn check_packed(name: &str, dt: Dt, tol: f32) {
    let f = fixture(name);
    let out = std::env::temp_dir().join(format!("moe-test-{name}-{}.moe", dt.name()));
    let src = Store::open(&f.dir).unwrap();
    src.pack(&out, dt, dt, |_| {}).unwrap();

    let m = load(&out);
    let mut st = State::new(&m, 32);
    let got = m.forward(&f.tokens, &mut st);
    let want = f.logits.last().unwrap();
    let d = max_abs_diff(&got, want);
    assert!(d < tol, "{name} packed {}: max |diff| = {d} (tol {tol})", dt.name());
    assert_eq!(argmax(&got), argmax(want), "{name} packed {}: argmax moved", dt.name());
    let _ = std::fs::remove_file(&out);
}

#[test]
fn gqa_incremental_matches_reference() {
    check_incremental("gqa");
}

#[test]
fn mla_incremental_matches_reference() {
    check_incremental("mla");
}

#[test]
fn gqa_batched_prefill_matches_reference() {
    check_batched("gqa");
}

#[test]
fn mla_batched_prefill_matches_reference() {
    check_batched("mla");
}

#[test]
fn packed_f32_is_lossless() {
    check_packed("gqa", Dt::F32, 5e-3);
    check_packed("mla", Dt::F32, 5e-3);
}

#[test]
fn packed_q8_preserves_prediction() {
    check_packed("gqa", Dt::Q8, 0.25);
    check_packed("mla", Dt::Q8, 0.25);
}

#[test]
fn architecture_is_detected_from_the_checkpoint() {
    let gqa = load(&fixture("gqa").dir);
    assert!(gqa.spec.mla.is_none());
    assert!(gqa.spec.qk_norm);
    assert_eq!((gqa.spec.heads, gqa.spec.kv_heads, gqa.spec.head_dim), (4, 2, 8));
    assert_eq!((gqa.spec.experts, gqa.spec.top_k), (4, 2));
    assert!(!gqa.spec.sigmoid);

    let mla = load(&fixture("mla").dir);
    let d = mla.spec.mla.expect("latent attention not detected");
    assert_eq!((d.kv_lora, d.qk_nope, d.qk_rope, d.v_head), (12, 8, 8, 8));
    assert!(mla.spec.sigmoid, "sigmoid gating not detected");
    assert_eq!(mla.spec.n_group, 2);
    // Layer 0 is dense, so only the routed layer contributes expert bytes.
    assert!(mla.expert_bytes() > 0);
}
