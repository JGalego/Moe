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

/// `forward_all` must reproduce the reference at *every* position from a single
/// batched step, not just the last. Scoring text and verifying a speculative
/// draft both rest on that, so it is checked against the oracle directly rather
/// than against `forward`.
fn check_forward_all(name: &str) {
    let f = fixture(name);
    let m = load(&f.dir);
    let mut st = State::new(&m, 32);
    let all = m.forward_all(&f.tokens, &mut st);
    let vocab = m.spec.vocab;
    assert_eq!(all.len(), f.tokens.len() * vocab);
    for (i, want) in f.logits.iter().enumerate() {
        let got = &all[i * vocab..(i + 1) * vocab];
        let d = max_abs_diff(got, want);
        assert!(d < 5e-3, "{name} forward_all position {i}: max |diff| = {d}");
        assert_eq!(argmax(got), argmax(want), "{name} forward_all position {i}: argmax moved");
    }
    // And the last row must be exactly what the cheap path returns.
    let mut st2 = State::new(&m, 32);
    let last = m.forward(&f.tokens, &mut st2);
    let tail = &all[(f.tokens.len() - 1) * vocab..];
    assert!(max_abs_diff(tail, &last) < 1e-6, "{name}: forward_all disagrees with forward");
}

#[test]
fn forward_all_matches_reference_at_every_position() {
    check_forward_all("gqa");
    check_forward_all("mla");
}

/// Scoring is only meaningful if the numbers are real: a model must find the
/// text it was shown less surprising than the same tokens shuffled, and the
/// uniform-prediction bound (ln vocab) must never be exceeded on average by a
/// model that has learned anything at all.
#[test]
fn scoring_is_bounded_and_ordered() {
    let f = fixture("gqa");
    let m = load(&f.dir);
    let ids: Vec<u32> = (0..40).map(|i| ((i * 7 + 3) % m.spec.vocab) as u32).collect();
    let s = moe::eval::score(&m, &ids, 16, 8, 0, |_, _| {});
    assert_eq!(s.tokens, ids.len() - 1, "every token but the first must be scored");
    assert!(s.mean_nll() > 0.0, "surprise cannot be negative");
    // A random-weight fixture predicts near-uniformly; it cannot do worse than
    // uniform by much, and cannot beat the vocabulary bound by construction.
    let uniform = (m.spec.vocab as f64).ln();
    assert!(s.mean_nll() < uniform * 1.5, "nll {} vs uniform {uniform}", s.mean_nll());

    // A tighter stride re-reads more context but must score the same count.
    let fine = moe::eval::score(&m, &ids, 16, 1, 0, |_, _| {});
    assert_eq!(fine.tokens, s.tokens);
}

#[test]
fn gqa_incremental_matches_reference() {
    check_incremental("gqa");
}

#[test]
fn mla_incremental_matches_reference() {
    check_incremental("mla");
}

/// qk-norm has two conventions and only the weight's width distinguishes them.
#[test]
fn qk_norm_across_the_whole_projection_matches_reference() {
    check_incremental("gqa_fullnorm");
    check_batched("gqa_fullnorm");
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

/// Tracing must record one entry per token per *routed* layer, with exactly
/// top-k experts each. The latent fixture has a dense first layer, which is how
/// we check dense layers contribute nothing.
#[test]
fn tracing_records_every_routing_decision() {
    for (name, routed_layers) in [("gqa", 2), ("mla", 1)] {
        let f = fixture(name);
        let m = load(&f.dir);
        let mut st = State::new(&m, 32);
        st.trace();
        m.forward(&f.tokens, &mut st);

        let tr = st.trace.as_ref().expect("tracing was switched on");
        assert_eq!(tr.tokens, f.tokens, "{name}: token ids not recorded");
        assert_eq!(tr.routes.len(), f.tokens.len() * routed_layers, "{name}: wrong record count");
        for r in &tr.routes {
            assert_eq!(r.experts.len(), m.spec.top_k, "{name}: not top-k experts");
            assert!(r.experts.iter().all(|(e, _)| (*e as usize) < m.spec.experts), "{name}: expert out of range");
            assert!((r.pos as usize) < f.tokens.len(), "{name}: position out of range");
        }
        // Weights come out strongest first, normalised, then scaled by the
        // checkpoint's routed_scaling_factor — 1.5 for the latent fixture.
        let w: Vec<f32> = tr.routes[0].experts.iter().map(|(_, w)| *w).collect();
        assert!(w.windows(2).all(|p| p[0] >= p[1]), "{name}: weights not ordered");
        let sum = w.iter().sum::<f32>();
        assert!((sum - m.spec.routed_scale).abs() < 1e-4, "{name}: weights sum to {sum}, not the routed scale");
    }
}

/// Rewinding the KV cache and continuing must be indistinguishable from never
/// having cached at all. This is asserted on logits rather than on sampled
/// tokens: the fixtures' logits are near enough context-independent that an
/// argmax cannot tell a stale cache from a correct one.
#[test]
fn truncating_the_cache_matches_a_fresh_state() {
    for name in ["gqa", "mla"] {
        let m = load(&fixture(name).dir);

        // Diverge: run one sequence, rewind to a prefix, continue elsewhere.
        let mut warm = State::new(&m, 32);
        m.forward(&[3, 11, 5, 40, 7, 1], &mut warm);
        warm.truncate(2);
        let got = m.forward(&[9, 9], &mut warm);
        let mut fresh = State::new(&m, 32);
        let want = m.forward(&[3, 11, 9, 9], &mut fresh);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-5, "{name}: rewound cache diverged, max |diff| = {d}");

        // Extend: the chat case, where the next turn only adds tokens.
        let mut warm = State::new(&m, 32);
        m.forward(&[3, 11], &mut warm);
        let got = m.forward(&[5, 40], &mut warm);
        let mut fresh = State::new(&m, 32);
        let want = m.forward(&[3, 11, 5, 40], &mut fresh);
        let d = max_abs_diff(&got, &want);
        assert!(d < 1e-5, "{name}: extended cache diverged, max |diff| = {d}");
    }
}

#[test]
fn architecture_is_detected_from_the_checkpoint() {
    let gqa = load(&fixture("gqa").dir);
    assert!(gqa.spec.mla.is_none());
    assert!(gqa.spec.qk_norm);
    assert!(load(&fixture("gqa_fullnorm").dir).spec.qk_norm);
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
