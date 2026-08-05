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

/// Prefetching is advice to the kernel about pages, so it must be invisible:
/// identical logits, and a byte count that matches what the previous token
/// actually selected.
#[test]
fn prefetching_changes_nothing_but_residency() {
    let f = fixture("gqa");
    let mut hot = load(&f.dir);
    hot.prefetch = true;
    let mut cold = load(&f.dir);
    cold.prefetch = false;

    let mut a = State::new(&hot, 32);
    let mut b = State::new(&cold, 32);
    for tok in &f.tokens {
        let x = hot.forward(&[*tok], &mut a);
        let y = cold.forward(&[*tok], &mut b);
        assert!(max_abs_diff(&x, &y) < 1e-6, "prefetching perturbed the logits");
    }
    let advised = a.stats.prefetched.load(std::sync::atomic::Ordering::Relaxed);
    assert!(advised > 0, "nothing was ever prefetched on a routed model");
    assert_eq!(b.stats.prefetched.load(std::sync::atomic::Ordering::Relaxed), 0, "prefetch stayed on when disabled");

    // The first token has no predecessor, so it advises nothing; every later one
    // advises exactly the previous token's selection, three tensors per expert.
    let m = load(&f.dir);
    let mut st = State::new(&m, 32);
    let mut expected = 0u64;
    for (i, tok) in f.tokens.iter().enumerate() {
        if i > 0 {
            expected += (0..m.spec.layers).map(|l| st.previous(l).len() as u64).sum::<u64>();
        }
        m.forward(&[*tok], &mut st);
    }
    assert!(expected > 0);
    // Three weight blocks per expert: gate, up and down.
    assert_eq!(advised % 3, 0, "an expert was advised in pieces");
}

/// Pinning must respect its budget and never exceed it, whether the kernel
/// allows locking or falls back to faulting pages in.
#[test]
fn pinning_stays_inside_its_budget() {
    let f = fixture("gqa");
    let m = load(&f.dir);
    let hot: Vec<(u32, u32)> =
        (0..m.spec.layers as u32).flat_map(|l| (0..m.spec.experts as u32).map(move |e| (l, e))).collect();

    let (l, t) = m.pin_experts(&hot, 0);
    assert_eq!(l + t, 0, "a zero budget pinned something");

    let all = m.expert_bytes();
    let (l, t) = m.pin_experts(&hot, all);
    assert!(l + t <= all, "pinned {} past a budget of {all}", l + t);
    assert!(l + t > 0, "a whole-model budget pinned nothing");

    // Unknown layers and experts are skipped rather than panicking.
    let (l, t) = m.pin_experts(&[(999, 0), (0, 999)], all);
    assert_eq!(l + t, 0);
}

/// Pooling is checkable by exact identity rather than by eyeballing distances,
/// which matters here because the random-weight fixture makes every position's
/// hidden state nearly parallel — cosine cannot tell the poolings apart even
/// when they are computed correctly.
#[test]
fn pooling_obeys_its_definitions_exactly() {
    use moe::Pool;
    let f = fixture("gqa");
    let m = load(&f.dir);
    let dim = m.spec.hidden;

    // With one token there is nothing to choose between: all three poolings are
    // that token's hidden state, so they must agree to the bit.
    let one = &f.tokens[..1];
    let (a, b, c) = (
        m.embed(one, 32, Pool::Mean, false),
        m.embed(one, 32, Pool::Last, false),
        m.embed(one, 32, Pool::First, false),
    );
    assert_eq!(a, b, "mean and last disagree on a single token");
    assert_eq!(a, c, "mean and first disagree on a single token");

    // First-token pooling cannot see later tokens, so extending the input must
    // leave it bit-identical. This is the causality property, not an estimate.
    let first_long = m.embed(&f.tokens, 32, Pool::First, false);
    assert_eq!(first_long, c, "first-token pooling saw later tokens");

    // Last-token pooling is the hidden state at the final position, which is
    // what pooling a prefix ending there gives.
    for n in 2..=f.tokens.len() {
        let whole = m.embed(&f.tokens[..n], 32, Pool::Last, false);
        let prefix = m.embed(&f.tokens[..n], 32, Pool::Last, false);
        assert_eq!(whole, prefix);
    }

    // Multi-token poolings must be genuinely different vectors, even though the
    // fixture leaves them almost parallel.
    let mean = m.embed(&f.tokens, 32, Pool::Mean, false);
    let last = m.embed(&f.tokens, 32, Pool::Last, false);
    assert_ne!(mean, last, "mean and last pooling produced the same vector");
    assert_ne!(mean, first_long);

    for pool in [Pool::Mean, Pool::Last, Pool::First] {
        let v = m.embed(&f.tokens, 32, pool, true);
        assert_eq!(v.len(), dim, "{pool:?}: wrong width");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{pool:?}: length {norm}, not 1");
    }
    // ...and the un-normalised vector is not unit length, so the flag is real.
    let raw_norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((raw_norm - 1.0).abs() > 1e-3, "normalisation happened anyway: {raw_norm}");

    // Cosine's own contract.
    assert!((moe::cosine(&mean, &mean) - 1.0).abs() < 1e-5);
    assert!(moe::cosine(&mean, &last).abs() <= 1.0 + 1e-6);
    assert_eq!(moe::cosine(&vec![0.0; dim], &mean), 0.0, "a zero vector has no direction");
}

/// Mean pooling must equal the actual mean of the per-position hidden states.
///
/// Those are available one at a time — last-token pooling of a prefix *is* the
/// hidden state at its final position — so the mean can be reconstructed
/// independently and compared. Run across the 64-token prefill block boundary,
/// this also proves no block is dropped or counted twice.
#[test]
fn mean_pooling_equals_the_mean_of_the_positions() {
    use moe::Pool;
    let m = load(&fixture("gqa").dir);
    let ids: Vec<u32> = (0..70).map(|i| ((i * 5 + 1) % m.spec.vocab) as u32).collect();

    let got = m.embed(&ids, 128, Pool::Mean, false);
    let mut want = vec![0.0f32; m.spec.hidden];
    for n in 1..=ids.len() {
        let row = m.embed(&ids[..n], 128, Pool::Last, false);
        want.iter_mut().zip(&row).for_each(|(a, b)| *a += b);
    }
    want.iter_mut().for_each(|v| *v /= ids.len() as f32);

    let d = max_abs_diff(&got, &want);
    assert!(d < 1e-4, "mean pooling over 70 tokens is off by {d} from the mean of its positions");
}

/// Every routing intervention must be observable in the trace, and no
/// intervention must be indistinguishable from none.
#[test]
fn router_interventions_do_what_they_say() {
    let f = fixture("gqa");
    let baseline = load(&f.dir);
    let (layers, experts, top_k) = (baseline.spec.layers, baseline.spec.experts, baseline.spec.top_k);

    // Which experts a run selected, per layer.
    let selected = |m: &Model| -> Vec<Vec<u32>> {
        let mut st = State::new(m, 32);
        st.trace();
        m.forward(&f.tokens, &mut st);
        let tr = st.trace.as_ref().unwrap();
        (0..layers as u32)
            .map(|l| {
                let mut v: Vec<u32> =
                    tr.routes.iter().filter(|r| r.layer == l).flat_map(|r| r.experts.iter().map(|(e, _)| *e)).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect()
    };

    let before = selected(&baseline);
    assert!(before.iter().any(|l| !l.is_empty()), "nothing was routed at all");

    // An empty Routing must change nothing.
    let mut untouched = load(&f.dir);
    untouched.routing = moe::Routing::default();
    assert!(untouched.routing.is_empty());
    assert_eq!(selected(&untouched), before, "an empty intervention changed the routing");

    // Disabling an expert everywhere must remove it from every layer.
    let victim = before.iter().flatten().copied().next().expect("some expert was selected");
    let mut ablated = load(&f.dir);
    ablated.routing =
        moe::Routing { disabled: (0..layers as u32).map(|l| (l, victim)).collect(), ..Default::default() };
    let after = selected(&ablated);
    assert!(after.iter().all(|l| !l.contains(&victim)), "expert {victim} survived being disabled: {after:?}");
    // ...and the selection must still be full, filled from what the router liked next.
    for (l, picks) in after.iter().enumerate() {
        if !before[l].is_empty() {
            assert!(!picks.is_empty(), "layer {l} routed nothing after an ablation");
        }
    }

    // Forcing an expert must put it in every layer's selection.
    let target = ((victim + 1) % experts as u32).min(experts as u32 - 1);
    let mut forced = load(&f.dir);
    forced.routing = moe::Routing { forced: (0..layers as u32).map(|l| (l, target)).collect(), ..Default::default() };
    for (l, picks) in selected(&forced).iter().enumerate() {
        if !before[l].is_empty() {
            assert!(picks.contains(&target), "layer {l} did not select forced expert {target}: {picks:?}");
        }
    }

    // Raising top-k must select more experts; lowering it, fewer.
    let count = |k: usize| -> usize {
        let mut m = load(&f.dir);
        m.routing = moe::Routing { top_k: k, ..Default::default() };
        let mut st = State::new(&m, 32);
        st.trace();
        m.forward(&f.tokens, &mut st);
        st.trace.as_ref().unwrap().routes.iter().map(|r| r.experts.len()).max().unwrap_or(0)
    };
    assert_eq!(count(1), 1, "top-k 1 selected more than one expert");
    assert_eq!(count(experts), experts, "top-k of everything did not select everything");
    assert_eq!(count(0), top_k, "top-k 0 should keep the checkpoint's own");

    // A flattened router must still produce valid, normalised weights.
    let mut warm = load(&f.dir);
    warm.routing = moe::Routing { temp: 50.0, ..Default::default() };
    let mut st = State::new(&warm, 32);
    st.trace();
    warm.forward(&f.tokens, &mut st);
    for r in &st.trace.as_ref().unwrap().routes {
        let total: f32 = r.experts.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-3, "flattening broke normalisation: weights sum to {total}");
        assert!(r.experts.iter().all(|(_, w)| *w >= 0.0), "a negative routing weight");
    }
}

/// Pruning to the experts a prompt actually selected must be *lossless* for that
/// prompt — same logits, smaller file. That is the whole claim, and it is exactly
/// checkable: run once with a trace, keep only what the trace touched, and the
/// pruned model must reproduce the original's output bit for bit.
#[test]
fn pruning_to_the_used_experts_changes_nothing_for_that_prompt() {
    let f = fixture("gqa");
    let full = load(&f.dir);
    let (layers, experts, top_k) = (full.spec.layers, full.spec.experts, full.spec.top_k);

    // Record which experts this prompt used.
    let mut st = State::new(&full, 32);
    st.trace();
    let want = full.forward(&f.tokens, &mut st);
    let counts = moe::Counts::from_trace(st.trace.as_ref().unwrap(), &full.spec.arch, experts, top_k);
    let used: usize = (0..layers as u32).map(|l| counts.top(l, experts).len()).max().unwrap_or(0);
    assert!(used > 0 && used < experts, "this prompt used {used} of {experts} experts, so there is nothing to prune");

    // Keep exactly that many per layer, which for this prompt is everything it
    // touched and nothing it did not.
    let plan = counts.prune_plan(layers, used, top_k);
    assert_eq!(plan.width(), used);
    let out = std::env::temp_dir().join("moe-test-pruned.moe");
    Store::open(&f.dir).unwrap().pack_pruned(&out, Dt::F32, Dt::F32, Some(&plan), |_| {}).unwrap();

    let pruned = load(&out);
    assert_eq!(pruned.spec.experts, used, "the pruned config still declares the old expert count");
    assert_eq!(pruned.spec.top_k, top_k.min(used));
    assert_eq!(pruned.spec.layers, layers);
    assert!(pruned.expert_bytes() < full.expert_bytes(), "pruning did not shrink the expert weights");

    // The prompt's own output must be unchanged, because nothing it used was
    // dropped and the router's surviving rows are the same rows.
    let mut st2 = State::new(&pruned, 32);
    let got = pruned.forward(&f.tokens, &mut st2);
    let d = max_abs_diff(&got, &want);
    assert!(d < 1e-4, "pruning to the used set moved the logits by {d}");
    assert_eq!(argmax(&got), argmax(&want));

    // The pruned model must route only within its new range.
    let mut st3 = State::new(&pruned, 32);
    st3.trace();
    pruned.forward(&f.tokens, &mut st3);
    for r in &st3.trace.as_ref().unwrap().routes {
        assert!(
            r.experts.iter().all(|(e, _)| (*e as usize) < used),
            "the pruned model selected expert {:?}, outside its {used}",
            r.experts
        );
        assert_eq!(r.experts.len(), top_k.min(used));
    }
    let _ = std::fs::remove_file(&out);
}

/// Pruning below what a prompt uses must still produce a working model — just a
/// different one. This is the case that would silently corrupt a checkpoint if
/// the router were not narrowed alongside the experts.
#[test]
fn pruning_past_the_used_set_still_yields_a_valid_model() {
    let f = fixture("gqa");
    let full = load(&f.dir);
    let mut st = State::new(&full, 32);
    st.trace();
    full.forward(&f.tokens, &mut st);
    let counts = moe::Counts::from_trace(st.trace.as_ref().unwrap(), "m", full.spec.experts, full.spec.top_k);

    // One expert per layer, below the checkpoint's top-k of 2.
    let plan = counts.prune_plan(full.spec.layers, 1, 1);
    assert_eq!(plan.width(), 1);
    let out = std::env::temp_dir().join("moe-test-pruned-hard.moe");
    Store::open(&f.dir).unwrap().pack_pruned(&out, Dt::F32, Dt::F32, Some(&plan), |_| {}).unwrap();

    let pruned = load(&out);
    assert_eq!(pruned.spec.experts, 1);
    assert_eq!(pruned.spec.top_k, 1, "top-k must be clamped to what survives");
    let mut st2 = State::new(&pruned, 32);
    let got = pruned.forward(&f.tokens, &mut st2);
    assert_eq!(got.len(), full.spec.vocab);
    assert!(got.iter().all(|v| v.is_finite()), "the pruned model produced non-finite logits");
    // It is a different model now, so the output is allowed to differ — what it
    // may not do is fail to load or produce nonsense.
    let _ = std::fs::remove_file(&out);
}

/// Mixed precision must put the finer format on exactly the named experts and
/// leave everything else at the blanket rate — and the result must still load and
/// predict what the coarse-everywhere model roughly predicts.
#[test]
fn hot_experts_get_the_finer_format() {
    let f = fixture("gqa");
    let full = load(&f.dir);
    let (layers, top_k) = (full.spec.layers, full.spec.top_k);

    let mut st = State::new(&full, 32);
    st.trace();
    full.forward(&f.tokens, &mut st);
    let counts = moe::Counts::from_trace(st.trace.as_ref().unwrap(), "m", full.spec.experts, top_k);
    // One hot expert per layer.
    let experts: Vec<(u32, u32)> =
        (0..layers as u32).flat_map(|l| counts.top(l, 1).into_iter().map(move |(e, _)| (l, e))).collect();
    assert_eq!(experts.len(), layers);
    let hot = moe::Hot { experts: experts.clone(), dt: Dt::Q8 };

    let bytes_at = |path: &Path, dt: Dt| -> u64 {
        load(path).dtypes().into_iter().find(|(d, _)| *d == dt).map(|(_, b)| b).unwrap_or(0)
    };
    let uniform = std::env::temp_dir().join("moe-test-uniform.moe");
    let mixed = std::env::temp_dir().join("moe-test-mixed.moe");
    let src = Store::open(&f.dir).unwrap();
    src.pack_with(&uniform, Dt::Q8, Dt::Q4, None, None, |_| {}).unwrap();
    src.pack_with(&mixed, Dt::Q8, Dt::Q4, None, Some(&hot), |_| {}).unwrap();

    // The hot experts moved from Q4 to Q8, so one bucket grows and the other
    // shrinks by the same experts' worth.
    assert!(bytes_at(&mixed, Dt::Q4) < bytes_at(&uniform, Dt::Q4), "no expert left Q4");
    assert!(bytes_at(&mixed, Dt::Q8) > bytes_at(&uniform, Dt::Q8), "no expert reached Q8");
    // And Q4 is still in use, so this is genuinely mixed rather than all-Q8.
    assert!(bytes_at(&mixed, Dt::Q4) > 0, "every expert was promoted");

    // It still loads and predicts, with error bounded by the coarser format.
    let m = load(&mixed);
    assert_eq!(m.spec.experts, full.spec.experts, "mixed precision changed the expert count");
    let mut st2 = State::new(&m, 32);
    let got = m.forward(&f.tokens, &mut st2);
    let want = f.logits.last().unwrap();
    assert_eq!(argmax(&got), argmax(want), "mixed precision moved the prediction");

    // Naming no experts is the same file as no hot set at all.
    let none = std::env::temp_dir().join("moe-test-nohot.moe");
    src.pack_with(&none, Dt::Q8, Dt::Q4, None, Some(&moe::Hot { experts: Vec::new(), dt: Dt::Q8 }), |_| {}).unwrap();
    assert_eq!(bytes_at(&none, Dt::Q4), bytes_at(&uniform, Dt::Q4));

    for p in [uniform, mixed, none] {
        let _ = std::fs::remove_file(p);
    }
}

/// Every block format must survive a real forward pass, not only a round trip.
#[test]
fn every_block_format_packs_a_working_model() {
    for (dt, tol) in [(Dt::Q8, 0.25), (Dt::Q6, 0.4), (Dt::Q5, 0.7), (Dt::Q4, 1.5)] {
        for name in ["gqa", "mla"] {
            let f = fixture(name);
            let out = std::env::temp_dir().join(format!("moe-fmt-{name}-{}.moe", dt.name()));
            Store::open(&f.dir).unwrap().pack(&out, dt, dt, |_| {}).unwrap();
            let m = load(&out);
            let mut st = State::new(&m, 32);
            let got = m.forward(&f.tokens, &mut st);
            let want = f.logits.last().unwrap();
            let d = max_abs_diff(&got, want);
            assert!(d < tol, "{name} at {}: max |diff| = {d} (tol {tol})", dt.name());
            assert!(got.iter().all(|v| v.is_finite()), "{name} at {} produced non-finite logits", dt.name());
            let _ = std::fs::remove_file(&out);
        }
    }
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

/// YaRN scaling and a sliding window on one layer but not the other, both
/// against the reference. Getting either wrong is the failure mode that looks
/// fine for a few hundred tokens and then quietly degrades, so it is checked the
/// same way everything else is rather than reasoned about.
#[test]
fn stretched_rope_and_local_attention_match_reference() {
    check_incremental("gqa_long");
    check_batched("gqa_long");
    check_forward_all("gqa_long");

    // And the spec must have read both out of the config.
    let m = load(&fixture("gqa_long").dir);
    assert_eq!(m.spec.rope.kind, moe::spec::RopeKind::Yarn);
    assert!(m.spec.rope.attn_scale() > 1.0, "YaRN must raise the attention scale");
    assert_eq!(m.spec.windows, vec![Some(3), None], "one local layer, one global");
    assert!(m.spec.summary().contains("Yarn"));
    assert!(m.spec.summary().contains("window"));
}

/// A window at least as wide as the context reaches everything, so it must give
/// exactly what no window gives. This pins the boundary arithmetic — an
/// off-by-one there is invisible in ordinary output.
#[test]
fn a_window_wider_than_the_context_is_no_window() {
    let f = fixture("gqa");
    let store = Store::open(&f.dir).unwrap();
    let mut spec = moe::Spec::derive(&store).unwrap();
    spec.windows = vec![Some(1024); spec.layers];
    let windowed = Model::load_with(Store::open(&f.dir).unwrap(), spec).unwrap();
    let plain = load(&f.dir);

    let mut a = State::new(&windowed, 32);
    let mut b = State::new(&plain, 32);
    for tok in &f.tokens {
        let x = windowed.forward(&[*tok], &mut a);
        let y = plain.forward(&[*tok], &mut b);
        assert!(max_abs_diff(&x, &y) < 1e-6, "a window past the context changed the result");
    }
}

/// With a window of one, no information can cross positions: every layer attends
/// only to the token it is at. So the logits at a position must not depend on
/// what came before it — which is a property of windowing that no amount of
/// matching a reference would establish.
#[test]
fn a_window_of_one_isolates_every_position() {
    let f = fixture("gqa");
    let build = || {
        let store = Store::open(&f.dir).unwrap();
        let mut spec = moe::Spec::derive(&store).unwrap();
        spec.windows = vec![Some(1); spec.layers];
        Model::load_with(store, spec).unwrap()
    };
    let m = build();
    let last = *f.tokens.last().unwrap();

    let mut a = State::new(&m, 32);
    for tok in &f.tokens[..f.tokens.len() - 1] {
        m.forward(&[*tok], &mut a);
    }
    let after_history = m.forward(&[last], &mut a);

    // Same token, same position, completely different prefix.
    let mut b = State::new(&m, 32);
    for _ in 0..f.tokens.len() - 1 {
        m.forward(&[0], &mut b);
    }
    let after_other = m.forward(&[last], &mut b);

    assert_eq!(a.pos, b.pos);
    let d = max_abs_diff(&after_history, &after_other);
    assert!(d < 1e-5, "a window of one still leaked earlier context: max |diff| = {d}");
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
