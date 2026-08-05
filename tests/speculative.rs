//! Speculation must be invisible in the output.
//!
//! The whole claim is that drafting changes *when* tokens are computed, never
//! which ones come out. Greedily that is checkable exactly: the same prompt,
//! seed and sampler must produce a byte-identical token sequence whether the
//! drafter is off, guessing two tokens ahead, or guessing sixteen. These tests
//! run on a repetitive prompt on purpose — that is where lookup drafting fires,
//! so a bug in the accept-and-rewind path has somewhere to show itself.

use moe::{draft::Lookup, generate::generate, Model, Plan, Sampler, State, Store};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn model(name: &str) -> Model {
    Model::load(Store::open(&fixture(name)).unwrap()).unwrap()
}

fn plan(lookahead: usize, temp: f32, max_tokens: usize) -> Plan {
    Plan {
        max_tokens,
        sampler: Sampler { temp, top_p: 1.0, ..Sampler::default() },
        seed: 7,
        lookahead,
        lookup: Lookup { max_ngram: 6, min_ngram: 2 },
        logprobs: 0,
    }
}

/// Generate from `prompt` and return the tokens plus the outcome's counters.
fn run(m: &Model, prompt: &[u32], p: &Plan) -> (Vec<u32>, moe::Outcome) {
    let ctx = prompt.len() + p.max_tokens + p.lookahead + 2;
    let mut st = State::new(m, ctx);
    let mut logits = Vec::new();
    for chunk in prompt.chunks(64) {
        logits = m.forward(chunk, &mut st);
    }
    let mut history = prompt.to_vec();
    let mut out = Vec::new();
    let outcome = generate(m, &mut st, &mut history, logits, p, None, |t| {
        out.push(t);
        true
    });
    // Whatever was committed must be exactly what the history grew by.
    assert_eq!(history.len(), prompt.len() + out.len(), "history and emissions disagree");
    assert_eq!(&history[prompt.len()..], &out[..]);
    (out, outcome)
}

/// A prompt that repeats, so the drafter has precedents to find.
fn repetitive(vocab: usize) -> Vec<u32> {
    let unit: Vec<u32> = (0..6).map(|i| ((i * 3 + 1) % vocab) as u32).collect();
    unit.repeat(5)
}

#[test]
fn greedy_output_is_identical_at_every_lookahead() {
    for name in ["gqa", "mla"] {
        let m = model(name);
        let prompt = repetitive(m.spec.vocab);
        let (baseline, plain) = run(&m, &prompt, &plan(0, 0.0, 24));
        assert!(!baseline.is_empty(), "{name}: nothing generated");
        assert_eq!(plain.drafted, 0, "{name}: speculation ran with lookahead 0");
        // The first token falls out of the prefill's own logits, so N tokens
        // cost N - 1 forward steps when nothing is drafted.
        let floor = baseline.len() - 1;
        assert_eq!(plain.steps, floor, "{name}: plain decoding costs one step per token after the first");

        for lookahead in [1usize, 2, 4, 8, 16] {
            let (got, spec) = run(&m, &prompt, &plan(lookahead, 0.0, 24));
            assert_eq!(got, baseline, "{name}: lookahead {lookahead} changed the output");
            assert!(spec.accepted <= spec.drafted);
            // Accepted drafts are tokens that cost no step of their own, so
            // speculating can never need more steps than not speculating.
            assert!(spec.steps <= floor, "{name}: lookahead {lookahead} used {} steps, plain used {floor}", spec.steps);
        }
    }
}

/// On text that repeats, the drafter must actually land some guesses — otherwise
/// the identity test above would be passing for the trivial reason that nothing
/// was ever speculated.
#[test]
fn drafting_pays_off_on_repetitive_text() {
    let m = model("gqa");
    let prompt = repetitive(m.spec.vocab);
    let (tokens, spec) = run(&m, &prompt, &plan(4, 0.0, 24));
    assert!(spec.drafted > 0, "nothing was drafted");
    assert!(spec.accepted > 0, "no draft was accepted on repetitive input");
    assert!(spec.steps < tokens.len(), "{} steps for {} tokens is no saving", spec.steps, tokens.len());
    assert!(spec.tokens_per_step() > 1.0);
    assert!(spec.acceptance() > 0.0 && spec.acceptance() <= 1.0);
}

/// Sampling cannot be checked for token-identity, but the *state* must still be
/// consistent: the cache has to hold exactly the committed tokens, so continuing
/// from it agrees with a fresh run over the same history.
#[test]
fn the_cache_matches_the_committed_history_after_speculating() {
    let m = model("gqa");
    let prompt = repetitive(m.spec.vocab);
    let p = plan(4, 0.8, 16);
    let ctx = prompt.len() + p.max_tokens + p.lookahead + 2;

    let mut st = State::new(&m, ctx);
    let mut logits = Vec::new();
    for chunk in prompt.chunks(64) {
        logits = m.forward(chunk, &mut st);
    }
    let mut history = prompt.to_vec();
    generate(&m, &mut st, &mut history, logits, &p, None, |_| true);
    assert_eq!(st.pos, history.len() - 1, "the last committed token is never forwarded");

    // Replay the whole history into a clean state and continue from both; the
    // next prediction must agree, which it only can if nothing speculative was
    // left behind in the cache.
    let mut fresh = State::new(&m, ctx);
    let mut a = Vec::new();
    for chunk in history[..history.len() - 1].chunks(64) {
        a = m.forward(chunk, &mut fresh);
    }
    assert_eq!(fresh.pos, st.pos);
    let last = *history.last().unwrap();
    let from_spec = m.forward(&[last], &mut st);
    let from_fresh = m.forward(&[last], &mut fresh);
    let d = from_spec.iter().zip(&from_fresh).fold(0.0f32, |mx, (x, y)| mx.max((x - y).abs()));
    assert!(d < 1e-4, "speculative state diverged from a replayed one: max |diff| = {d}");
    let _ = a;
}

/// A draft may never carry generation past what was asked for, and the stop
/// reasons must stay truthful when it ends inside an accepted run.
#[test]
fn speculation_respects_the_token_budget() {
    let m = model("gqa");
    let prompt = repetitive(m.spec.vocab);
    for max in [1usize, 2, 3, 5, 11] {
        let (got, o) = run(&m, &prompt, &plan(8, 0.0, max));
        assert_eq!(got.len(), max, "asked for {max}, got {}", got.len());
        assert_eq!(o.tokens, max);
    }
}

/// Returning false from the callback must stop immediately — the mechanism the
/// server's stop sequences rely on.
#[test]
fn the_callback_can_halt_mid_draft() {
    let m = model("gqa");
    let prompt = repetitive(m.spec.vocab);
    let mut st = State::new(&m, prompt.len() + 40);
    let mut logits = Vec::new();
    for chunk in prompt.chunks(64) {
        logits = m.forward(chunk, &mut st);
    }
    let mut history = prompt.to_vec();
    let mut seen = 0;
    let o = generate(&m, &mut st, &mut history, logits, &plan(8, 0.0, 20), None, |_| {
        seen += 1;
        seen < 3
    });
    assert_eq!(seen, 3);
    assert_eq!(o.tokens, 3);
    assert_eq!(o.stop, moe::Stop::Caller);
}
