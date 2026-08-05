//! The decode loop, in one place.
//!
//! `moe run` and `moe serve` used to carry a copy each. They now share this,
//! which is what lets speculation exist at all: the loop below verifies a
//! drafted continuation in the same step that forwards the token before it, and
//! a single implementation means the CLI and the server cannot drift on which
//! tokens they emit.
//!
//! The structure is one invariant: `pending` is a token that has been committed
//! and handed to the caller but is not yet in the KV cache. Every round forwards
//! it — together with any draft riding behind it — and ends with the next
//! committed token in its place. With no draft the round is an ordinary
//! single-token decode, so speculation is not a second code path.

use crate::draft::Lookup;
use crate::grammar::Guide;
use crate::model::{Model, State};
use crate::sample::{Rng, Sampler};

/// What to generate and how.
#[derive(Clone)]
pub struct Plan {
    pub max_tokens: usize,
    pub sampler: Sampler,
    pub seed: u64,
    /// Tokens to draft per step. 0 disables speculation entirely.
    pub lookahead: usize,
    pub lookup: Lookup,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan { max_tokens: 128, sampler: Sampler::default(), seed: 0, lookahead: 0, lookup: Lookup::default() }
    }
}

/// Why generation ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Stop {
    /// `max_tokens` reached.
    #[default]
    Length,
    /// The checkpoint's own end-of-sequence token.
    Eos,
    /// The caller's callback asked to stop.
    Caller,
    /// The context window ran out.
    Context,
    /// A constraint was satisfied: the document is complete.
    Complete,
    /// A constraint left no token that could legally follow. Only reachable if
    /// the grammar admits no continuation at all, which a satisfiable schema
    /// cannot do before [`Stop::Complete`].
    Stuck,
}

impl Stop {
    /// The `finish_reason` an OpenAI client expects.
    pub fn reason(self) -> &'static str {
        match self {
            Stop::Length | Stop::Context | Stop::Stuck => "length",
            Stop::Eos | Stop::Caller | Stop::Complete => "stop",
        }
    }
}

#[derive(Default, Debug)]
pub struct Outcome {
    pub tokens: usize,
    /// Forward steps taken. Fewer steps than tokens means speculation paid.
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub stop: Stop,
}

impl Outcome {
    /// Share of drafted tokens the model agreed with. This is the number that
    /// says whether speculation is helping on this workload.
    pub fn acceptance(&self) -> f32 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f32 / self.drafted as f32
        }
    }

    /// Tokens committed per forward step. 1.0 is plain decoding.
    pub fn tokens_per_step(&self) -> f32 {
        self.tokens as f32 / self.steps.max(1) as f32
    }
}

/// Generate from a state that has already been prefilled.
///
/// `logits` must be the prediction following `history`, as returned by the
/// prefill. `history` is extended with everything committed. `on_token` sees
/// each committed token in order and returns `false` to stop — which is how a
/// caller implements stop sequences without this loop knowing about text.
pub fn generate(
    m: &Model,
    st: &mut State,
    history: &mut Vec<u32>,
    mut logits: Vec<f32>,
    plan: &Plan,
    mut guide: Option<&mut Guide>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Outcome {
    let vocab = m.spec.vocab;
    let mut rng = Rng::new(plan.seed);
    let mut out = Outcome::default();
    if plan.max_tokens == 0 {
        return out;
    }

    // Commit the first token from the prefill's own logits.
    if let Some(g) = guide.as_deref_mut() {
        if g.mask(&mut logits) == 0 {
            out.stop = Stop::Stuck;
            return out;
        }
    }
    let mut pending = plan.sampler.pick(&mut logits, history, &mut rng);
    // A constraint masks every control token, so under one this cannot fire.
    if m.spec.eos.contains(&pending) {
        out.stop = Stop::Eos;
        return out;
    }
    if st.pos >= st.ctx {
        out.stop = Stop::Context;
        return out;
    }
    history.push(pending);
    out.tokens += 1;
    if !on_token(pending) {
        out.stop = Stop::Caller;
        return out;
    }
    if let Some(g) = guide.as_deref_mut() {
        g.accept(pending);
        if g.complete() {
            out.stop = Stop::Complete;
            return out;
        }
    }

    let mut row = vec![0.0f32; vocab];
    while out.tokens < plan.max_tokens {
        // Never draft past the context window: the batch has to fit.
        let room = st.ctx.saturating_sub(st.pos + 1);
        if room == 0 && st.pos + 1 > st.ctx {
            out.stop = Stop::Context;
            break;
        }
        let want = plan.lookahead.min(room).min(plan.max_tokens - out.tokens);
        let draft = if want == 0 { Vec::new() } else { plan.lookup.propose(history, want) };

        let base = st.pos;
        let mut batch = Vec::with_capacity(1 + draft.len());
        batch.push(pending);
        batch.extend_from_slice(&draft);
        let all = m.forward_all(&batch, st);
        out.steps += 1;
        out.drafted += draft.len();

        // Row `i` is the prediction made after `batch[i]`, so row `i` is what
        // judges `draft[i]`.
        let mut taken = 0usize;
        let mut corrected = None;
        let mut finish = None;
        while taken < draft.len() {
            row.copy_from_slice(&all[taken * vocab..(taken + 1) * vocab]);
            // A constraint masks the row before verification, so a draft that
            // would break the grammar has probability zero and is rejected —
            // speculation needs no separate rule to stay within the shape.
            if let Some(g) = guide.as_deref_mut() {
                if g.mask(&mut row) == 0 {
                    finish = Some(Stop::Stuck);
                    break;
                }
            }
            match plan.sampler.verify(&mut row, history, draft[taken], &mut rng) {
                Ok(()) => {
                    let tok = draft[taken];
                    taken += 1;
                    out.accepted += 1;
                    if m.spec.eos.contains(&tok) {
                        finish = Some(Stop::Eos);
                        break;
                    }
                    history.push(tok);
                    out.tokens += 1;
                    if !on_token(tok) {
                        finish = Some(Stop::Caller);
                        break;
                    }
                    if let Some(g) = guide.as_deref_mut() {
                        g.accept(tok);
                        if g.complete() {
                            finish = Some(Stop::Complete);
                            break;
                        }
                    }
                    if out.tokens >= plan.max_tokens {
                        finish = Some(Stop::Length);
                        break;
                    }
                }
                // A rejected guess still yields the token to commit instead, and
                // it must be used: re-sampling the row would draw from the
                // uncorrected distribution and bias the output.
                Err(replacement) => {
                    corrected = Some(replacement);
                    break;
                }
            }
        }
        // Positions past the accepted prefix were speculative; drop them so the
        // cache holds exactly the committed sequence.
        st.truncate(base + 1 + taken);
        if let Some(stop) = finish {
            out.stop = stop;
            return out;
        }

        let next = match corrected {
            Some(t) => t,
            None => {
                row.copy_from_slice(&all[taken * vocab..(taken + 1) * vocab]);
                if let Some(g) = guide.as_deref_mut() {
                    if g.mask(&mut row) == 0 {
                        out.stop = Stop::Stuck;
                        return out;
                    }
                }
                plan.sampler.pick(&mut row, history, &mut rng)
            }
        };
        if m.spec.eos.contains(&next) {
            out.stop = Stop::Eos;
            return out;
        }
        if st.pos >= st.ctx {
            out.stop = Stop::Context;
            return out;
        }
        history.push(next);
        out.tokens += 1;
        if !on_token(next) {
            out.stop = Stop::Caller;
            return out;
        }
        if let Some(g) = guide.as_deref_mut() {
            g.accept(next);
            if g.complete() {
                out.stop = Stop::Complete;
                return out;
            }
        }
        pending = next;
    }
    out
}
