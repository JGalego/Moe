//! Scoring text: how surprised a model is by tokens it did not choose.
//!
//! This is the number that makes a quantisation claim checkable. Packing a
//! checkpoint to Q4 shrinks it 3.56x, and a compression ratio says nothing at
//! all about whether the model still works; the negative log-likelihood of held-
//! out text does, and it is the same measurement for both files.
//!
//! Two things matter for the number to mean anything. Every token is scored with
//! as much left context as the window allows, which is what the stride schedule
//! below is for — a naive chunking scores the first token of each block on no
//! context and quietly inflates the result. And bits-per-byte is reported beside
//! perplexity, because perplexity is per *token* and therefore not comparable
//! across two models that tokenise the same text differently.

use crate::model::{Model, State};

/// Running total of surprise over a corpus.
#[derive(Default, Clone, Copy)]
pub struct Score {
    /// Summed negative log-likelihood, in nats.
    pub nll: f64,
    /// Tokens scored (the first token of the corpus is never a target).
    pub tokens: usize,
    /// Bytes those tokens decode to, for the tokenisation-independent figure.
    pub bytes: usize,
}

impl Score {
    /// Mean negative log-likelihood per token, in nats.
    pub fn mean_nll(&self) -> f64 {
        self.nll / self.tokens.max(1) as f64
    }

    /// `exp` of the mean NLL. Comparable only between models sharing a vocabulary.
    pub fn perplexity(&self) -> f64 {
        self.mean_nll().exp()
    }

    pub fn bits_per_token(&self) -> f64 {
        self.mean_nll() / std::f64::consts::LN_2
    }

    /// Bits per byte of the original text — comparable across tokenizers, and
    /// the honest way to put two checkpoints side by side.
    pub fn bits_per_byte(&self) -> f64 {
        self.nll / std::f64::consts::LN_2 / self.bytes.max(1) as f64
    }
}

/// `-log p(target)` from raw logits, via a numerically safe log-sum-exp.
pub fn nll_of(logits: &[f32], target: u32) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let sum: f64 = logits.iter().map(|l| ((*l - max) as f64).exp()).sum();
    let lse = max as f64 + sum.ln();
    lse - logits[target as usize % logits.len()] as f64
}

/// How much of a window is fresh, given the stride schedule.
///
/// Windows overlap so that every token is predicted from a full `ctx` of
/// context wherever the corpus allows it. A token already scored by an earlier
/// window is re-run for context but not counted twice.
pub struct Windows {
    pub total: usize,
    pub ctx: usize,
    pub stride: usize,
}

impl Windows {
    /// `(start, first scored offset within the window, window length)` per step.
    ///
    /// Windows are laid out so their target ranges are contiguous and disjoint:
    /// each one picks up exactly where the last left off, and scores as far as
    /// its right edge reaches. A stride shorter than the window is what buys the
    /// context — the tokens before a window's first target are still forwarded,
    /// they are just not counted again.
    pub fn plan(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        if self.total < 2 {
            return out;
        }
        let ctx = self.ctx.max(2);
        // A stride past the window would leave a gap no later window covers.
        let stride = self.stride.clamp(1, ctx);
        // Highest absolute target scored so far; 0 means none, since token 0 is
        // predicted by nothing.
        let mut prev_end = 0usize;
        let mut start = 0usize;
        loop {
            let len = ctx.min(self.total - start);
            // Target `prev_end + 1` is predicted by position `prev_end`.
            out.push((start, prev_end - start, len));
            prev_end = (start + len).min(self.total - 1);
            if prev_end >= self.total - 1 {
                break;
            }
            start += stride;
        }
        out
    }
}

/// Score `tokens` with a fresh state per window, reporting progress by window.
///
/// `bytes` is the byte length of the text the tokens came from, carried through
/// so bits-per-byte is available; pass 0 when scoring raw ids.
pub fn score(
    m: &Model,
    tokens: &[u32],
    ctx: usize,
    stride: usize,
    bytes: usize,
    mut on_window: impl FnMut(usize, usize),
) -> Score {
    let mut acc = Score { nll: 0.0, tokens: 0, bytes };
    let plan = Windows { total: tokens.len(), ctx, stride }.plan();
    for (wi, (start, from, len)) in plan.iter().enumerate() {
        let window = &tokens[*start..*start + *len];
        let mut st = State::new(m, *len);
        let mut done = 0usize;
        // Block the window so the logits buffer stays `block * vocab`, not
        // `ctx * vocab` — the latter is hundreds of megabytes at a real context.
        while done < window.len() {
            let take = 64.min(window.len() - done);
            let logits = m.forward_all(&window[done..done + take], &mut st);
            for i in 0..take {
                let offset = done + i;
                let target = *start + offset + 1;
                if offset < *from || target >= tokens.len() {
                    continue;
                }
                acc.nll += nll_of(&logits[i * m.spec.vocab..(i + 1) * m.spec.vocab], tokens[target]);
                acc.tokens += 1;
            }
            done += take;
        }
        on_window(wi + 1, plan.len());
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nll_is_the_negative_log_softmax() {
        // Two equal logits: each has probability 1/2, so NLL is ln 2.
        let d = nll_of(&[1.0, 1.0], 0);
        assert!((d - std::f64::consts::LN_2).abs() < 1e-6, "{d}");
        // A dominant logit approaches zero surprise.
        assert!(nll_of(&[40.0, 0.0], 0) < 1e-6);
        // ...and the alternative is correspondingly expensive.
        assert!(nll_of(&[40.0, 0.0], 1) > 39.0);
    }

    /// Shifting every logit by a constant cannot change a probability.
    #[test]
    fn nll_is_shift_invariant() {
        let a = nll_of(&[1.0, 2.0, 3.0], 1);
        let b = nll_of(&[1001.0, 1002.0, 1003.0], 1);
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    /// Every token but the first must be scored exactly once.
    #[test]
    fn windows_cover_each_target_once() {
        for total in [2usize, 5, 17, 64, 100] {
            for ctx in [2usize, 4, 8, 33] {
                for stride in [1usize, 2, 3, ctx / 2 + 1] {
                    let plan = Windows { total, ctx, stride }.plan();
                    let mut hits = vec![0usize; total];
                    for (start, from, len) in plan {
                        for offset in from..len {
                            let target = start + offset + 1;
                            if target < total {
                                hits[target] += 1;
                            }
                        }
                    }
                    assert!(hits[1..].iter().all(|h| *h == 1), "total {total} ctx {ctx} stride {stride}: {hits:?}");
                    assert_eq!(hits[0], 0, "token 0 is never a target");
                }
            }
        }
    }

    #[test]
    fn a_corpus_shorter_than_two_tokens_scores_nothing() {
        assert!(Windows { total: 0, ctx: 8, stride: 4 }.plan().is_empty());
        assert!(Windows { total: 1, ctx: 8, stride: 4 }.plan().is_empty());
    }

    /// Every scored position must see all the context the window can give it.
    #[test]
    fn later_windows_score_only_their_fresh_tail() {
        let plan = Windows { total: 20, ctx: 8, stride: 4 }.plan();
        assert_eq!(plan[0], (0, 0, 8));
        // The first window reaches target 8 at its right edge, so the second
        // starts scoring at token 9 — predicted by position 8, which sits four
        // tokens into a window that begins at 4 and therefore has full context.
        assert_eq!(plan[1].0, 4);
        assert_eq!(plan[1].0 + plan[1].1, 8);
    }

    #[test]
    fn score_totals_divide_out() {
        let s = Score { nll: 4.0 * std::f64::consts::LN_2, tokens: 4, bytes: 8 };
        assert!((s.bits_per_token() - 1.0).abs() < 1e-9);
        assert!((s.bits_per_byte() - 0.5).abs() < 1e-9);
        assert!((s.perplexity() - 2.0).abs() < 1e-9);
    }
}
