//! Guessing what comes next without running the model.
//!
//! Decoding one token at a time on a CPU is bandwidth-bound: the step is
//! dominated by reading weights, and reading them for eight tokens costs barely
//! more than for one. So if something can guess the next few tokens cheaply, the
//! model can check all of them in a single step and keep whichever prefix it
//! agrees with. The engine already has one code path for `t` tokens, which is
//! exactly the machinery a verification step needs.
//!
//! The guesser here needs no second model. It looks for the last few tokens
//! occurring earlier in the same sequence and proposes whatever followed them,
//! which is free and works precisely where token-at-a-time decoding is most
//! wasteful — quoting a prompt back, editing code, filling a template,
//! continuing a list. On genuinely novel prose it proposes badly and the
//! acceptance rate falls; nothing is ever wrong, only wasted, because every
//! guess is checked against the true distribution before it is emitted.

/// Propose continuations by finding where the tail of a sequence occurred before.
#[derive(Clone, Copy)]
pub struct Lookup {
    /// Longest suffix to try matching. Longer matches are more reliable, so the
    /// search starts here and works down.
    pub max_ngram: usize,
    /// Shortest suffix worth trusting. Below about 2 the matches are noise.
    pub min_ngram: usize,
}

impl Default for Lookup {
    fn default() -> Lookup {
        Lookup { max_ngram: 8, min_ngram: 2 }
    }
}

impl Lookup {
    /// Up to `want` tokens that plausibly follow `history`.
    ///
    /// Tries the longest suffix first and takes the *most recent* earlier
    /// occurrence, on the reasoning that a repeat is usually continuing the
    /// nearest precedent rather than the oldest one.
    pub fn propose(&self, history: &[u32], want: usize) -> Vec<u32> {
        if want == 0 || history.len() < 2 {
            return Vec::new();
        }
        let hi = self.max_ngram.min(history.len() - 1);
        for n in (self.min_ngram.max(1)..=hi).rev() {
            let tail = &history[history.len() - n..];
            // Candidate starts must leave at least one token after the match,
            // and must not be the tail itself.
            let last_start = history.len() - n - 1;
            for start in (0..=last_start).rev() {
                if &history[start..start + n] == tail {
                    let from = start + n;
                    let take = want.min(history.len() - from);
                    if take > 0 {
                        return history[from..from + take].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposes_what_followed_the_last_repeat() {
        let l = Lookup { max_ngram: 4, min_ngram: 2 };
        // "1 2 3" occurred before and was followed by "4 5".
        let h = [1u32, 2, 3, 4, 5, 9, 1, 2, 3];
        assert_eq!(l.propose(&h, 2), vec![4, 5]);
        assert_eq!(l.propose(&h, 1), vec![4]);
        // Asking for more than the history holds yields what there is.
        assert_eq!(l.propose(&h, 9), vec![4, 5, 9, 1, 2, 3]);
    }

    #[test]
    fn prefers_the_most_recent_precedent() {
        let l = Lookup { max_ngram: 2, min_ngram: 1 };
        // "7" is followed by 1 early on and by 2 later; the later wins.
        let h = [7u32, 1, 0, 7, 2, 0, 7];
        assert_eq!(l.propose(&h, 1), vec![2]);
    }

    #[test]
    fn prefers_a_longer_match_over_a_nearer_short_one() {
        let l = Lookup { max_ngram: 3, min_ngram: 1 };
        // Suffix "5 6": the 3-gram "4 5 6" matched earlier and was followed by
        // 7, which beats the nearer bare "6" followed by 9.
        let h = [4u32, 5, 6, 7, 0, 6, 9, 0, 4, 5, 6];
        assert_eq!(l.propose(&h, 1), vec![7]);
    }

    #[test]
    fn novel_text_proposes_nothing() {
        let l = Lookup::default();
        let h: Vec<u32> = (0..40).collect();
        assert!(l.propose(&h, 4).is_empty());
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        let l = Lookup::default();
        assert!(l.propose(&[], 4).is_empty());
        assert!(l.propose(&[1], 4).is_empty());
        assert!(l.propose(&[1, 1, 1, 1], 0).is_empty());
        // A min_ngram longer than the history cannot match anything.
        assert!(Lookup { max_ngram: 8, min_ngram: 8 }.propose(&[1, 2, 3], 2).is_empty());
    }

    /// Whatever comes back must be a real slice of the history, so a draft can
    /// never introduce a token the vocabulary has not already produced.
    #[test]
    fn proposals_only_ever_replay_seen_tokens() {
        let l = Lookup::default();
        let h = [3u32, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 9, 2];
        let p = l.propose(&h, 5);
        assert!(p.iter().all(|t| h.contains(t)), "{p:?}");
    }
}
