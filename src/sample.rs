//! Logit post-processing and sampling.

/// xorshift64*, so runs are reproducible from `--seed` without a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u32 << 24) as f32
    }
}

#[derive(Clone)]
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub rng: u64,
}

impl Default for Sampler {
    fn default() -> Sampler {
        Sampler { temp: 0.0, top_p: 0.95, top_k: 0, repeat_penalty: 1.0, repeat_window: 64, rng: 0 }
    }
}

impl Sampler {
    /// Pick the next token id from raw logits.
    pub fn pick(&self, logits: &mut [f32], history: &[u32], rng: &mut Rng) -> u32 {
        let (idx, probs) = self.distribution(logits, history);
        draw(&idx, &probs, rng)
    }

    /// Verify a token some cheaper process guessed would come next.
    ///
    /// `Ok(())` keeps the guess; `Err(t)` rejects it and returns the token to
    /// commit instead. The accepted stream is distributed exactly as [`pick`]
    /// would have produced it, which is what makes speculation lossless rather
    /// than an approximation:
    ///
    /// * Greedy decoding accepts precisely when the guess *is* the argmax, so
    ///   the output is bit-identical to not speculating at all.
    /// * With temperature, the guess is accepted with probability `p(guess)`
    ///   under the true distribution, and a rejection draws from that
    ///   distribution with the guess removed and the rest renormalised. For a
    ///   drafter that proposes one token with certainty — which any lookup
    ///   drafter is — that is the standard rejection correction, and it leaves
    ///   the marginal distribution of every emitted token untouched.
    pub fn verify(&self, logits: &mut [f32], history: &[u32], guess: u32, rng: &mut Rng) -> Result<(), u32> {
        let (idx, probs) = self.distribution(logits, history);
        let p = idx.iter().position(|i| *i == guess).map(|k| probs[k]).unwrap_or(0.0);
        // next_f32 is in [0, 1), so a certainty accepts unconditionally.
        if rng.next_f32() < p {
            return Ok(());
        }
        // The residual: everything but the guess, renormalised.
        let keep: Vec<(u32, f32)> = idx.iter().copied().zip(probs).filter(|(i, _)| *i != guess).collect();
        let z: f32 = keep.iter().map(|(_, p)| *p).sum();
        if keep.is_empty() || z <= 0.0 {
            // The guess held all the mass, so there is nothing else to move to.
            return Ok(());
        }
        let (ids, ps): (Vec<u32>, Vec<f32>) = keep.into_iter().map(|(i, p)| (i, p / z)).unzip();
        Err(draw(&ids, &ps, rng))
    }

    /// The candidate ids and probabilities this sampler would draw from, after
    /// the repeat penalty, temperature, top-k and top-p. Greedy decoding is the
    /// degenerate case: one candidate with probability 1.
    pub fn distribution(&self, logits: &mut [f32], history: &[u32]) -> (Vec<u32>, Vec<f32>) {
        if self.repeat_penalty != 1.0 {
            for t in history.iter().rev().take(self.repeat_window) {
                let l = &mut logits[*t as usize];
                *l = if *l > 0.0 { *l / self.repeat_penalty } else { *l * self.repeat_penalty };
            }
        }
        if self.temp <= 0.0 {
            let mut best = 0usize;
            for (i, v) in logits.iter().enumerate() {
                if *v > logits[best] {
                    best = i;
                }
            }
            return (vec![best as u32], vec![1.0]);
        }

        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let k = if self.top_k == 0 { logits.len() } else { self.top_k.min(logits.len()) };
        idx.select_nth_unstable_by(k - 1, |a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
        idx.truncate(k);
        idx.sort_unstable_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));

        let max = logits[idx[0] as usize];
        let mut probs: Vec<f32> = idx.iter().map(|i| ((logits[*i as usize] - max) / self.temp).exp()).collect();
        let z: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= z);

        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut acc = 0.0;
            let cut = probs
                .iter()
                .position(|p| {
                    acc += p;
                    acc >= self.top_p
                })
                .map_or(probs.len(), |i| i + 1);
            probs.truncate(cut);
            idx.truncate(cut);
            let z: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= z);
        }
        (idx, probs)
    }
}

/// Draw one id from a normalised categorical distribution.
fn draw(idx: &[u32], probs: &[f32], rng: &mut Rng) -> u32 {
    if idx.len() == 1 {
        return idx[0];
    }
    let r = rng.next_f32();
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return idx[i];
        }
    }
    *idx.last().unwrap_or(&0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let s = Sampler::default();
        let mut rng = Rng::new(1);
        assert_eq!(s.pick(&mut [0.1, 5.0, -2.0], &[], &mut rng), 1);
    }

    #[test]
    fn temperature_zero_ignores_seed() {
        let s = Sampler::default();
        let mut logits = vec![1.0, 2.0, 3.0, 0.5];
        let a = s.pick(&mut logits.clone(), &[], &mut Rng::new(7));
        let b = s.pick(&mut logits, &[], &mut Rng::new(99));
        assert_eq!(a, b);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let s = Sampler { temp: 1.0, top_k: 1, ..Sampler::default() };
        let mut rng = Rng::new(3);
        assert_eq!(s.pick(&mut [0.0, 0.2, 9.0], &[], &mut rng), 2);
    }

    #[test]
    fn repeat_penalty_demotes_history() {
        let s = Sampler { repeat_penalty: 2.0, ..Sampler::default() };
        let mut rng = Rng::new(1);
        assert_eq!(s.pick(&mut [3.0, 2.0], &[0], &mut rng), 1);
    }

    /// Greedily, a guess is accepted exactly when it is the argmax — which is
    /// what makes speculative decoding bit-identical rather than merely close.
    #[test]
    fn greedy_verification_is_exactly_the_argmax() {
        let s = Sampler::default();
        let logits = [0.5f32, 9.0, -1.0];
        let mut rng = Rng::new(4);
        assert_eq!(s.verify(&mut logits.clone(), &[], 1, &mut rng), Ok(()));
        assert_eq!(s.verify(&mut logits.clone(), &[], 0, &mut rng), Err(1));
        assert_eq!(s.verify(&mut logits.clone(), &[], 2, &mut rng), Err(1));
    }

    /// A token outside the truncated candidate set has probability zero, so it
    /// can never be accepted — top-k must bind on verification as it does on
    /// sampling, or speculation would smuggle tokens past the cutoff.
    #[test]
    fn truncation_binds_on_verification() {
        let s = Sampler { temp: 1.0, top_k: 1, ..Sampler::default() };
        let mut rng = Rng::new(9);
        assert_eq!(s.verify(&mut [0.0, 0.2, 9.0], &[], 0, &mut rng), Err(2));
        assert_eq!(s.verify(&mut [0.0, 0.2, 9.0], &[], 2, &mut rng), Ok(()));
    }

    /// The distribution of accepted-or-corrected tokens must match plain
    /// sampling. Verified empirically over many draws, because this is the one
    /// property that cannot be checked by reading the code.
    #[test]
    fn verification_preserves_the_sampled_distribution() {
        let s = Sampler { temp: 1.0, top_p: 1.0, ..Sampler::default() };
        let logits = [1.0f32, 2.0, 0.5, -0.5];
        const N: usize = 60_000;

        let mut direct = [0usize; 4];
        let mut rng = Rng::new(11);
        for _ in 0..N {
            direct[s.pick(&mut logits.clone(), &[], &mut rng) as usize] += 1;
        }

        // Draft a fixed token every time — the worst case for a naive scheme,
        // which would over-represent whatever was guessed.
        for guess in 0..4u32 {
            let mut spec = [0usize; 4];
            let mut rng = Rng::new(11);
            for _ in 0..N {
                match s.verify(&mut logits.clone(), &[], guess, &mut rng) {
                    Ok(()) => spec[guess as usize] += 1,
                    Err(t) => spec[t as usize] += 1,
                }
            }
            for i in 0..4 {
                let (a, b) = (direct[i] as f32 / N as f32, spec[i] as f32 / N as f32);
                assert!((a - b).abs() < 0.01, "guess {guess}, token {i}: {a:.3} direct vs {b:.3} speculative");
            }
        }
    }
}
