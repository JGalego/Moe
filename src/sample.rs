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
            return best as u32;
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

        let r = rng.next_f32();
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return idx[i];
            }
        }
        *idx.last().unwrap()
    }
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
}
