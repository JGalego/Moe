//! Moe — CPU inference for sparse mixture-of-experts language models.
//!
//! The engine is deliberately small. Weights are memory mapped and read in
//! whatever format they already have (f32/f16/bf16, or 4/8-bit blocks after
//! packing), the model's shape is detected from the checkpoint rather than
//! hard-coded, and a forward step is the same code whether it is prefilling a
//! prompt or decoding one token.
//!
//! ```no_run
//! use moe::{Model, State, Store, Tokenizer};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Model::load(Store::open("model.moe".as_ref())?)?;
//! let tok = Tokenizer::load("model.moe".as_ref())?;
//! let mut state = State::new(&model, 4096);
//! let ids = tok.encode("The capital of France is", model.spec.bos);
//! let logits = model.forward(&ids, &mut state);
//! # let _ = logits; Ok(())
//! # }
//! ```

pub mod model;
pub mod quant;
pub mod sample;
pub mod spec;
pub mod store;
pub mod tokenizer;

pub use model::{Model, State, Stats};
pub use quant::{Dt, QT};
pub use sample::{Rng, Sampler};
pub use spec::Spec;
pub use store::Store;
pub use tokenizer::Tokenizer;

/// Peak resident set size in bytes, or 0 where the OS does not report it.
pub fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// Human-readable byte count.
pub fn human(bytes: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", bytes, U[0])
    } else {
        format!("{v:.2} {}", U[i])
    }
}
