//! `tokenizer.json`, which arrives inside every downloaded checkpoint.
//!
//! Parsing is only half of it: the tokenizer must also survive being *used*, so
//! a vocabulary that loads is then asked to encode and decode. Two real bugs
//! lived here — a vocabulary id trusted as an allocation size, and an added token
//! with empty content that made encoding loop forever.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    let Ok(j) = serde_json::from_str::<serde_json::Value>(text) else { return };
    if let Ok(tok) = moe::Tokenizer::from_json(&j) {
        let ids = tok.encode("hello world 123\n\ttabs", Some(0));
        let _ = tok.decode(&ids);
        let _ = tok.decode(&[0, 1, u32::MAX]);
        let _ = tok.is_special(u32::MAX);
    }
});
