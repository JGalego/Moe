//! GGUF headers, straight from arbitrary bytes.
//!
//! This is the one parser that takes a slice rather than a path, so it fuzzes at
//! full speed. Every offset and length in a GGUF header is attacker-controlled
//! and every one of them indexes a memory map.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(g) = moe::gguf::Gguf::parse(data) {
        // A header that parsed must also survive being used.
        let _ = g.config();
        let _ = g.tokenizer();
        let _ = g.chat_template();
        for t in &g.tensors {
            let _ = moe::gguf::rename(&t.name);
        }
    }
});
