//! Chat templates: Jinja from a downloaded `tokenizer_config.json`.
//!
//! The interpreter has loops and a recursive expression parser, so both
//! non-termination and stack exhaustion are in scope. Depth is bounded; this is
//! how that stays true.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(src) = std::str::from_utf8(data) else { return };
    if let Ok(t) = moe::Template::parse(src) {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "s"}),
            serde_json::json!({"role": "user", "content": "u"}),
        ];
        let _ = t.render(&messages, true, "<s>", "</s>");
    }
});
