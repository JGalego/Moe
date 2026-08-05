//! The JSON grammar, from both ends: an arbitrary schema, and arbitrary bytes fed
//! through the automaton it compiles to.
//!
//! The automaton runs inside the sampling loop over every candidate token, so a
//! panic here is a panic in the middle of generation.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // The first byte picks the schema; the rest is the document.
    let (pick, body) = data.split_first().unwrap_or((&0, &[]));
    let schema = match pick % 4 {
        0 => serde_json::json!({}),
        1 => serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}),
        2 => serde_json::json!({"type": "array", "items": {"enum": [1, 12, "x", null]}, "minItems": 1}),
        // A schema straight out of the input, so compilation is fuzzed too.
        _ => match std::str::from_utf8(body).ok().and_then(|s| serde_json::from_str(s).ok()) {
            Some(v) => v,
            None => serde_json::json!({"type": "string"}),
        },
    };
    let Ok(g) = moe::Grammar::from_schema(&schema) else { return };
    let mut m = g.start();
    for b in body {
        if !g.feed(&mut m, *b) {
            break;
        }
    }
    let _ = g.complete(&m);
});
