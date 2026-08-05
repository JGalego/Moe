//! Constrained decoding, end to end through a real forward pass.
//!
//! The grammar's own tests check the automaton against text. These check the
//! thing that actually matters: that a model driven by masked logits cannot
//! produce output the grammar would reject. The weights are the random fixture
//! ones, so the model has no idea what JSON is — which is the point. If valid
//! JSON comes out of a model that has never seen any, it came out because the
//! mask made everything else unreachable.

use moe::{draft::Lookup, generate::generate, Grammar, Guide, Model, Plan, Sampler, State, Store, Tokenizer};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gqa")
}

fn model() -> Model {
    Model::load(Store::open(&fixture()).unwrap()).unwrap()
}

/// A byte-level vocabulary of exactly the characters JSON is made of, one token
/// each, sized to the fixture's 48-token vocabulary. Single-byte tokens are the
/// hardest case for the masker: every structural decision is a separate step.
fn tokenizer() -> Tokenizer {
    let chars = "{}[]\":,0123456789truefalsnq abcXY.-";
    let mut vocab = serde_json::Map::new();
    let table = byte_level_table();
    for (i, c) in chars.chars().collect::<Vec<_>>().iter().enumerate() {
        // The engine's vocabulary is 48 wide; keep inside it.
        assert!(i < 48, "test vocabulary outgrew the fixture");
        vocab.insert(table[*c as usize].to_string(), serde_json::json!(i));
    }
    let n = vocab.len();
    // One control token, to prove specials are masked out under a constraint.
    let j = serde_json::json!({
        "model": {"vocab": vocab, "merges": []},
        "added_tokens": [{"id": n, "content": "<|eos|>", "special": true}],
    });
    Tokenizer::from_json(&j).unwrap()
}

/// The GPT-2 byte-to-printable-codepoint bijection, as the tokenizer expects
/// vocabulary keys to be written.
fn byte_level_table() -> Vec<char> {
    let mut map = vec!['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
    }
    map
}

/// Generate under `grammar` and return the decoded text plus why it stopped.
fn constrained(g: Grammar, max_tokens: usize, lookahead: usize, temp: f32, seed: u64) -> (String, moe::Stop) {
    let m = model();
    let tok = tokenizer();
    let mut guide = Guide::new(g, &tok);
    let prompt: Vec<u32> = vec![0, 1, 2];
    let ctx = prompt.len() + max_tokens + lookahead + 2;
    let mut st = State::new(&m, ctx);
    let mut logits = Vec::new();
    for chunk in prompt.chunks(64) {
        logits = m.forward(chunk, &mut st);
    }
    let plan = Plan {
        max_tokens,
        sampler: Sampler { temp, top_p: 1.0, ..Sampler::default() },
        seed,
        lookahead,
        lookup: Lookup { max_ngram: 4, min_ngram: 2 },
        logprobs: 0,
    };
    let mut out = Vec::new();
    let o = generate(&m, &mut st, &mut prompt.clone(), logits, &plan, Some(&mut guide), |t| {
        out.push(t);
        true
    });
    (tok.decode(&out), o.stop)
}

#[test]
fn free_json_output_always_parses() {
    // Many seeds, because a mask bug may only show on a particular path.
    for seed in 0..24u64 {
        for temp in [0.0f32, 1.0] {
            let (text, stop) = constrained(Grammar::json(), 40, 0, temp, seed);
            assert!(!text.is_empty(), "seed {seed} produced nothing");
            // Either it finished a document, or it ran out of budget mid-way.
            if stop == moe::Stop::Complete {
                serde_json::from_str::<serde_json::Value>(&text)
                    .unwrap_or_else(|e| panic!("seed {seed} temp {temp} emitted invalid JSON {text:?}: {e}"));
            } else {
                // An unfinished document must still be a legal prefix, which is
                // exactly what the grammar says it is.
                let g = Grammar::json();
                let mut mach = g.start();
                assert!(g.feed_all(&mut mach, text.as_bytes()), "seed {seed}: {text:?} is not a viable prefix");
            }
        }
    }
}

#[test]
fn a_schema_is_obeyed_to_the_letter() {
    let src = serde_json::json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}, "ok": {"type": "boolean"}},
        "required": ["n", "ok"],
    });
    let g = Grammar::from_schema(&src).unwrap();
    for seed in 0..16u64 {
        let (text, stop) = constrained(g.clone(), 60, 0, 1.0, seed);
        if stop != moe::Stop::Complete {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("seed {seed} emitted invalid JSON {text:?}: {e}"));
        assert!(v["n"].is_i64(), "seed {seed}: n is not an integer in {text}");
        assert!(v["ok"].is_boolean(), "seed {seed}: ok is not a boolean in {text}");
        assert_eq!(v.as_object().unwrap().len(), 2, "seed {seed}: extra properties in {text}");
        // Keys come out in the declared order, which is what makes the automaton
        // deterministic; check the bytes, not just the parse.
        assert!(text.find("\"n\"").unwrap() < text.find("\"ok\"").unwrap(), "order not held in {text}");
    }
}

/// Generation must stop the moment the document is complete rather than running
/// to the token budget and trailing garbage.
#[test]
fn a_completed_document_ends_generation() {
    let g = Grammar::from_schema(&serde_json::json!({"type": "boolean"})).unwrap();
    let (text, stop) = constrained(g, 40, 0, 0.0, 1);
    assert_eq!(stop, moe::Stop::Complete);
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    assert!(v.is_boolean(), "{text:?}");
    // Nothing may follow the value that completed the document.
    assert_eq!(text.trim(), if v == true { "true" } else { "false" });
}

/// An enum can only produce one of its members, whatever the model prefers.
#[test]
fn an_enum_admits_only_its_members() {
    let g = Grammar::from_schema(&serde_json::json!({"enum": ["abc", "XY"]})).unwrap();
    for seed in 0..12u64 {
        let (text, stop) = constrained(g.clone(), 20, 0, 1.0, seed);
        if stop == moe::Stop::Complete {
            // Leading whitespace is legal JSON, so compare the value, not the bytes.
            let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert!(v == "abc" || v == "XY", "seed {seed}: {text:?}");
        }
    }
}

/// Speculation and constraints have to compose: a drafted token that would break
/// the grammar is masked to zero probability and therefore always rejected, so
/// the output stays valid and identical to the unspeculated one.
#[test]
fn speculation_cannot_smuggle_invalid_tokens_past_a_constraint() {
    let g = Grammar::json();
    for seed in 0..12u64 {
        let (plain, plain_stop) = constrained(g.clone(), 40, 0, 0.0, seed);
        for lookahead in [2usize, 4, 8] {
            let (spec, spec_stop) = constrained(g.clone(), 40, lookahead, 0.0, seed);
            assert_eq!(spec, plain, "seed {seed} lookahead {lookahead} changed constrained output");
            assert_eq!(spec_stop, plain_stop);
        }
        if plain_stop == moe::Stop::Complete {
            serde_json::from_str::<serde_json::Value>(&plain).expect("constrained output must parse");
        }
    }
}

/// Control tokens carry no document bytes, so a constraint must make them
/// unreachable — otherwise a model could end a document early with an EOS the
/// grammar never saw.
#[test]
fn control_tokens_are_unreachable_under_a_constraint() {
    let tok = tokenizer();
    let mut guide = Guide::new(Grammar::json(), &tok);
    let mut logits = vec![0.0f32; 48];
    // Make the special token overwhelmingly attractive.
    let special = (0..48u32).find(|i| tok.is_special(*i)).expect("the fixture tokenizer has a special token");
    logits[special as usize] = 100.0;
    let allowed = guide.mask(&mut logits);
    assert!(allowed > 0, "the mask left nothing at all");
    assert_eq!(logits[special as usize], f32::NEG_INFINITY, "a control token survived the mask");
}
