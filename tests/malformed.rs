//! Every parser that touches a file must fail, never panic.
//!
//! Three of them read bytes the user did not write: the packed `.moe` header, the
//! safetensors header, and GGUF. A fourth reads a `tokenizer.json`, and a fifth a
//! chat template — both of which arrive inside downloaded checkpoints. A crash on
//! any of them is a bug, and on a memory-mapped file a bounds mistake is worse
//! than a crash, so these are swept deterministically rather than left to a
//! fuzzer someone has to remember to run.
//!
//! `fuzz/` holds proper libFuzzer targets for the same surfaces; this is the part
//! that runs on every commit.

use moe::{Grammar, Model, Store, Template, Tokenizer};
use std::path::{Path, PathBuf};

/// A deterministic byte source, so a failure here is reproducible from its seed.
struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        // splitmix64.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }

    /// Corrupt one byte of `src`, which is how a header ends up almost-valid.
    fn mutate(&mut self, src: &[u8]) -> Vec<u8> {
        let mut out = src.to_vec();
        if !out.is_empty() {
            let at = (self.next() as usize) % out.len();
            out[at] = self.byte();
        }
        out
    }
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("moe-malformed-{name}"))
}

/// Open `bytes` as a model. Returning is a pass; panicking fails the test.
fn probe(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
    if let Ok(store) = Store::open(path) {
        // A store that opened must also survive being interrogated: the offsets
        // it accepted are used here for the first time.
        let _ = store.bytes();
        let names: Vec<String> = store.names().map(String::from).collect();
        for n in &names {
            let _ = store.get(n);
            let _ = store.shape(n);
            // Ask for slabs and rows the file never promised.
            let _ = store.view(n, 0, 0..1);
            let _ = store.view(n, usize::MAX, 0..1);
            let _ = store.view(n, 1, 0..usize::MAX);
        }
        let _ = Model::load(store);
    }
}

#[test]
fn arbitrary_bytes_never_panic_a_loader() {
    let path = scratch("random.moe");
    let mut r = Rand(0xC0FFEE);
    // Pure noise, and noise behind each magic, so the header parsers are reached
    // rather than rejected at the first four bytes.
    for len in [0usize, 1, 4, 8, 16, 17, 64, 300] {
        for _ in 0..40 {
            probe(&path, &r.bytes(len));
            let mut with_magic = b"MOEF".to_vec();
            with_magic.extend(r.bytes(len));
            probe(&path, &with_magic);
            let mut gguf = b"GGUF".to_vec();
            gguf.extend(r.bytes(len));
            probe(&path, &gguf);
        }
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_corrupted_packed_model_is_refused_not_trusted() {
    let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gqa");
    let good = scratch("good.moe");
    Store::open(&f).unwrap().pack(&good, moe::Dt::Q8, moe::Dt::Q8, |_| {}).unwrap();
    let whole = std::fs::read(&good).unwrap();
    // It must load before it is broken, or this proves nothing.
    assert!(Model::load(Store::open(&good).unwrap()).is_ok());

    let path = scratch("broken.moe");
    let mut r = Rand(7);
    // Single-byte corruptions, concentrated in the header where the offsets are.
    for _ in 0..400 {
        probe(&path, &r.mutate(&whole));
    }
    // And every truncation, which is what an interrupted download leaves.
    for cut in 0..whole.len().min(600) {
        probe(&path, &whole[..cut]);
    }
    // Header lengths are the field that used to panic; try the extremes directly.
    for hlen in [0u64, 1, 15, u64::MAX, u64::MAX - 16, 1 << 40, 1 << 62] {
        let mut bad = whole.clone();
        bad[8..16].copy_from_slice(&hlen.to_le_bytes());
        probe(&path, &bad);
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&good);
}

#[test]
fn a_corrupted_safetensors_shard_is_refused() {
    let f = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gqa");
    let whole = std::fs::read(f.join("model.safetensors")).unwrap();
    let dir = scratch("shard");
    let _ = std::fs::create_dir_all(&dir);
    std::fs::copy(f.join("config.json"), dir.join("config.json")).unwrap();
    let shard = dir.join("model.safetensors");

    let mut r = Rand(11);
    for _ in 0..200 {
        std::fs::write(&shard, r.mutate(&whole)).unwrap();
        if let Ok(store) = Store::open(&dir) {
            let names: Vec<String> = store.names().map(String::from).collect();
            for n in &names {
                let _ = store.get(n);
            }
            let _ = Model::load(store);
        }
    }
    // A header length past the end of the file.
    for hlen in [u64::MAX, 1 << 50, whole.len() as u64] {
        let mut bad = whole.clone();
        bad[..8].copy_from_slice(&hlen.to_le_bytes());
        std::fs::write(&shard, &bad).unwrap();
        let _ = Store::open(&dir);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `tokenizer.json` arrives inside a downloaded checkpoint, so it is untrusted
/// too — and it must survive being *used*, not only parsed.
#[test]
fn arbitrary_tokenizers_never_panic() {
    let mut r = Rand(23);
    for _ in 0..200 {
        let noise = String::from_utf8_lossy(&r.bytes(48)).into_owned();
        for j in [
            serde_json::json!({}),
            serde_json::json!({"model": {}}),
            serde_json::json!({"model": {"vocab": {}}}),
            serde_json::json!({"model": {"vocab": {noise.clone(): 0}, "merges": [noise.clone()]}}),
            // Ids far outside the vocabulary, and merges of nothing.
            serde_json::json!({"model": {"vocab": {"a": 4_000_000_000u32}, "merges": ["", " ", "a"]}}),
            serde_json::json!({"model": {"vocab": {"a": 0}}, "added_tokens": [{"id": 99, "content": ""}]}),
            serde_json::json!({"model": {"vocab": {"a": 0}}, "added_tokens": [{"id": u64::MAX}]}),
        ] {
            if let Ok(tok) = Tokenizer::from_json(&j) {
                let _ = tok.encode(&noise, Some(0));
                let _ = tok.encode("", None);
                let _ = tok.decode(&[0, 1, u32::MAX]);
                let _ = tok.decode_one(u32::MAX);
                let _ = tok.is_special(u32::MAX);
                let _ = tok.vocab_size();
            }
        }
    }
}

/// The grammar drives sampling, so it sees whatever bytes a model emits.
#[test]
fn arbitrary_bytes_never_panic_the_grammar() {
    let mut r = Rand(31);
    let schemas = [
        serde_json::json!({}),
        serde_json::json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}),
        serde_json::json!({"type": "array", "items": {"enum": [1, 12, "x"]}, "minItems": 1}),
    ];
    for s in &schemas {
        let g = Grammar::from_schema(s).unwrap();
        for _ in 0..300 {
            let mut m = g.start();
            for b in r.bytes(64) {
                if !g.feed(&mut m, b) {
                    break;
                }
            }
            let _ = g.complete(&m);
        }
        // Deeply nested and pathological inputs, not only noise.
        for src in ["[".repeat(200), "{".repeat(200), "\\".repeat(100), "1e".repeat(80), "\"".repeat(99)] {
            let mut m = g.start();
            let _ = g.feed_all(&mut m, src.as_bytes());
            let _ = g.complete(&m);
        }
    }
}

/// A chat template is Jinja from a downloaded config, so a malformed one must be
/// a parse error and a hostile one must not run forever.
#[test]
fn arbitrary_templates_never_panic() {
    let mut r = Rand(41);
    let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
    for _ in 0..300 {
        let src = String::from_utf8_lossy(&r.bytes(40)).into_owned();
        if let Ok(t) = Template::parse(&src) {
            let _ = t.render(&messages, true, "<s>", "</s>");
        }
    }
    for src in [
        "{{",
        "{%",
        "{#",
        "{{ }}",
        "{% %}",
        "{% if %}",
        "{% for %}",
        "{{ [ }}",
        "{{ 'a",
        "{% endif %}",
        "{% for x in x %}{% endfor %}",
        "{{ x.y.z.w }}",
        "{{ x[0][0][0] }}",
        "{{ ((((((1)))))) }}",
    ] {
        if let Ok(t) = Template::parse(src) {
            let _ = t.render(&messages, true, "", "");
        }
    }
    // Nesting past the interpreter's depth limit must be refused, not overflow.
    let deep = "{% if true %}".repeat(200) + &"{% endif %}".repeat(200);
    if let Ok(t) = Template::parse(&deep) {
        assert!(t.render(&messages, true, "", "").is_err(), "unbounded nesting was accepted");
    }
}
