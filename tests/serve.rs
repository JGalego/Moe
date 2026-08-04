//! The server has no independent reference to check against, so it is checked
//! against the engine underneath it: the same prompt must produce the same
//! tokens over HTTP as it does through `Model::forward` directly, and reusing
//! the KV cache across requests must not change a single one of them.
//!
//! Prompts go in as token ids, so these run on the tiny fixtures with no
//! tokenizer involved.

use moe::{http, Model, Rng, Sampler, Server, State, Store};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gqa")
}

fn model() -> Model {
    Model::load(Store::open(&fixture_dir()).unwrap()).unwrap()
}

/// Boot a server on an ephemeral port; returns the port.
fn boot(prefix_cache: bool) -> u16 {
    let server = Server::new(model(), None, None, "fixture".into(), 64, prefix_cache);
    let listener = http::bind("127.0.0.1", 0).unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::sync::Arc::new(server);
    std::thread::spawn(move || http::run(listener, move |req, conn| server.handle(req, conn)));
    port
}

fn post(port: u16, path: &str, body: &str) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or(out)
}

fn get(port: u16, path: &str) -> String {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or(out)
}

fn completion(port: u16, prompt: &[u32], max_tokens: usize) -> String {
    let body = serde_json::json!({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0});
    let raw = post(port, "/v1/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}: {raw}"));
    v["choices"][0]["text"].as_str().unwrap_or_else(|| panic!("no text in {raw}")).to_string()
}

/// What the engine produces for the same prompt, greedily, with no server in
/// the way. Without a tokenizer the server prints raw ids, so this does too.
fn engine_reference(prompt: &[u32], max_tokens: usize) -> String {
    let m = model();
    let mut st = State::new(&m, 64);
    let sampler = Sampler::default();
    let mut rng = Rng::new(0);
    let mut history = prompt.to_vec();
    let mut logits = Vec::new();
    for chunk in prompt.chunks(64) {
        logits = m.forward(chunk, &mut st);
    }
    let mut out = String::new();
    for _ in 0..max_tokens {
        let next = sampler.pick(&mut logits, &history, &mut rng);
        if m.spec.eos.contains(&next) {
            break;
        }
        out.push_str(&format!("{next} "));
        history.push(next);
        logits = m.forward(&[next], &mut st);
    }
    out
}

#[test]
fn completions_match_the_engine() {
    let port = boot(true);
    let prompt = [3u32, 11, 5, 40];
    assert_eq!(completion(port, &prompt, 5), engine_reference(&prompt, 5));
}

/// Exercises the reuse paths through the API — extend, diverge, and walk
/// backwards — against a server that never caches.
///
/// This checks the plumbing, not the arithmetic: these fixtures have random
/// weights, so their logits barely move with context and every prompt samples
/// the same tokens. The numerical guarantee is
/// `truncating_the_cache_matches_a_fresh_state` in oracle.rs, which compares
/// logits.
#[test]
fn prefix_reuse_paths_stay_consistent() {
    let warm = boot(true);
    let cold = boot(false);

    // Each turn extends the last, the way a chat grows.
    let turns: [&[u32]; 3] = [&[3, 11], &[3, 11, 5, 40], &[3, 11, 5, 40, 7, 1]];
    for turn in turns {
        let got = completion(warm, turn, 4);
        assert_eq!(got, completion(cold, turn, 4), "cache changed the output for {turn:?}");
        assert_eq!(got, engine_reference(turn, 4), "output drifted from the engine for {turn:?}");
    }

    // A prompt that diverges from the cached prefix must invalidate the tail.
    let diverged: &[u32] = &[3, 11, 9, 9];
    assert_eq!(completion(warm, diverged, 4), engine_reference(diverged, 4));

    // And a shorter prompt, which walks the cursor backwards.
    let shorter: &[u32] = &[3, 11];
    assert_eq!(completion(warm, shorter, 4), engine_reference(shorter, 4));
}

#[test]
fn metadata_endpoints_answer() {
    let port = boot(true);
    let health: serde_json::Value = serde_json::from_str(&get(port, "/health")).unwrap();
    assert_eq!(health["status"], "ok");

    let models: serde_json::Value = serde_json::from_str(&get(port, "/v1/models")).unwrap();
    assert_eq!(models["data"][0]["id"], "fixture");

    assert!(get(port, "/nope").contains("not found"));
}

#[test]
fn bad_requests_are_refused_not_crashed() {
    let port = boot(true);
    for (path, body, expect) in [
        ("/v1/completions", "{not json", "invalid JSON"),
        ("/v1/completions", r#"{"prompt":{}}"#, "must be a string or an array"),
        ("/v1/completions", r#"{"prompt":[]}"#, "must be a string or an array"),
        // No tokenizer is loaded for the fixtures, so text has nothing to encode with.
        ("/v1/completions", r#"{"prompt":"text"}"#, "no tokenizer"),
        // A base model has no chat template, and guessing one would mis-prompt it.
        ("/v1/chat/completions", r#"{"messages":[{"role":"user","content":"hi"}]}"#, "no chat template"),
        // Asking for more than the window holds is caught before any work.
        ("/v1/completions", r#"{"prompt":[1,2],"max_tokens":9999}"#, "context window"),
    ] {
        let raw = post(port, path, body);
        assert!(raw.contains(expect), "expected {expect:?} in response to {body}, got: {raw}");
    }
}

#[test]
fn streaming_deltas_reassemble_into_the_whole_completion() {
    let port = boot(true);
    let prompt = [3u32, 11, 5];
    let body = serde_json::json!({"prompt": prompt, "max_tokens": 5, "temperature": 0, "stream": true});
    let raw = post(port, "/v1/completions", &body.to_string());

    let mut joined = String::new();
    let mut saw_done = false;
    let mut finish = None;
    for line in raw.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            saw_done = true;
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        joined.push_str(v["choices"][0]["text"].as_str().unwrap_or(""));
        if let Some(f) = v["choices"][0]["finish_reason"].as_str() {
            finish = Some(f.to_string());
        }
    }
    assert!(saw_done, "stream never terminated with [DONE]");
    assert_eq!(finish.as_deref(), Some("length"));
    assert_eq!(joined, engine_reference(&prompt, 5), "streamed deltas differ from the engine");
}
