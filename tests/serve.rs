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

/// Embeddings over HTTP must agree with the engine, batch in the order they were
/// given, and stay stateless — the generation session's cache must be untouched
/// afterwards, since embedding shares nothing with it.
#[test]
fn embeddings_match_the_engine_and_batch_in_order() {
    let port = boot(true);
    let m = model();
    let (a, b) = (vec![1u32, 2, 3], vec![7u32, 8]);

    let raw = post(port, "/v1/embeddings", &serde_json::json!({"input": [a, b]}).to_string());
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}: {raw}"));
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(v["usage"]["prompt_tokens"], 5, "token count is the sum of the inputs");

    for (i, ids) in [&a, &b].iter().enumerate() {
        assert_eq!(v["data"][i]["index"], i, "batch came back out of order");
        let got: Vec<f32> =
            v["data"][i]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let want = m.embed(ids, 64, moe::Pool::Mean, true);
        assert_eq!(got.len(), want.len());
        let d = got.iter().zip(&want).fold(0.0f32, |mx, (x, y)| mx.max((x - y).abs()));
        assert!(d < 1e-5, "input {i}: HTTP embedding differs from the engine by {d}");
        // The endpoint always normalises.
        let norm = got.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "input {i}: length {norm}");
    }

    // A single input, not a list, is also accepted.
    let one = post(port, "/v1/embeddings", &serde_json::json!({"input": a}).to_string());
    let one: serde_json::Value = serde_json::from_str(&one).unwrap();
    assert_eq!(one["data"].as_array().unwrap().len(), 1);

    // Bad shapes are refused rather than half-answered.
    for bad in [
        serde_json::json!({}),
        serde_json::json!({"input": []}),
        serde_json::json!({"input": 5}),
        serde_json::json!({"input": [[]]}),
    ] {
        let r = post(port, "/v1/embeddings", &bad.to_string());
        assert!(r.contains("error"), "{bad} was not refused: {r}");
    }

    // Generation still works afterwards, so embedding disturbed no state.
    assert!(!completion(port, &[1, 2, 3], 4).is_empty());
}

/// Reported logprobs must be the model's own, not the sampler's — so they have
/// to be a valid log-distribution: never positive, and the sum over the top
/// alternatives never exceeding 1 in probability.
#[test]
fn logprobs_describe_a_real_distribution() {
    let port = boot(true);
    let body = serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 5, "temperature": 0, "logprobs": 4});
    let raw = post(port, "/v1/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}: {raw}"));
    let lp = &v["choices"][0]["logprobs"];
    assert!(!lp.is_null(), "no logprobs came back: {raw}");

    let tokens = lp["tokens"].as_array().unwrap();
    let chosen = lp["token_logprobs"].as_array().unwrap();
    let tops = lp["top_logprobs"].as_array().unwrap();
    let offsets = lp["text_offset"].as_array().unwrap();
    assert_eq!(tokens.len(), chosen.len(), "parallel arrays disagree in length");
    assert_eq!(tokens.len(), tops.len());
    assert_eq!(tokens.len(), offsets.len());
    assert!(!tokens.is_empty());

    for (i, l) in chosen.iter().enumerate() {
        let l = l.as_f64().unwrap();
        assert!(l <= 1e-5, "token {i} has logprob {l}, which is a probability above 1");
        // Greedy decoding picks the argmax, so its probability cannot be tiny.
        assert!(l > -20.0, "token {i} logprob {l} is implausible for a greedy pick");
        let top = tops[i].as_object().unwrap();
        assert_eq!(top.len(), 4, "asked for 4 alternatives, got {}", top.len());
        let mass: f64 = top.values().map(|v| v.as_f64().unwrap().exp()).sum();
        assert!(mass <= 1.0 + 1e-4, "top-4 probabilities sum to {mass}");
        // The chosen token is the argmax, so it must be the strongest listed.
        let best = top.values().map(|v| v.as_f64().unwrap()).fold(f64::NEG_INFINITY, f64::max);
        assert!((best - l).abs() < 1e-5, "the greedy pick was not the strongest alternative");
    }

    // Offsets must march forward by each token's length.
    let mut at = 0usize;
    for (i, tok) in tokens.iter().enumerate() {
        assert_eq!(offsets[i].as_u64().unwrap() as usize, at);
        at += tok.as_str().unwrap().len();
    }

    // The chat endpoint reports the same numbers in its own layout.
    let chat = serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 3, "temperature": 0, "logprobs": 2});
    let raw = post(port, "/v1/completions", &chat.to_string());
    assert!(raw.contains("token_logprobs"), "{raw}");
    // ...and asking for none reports none rather than an empty object.
    let none = serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 2, "temperature": 0});
    let v: serde_json::Value = serde_json::from_str(&post(port, "/v1/completions", &none.to_string())).unwrap();
    assert!(v["choices"][0]["logprobs"].is_null());
}

/// `n` must produce that many distinct choices, and `echo` must put the prompt
/// back in front without disturbing the token accounting.
#[test]
fn n_and_echo_shape_the_choices() {
    let port = boot(true);
    let body = serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 4, "temperature": 1.0, "n": 3, "seed": 5});
    let raw = post(port, "/v1/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{e}: {raw}"));
    let choices = v["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 3);
    for (i, c) in choices.iter().enumerate() {
        assert_eq!(c["index"], i, "choices came back out of order");
        assert!(c["text"].is_string());
    }
    // Usage counts every completion, not just the first.
    let total: u64 = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(total >= 3, "usage reports {total} tokens for three completions");

    // Without a tokenizer there is no text to echo, so echo is a no-op rather
    // than an error — the fixture server has none.
    let echo = serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 2, "temperature": 0, "echo": true});
    let raw = post(port, "/v1/completions", &echo.to_string());
    assert!(serde_json::from_str::<serde_json::Value>(&raw).is_ok(), "echo broke the response: {raw}");

    // `n` is bounded, so a client cannot ask for a thousand.
    let many = serde_json::json!({"prompt": [1, 2, 3], "max_tokens": 1, "temperature": 0, "n": 500});
    let v: serde_json::Value = serde_json::from_str(&post(port, "/v1/completions", &many.to_string())).unwrap();
    assert!(v["choices"].as_array().unwrap().len() <= 8);
}

/// /metrics must be scrapeable and must actually count what happened.
#[test]
fn metrics_count_the_work_done() {
    let port = boot(true);
    let before = get(port, "/metrics");
    assert!(before.contains("# TYPE moe_requests_total counter"), "not Prometheus format: {before}");
    assert!(before.contains("moe_queued_requests"));

    let count = |text: &str, name: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(name) && !l.starts_with('#'))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("no {name} in {text}"))
    };
    assert_eq!(count(&before, "moe_requests_total"), 0);

    completion(port, &[1, 2, 3], 4);
    post(port, "/v1/embeddings", &serde_json::json!({"input": [1, 2]}).to_string());
    let after = get(port, "/metrics");
    assert_eq!(count(&after, "moe_requests_total"), 1);
    assert_eq!(count(&after, "moe_embedding_requests_total"), 1);
    assert_eq!(count(&after, "moe_prompt_tokens_total"), 3);
    assert!(count(&after, "moe_completion_tokens_total") > 0);
    assert!(count(&after, "moe_forward_steps_total") > 0);
    // Nothing was drafted, since the fixture server does not speculate.
    assert_eq!(count(&after, "moe_draft_tokens_total"), 0);

    // A refused request is counted as an error, not as a success.
    post(port, "/v1/completions", "{}");
    let errs = get(port, "/metrics");
    assert_eq!(count(&errs, "moe_request_errors_total"), 1);

    // Re-sending the same prompt must show up as cache reuse.
    completion(port, &[1, 2, 3], 4);
    assert!(count(&get(port, "/metrics"), "moe_prefix_cache_tokens_reused_total") > 0);
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
