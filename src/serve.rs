//! An OpenAI-compatible HTTP front end for one model.
//!
//! Deliberately serial: at batch size one the engine is memory-bandwidth-bound
//! and already uses every core, so two concurrent generations would halve each
//! other's throughput rather than add any. Requests queue on a single session
//! instead, which also lets that session keep its KV cache warm between them —
//! the reason a second chat turn only prefills the message that was added.

use crate::http::{self, Conn, Request};
use crate::model::{Model, State};
use crate::sample::{Rng, Sampler};
use crate::tokenizer::{Stream, Tokenizer};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// A chat template reduced to what these models actually need: a wrapper per
/// role, and the opening of the assistant's turn.
#[derive(Clone, Copy)]
pub struct ChatFormat {
    pub name: &'static str,
    /// Emitted once, before any turn.
    prefix: &'static str,
    system: (&'static str, &'static str),
    user: (&'static str, &'static str),
    assistant: (&'static str, &'static str),
    generation: &'static str,
    /// Mistral has no system role; its system text joins the first user turn.
    fold_system: bool,
}

const CHATML: ChatFormat = ChatFormat {
    name: "chatml",
    prefix: "",
    system: ("<|im_start|>system\n", "<|im_end|>\n"),
    user: ("<|im_start|>user\n", "<|im_end|>\n"),
    assistant: ("<|im_start|>assistant\n", "<|im_end|>\n"),
    generation: "<|im_start|>assistant\n",
    fold_system: false,
};

const LLAMA3: ChatFormat = ChatFormat {
    name: "llama3",
    prefix: "<|begin_of_text|>",
    system: ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
    user: ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"),
    assistant: ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"),
    generation: "<|start_header_id|>assistant<|end_header_id|>\n\n",
    fold_system: false,
};

const MISTRAL: ChatFormat = ChatFormat {
    name: "mistral",
    prefix: "<s>",
    system: ("", ""),
    user: ("[INST] ", " [/INST]"),
    assistant: ("", "</s> "),
    generation: "",
    fold_system: true,
};

const FORMATS: [ChatFormat; 3] = [CHATML, LLAMA3, MISTRAL];

impl ChatFormat {
    pub fn by_name(name: &str) -> Option<ChatFormat> {
        FORMATS.iter().find(|f| f.name == name).copied()
    }

    /// Pick a format from the control tokens the vocabulary carries — the same
    /// introspection the rest of the engine uses. A checkpoint with none of
    /// them is a base model, and guessing would silently mis-prompt it.
    pub fn detect(tok: &Tokenizer) -> Option<ChatFormat> {
        for (marker, fmt) in [("<|im_start|>", CHATML), ("<|start_header_id|>", LLAMA3), ("[/INST]", MISTRAL)] {
            if tok.has_token(marker) {
                return Some(fmt);
            }
        }
        None
    }

    /// Render OpenAI `messages` into a prompt.
    fn render(&self, messages: &[Value]) -> Result<String, String> {
        let mut out = self.prefix.to_string();
        let mut pending_system = String::new();
        for m in messages {
            let role = m["role"].as_str().unwrap_or("user");
            let content = m["content"].as_str().unwrap_or("");
            let wrap = match role {
                "system" => {
                    if self.fold_system {
                        pending_system = format!("{content}\n\n");
                        continue;
                    }
                    self.system
                }
                "assistant" => self.assistant,
                "user" | "tool" | "function" => self.user,
                other => return Err(format!("unsupported role: {other}")),
            };
            let body = if role == "user" && !pending_system.is_empty() {
                format!("{}{content}", std::mem::take(&mut pending_system))
            } else {
                content.to_string()
            };
            out.push_str(wrap.0);
            out.push_str(&body);
            out.push_str(wrap.1);
        }
        out.push_str(self.generation);
        Ok(out)
    }
}

struct Session {
    state: State,
    /// Token ids that produced the cache as it stands.
    history: Vec<u32>,
}

pub struct Server {
    model: Model,
    tok: Option<Tokenizer>,
    chat: Option<ChatFormat>,
    name: String,
    session: Mutex<Session>,
    queued: AtomicUsize,
    pub max_queue: usize,
    prefix_cache: bool,
    pub cors: bool,
}

struct Params {
    max_tokens: usize,
    sampler: Sampler,
    seed: u64,
    stop: Vec<String>,
}

struct Gen {
    text: String,
    tokens: usize,
    finish: &'static str,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Largest index <= `i` that is a char boundary.
fn floor_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

impl Server {
    pub fn new(
        model: Model,
        tok: Option<Tokenizer>,
        chat: Option<ChatFormat>,
        name: String,
        ctx: usize,
        prefix_cache: bool,
    ) -> Server {
        let state = State::new(&model, ctx);
        Server {
            model,
            tok,
            chat,
            name,
            session: Mutex::new(Session { state, history: Vec::new() }),
            queued: AtomicUsize::new(0),
            max_queue: 32,
            prefix_cache,
            cors: false,
        }
    }

    pub fn chat_format(&self) -> Option<ChatFormat> {
        self.chat
    }

    fn encode(&self, text: &str, bos: bool) -> Result<Vec<u32>, String> {
        let tok = self.tok.as_ref().ok_or("no tokenizer: send prompt as an array of token ids")?;
        Ok(tok.encode(text, if bos { self.model.spec.bos } else { None }))
    }

    fn params(&self, body: &Value) -> Params {
        let f = |k: &str, d: f32| body[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let stop = match &body["stop"] {
            Value::String(s) => vec![s.clone()],
            Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            _ => Vec::new(),
        };
        Params {
            max_tokens: body["max_tokens"].as_u64().unwrap_or(256).min(8192) as usize,
            sampler: Sampler {
                temp: f("temperature", 0.0),
                top_p: f("top_p", 1.0),
                top_k: body["top_k"].as_u64().unwrap_or(0) as usize,
                // OpenAI's frequency_penalty is additive; ours is a divisor, so
                // this is a rough mapping rather than a faithful one.
                repeat_penalty: 1.0 + f("frequency_penalty", 0.0).max(0.0),
                ..Sampler::default()
            },
            seed: body["seed"].as_u64().unwrap_or_else(|| now().wrapping_mul(2654435761)),
            stop,
        }
    }

    /// Reuse whatever of `want` the cache already holds, and return how much.
    fn rewind(&self, sess: &mut Session, want: &[u32]) -> usize {
        if !self.prefix_cache {
            sess.state.reset();
            sess.history.clear();
            return 0;
        }
        let common = want.iter().zip(&sess.history).take_while(|(a, b)| a == b).count();
        // Always leave at least one token to forward: logits come from running
        // a position, not from having run it earlier.
        let keep = common.min(want.len().saturating_sub(1));
        sess.state.truncate(keep);
        sess.history.truncate(keep);
        keep
    }

    fn generate(&self, prompt: &[u32], p: &Params, mut on_delta: impl FnMut(&str)) -> Result<Gen, String> {
        let mut sess = self.session.lock().map_err(|_| "session poisoned")?;
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() + p.max_tokens > sess.state.ctx {
            return Err(format!(
                "prompt of {} tokens plus max_tokens {} exceeds the context window of {}",
                prompt.len(),
                p.max_tokens,
                sess.state.ctx
            ));
        }

        let keep = self.rewind(&mut sess, prompt);
        let mut logits = Vec::new();
        for chunk in prompt[keep..].chunks(64) {
            logits = self.model.forward(chunk, &mut sess.state);
            sess.history.extend_from_slice(chunk);
        }

        let mut rng = Rng::new(p.seed);
        let mut text = String::new();
        let mut stream = Stream::default();
        let mut emitted = 0usize;
        let hold = p.stop.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
        let mut out = Gen { text: String::new(), tokens: 0, finish: "length" };

        for _ in 0..p.max_tokens {
            let next = p.sampler.pick(&mut logits, &sess.history, &mut rng);
            if self.model.spec.eos.contains(&next) {
                out.finish = "stop";
                break;
            }
            out.tokens += 1;
            match self.tok.as_ref() {
                Some(t) => text.push_str(&stream.push(t, next)),
                None => text.push_str(&format!("{next} ")),
            }
            if let Some(at) = p.stop.iter().filter_map(|s| text.find(s.as_str())).min() {
                text.truncate(at);
                out.finish = "stop";
                break;
            }
            // Hold back enough tail that a stop sequence can never be emitted
            // before we have seen all of it.
            let safe = floor_boundary(&text, text.len().saturating_sub(hold));
            if safe > emitted {
                on_delta(&text[emitted..safe]);
                emitted = safe;
            }
            sess.history.push(next);
            logits = self.model.forward(&[next], &mut sess.state);
        }
        if emitted < text.len() {
            on_delta(&text[emitted..]);
        }
        out.text = text;
        Ok(out)
    }

    fn completion_body(&self, chat: bool, g: &Gen, prompt_tokens: usize) -> String {
        let (object, choice) = if chat {
            (
                "chat.completion",
                json!({"index": 0, "message": {"role": "assistant", "content": g.text}, "finish_reason": g.finish}),
            )
        } else {
            ("text_completion", json!({"index": 0, "text": g.text, "finish_reason": g.finish}))
        };
        json!({
            "id": format!("cmpl-{}", now()),
            "object": object,
            "created": now(),
            "model": self.name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": g.tokens,
                "total_tokens": prompt_tokens + g.tokens,
            },
        })
        .to_string()
    }

    fn chunk_body(&self, chat: bool, delta: &str, first: bool, finish: Option<&str>) -> String {
        let choice = if chat {
            let mut d = json!({});
            if first {
                d["role"] = json!("assistant");
            }
            if !delta.is_empty() {
                d["content"] = json!(delta);
            }
            json!({"index": 0, "delta": d, "finish_reason": finish})
        } else {
            json!({"index": 0, "text": delta, "finish_reason": finish})
        };
        json!({
            "id": format!("cmpl-{}", now()),
            "object": if chat { "chat.completion.chunk" } else { "text_completion" },
            "created": now(),
            "model": self.name,
            "choices": [choice],
        })
        .to_string()
    }

    fn prompt_from(&self, body: &Value, chat: bool) -> Result<Vec<u32>, String> {
        if chat {
            let messages = body["messages"].as_array().ok_or("messages must be an array")?;
            let fmt = self.chat.ok_or(
                "this checkpoint declares no chat template, so its prompt format is unknown. \
                 Pass --chat-format (chatml, llama3, mistral) or use /v1/completions",
            )?;
            return self.encode(&fmt.render(messages)?, false);
        }
        match &body["prompt"] {
            Value::String(s) => self.encode(s, true),
            // Token ids let a client drive the model with no tokenizer at all.
            Value::Array(a) if !a.is_empty() && a.iter().all(|v| v.is_u64()) => {
                Ok(a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())
            }
            Value::Array(a) if a.len() == 1 && a[0].is_string() => self.encode(a[0].as_str().unwrap_or(""), true),
            _ => Err("prompt must be a string or an array of token ids".into()),
        }
    }

    fn completions(&self, req: &Request, conn: &mut Conn, chat: bool) {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return drop(conn.error(400, &format!("invalid JSON: {e}"))),
        };
        let prompt = match self.prompt_from(&body, chat) {
            Ok(p) => p,
            Err(e) => return drop(conn.error(400, &e)),
        };
        let mut p = self.params(&body);
        // A chat turn ends at the format's own closing marker, whether or not
        // the checkpoint made that marker its EOS token.
        if chat {
            if let Some(end) = self.chat.map(|f| f.assistant.1).filter(|e| !e.is_empty()) {
                p.stop.push(end.to_string());
            }
        }
        let streaming = body["stream"].as_bool().unwrap_or(false);

        if !streaming {
            match self.generate(&prompt, &p, |_| {}) {
                Ok(g) => drop(conn.json(200, &self.completion_body(chat, &g, prompt.len()))),
                Err(e) => drop(conn.error(400, &e)),
            }
            return;
        }

        if conn.sse_open().is_err() {
            return;
        }
        let mut first = true;
        let mut failed = None;
        let result = self.generate(&prompt, &p, |delta| {
            let body = self.chunk_body(chat, delta, first, None);
            first = false;
            if conn.sse(&body).is_err() {
                failed = Some(());
            }
        });
        match result {
            Ok(g) => {
                let _ = conn.sse(&self.chunk_body(chat, "", first, Some(g.finish)));
                let _ = conn.sse_close();
            }
            // The stream is already open, so an error can only be reported inside it.
            Err(e) => {
                let _ = conn.sse(&json!({"error": {"message": e}}).to_string());
                let _ = conn.sse_close();
            }
        }
    }

    pub fn handle(&self, req: &Request, conn: &mut Conn) {
        conn.cors = self.cors;
        if req.method == "OPTIONS" {
            return drop(conn.text(200, ""));
        }
        let route = req.route();
        if req.method == "GET" {
            return match route {
                "/health" => drop(conn.json(200, &json!({"status": "ok", "model": self.name}).to_string())),
                "/v1/models" => {
                    let body = json!({"object": "list", "data": [
                        {"id": self.name, "object": "model", "created": 0, "owned_by": "moe"}
                    ]});
                    drop(conn.json(200, &body.to_string()))
                }
                _ => drop(conn.error(404, "not found")),
            };
        }
        if req.method != "POST" {
            return drop(conn.error(405, "method not allowed"));
        }

        // Queue rather than run concurrently, but bound the queue so a burst
        // fails fast instead of piling up behind a slow generation.
        let waiting = self.queued.fetch_add(1, Ordering::SeqCst);
        if waiting >= self.max_queue {
            self.queued.fetch_sub(1, Ordering::SeqCst);
            return drop(conn.error(503, "server busy: too many queued requests"));
        }
        match route {
            "/v1/chat/completions" => self.completions(req, conn, true),
            "/v1/completions" => self.completions(req, conn, false),
            _ => drop(conn.error(404, "not found")),
        }
        self.queued.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Bind and serve until killed.
pub fn run(server: Server, host: &str, port: u16) -> std::io::Result<()> {
    let listener = http::bind(host, port)?;
    let addr = listener.local_addr()?;
    eprintln!("moe serve: http://{addr}  (model {})", server.name);
    match server.chat_format() {
        Some(f) => eprintln!("  chat format: {}", f.name),
        None => eprintln!("  chat format: none detected — /v1/chat/completions will refuse; use /v1/completions"),
    }
    let server = std::sync::Arc::new(server);
    http::run(listener, move |req, conn| server.handle(req, conn));
    Ok(())
}
