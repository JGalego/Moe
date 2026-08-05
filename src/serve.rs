//! An OpenAI-compatible HTTP front end for one model.
//!
//! Deliberately serial: at batch size one the engine is memory-bandwidth-bound
//! and already uses every core, so two concurrent generations would halve each
//! other's throughput rather than add any. Requests queue on a single session
//! instead, which also lets that session keep its KV cache warm between them —
//! the reason a second chat turn only prefills the message that was added.

use crate::chat::Template;
use crate::draft::Lookup;
use crate::generate::{generate, Plan};
use crate::grammar::{Grammar, Guide};
use crate::http::{self, Conn, Request};
use crate::model::{Model, Pool, State};
use crate::sample::Sampler;
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

/// How a chat turn becomes a prompt: the checkpoint's own template if it ships
/// one the engine can render, otherwise a format inferred from the vocabulary.
pub enum Prompting {
    /// The template `tokenizer_config.json` declared.
    Template(Template),
    /// One of the three built-in shapes, matched on a control token.
    Detected(ChatFormat),
}

impl Prompting {
    /// Prefer the checkpoint's declaration; fall back to detection when it is
    /// absent or uses Jinja beyond what this engine renders. Falling back is
    /// better than failing, and both are better than rendering it wrongly.
    pub fn resolve(template: Option<&str>, tok: Option<&Tokenizer>) -> Option<Prompting> {
        if let Some(src) = template {
            match Template::parse(src) {
                Ok(t) => return Some(Prompting::Template(t)),
                Err(e) => {
                    eprintln!("moe: the checkpoint's chat template is beyond this engine ({e}); detecting instead")
                }
            }
        }
        tok.and_then(ChatFormat::detect).map(Prompting::Detected)
    }

    pub fn name(&self) -> &str {
        match self {
            Prompting::Template(_) => "the checkpoint's own template",
            Prompting::Detected(f) => f.name,
        }
    }

    /// The marker that closes an assistant turn, which is also where generation
    /// should stop whether or not the checkpoint made it an EOS token.
    fn turn_end(&self) -> Option<&str> {
        match self {
            Prompting::Detected(f) => Some(f.assistant.1).filter(|e| !e.is_empty()),
            // A template's stop marker is the checkpoint's EOS, which the decode
            // loop already honours.
            Prompting::Template(_) => None,
        }
    }

    fn render(&self, messages: &[Value], bos: &str, eos: &str) -> Result<String, String> {
        match self {
            Prompting::Template(t) => t.render(messages, true, bos, eos),
            Prompting::Detected(f) => f.render(messages),
        }
    }
}

pub struct Server {
    model: Model,
    tok: Option<Tokenizer>,
    chat: Option<Prompting>,
    name: String,
    session: Mutex<Session>,
    queued: AtomicUsize,
    pub max_queue: usize,
    prefix_cache: bool,
    pub cors: bool,
    /// Tokens to draft per step; 0 disables speculation.
    pub lookahead: usize,
    /// How /v1/embeddings pools hidden states.
    pub pool: Pool,
}

struct Params {
    plan: Plan,
    stop: Vec<String>,
    /// The shape the answer must have, from `response_format`.
    shape: Option<Grammar>,
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
        chat: Option<Prompting>,
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
            lookahead: 0,
            pool: Pool::Mean,
        }
    }

    pub fn chat_format(&self) -> Option<&Prompting> {
        self.chat.as_ref()
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
            plan: Plan {
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
                lookahead: self.lookahead,
                lookup: Lookup::default(),
            },
            stop,
            shape: Self::shape_of(&body["response_format"]),
        }
    }

    /// `response_format` in the shape OpenAI clients send it: `json_object` for
    /// any object, `json_schema` for one with a declared shape. An unparseable
    /// schema leaves the request unconstrained rather than failing it, since
    /// refusing would be worse than answering in free text.
    fn shape_of(rf: &Value) -> Option<Grammar> {
        match rf["type"].as_str()? {
            "json_object" => Some(Grammar::json()),
            "json_schema" => {
                let s = rf["json_schema"].get("schema").unwrap_or(&rf["json_schema"]);
                match Grammar::from_schema(s) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        eprintln!("moe: response_format schema unusable ({e}); answering unconstrained");
                        None
                    }
                }
            }
            _ => None,
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
        if prompt.len() + p.plan.max_tokens > sess.state.ctx {
            return Err(format!(
                "prompt of {} tokens plus max_tokens {} exceeds the context window of {}",
                prompt.len(),
                p.plan.max_tokens,
                sess.state.ctx
            ));
        }

        let keep = self.rewind(&mut sess, prompt);
        let mut logits = Vec::new();
        for chunk in prompt[keep..].chunks(64) {
            logits = self.model.forward(chunk, &mut sess.state);
            sess.history.extend_from_slice(chunk);
        }

        let mut text = String::new();
        let mut stream = Stream::default();
        let mut emitted = 0usize;
        let hold = p.stop.iter().map(|s| s.len()).max().unwrap_or(1).saturating_sub(1);
        let mut hit_stop = false;
        // A shape needs the vocabulary to mask against, so a request asking for
        // JSON from a checkpoint with no tokenizer beside it cannot be honoured.
        let mut guide = match (&p.shape, self.tok.as_ref()) {
            (Some(g), Some(t)) => Some(Guide::new(g.clone(), t)),
            (Some(_), None) => return Err("response_format needs a tokenizer to constrain against".into()),
            (None, _) => None,
        };
        let Session { state, history } = &mut *sess;

        // Stop sequences live out here rather than in the decode loop: they are
        // a property of the decoded text, not of the tokens, and returning false
        // is how the loop is told to end.
        let outcome = generate(&self.model, state, history, logits, &p.plan, guide.as_mut(), |next| {
            match self.tok.as_ref() {
                Some(t) => text.push_str(&stream.push(t, next)),
                None => text.push_str(&format!("{next} ")),
            }
            if let Some(at) = p.stop.iter().filter_map(|s| text.find(s.as_str())).min() {
                text.truncate(at);
                hit_stop = true;
                return false;
            }
            // Hold back enough tail that a stop sequence can never be emitted
            // before we have seen all of it.
            let safe = floor_boundary(&text, text.len().saturating_sub(hold));
            if safe > emitted {
                on_delta(&text[emitted..safe]);
                emitted = safe;
            }
            true
        });
        if emitted < text.len() {
            on_delta(&text[emitted..]);
        }
        let finish = if hit_stop { "stop" } else { outcome.stop.reason() };
        Ok(Gen { text, tokens: outcome.tokens, finish })
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
            let fmt = self.chat.as_ref().ok_or(
                "this checkpoint declares no chat template, so its prompt format is unknown. \
                 Pass --chat-format (chatml, llama3, mistral) or use /v1/completions",
            )?;
            // A template emits its own BOS if it wants one, so never add a second.
            let bos = self.tok.as_ref().zip(self.model.spec.bos).map(|(t, b)| t.decode_one(b)).unwrap_or_default();
            let eos =
                self.tok.as_ref().zip(self.model.spec.eos.first()).map(|(t, e)| t.decode_one(*e)).unwrap_or_default();
            return self.encode(&fmt.render(messages, &bos, &eos)?, false);
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

    /// `POST /v1/embeddings`.
    ///
    /// Embedding is stateless — no cache to reuse, no tokens to sample — so a
    /// batch is just a loop, and it does not touch the generation session at all.
    fn embeddings(&self, req: &Request, conn: &mut Conn) {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return drop(conn.error(400, &format!("invalid JSON: {e}"))),
        };
        // An empty batch is a client mistake, not a request for zero vectors.
        if matches!(&body["input"], Value::Array(a) if a.is_empty()) {
            return drop(conn.error(400, "input is empty"));
        }
        // `input` is a string, a list of strings, token ids, or a list of those.
        let inputs: Vec<Result<Vec<u32>, String>> = match &body["input"] {
            Value::String(s) => vec![self.encode(s, true)],
            Value::Array(a) if a.iter().all(|v| v.is_u64()) && !a.is_empty() => {
                vec![Ok(a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect())]
            }
            Value::Array(a) => a
                .iter()
                .map(|v| match v {
                    Value::String(s) => self.encode(s, true),
                    Value::Array(ids) => Ok(ids.iter().map(|i| i.as_u64().unwrap_or(0) as u32).collect()),
                    _ => Err("each input must be a string or an array of token ids".into()),
                })
                .collect(),
            _ => return drop(conn.error(400, "input must be a string, an array of strings, or token ids")),
        };

        let ctx = match self.session.lock() {
            Ok(s) => s.state.ctx,
            Err(_) => return drop(conn.error(400, "session poisoned")),
        };
        let mut data = Vec::new();
        let mut total = 0usize;
        for (i, ids) in inputs.into_iter().enumerate() {
            let ids = match ids {
                Ok(v) if v.is_empty() => return drop(conn.error(400, "empty input")),
                Ok(v) if v.len() > ctx => {
                    return drop(conn.error(400, &format!("input {i} is {} tokens, over the {ctx} window", v.len())))
                }
                Ok(v) => v,
                Err(e) => return drop(conn.error(400, &e)),
            };
            total += ids.len();
            let v = self.model.embed(&ids, ctx, self.pool, true);
            data.push(json!({"object": "embedding", "index": i, "embedding": v}));
        }
        let out = json!({
            "object": "list",
            "data": data,
            "model": self.name,
            "usage": {"prompt_tokens": total, "total_tokens": total},
        });
        drop(conn.json(200, &out.to_string()))
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
            if let Some(end) = self.chat.as_ref().and_then(|c| c.turn_end()) {
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
            "/v1/embeddings" => self.embeddings(req, conn),
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
        Some(f) => eprintln!("  chat format: {}", f.name()),
        None => eprintln!("  chat format: none detected — /v1/chat/completions will refuse; use /v1/completions"),
    }
    let server = std::sync::Arc::new(server);
    http::run(listener, move |req, conn| server.handle(req, conn));
    Ok(())
}
