use moe::{human, peak_rss, resolve, tokenizer::Stream, Dt, Model, Rng, Sampler, State, Store, Tokenizer};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
moe — CPU inference for sparse mixture-of-experts models

USAGE
  moe run   <model> [options]        generate text
  moe pull  <model>                  download a model into the cache
  moe pack  <model> -o <out.moe>     re-quantise a checkpoint for fast loading
  moe info  <model>                  show detected architecture and footprint
  moe bench <model> [options]        measure prefill and decode throughput
  moe tokenize <model> -p TEXT       show token ids (--decode 1,2,3 reverses it)
  moe serve <model> [options]        OpenAI-compatible HTTP server
  moe route <model|trace> [options]  analyse which experts the routing selected
  moe eval  <model> --text PATH      perplexity and bits/byte on held-out text

<model> is any of:
  ./model.moe                        a packed model file
  ~/models/mixtral                   a directory of safetensors
  mistralai/Mixtral-8x7B-v0.1        a Hugging Face repo, optionally @revision
  https://host/path/model.moe        a direct download

Remote models are cached; set MOE_CACHE to move the cache, HF_TOKEN to reach
gated repos, and --offline to refuse to download.

RUN OPTIONS
  -p, --prompt TEXT         prompt text
      --prompt-file PATH    read the prompt from a file
      --ids 1,2,3           feed token ids directly (no tokenizer needed)
  -n, --tokens N            tokens to generate            [128]
      --ctx N               context window                [min(model, 4096)]
      --temp F              sampling temperature, 0 = greedy   [0]
      --top-p F             nucleus cutoff                [0.95]
      --top-k N             candidate cutoff, 0 = off     [0]
      --repeat-penalty F    penalty on recent tokens      [1.0]
      --seed N              sampler seed                  [0]
      --tokenizer PATH      tokenizer.json or its directory
      --warm GB             fault in this many GB of weights before decoding
      --no-stream           print only the finished text
      --stats               per-run routing and throughput detail
      --trace PATH          write every routing decision as JSONL
      --draft N             speculate N tokens per step, 0 = off       [0]
      --draft-ngram N       longest suffix the drafter matches         [8]
      --json                emit only valid JSON, enforced while decoding
      --schema PATH         emit only JSON matching this JSON Schema

SERVE OPTIONS
      --port N              listen port                   [8080]
      --host ADDR           bind address                  [127.0.0.1]
      --ctx N               context window per session    [4096]
      --chat-format NAME    chatml | llama3 | mistral     [detected from vocab]
      --max-queue N         requests allowed to wait before 503     [32]
      --no-prefix-cache     re-prefill every request instead of reusing the cache
      --draft N             speculate N tokens per step, 0 = off       [0]
      --cors                allow browser origins

ROUTE OPTIONS
  Takes a model plus a prompt, or one or two *.jsonl traces from --trace.
  -p, --prompt TEXT         prompt to route (--ids 1,2,3 for raw ids)
      --vs TEXT             second prompt: report the difference
      --vs-ids 1,2,3        second prompt as raw ids
  -n, --tokens N            also route this many generated tokens    [0]
      --top N               busiest experts to list per layer        [5]
  -o, --out PATH            write the heatmap as SVG

EVAL OPTIONS
      --text PATH           text to score (- for stdin)
      --ids 1,2,3           score raw token ids instead
      --vs MODEL            score a second model on the same tokens and diff
      --ctx N               window size                   [min(model, 2048)]
      --stride N            window step; smaller is slower and more accurate
                                                          [--ctx / 2]
      --limit N             stop after N tokens           [all]

PACK OPTIONS
  -o, --out PATH            output file                   [./<name>.moe]
      --quant FMT           dense weights: q4 q8 f16 f32  [q8]
      --expert-quant FMT    routed experts                [--quant]

GLOBAL
      --threads N           worker threads                [all cores]
      --offline             use the cache only, never download
  -h, --help                this message
  -V, --version             version
";

struct Args {
    cmd: String,
    pos: Vec<String>,
    opt: HashMap<String, String>,
}

impl Args {
    fn parse() -> Args {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let (mut pos, mut opt) = (Vec::new(), HashMap::new());
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(name) = a.strip_prefix("--").or_else(|| a.strip_prefix('-').filter(|s| !s.is_empty())) {
                let (key, inline) = match name.split_once('=') {
                    Some((k, v)) => (k.to_string(), Some(v.to_string())),
                    None => (name.to_string(), None),
                };
                let key = match key.as_str() {
                    "p" => "prompt",
                    "n" => "tokens",
                    "o" => "out",
                    "h" => "help",
                    "V" => "version",
                    k => k,
                }
                .to_string();
                let flag = matches!(
                    key.as_str(),
                    "help" | "no-stream" | "stats" | "version" | "offline" | "cors" | "no-prefix-cache" | "json"
                );
                let val = match inline {
                    Some(v) => v,
                    None if flag => "1".into(),
                    None => {
                        i += 1;
                        raw.get(i).cloned().unwrap_or_default()
                    }
                };
                opt.insert(key, val);
            } else {
                pos.push(a.clone());
            }
            i += 1;
        }
        let cmd = if pos.is_empty() { String::new() } else { pos.remove(0) };
        Args { cmd, pos, opt }
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.opt.get(k).map(|s| s.as_str())
    }

    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        self.get(k).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    fn on(&self, k: &str) -> bool {
        self.opt.contains_key(k)
    }
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("moe: {msg}");
    std::process::exit(1)
}

fn main() {
    // Let `moe info model | head` end quietly instead of panicking on EPIPE.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let args = Args::parse();
    if args.on("version") {
        println!("moe {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.on("help") || args.cmd.is_empty() {
        print!("{USAGE}");
        return;
    }
    let threads: usize = args.num("threads", 0);
    if threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }
    let spec = args.pos.first().cloned().unwrap_or_else(|| fail("expected a model"));
    if args.cmd == "pull" {
        let path = fetched(&args, &spec);
        println!("{}", path.display());
        return;
    }
    // `route` also accepts trace files, which are not models to resolve.
    if args.cmd == "route" && spec.ends_with(".jsonl") {
        return route_traces(&args);
    }
    let path = fetched(&args, &spec);

    match args.cmd.as_str() {
        "run" => run(&args, &path),
        "bench" => bench(&args, &path),
        "info" => info(&path),
        "pack" => pack(&args, &path, &spec),
        "tokenize" => tokenize(&args, &path),
        "serve" => serve(&args, &path, &spec),
        "route" => route_model(&args, &path),
        "eval" => eval(&args, &path),
        other => fail(format!("unknown command '{other}' (try --help)")),
    }
}

/// Resolve a model spec to a local path, downloading it if necessary.
fn fetched(args: &Args, spec: &str) -> PathBuf {
    resolve(spec, args.on("offline")).unwrap_or_else(|e| fail(e))
}

/// Peak RSS as a display fragment, empty where the OS does not report it.
fn rss_note(sep: &str) -> String {
    match peak_rss() {
        0 => String::new(),
        b => format!("{sep}peak rss {}", human(b)),
    }
}

fn load(path: &Path) -> Model {
    let store = Store::open(path).unwrap_or_else(|e| fail(e));
    Model::load(store).unwrap_or_else(|e| fail(e))
}

fn info(path: &Path) {
    let m = load(path);
    let (dense, experts) = m.footprint();
    println!("{}", m.spec.summary());
    println!("\nweights    {} tensors, {}", m.store.len(), human(dense + experts));
    println!("  dense    {}", human(dense));
    println!("  experts  {} ({:.0}%)", human(experts), 100.0 * experts as f64 / (dense + experts).max(1) as f64);
    let dt: Vec<String> = m.dtypes().iter().map(|(d, b)| format!("{} {}", d.name(), human(*b))).collect();
    println!("  formats  {}", dt.join(", "));
    let (kd, vd) = m.kv_dims();
    println!("kv cache   {} per 1k tokens", human((m.spec.layers * (kd + vd) * 4 * 1024) as u64));
    println!("source     {} ({})", m.store.path.display(), if m.store.packed { "packed" } else { "safetensors" });
}

fn pack(args: &Args, path: &Path, spec: &str) {
    let dt = |k: &str, d: Dt| {
        args.get(k).map(|v| Dt::parse(v).unwrap_or_else(|| fail(format!("unknown format '{v}'")))).unwrap_or(d)
    };
    let weight = dt("quant", Dt::Q8);
    let expert = dt("expert-quant", weight);
    // Default to <name>.moe in the working directory, never inside the cache.
    let out = args.get("out").map(PathBuf::from).unwrap_or_else(|| {
        let name = spec.trim_end_matches(['/', '\\']).rsplit(['/', '\\']).next().unwrap_or("model");
        PathBuf::from(format!("{}.moe", name.trim_end_matches(".moe")))
    });
    let store = Store::open(path).unwrap_or_else(|e| fail(e));
    println!("packing {} -> {} (dense {}, experts {})", path.display(), out.display(), weight.name(), expert.name());
    let t0 = Instant::now();
    store.pack(&out, weight, expert, |line| println!("{line}")).unwrap_or_else(|e| fail(e));
    let after = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "done in {:.1}s: {} -> {} ({:.2}x)",
        t0.elapsed().as_secs_f32(),
        human(store.bytes()),
        human(after),
        store.bytes() as f64 / after.max(1) as f64
    );
}

/// Prefer an explicit `--tokenizer`, then one embedded in a packed model, then
/// `tokenizer.json` beside the weights.
fn tokenizer_for(args: &Args, path: &Path, store: Option<&Store>) -> Option<Tokenizer> {
    let given = args.get("tokenizer").map(PathBuf::from);
    if given.is_none() {
        if let Some(j) = store.and_then(|s| s.tokenizer.as_ref()) {
            match Tokenizer::from_json(j) {
                Ok(t) => return Some(t),
                Err(e) => eprintln!("moe: embedded tokenizer unusable ({e})"),
            }
        }
    }
    let near = |p: &Path| {
        let dir = if p.is_dir() { p.to_path_buf() } else { p.parent().unwrap_or(Path::new(".")).to_path_buf() };
        dir.join("tokenizer.json")
    };
    let cand: Vec<PathBuf> = match given {
        // An explicit path may be the file itself or the directory holding it.
        Some(p) if p.is_file() => vec![p],
        Some(p) => vec![near(&p), p],
        None if path.is_file() && path.extension().is_some_and(|e| e == "json") => vec![path.to_path_buf()],
        None => vec![near(path)],
    };
    cand.iter().find(|p| p.is_file()).and_then(|p| match Tokenizer::load(p) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("moe: tokenizer unavailable ({e})");
            None
        }
    })
}

fn prompt_ids(args: &Args, tok: Option<&Tokenizer>, bos: Option<u32>) -> Vec<u32> {
    if let Some(list) = args.get("ids") {
        return list.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    }
    let text = match (args.get("prompt"), args.get("prompt-file")) {
        (Some(t), _) => t.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).unwrap_or_else(|e| fail(format!("{f}: {e}"))),
        _ => fail("give one of --prompt, --prompt-file or --ids"),
    };
    match tok {
        Some(t) => t.encode(&text, bos),
        None => fail("no tokenizer found; pass --tokenizer or use --ids"),
    }
}

/// The shape `--json` or `--schema` asks the output to have.
fn shape_of(args: &Args) -> Option<moe::Grammar> {
    if let Some(path) = args.get("schema") {
        let raw = std::fs::read(path).unwrap_or_else(|e| fail(format!("{path}: {e}")));
        let v = serde_json::from_slice(&raw).unwrap_or_else(|e| fail(format!("{path}: {e}")));
        return Some(moe::Grammar::from_schema(&v).unwrap_or_else(|e| fail(format!("{path}: {e}"))));
    }
    args.on("json").then(moe::Grammar::json)
}

/// Prefill in blocks so long prompts stay batched without a huge scratch.
fn prefill(m: &Model, ids: &[u32], st: &mut State) -> Vec<f32> {
    let mut logits = Vec::new();
    for chunk in ids.chunks(64) {
        logits = m.forward(chunk, st);
    }
    logits
}

fn run(args: &Args, path: &Path) {
    let t_load = Instant::now();
    let m = load(path);
    let tok = tokenizer_for(args, path, Some(&m.store));
    let ids = prompt_ids(args, tok.as_ref(), m.spec.bos);
    if ids.is_empty() {
        fail("empty prompt");
    }
    let n_gen: usize = args.num("tokens", 128);
    let ctx =
        args.get("ctx").and_then(|v| v.parse().ok()).unwrap_or_else(|| m.spec.max_ctx.min(4096)).max(ids.len() + n_gen);
    let mut st = State::new(&m, ctx);
    if args.get("trace").is_some() {
        st.trace();
    }
    let load_s = t_load.elapsed().as_secs_f32();

    let warm: f64 = args.num("warm", 0.0);
    if warm > 0.0 {
        let t = Instant::now();
        let done = m.store.warm((warm * 1e9) as u64);
        eprintln!("warmed {} in {:.1}s", human(done), t.elapsed().as_secs_f32());
    }

    let sampler = Sampler {
        temp: args.num("temp", 0.0),
        top_p: args.num("top-p", 0.95),
        top_k: args.num("top-k", 0),
        repeat_penalty: args.num("repeat-penalty", 1.0),
        ..Sampler::default()
    };
    let plan = moe::Plan {
        max_tokens: n_gen,
        sampler,
        seed: args.num("seed", 0u64),
        lookahead: args.num("draft", 0usize),
        lookup: moe::Lookup { max_ngram: args.num("draft-ngram", 8usize).max(1), min_ngram: 2 },
    };
    let stream = !args.on("no-stream");
    let mut guide = shape_of(args).map(|g| {
        let t = tok.as_ref().unwrap_or_else(|| fail("constrained decoding needs a tokenizer to mask against"));
        moe::Guide::new(g, t)
    });

    if stream {
        if let (Some(t), true) = (tok.as_ref(), args.get("ids").is_none()) {
            let echo: Vec<u32> = ids.iter().copied().filter(|i| !t.is_special(*i)).collect();
            print!("{}", t.decode(&echo));
            let _ = std::io::stdout().flush();
        }
    }

    let t0 = Instant::now();
    let logits = prefill(&m, &ids, &mut st);
    let prefill_s = t0.elapsed().as_secs_f32();

    let mut history = ids.clone();
    let mut out = Vec::new();
    let mut text = Stream::default();
    let t1 = Instant::now();
    let gen = moe::generate::generate(&m, &mut st, &mut history, logits, &plan, guide.as_mut(), |next| {
        out.push(next);
        if stream {
            match tok.as_ref() {
                Some(t) if !t.is_special(next) => print!("{}", text.push(t, next)),
                Some(_) => {}
                None => print!("{next} "),
            }
            let _ = std::io::stdout().flush();
        }
        true
    });
    let decode_s = t1.elapsed().as_secs_f32();

    if !stream {
        match tok.as_ref() {
            Some(t) => println!("{}", t.decode(&out)),
            None => println!("{out:?}"),
        }
    }
    println!();
    eprintln!(
        "\nload {load_s:.2}s | prefill {} tok in {prefill_s:.2}s ({:.1} tok/s) | decode {} tok in {decode_s:.2}s ({:.2} tok/s){}",
        ids.len(),
        ids.len() as f32 / prefill_s.max(1e-6),
        out.len(),
        out.len() as f32 / decode_s.max(1e-6),
        rss_note(" | "),
    );
    // Speculation is only worth reporting when it was asked for; the ratio of
    // tokens to forward steps is the part that turns into wall clock.
    if plan.lookahead > 0 {
        eprintln!(
            "draft {} of {} accepted ({:.0}%) | {:.2} tokens per forward step",
            gen.accepted,
            gen.drafted,
            100.0 * gen.acceptance(),
            gen.tokens_per_step(),
        );
    }
    if let Some(path) = args.get("trace") {
        match write_trace(path, &m, &st, tok.as_ref()) {
            Ok(n) => eprintln!("wrote {n} routing records to {path}"),
            Err(e) => eprintln!("moe: {path}: {e}"),
        }
    }
    if args.on("stats") {
        eprintln!(
            "experts activated {} | repeated previous token's choice {:.0}% | expert bytes touched {} | kv cache {}",
            st.stats.routed.load(std::sync::atomic::Ordering::Relaxed),
            100.0 * st.stats.reuse_rate(),
            human(st.stats.expert_bytes.load(std::sync::atomic::Ordering::Relaxed)),
            human(st.kv_bytes()),
        );
    }
}

fn serve(args: &Args, path: &Path, spec: &str) {
    let m = load(path);
    let tok = tokenizer_for(args, path, Some(&m.store));
    let chat = match args.get("chat-format") {
        Some(name) => Some(
            moe::ChatFormat::by_name(name)
                .unwrap_or_else(|| fail(format!("unknown chat format '{name}' (chatml, llama3, mistral)"))),
        ),
        None => tok.as_ref().and_then(moe::ChatFormat::detect),
    };
    let ctx = args.num("ctx", 4096usize).min(m.spec.max_ctx);
    let name = spec.trim_end_matches(['/', '\\']).rsplit(['/', '\\']).next().unwrap_or("model").to_string();
    let mut server = moe::Server::new(m, tok, chat, name, ctx, !args.on("no-prefix-cache"));
    server.cors = args.on("cors");
    server.max_queue = args.num("max-queue", 32usize).max(1);
    server.lookahead = args.num("draft", 0usize);
    let host = args.get("host").unwrap_or("127.0.0.1").to_string();
    let port = args.num("port", 8080u16);
    moe::serve::run(server, &host, port).unwrap_or_else(|e| fail(format!("{host}:{port}: {e}")));
}

fn tokenize(args: &Args, path: &Path) {
    // A packed model carries its own tokenizer; a bare tokenizer.json works too.
    let store =
        if path.is_file() && path.extension().is_some_and(|e| e == "moe") { Store::open(path).ok() } else { None };
    let tok = tokenizer_for(args, path, store.as_ref()).unwrap_or_else(|| fail("no tokenizer found"));
    if let Some(list) = args.get("decode") {
        let ids: Vec<u32> = list.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        print!("{}", tok.decode(&ids));
        return;
    }
    let text = match (args.get("prompt"), args.get("prompt-file")) {
        (Some(t), _) => t.to_string(),
        (None, Some(f)) => std::fs::read_to_string(f).unwrap_or_else(|e| fail(format!("{f}: {e}"))),
        _ => fail("give --prompt or --prompt-file"),
    };
    let ids = tok.encode(&text, None);
    println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
    if args.on("stats") {
        for id in &ids {
            eprintln!("{id:>7}  {:?}", tok.decode_one(*id));
        }
    }
}

/// One JSON object per (token, routed layer): which experts it chose and with
/// what weight. Flat rather than nested so it streams and greps.
fn write_trace(path: &str, m: &Model, st: &State, tok: Option<&Tokenizer>) -> std::io::Result<usize> {
    let Some(tr) = st.trace.as_ref() else { return Ok(0) };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    // A header line first, so the file explains itself without the model.
    writeln!(
        f,
        "{}",
        serde_json::json!({
            "model": m.spec.arch,
            "layers": m.spec.layers,
            "experts": m.spec.experts,
            "top_k": m.spec.top_k,
        })
    )?;
    for r in &tr.routes {
        let id = tr.tokens.get(r.pos as usize).copied();
        let line = serde_json::json!({
            "pos": r.pos,
            "layer": r.layer,
            "token": id,
            "text": id.and_then(|i| tok.map(|k| k.decode_one(i))),
            "experts": r.experts.iter().map(|(e, w)| serde_json::json!([e, w])).collect::<Vec<_>>(),
        });
        writeln!(f, "{line}")?;
    }
    Ok(tr.routes.len())
}

/// `moe eval <model> --text PATH` — how surprised the model is by text it has
/// not been given the answer to. With `--vs`, the same tokens through a second
/// model, which is how a packed file's quality cost gets a number rather than a
/// compression ratio.
fn eval(args: &Args, path: &Path) {
    let m = load(path);
    let tok = tokenizer_for(args, path, Some(&m.store));

    let (mut ids, bytes) = match (args.get("text"), args.get("ids")) {
        (Some(src), _) => {
            let text = if src == "-" {
                let mut s = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)
                    .unwrap_or_else(|e| fail(format!("stdin: {e}")));
                s
            } else {
                std::fs::read_to_string(src).unwrap_or_else(|e| fail(format!("{src}: {e}")))
            };
            let t = tok.as_ref().unwrap_or_else(|| fail("no tokenizer found; pass --tokenizer, or give --ids"));
            (t.encode(&text, m.spec.bos), text.len())
        }
        (None, Some(list)) => (list.split(',').filter_map(|s| s.trim().parse().ok()).collect::<Vec<u32>>(), 0),
        _ => fail("give --text PATH (or - for stdin), or --ids"),
    };
    if let Some(limit) = args.get("limit").and_then(|v| v.parse::<usize>().ok()) {
        ids.truncate(limit);
    }
    if ids.len() < 2 {
        fail("need at least two tokens to score one prediction");
    }

    let ctx = args.get("ctx").and_then(|v| v.parse().ok()).unwrap_or_else(|| m.spec.max_ctx.min(2048)).max(2);
    let stride: usize = args.num("stride", (ctx / 2).max(1));

    let report = |label: &str, m: &Model, s: moe::Score, secs: f32| {
        println!(
            "{label:<28} ppl {:>8.3}   nll {:.4}   bits/token {:.3}{}\n{:<28} {} tokens in {secs:.1}s ({:.1} tok/s)",
            s.perplexity(),
            s.mean_nll(),
            s.bits_per_token(),
            if s.bytes > 0 { format!("   bits/byte {:.3}", s.bits_per_byte()) } else { String::new() },
            "",
            s.tokens,
            s.tokens as f32 / secs.max(1e-6),
        );
        let _ = m;
    };

    let run = |m: &Model, label: &str| -> moe::Score {
        let t0 = Instant::now();
        let s = moe::eval::score(m, &ids, ctx, stride, bytes, |i, n| {
            eprint!("\r{label}: window {i}/{n}");
            let _ = std::io::stderr().flush();
        });
        eprint!("\r\x1b[K");
        report(label, m, s, t0.elapsed().as_secs_f32());
        s
    };

    println!("scoring {} tokens, window {ctx}, stride {stride}\n", ids.len());
    let a = run(&m, &name_of(path));
    if let Some(other) = args.get("vs") {
        let p2 = fetched(args, other);
        let m2 = load(&p2);
        if m2.spec.vocab != m.spec.vocab {
            eprintln!(
                "moe: vocabularies differ ({} vs {}); compare bits/byte, not perplexity",
                m.spec.vocab, m2.spec.vocab
            );
        }
        let b = run(&m2, &name_of(&p2));
        // The delta is the whole point of --vs, so state it rather than leaving
        // it to be read off two lines.
        println!(
            "\ndelta                        ppl {:+.3} ({:+.2}%)   nll {:+.4}{}",
            b.perplexity() - a.perplexity(),
            100.0 * (b.perplexity() / a.perplexity() - 1.0),
            b.mean_nll() - a.mean_nll(),
            if a.bytes > 0 {
                format!("   bits/byte {:+.4}", b.bits_per_byte() - a.bits_per_byte())
            } else {
                String::new()
            },
        );
    }
}

/// A short display name for a resolved model path.
fn name_of(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| p.display().to_string())
}

/// The basename of a path, for labelling a diff without its directory.
fn base(p: &str) -> &str {
    p.rsplit(['/', '\\']).next().unwrap_or(p)
}

fn write_svg(args: &Args, doc: &str) {
    if let Some(out) = args.get("out") {
        match std::fs::write(out, doc) {
            Ok(()) => eprintln!("-> {out}"),
            Err(e) => fail(format!("{out}: {e}")),
        }
    }
}

/// `moe route a.jsonl [b.jsonl]` — analyse traces already on disk.
fn route_traces(args: &Args) {
    let top: usize = args.num("top", 5);
    let read = |p: &str| {
        let c = moe::Counts::read(Path::new(p)).unwrap_or_else(|e| fail(format!("{p}: {e}")));
        if c.is_empty() {
            fail(format!("{p}: no routing records (a dense model traces nothing)"));
        }
        c
    };
    let a = read(&args.pos[0]);
    match args.pos.get(1) {
        Some(second) => {
            let d = moe::route::Diff { a, b: read(second) };
            print!("{}", d.report(top));
            write_svg(args, &moe::route::diffmap(&d, (base(&args.pos[0]), base(second)), args.get("title")));
        }
        None => {
            print!("{}", a.report(top));
            write_svg(args, &moe::route::heatmap(&a, args.get("title")));
        }
    }
}

/// One side of a routing comparison: prompt text, or raw ids for a checkpoint
/// with no tokenizer beside it.
enum Side<'a> {
    Text(&'a str),
    Ids(&'a str),
}

/// `moe route <model> -p TEXT [--vs TEXT]` — route a prompt and analyse it in
/// one step, so seeing where two prompts diverge needs no intermediate files.
fn route_model(args: &Args, path: &Path) {
    let m = load(path);
    if m.spec.experts == 0 {
        fail("this checkpoint is dense: it has no routing to analyse");
    }
    let tok = tokenizer_for(args, path, Some(&m.store));
    let n_gen: usize = args.num("tokens", 0);
    let top: usize = args.num("top", 5);

    // Each prompt gets its own state, so the two runs cannot see each other.
    let count = |side: Side| -> moe::Counts {
        let ids = match side {
            Side::Ids(list) => list.split(',').filter_map(|s| s.trim().parse().ok()).collect(),
            Side::Text(text) => match tok.as_ref() {
                Some(t) => t.encode(text, m.spec.bos),
                None => fail("no tokenizer found; pass --tokenizer, or give --ids"),
            },
        };
        if ids.is_empty() {
            fail("empty prompt");
        }
        let ctx = (ids.len() + n_gen + 1).max(8);
        let mut st = State::new(&m, ctx);
        st.trace();
        let mut logits = prefill(&m, &ids, &mut st);
        let sampler = Sampler::default();
        let mut rng = Rng::new(0);
        let mut history = ids.clone();
        for _ in 0..n_gen {
            let next = sampler.pick(&mut logits, &history, &mut rng);
            if m.spec.eos.contains(&next) || st.pos + 1 >= ctx {
                break;
            }
            history.push(next);
            logits = m.forward(&[next], &mut st);
        }
        moe::Counts::from_trace(st.trace.as_ref().unwrap(), &m.spec.arch, m.spec.experts, m.spec.top_k)
    };

    let first = match (args.get("prompt"), args.get("ids")) {
        (Some(t), _) => Side::Text(t),
        (None, Some(l)) => Side::Ids(l),
        _ => fail("give --prompt or --ids (or pass a *.jsonl trace)"),
    };
    let a = count(first);
    let second = args.get("vs").map(Side::Text).or_else(|| args.get("vs-ids").map(Side::Ids));
    match second {
        Some(other) => {
            let d = moe::route::Diff { a, b: count(other) };
            print!("{}", d.report(top));
            write_svg(args, &moe::route::diffmap(&d, ("first", "second"), args.get("title")));
        }
        None => {
            print!("{}", a.report(top));
            write_svg(args, &moe::route::heatmap(&a, args.get("title")));
        }
    }
}

fn bench(args: &Args, path: &Path) {
    let m = load(path);
    let plen: usize = args.num("prompt-len", 32);
    let n: usize = args.num("tokens", 16);
    let ids: Vec<u32> = (0..plen).map(|i| ((i * 7 + 13) % m.spec.vocab) as u32).collect();
    let mut st = State::new(&m, plen + n + 8);

    let t0 = Instant::now();
    let mut logits = prefill(&m, &ids, &mut st);
    let pf = t0.elapsed().as_secs_f32();

    let t1 = Instant::now();
    for i in 0..n {
        let next = (i % m.spec.vocab) as u32;
        std::hint::black_box(&logits);
        logits = m.forward(&[next], &mut st);
    }
    let dc = t1.elapsed().as_secs_f32();

    println!("{}", m.spec.summary());
    println!(
        "\nprefill  {plen} tok in {pf:.2}s   {:.1} tok/s\ndecode   {n} tok in {dc:.2}s   {:.2} tok/s   {:.0} ms/tok",
        plen as f32 / pf.max(1e-6),
        n as f32 / dc.max(1e-6),
        1000.0 * dc / n as f32
    );
    println!(
        "threads  {}   expert reuse {:.0}%{}",
        rayon::current_num_threads(),
        100.0 * st.stats.reuse_rate(),
        rss_note("   ")
    );
}
