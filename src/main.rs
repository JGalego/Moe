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

SERVE OPTIONS
      --port N              listen port                   [8080]
      --host ADDR           bind address                  [127.0.0.1]
      --ctx N               context window per session    [4096]
      --chat-format NAME    chatml | llama3 | mistral     [detected from vocab]
      --max-queue N         requests allowed to wait before 503     [32]
      --no-prefix-cache     re-prefill every request instead of reusing the cache
      --cors                allow browser origins

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
                    "help" | "no-stream" | "stats" | "version" | "offline" | "cors" | "no-prefix-cache"
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
    let path = fetched(&args, &spec);

    match args.cmd.as_str() {
        "run" => run(&args, &path),
        "bench" => bench(&args, &path),
        "info" => info(&path),
        "pack" => pack(&args, &path, &spec),
        "tokenize" => tokenize(&args, &path),
        "serve" => serve(&args, &path, &spec),
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
    let mut rng = Rng::new(args.num("seed", 0u64));
    let stream = !args.on("no-stream");

    if stream {
        if let (Some(t), true) = (tok.as_ref(), args.get("ids").is_none()) {
            let echo: Vec<u32> = ids.iter().copied().filter(|i| !t.is_special(*i)).collect();
            print!("{}", t.decode(&echo));
            let _ = std::io::stdout().flush();
        }
    }

    let t0 = Instant::now();
    let mut logits = prefill(&m, &ids, &mut st);
    let prefill_s = t0.elapsed().as_secs_f32();

    let mut history = ids.clone();
    let mut out = Vec::new();
    let mut text = Stream::default();
    let t1 = Instant::now();
    for _ in 0..n_gen {
        let next = sampler.pick(&mut logits, &history, &mut rng);
        if m.spec.eos.contains(&next) || st.pos >= ctx {
            break;
        }
        history.push(next);
        out.push(next);
        if stream {
            if let Some(t) = tok.as_ref() {
                if !t.is_special(next) {
                    print!("{}", text.push(t, next));
                    let _ = std::io::stdout().flush();
                }
            } else {
                print!("{next} ");
                let _ = std::io::stdout().flush();
            }
        }
        logits = m.forward(&[next], &mut st);
    }
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
