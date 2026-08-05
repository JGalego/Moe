//! Model resolution: a local path, a URL, or a Hugging Face repo id all become
//! a local path. Downloads land in a shared cache, so the second run is free.
//!
//! ```text
//! moe run ./model.moe                 a file
//! moe run ~/models/mixtral            a directory of safetensors
//! moe run mistralai/Mixtral-8x7B-v0.1 a Hub repo (optionally @revision)
//! moe run https://host/model.moe      a direct download
//! ```

use std::io;
use std::path::{Path, PathBuf};

/// Where a model spec points.
#[derive(Debug, PartialEq)]
pub enum Source {
    Local(PathBuf),
    Url(String),
    Hub { repo: String, rev: String },
}

/// Files worth downloading from a Hub repo, in the absence of a packed model.
///
/// `tokenizer_config.json` and `chat_template.jinja` are here because a
/// checkpoint's prompt format lives in one or the other, and a format that is
/// never downloaded is a format the engine silently falls back from. They are a
/// few kilobytes next to gigabytes of weights.
#[cfg(feature = "fetch")]
const WANTED: [&str; 4] = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"];

/// Classify a model spec. Anything that exists on disk wins; after that a bare
/// `owner/name` is a repo id, which is why `hf:` exists to force the issue.
pub fn parse(spec: &str) -> Source {
    let path = Path::new(spec);
    if path.exists() {
        return Source::Local(path.to_path_buf());
    }
    if let Some(rest) = spec.strip_prefix("hf:") {
        return hub(rest);
    }
    if spec.starts_with("https://") || spec.starts_with("http://") {
        // A Hub model page is a repo, not a file to download.
        if let Some(rest) = spec.split_once("huggingface.co/").map(|(_, r)| r) {
            let trimmed = rest.trim_end_matches('/');
            if trimmed.split('/').count() == 2 && !trimmed.contains('?') {
                return hub(trimmed);
            }
        }
        return Source::Url(spec.to_string());
    }
    if spec.split('/').count() == 2 && !spec.contains('\\') && !spec.starts_with('.') {
        return hub(spec);
    }
    Source::Local(path.to_path_buf())
}

fn hub(spec: &str) -> Source {
    let (repo, rev) = spec.split_once('@').unwrap_or((spec, "main"));
    Source::Hub { repo: repo.trim_end_matches('/').to_string(), rev: rev.to_string() }
}

/// Root of the download cache: `$MOE_CACHE`, else the platform cache directory.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MOE_CACHE") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local).join("moe");
        }
    } else if cfg!(target_os = "macos") {
        if let Some(h) = &home {
            return h.join("Library/Caches/moe");
        }
    } else if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("moe");
    }
    home.unwrap_or_else(|| PathBuf::from(".")).join(".cache/moe")
}

/// Turn a spec into a local path, downloading if needed.
pub fn resolve(spec: &str, offline: bool) -> io::Result<PathBuf> {
    match parse(spec) {
        Source::Local(p) if p.exists() => Ok(p),
        Source::Local(p) => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} does not exist (a Hub model looks like owner/name)", p.display()),
        )),
        other => download(other, offline),
    }
}

#[cfg(not(feature = "fetch"))]
fn download(_: Source, _: bool) -> io::Result<PathBuf> {
    Err(io::Error::other("this build has downloads disabled; pass a local path"))
}

#[cfg(feature = "fetch")]
pub use net::download;

#[cfg(feature = "fetch")]
mod net {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::time::Instant;

    fn err<T>(msg: impl Into<String>) -> io::Result<T> {
        Err(io::Error::other(msg.into()))
    }

    /// An agent that honours `HTTPS_PROXY` (ureq's default) and `SSL_CERT_FILE`,
    /// so corporate proxies and custom roots work without a rebuild.
    fn agent() -> ureq::Agent {
        let mut tls = ureq::tls::TlsConfig::builder();
        let bundle = std::env::var_os("MOE_CA_BUNDLE").or_else(|| std::env::var_os("SSL_CERT_FILE"));
        if let Some(pem) = bundle.and_then(|p| fs::read(p).ok()) {
            let certs: Vec<_> = ureq::tls::parse_pem(&pem)
                .filter_map(|item| match item {
                    Ok(ureq::tls::PemItem::Certificate(c)) => Some(c),
                    _ => None,
                })
                .collect();
            if !certs.is_empty() {
                tls = tls.root_certs(ureq::tls::RootCerts::Specific(Arc::new(certs)));
            }
        }
        ureq::Agent::config_builder().tls_config(tls.build()).timeout_global(None).build().into()
    }

    fn token() -> Option<String> {
        ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"].iter().find_map(|k| std::env::var(k).ok()).filter(|t| !t.is_empty())
    }

    fn get(agent: &ureq::Agent, url: &str) -> io::Result<ureq::http::Response<ureq::Body>> {
        let mut req = agent.get(url);
        if url.contains("huggingface.co") {
            if let Some(t) = token() {
                req = req.header("Authorization", format!("Bearer {t}"));
            }
        }
        match req.call() {
            Ok(r) => Ok(r),
            Err(ureq::Error::StatusCode(401 | 403)) => {
                err(format!("{url}: not accessible. If the model is gated or private, set HF_TOKEN"))
            }
            Err(ureq::Error::StatusCode(404)) => err(format!("{url}: not found")),
            Err(e) => err(format!("{url}: {e}")),
        }
    }

    /// Fetch one file to `dest`, skipping it when a complete copy is already
    /// there. Writes to a `.part` file first so an interrupted run cannot leave
    /// a truncated model behind.
    fn file(agent: &ureq::Agent, url: &str, dest: &Path, size: Option<u64>, label: &str) -> io::Result<()> {
        if let (Ok(meta), Some(want)) = (fs::metadata(dest), size) {
            if meta.len() == want {
                return Ok(());
            }
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let part = dest.with_extension("part");
        let resp = get(agent, url)?;
        let total = size.or_else(|| resp.headers().get("content-length")?.to_str().ok()?.parse().ok());
        let mut body = resp.into_body().into_reader();
        let mut out = io::BufWriter::with_capacity(1 << 20, fs::File::create(&part)?);

        let (start, mut done, mut last) = (Instant::now(), 0u64, Instant::now());
        let mut buf = vec![0u8; 1 << 18];
        loop {
            let n = body.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            if last.elapsed().as_millis() > 200 {
                progress(label, done, total, start);
                last = Instant::now();
            }
        }
        out.flush()?;
        drop(out);
        if interactive() {
            progress(label, done, total, start);
            eprintln!();
        } else {
            eprintln!("  {label}  {}", crate::human(done));
        }
        fs::rename(&part, dest)?;
        Ok(())
    }

    /// Live progress only makes sense on a terminal; piped output gets one
    /// summary line per file instead of a smear of carriage returns.
    fn interactive() -> bool {
        #[cfg(unix)]
        {
            unsafe { libc::isatty(2) == 1 }
        }
        #[cfg(not(unix))]
        {
            true
        }
    }

    fn progress(label: &str, done: u64, total: Option<u64>, start: Instant) {
        if !interactive() {
            return;
        }
        let rate = done as f64 / start.elapsed().as_secs_f64().max(1e-3);
        match total {
            Some(t) if t > 0 => eprint!(
                "\r  {label}  {:>3}%  {} / {}  {}/s      ",
                done * 100 / t,
                crate::human(done),
                crate::human(t),
                crate::human(rate as u64)
            ),
            _ => eprint!("\r  {label}  {}  {}/s      ", crate::human(done), crate::human(rate as u64)),
        }
    }

    pub fn download(src: Source, offline: bool) -> io::Result<PathBuf> {
        match src {
            Source::Local(p) => Ok(p),
            Source::Url(url) => {
                let name = url.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or("model");
                let dest = cache_dir().join("url").join(stem(&url)).join(sanitize(name));
                if dest.exists() {
                    return Ok(dest);
                }
                if offline {
                    return err(format!("{url} is not cached and --offline was given"));
                }
                eprintln!("fetching {url}");
                file(&agent(), &url, &dest, None, name)?;
                Ok(dest)
            }
            Source::Hub { repo, rev } => hub_snapshot(&repo, &rev, offline),
        }
    }

    /// Download the parts of a Hub repo that inference actually reads.
    fn hub_snapshot(repo: &str, rev: &str, offline: bool) -> io::Result<PathBuf> {
        let dir = cache_dir().join("hub").join(repo.replace('/', "--")).join(sanitize(rev));
        if offline {
            return if dir.exists() {
                Ok(local_model(&dir))
            } else {
                err(format!("{repo} is not cached at {} and --offline was given", dir.display()))
            };
        }

        let agent = agent();
        let api = format!("https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1");
        let listing =
            get(&agent, &api)?.into_body().read_to_string().map_err(|e| io::Error::other(format!("{api}: {e}")))?;
        let tree: serde_json::Value =
            serde_json::from_str(&listing).map_err(|e| io::Error::other(format!("{api}: {e}")))?;
        let entries: Vec<(String, u64)> = tree
            .as_array()
            .map(|a| a.iter())
            .into_iter()
            .flatten()
            .filter(|e| e["type"] == "file")
            .filter_map(|e| Some((e["path"].as_str()?.to_string(), e["size"].as_u64().unwrap_or(0))))
            // Top level only: repos keep conversions and demos in subdirectories.
            .filter(|(p, _)| !p.contains('/'))
            .collect();

        // A packed model in the repo is all we need; otherwise take the weights.
        let packed: Vec<_> = entries.iter().filter(|(p, _)| p.ends_with(".moe")).cloned().collect();
        let picked: Vec<_> = if packed.is_empty() {
            entries
                .iter()
                .filter(|(p, _)| p.ends_with(".safetensors") || WANTED.contains(&p.as_str()))
                .cloned()
                .collect()
        } else {
            packed
        };
        if !picked.iter().any(|(p, _)| p.ends_with(".safetensors") || p.ends_with(".moe")) {
            return err(format!("{repo} has no .safetensors or .moe files at the top level"));
        }

        let total: u64 = picked.iter().map(|(_, s)| s).sum();
        let have: u64 = picked
            .iter()
            .filter(|(p, s)| fs::metadata(dir.join(p)).map(|m| m.len() == *s).unwrap_or(false))
            .map(|(_, s)| s)
            .sum();
        if have < total {
            eprintln!("fetching {repo}@{rev} ({} of {})", crate::human(total - have), crate::human(total));
        }
        for (path, size) in &picked {
            let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{path}");
            file(&agent, &url, &dir.join(path), Some(*size), path)?;
        }
        Ok(local_model(&dir))
    }

    /// A directory holding exactly one packed model resolves to that file.
    fn local_model(dir: &Path) -> PathBuf {
        let packed: Vec<_> = fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "moe"))
            .collect();
        match packed.len() {
            1 => packed.into_iter().next().unwrap(),
            _ => dir.to_path_buf(),
        }
    }

    fn sanitize(s: &str) -> String {
        s.chars().map(|c| if c.is_ascii_alphanumeric() || "-._".contains(c) { c } else { '-' }).collect()
    }

    /// Short stable directory name for a URL.
    fn stem(url: &str) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in url.as_bytes() {
            h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
        }
        format!("{h:016x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Hub download must bring the files the engine reads, and no more. The
    /// prompt format lives in `tokenizer_config.json`, and leaving it behind
    /// makes the chat template silently unavailable — the checkpoint appears to
    /// declare no format, and the engine falls back to guessing one.
    #[cfg(feature = "fetch")]
    #[test]
    fn the_download_list_covers_the_prompt_format() {
        assert!(WANTED.contains(&"config.json"), "architecture detection needs it");
        assert!(WANTED.contains(&"tokenizer.json"), "text in and out needs it");
        assert!(WANTED.contains(&"tokenizer_config.json"), "the chat template lives here");
        assert!(WANTED.contains(&"chat_template.jinja"), "...or here, in the newer convention");
        // Everything else in a repo is weights, conversions or demos, which are
        // picked by extension rather than by name.
        assert!(WANTED.iter().all(|w| w.ends_with(".json") || w.ends_with(".jinja")));
    }

    #[test]
    fn specs_are_classified() {
        assert_eq!(parse("owner/name"), Source::Hub { repo: "owner/name".into(), rev: "main".into() });
        assert_eq!(parse("owner/name@v2"), Source::Hub { repo: "owner/name".into(), rev: "v2".into() });
        assert_eq!(parse("hf:owner/name"), Source::Hub { repo: "owner/name".into(), rev: "main".into() });
        // A Hub model page is the repo, but a file under it is a download.
        assert_eq!(
            parse("https://huggingface.co/owner/name"),
            Source::Hub { repo: "owner/name".into(), rev: "main".into() }
        );
        assert!(matches!(parse("https://huggingface.co/owner/name/resolve/main/x.moe"), Source::Url(_)));
        assert!(matches!(parse("https://example.com/m.moe"), Source::Url(_)));
        // Paths, including Windows ones and anything with more segments.
        assert!(matches!(parse("./models/x"), Source::Local(_)));
        assert!(matches!(parse("C:\\models\\x"), Source::Local(_)));
        assert!(matches!(parse("a/b/c"), Source::Local(_)));
        // An existing path wins over the repo-id shape.
        assert!(matches!(parse("src/lib.rs"), Source::Local(_)));
    }

    #[test]
    fn cache_dir_follows_the_environment() {
        // Reading the environment rather than setting it: tests share a process.
        let dir = cache_dir();
        assert!(dir.components().count() > 1, "{dir:?}");
        match std::env::var_os("MOE_CACHE") {
            Some(override_) => assert_eq!(dir, PathBuf::from(override_)),
            None => assert!(dir.to_string_lossy().contains("moe"), "{dir:?}"),
        }
    }
}
