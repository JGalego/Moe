//! Weight storage. One lookup API over two backends:
//!
//! * a Hugging Face directory of `*.safetensors` shards, read in place;
//! * a packed `.moe` file, which is the same tensors re-quantised and laid out
//!   so that each expert is one contiguous range.
//!
//! Both are memory mapped and never copied. Maps are intentionally leaked: they
//! live for the whole process, which lets every weight view be `&'static` and
//! therefore trivially shareable across threads.

use crate::quant::{quantize, Dt, QT};
use memmap2::Mmap;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"MOEF";
const VERSION: u32 = 1;

/// A tensor's location and shape. `slabs` is the leading dimension of a fused
/// 3-D expert stack (`[experts, rows, cols]`) and 1 for an ordinary matrix; in
/// both cases the payload is `slabs * rows` contiguous rows of width `cols`.
#[derive(Clone, Copy)]
struct Rec {
    map: usize,
    off: usize,
    dt: Dt,
    slabs: usize,
    rows: usize,
    cols: usize,
}

impl Rec {
    fn len(&self) -> usize {
        self.slabs * self.rows * self.dt.row_bytes(self.cols)
    }
}

pub struct Store {
    maps: Vec<&'static [u8]>,
    index: BTreeMap<String, Rec>,
    /// The model's original `config.json`, carried through packing.
    pub config: Value,
    /// `tokenizer.json` embedded at pack time, so a `.moe` is one portable file.
    pub tokenizer: Option<Value>,
    /// The checkpoint's own chat template, from `tokenizer_config.json` or a
    /// `chat_template.jinja` beside it. Carried through packing for the same
    /// reason the tokenizer is: a prompt format is part of the model.
    pub chat_template: Option<String>,
    pub path: PathBuf,
    pub packed: bool,
}

fn err<T>(msg: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, msg.into()))
}

fn map_file(path: &Path) -> io::Result<&'static [u8]> {
    let f = File::open(path)?;
    let m = unsafe { Mmap::map(&f)? };
    let m: &'static Mmap = Box::leak(Box::new(m));
    Ok(&m[..])
}

/// Parse a safetensors header, returning `(records, data_start)`.
fn safetensors_index(buf: &[u8], map: usize) -> io::Result<(BTreeMap<String, Rec>, usize)> {
    if buf.len() < 8 {
        return err("truncated safetensors file");
    }
    let hlen = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    let start = 8 + hlen;
    if buf.len() < start {
        return err("safetensors header longer than file");
    }
    let head: Value = serde_json::from_slice(&buf[8..start])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad header: {e}")))?;
    let mut out = BTreeMap::new();
    for (name, v) in head.as_object().map(|o| o.iter()).into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let Some(dt) = v["dtype"].as_str().and_then(Dt::parse) else { continue };
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();
        let (slabs, rows, cols) = match shape.len() {
            1 => (1, 1, shape[0]),
            2 => (1, shape[0], shape[1]),
            3 => (shape[0], shape[1], shape[2]),
            _ => continue, // scalars and 4-D tensors have no role here
        };
        let off = v["data_offsets"][0].as_u64().unwrap_or(0) as usize + start;
        out.insert(name.clone(), Rec { map, off, dt, slabs, rows, cols });
    }
    Ok((out, start))
}

impl Store {
    /// Open a Hugging Face model directory or a packed `.moe` file.
    pub fn open(path: &Path) -> io::Result<Store> {
        if path.is_dir() {
            Store::open_hf(path)
        } else {
            Store::open_packed(path)
        }
    }

    fn open_hf(dir: &Path) -> io::Result<Store> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return err(format!("no .safetensors files in {}", dir.display()));
        }
        let (mut maps, mut index) = (Vec::new(), BTreeMap::new());
        for p in &shards {
            let buf = map_file(p)?;
            let (recs, _) = safetensors_index(buf, maps.len())?;
            index.extend(recs);
            maps.push(buf);
        }
        let cfg = dir.join("config.json");
        let config = if cfg.exists() {
            let mut s = String::new();
            File::open(&cfg)?.read_to_string(&mut s)?;
            serde_json::from_str(&s).unwrap_or(Value::Null)
        } else {
            Value::Null
        };
        Ok(Store {
            maps,
            index,
            config,
            tokenizer: None,
            chat_template: chat_template_in(dir),
            path: dir.into(),
            packed: false,
        })
    }

    fn open_packed(file: &Path) -> io::Result<Store> {
        let buf = map_file(file)?;
        if buf.len() < 16 || &buf[..4] != MAGIC {
            return err(format!("{} is not a .moe file", file.display()));
        }
        let ver = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if ver != VERSION {
            return err(format!("packed format v{ver}, this build speaks v{VERSION}"));
        }
        let hlen = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
        let head: Value = serde_json::from_slice(&buf[16..16 + hlen])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bad header: {e}")))?;
        let data = 16 + hlen;
        let mut index = BTreeMap::new();
        for (name, v) in head["tensors"].as_object().map(|o| o.iter()).into_iter().flatten() {
            let dt = Dt::parse(v["dt"].as_str().unwrap_or("")).unwrap_or(Dt::F32);
            let dims: Vec<usize> = v["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default();
            let (slabs, rows, cols) = match dims.len() {
                3 => (dims[0], dims[1], dims[2]),
                2 => (1, dims[0], dims[1]),
                _ => continue,
            };
            let off = v["off"].as_u64().unwrap_or(0) as usize + data;
            index.insert(name.clone(), Rec { map: 0, off, dt, slabs, rows, cols });
        }
        Ok(Store {
            maps: vec![buf],
            index,
            config: head["config"].clone(),
            tokenizer: match &head["tokenizer"] {
                Value::Null => None,
                v => Some(v.clone()),
            },
            chat_template: head["chat_template"].as_str().map(String::from),
            path: file.into(),
            packed: true,
        })
    }

    /// Whole tensor, with any leading expert dimension flattened into rows.
    pub fn get(&self, name: &str) -> Option<QT> {
        let r = *self.index.get(name)?;
        self.view(name, 0, 0..r.slabs * r.rows)
    }

    /// `(slabs, rows, cols)` of a tensor.
    pub fn shape(&self, name: &str) -> Option<(usize, usize, usize)> {
        self.index.get(name).map(|r| (r.slabs, r.rows, r.cols))
    }

    /// A row range inside one slab — how a single expert is addressed inside a
    /// fused `[experts, 2*inter, hidden]` stack, at zero copy.
    pub fn view(&self, name: &str, slab: usize, rows: std::ops::Range<usize>) -> Option<QT> {
        let r = self.index.get(name)?;
        if rows.end > r.slabs * r.rows || rows.start >= rows.end {
            return None;
        }
        let stride = r.dt.row_bytes(r.cols);
        let base = r.off + (slab * r.rows + rows.start) * stride;
        let n = rows.end - rows.start;
        let bytes = self.maps[r.map].get(base..base + n * stride)?;
        Some(QT::new(r.dt, n, r.cols, bytes))
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// First tensor that exists among several candidate names.
    pub fn any(&self, names: &[&str]) -> Option<QT> {
        names.iter().find_map(|n| self.get(n))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Total weight bytes as stored on disk.
    pub fn bytes(&self) -> u64 {
        self.index.values().map(|r| r.len() as u64).sum()
    }

    /// Ask the kernel to fault in the first `budget` bytes of every map, so the
    /// first token does not pay for page-ins one at a time.
    pub fn warm(&self, budget: u64) -> u64 {
        let mut done = 0u64;
        let mut sink = 0u64;
        for m in &self.maps {
            let take = (budget - done).min(m.len() as u64) as usize;
            for i in (0..take).step_by(4096) {
                sink = sink.wrapping_add(unsafe { std::ptr::read_volatile(&m[i]) } as u64);
            }
            done += take as u64;
            if done >= budget {
                break;
            }
        }
        std::hint::black_box(sink);
        done
    }

    /// Re-quantise into a packed `.moe` file.
    ///
    /// 2-D weights whose width is a multiple of the block size go to `weight_dt`
    /// (experts to `expert_dt`); norms, biases and router gates stay f32, where
    /// they cost nothing and quantisation noise would be amplified.
    pub fn pack(&self, out: &Path, weight_dt: Dt, expert_dt: Dt, log: impl FnMut(&str)) -> io::Result<()> {
        self.pack_pruned(out, weight_dt, expert_dt, None, log)
    }

    /// Re-quantise into a packed `.moe`, optionally keeping only some experts.
    ///
    /// With a [`Prune`], each MoE layer keeps the experts it names, renumbered
    /// into a contiguous range, and the router is narrowed to match. The result is
    /// a smaller, self-consistent model rather than a checkpoint with holes in it:
    /// nothing dangling, nothing that can select an expert that is no longer
    /// there.
    pub fn pack_pruned(
        &self,
        out: &Path,
        weight_dt: Dt,
        expert_dt: Dt,
        prune: Option<&Prune>,
        log: impl FnMut(&str),
    ) -> io::Result<()> {
        self.pack_with(out, weight_dt, expert_dt, prune, None, log)
    }

    /// The full form, with per-expert precision as well as pruning.
    ///
    /// `hot` names experts that get a finer format than the rest. Routing is
    /// skewed, so most of a checkpoint's bytes belong to experts a workload
    /// rarely selects — those can be cheap, while the handful it leans on stay
    /// accurate. That is a better trade than one rate for all of them.
    pub fn pack_with(
        &self,
        out: &Path,
        weight_dt: Dt,
        expert_dt: Dt,
        prune: Option<&Prune>,
        hot: Option<&Hot>,
        mut log: impl FnMut(&str),
    ) -> io::Result<()> {
        let mut tensors = serde_json::Map::new();
        let mut plan: Vec<Item> = Vec::new();
        let mut off = 0u64;
        for (name, r) in &self.index {
            if name.contains("rotary_emb") || name.ends_with(".inv_freq") {
                continue;
            }
            // Pruning renames, re-slices or drops a tensor; everything else is
            // copied straight through.
            let (dest, rows, cols, shape_override) = match prune.map(|p| p.plan(name, r)) {
                Some(Selection::Drop) => continue,
                Some(Selection::Keep { name, rows, cols, slabs }) => (name, rows, cols, slabs),
                None => (name.clone(), None, None, None),
            };
            let target = if r.rows == 1 || !is_big_weight(name) {
                Dt::F32
            } else if is_expert(name) {
                // A hot expert overrides the blanket expert format.
                hot.and_then(|h| h.format_for(name)).unwrap_or(expert_dt)
            } else {
                weight_dt
            };
            let dt = if target.fits(r.cols) { target } else { Dt::F32 };
            let out_cols = cols.as_ref().map(|c| c.len()).unwrap_or(r.cols);
            // A narrowed row may no longer be a whole number of blocks.
            let dt = if dt.fits(out_cols) { dt } else { Dt::F32 };
            let out_rows = rows.as_ref().map(|v| v.len()).unwrap_or(r.slabs * r.rows);
            let len = (out_rows * dt.row_bytes(out_cols)) as u64;
            let shape = match shape_override {
                // A fused expert stack keeps its three dimensions, with a
                // narrower leading one.
                Some(slabs) => json!([slabs, out_rows / slabs.max(1), out_cols]),
                None if r.slabs > 1 => json!([r.slabs, r.rows, out_cols]),
                None => json!([out_rows, out_cols]),
            };
            tensors.insert(dest.clone(), json!({"dt": dt.name(), "shape": shape, "off": off, "len": len}));
            plan.push(Item { dest, src: name.clone(), rec: *r, dt, rows, cols });
            off += len;
        }
        // Carry the tokenizer along so the packed model needs nothing beside it.
        let tokenizer = self
            .tokenizer
            .clone()
            .or_else(|| {
                std::fs::read(self.path.join("tokenizer.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())
            })
            .unwrap_or(Value::Null);
        let chat = self
            .chat_template
            .clone()
            .or_else(|| chat_template_in(&self.path))
            .map(Value::String)
            .unwrap_or(Value::Null);
        // The config has to agree with the tensors, or the pruned model would
        // detect an expert count it no longer has.
        let mut config = self.config.clone();
        if let Some(p) = prune {
            p.rewrite(&mut config);
        }
        let header = json!({
            "config": config,
            "tokenizer": tokenizer,
            "chat_template": chat,
            "tensors": Value::Object(tensors),
        });
        let hbytes = serde_json::to_vec(&header)?;

        let mut f = io::BufWriter::with_capacity(1 << 22, File::create(out)?);
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(hbytes.len() as u64).to_le_bytes())?;
        f.write_all(&hbytes)?;

        let total = plan.len();
        let (mut src, mut dst, mut narrowed) = (Vec::new(), Vec::new(), Vec::new());
        for (i, item) in plan.iter().enumerate() {
            let t = self.get(&item.src).unwrap();
            let r = &item.rec;
            src.resize(r.cols, 0.0);
            let out_cols = item.cols.as_ref().map(|c| c.len()).unwrap_or(r.cols);
            narrowed.resize(out_cols, 0.0);
            dst.resize(item.dt.row_bytes(out_cols), 0);
            let all: Vec<usize> = (0..r.slabs * r.rows).collect();
            let rows = item.rows.as_deref().unwrap_or(&all);
            for row in rows {
                t.dequant_row(*row, &mut src);
                let row_data: &[f32] = match &item.cols {
                    Some(keep) => {
                        for (o, c) in narrowed.iter_mut().zip(keep) {
                            *o = src[*c];
                        }
                        &narrowed
                    }
                    None => &src,
                };
                quantize(item.dt, row_data, &mut dst);
                f.write_all(&dst)?;
            }
            if i % 64 == 0 || i + 1 == total {
                log(&format!("  [{:>5}/{}] {} -> {}", i + 1, total, item.dest, item.dt.name()));
            }
        }
        f.flush()
    }
}

/// One tensor's write plan: where it comes from, what it is called, and which
/// rows and columns of it survive.
struct Item {
    dest: String,
    src: String,
    rec: Rec,
    dt: Dt,
    /// Rows of the flattened source to write, in order. `None` means all.
    rows: Option<Vec<usize>>,
    /// Columns to keep. `None` means all.
    cols: Option<Vec<usize>>,
}

/// What pruning does to one tensor.
enum Selection {
    Drop,
    Keep { name: String, rows: Option<Vec<usize>>, cols: Option<Vec<usize>>, slabs: Option<usize> },
}

/// Experts worth spending more bits on, and how many.
///
/// Built from a trace: the experts a workload actually leans on. Applies only to
/// checkpoints that store experts as separate tensors — a fused
/// `[experts, rows, cols]` stack is one tensor with one format, so there is
/// nothing per-expert to vary, and it silently keeps the blanket rate.
#[derive(Clone, Debug)]
pub struct Hot {
    /// `(layer, expert)` pairs to store at `dt`.
    pub experts: Vec<(u32, u32)>,
    pub dt: Dt,
}

impl Hot {
    fn format_for(&self, name: &str) -> Option<Dt> {
        let (layer, at) = Prune::layer_of(name)?;
        let suffix = &name[at..];
        for marker in ["mlp.experts.", "block_sparse_moe.experts."] {
            if let Some(rest) = suffix.strip_prefix(marker) {
                let num = rest.split('.').next()?;
                let e: u32 = num.parse().ok()?;
                if self.experts.contains(&(layer as u32, e)) {
                    return Some(self.dt);
                }
                return None;
            }
        }
        None
    }
}

/// Which experts each layer keeps.
///
/// A sparse checkpoint is mostly experts a given workload never selects, and
/// `moe route` says exactly which. Dropping the rest yields a smaller model
/// specialised to that workload — Mixtral, but only the parts that answer code
/// questions. The count has to be uniform across layers because a config
/// declares one expert count, but the *sets* differ per layer, which is where the
/// specialisation lives.
#[derive(Clone, Debug)]
pub struct Prune {
    /// `keep[layer]` is the experts to retain, ascending. Their position here is
    /// the index they are renumbered to.
    pub keep: Vec<Vec<u32>>,
    /// Experts per token in the pruned model, which cannot exceed what is kept.
    pub top_k: usize,
}

impl Prune {
    /// How many experts each layer will have. Uniform by construction.
    pub fn width(&self) -> usize {
        self.keep.iter().map(|k| k.len()).max().unwrap_or(0)
    }

    /// Which layer a per-layer tensor name belongs to, and where its suffix starts.
    fn layer_of(name: &str) -> Option<(usize, usize)> {
        // `...layers.<n>.rest` — find the number between two dots.
        let bytes = name.as_bytes();
        let mut i = 0;
        while let Some(dot) = name[i..].find('.') {
            let start = i + dot + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start && bytes.get(end) == Some(&b'.') {
                if let Ok(l) = name[start..end].parse() {
                    return Some((l, end + 1));
                }
            }
            i = start;
        }
        None
    }

    fn kept(&self, layer: usize) -> Option<&Vec<u32>> {
        self.keep.get(layer).filter(|k| !k.is_empty())
    }

    fn plan(&self, name: &str, r: &Rec) -> Selection {
        let keep_all = || Selection::Keep { name: name.to_string(), rows: None, cols: None, slabs: None };
        let Some((layer, at)) = Prune::layer_of(name) else { return keep_all() };
        let Some(keep) = self.kept(layer) else { return keep_all() };
        let suffix = &name[at..];

        // The router: one row per expert, so narrow it to the kept rows in the
        // same order the experts were renumbered.
        if suffix.ends_with("mlp.gate.weight") || suffix.ends_with("block_sparse_moe.gate.weight") {
            return Selection::Keep {
                name: name.to_string(),
                rows: Some(keep.iter().map(|e| *e as usize).collect()),
                cols: None,
                slabs: None,
            };
        }
        // The gate's per-expert bias is a single row indexed by expert, so the
        // selection is over columns instead.
        if suffix.contains("e_score_correction_bias") {
            return Selection::Keep {
                name: name.to_string(),
                rows: None,
                cols: Some(keep.iter().map(|e| *e as usize).collect()),
                slabs: None,
            };
        }
        // A fused stack: keep whole slabs, in the new order.
        if suffix.starts_with("mlp.experts.") && r.slabs > 1 {
            let rows: Vec<usize> =
                keep.iter().flat_map(|e| (0..r.rows).map(move |row| *e as usize * r.rows + row)).collect();
            return Selection::Keep { name: name.to_string(), rows: Some(rows), cols: None, slabs: Some(keep.len()) };
        }
        // Per-expert tensors: drop the unwanted, renumber the rest.
        for marker in ["mlp.experts.", "block_sparse_moe.experts."] {
            if let Some(rest) = suffix.strip_prefix(marker) {
                let (num, tail) = rest.split_once('.').unwrap_or((rest, ""));
                if let Ok(e) = num.parse::<u32>() {
                    return match keep.iter().position(|k| *k == e) {
                        Some(new) => Selection::Keep {
                            name: format!("{}{marker}{new}.{tail}", &name[..at]),
                            rows: None,
                            cols: None,
                            slabs: None,
                        },
                        None => Selection::Drop,
                    };
                }
            }
        }
        keep_all()
    }

    /// Bring the config into line with the tensors that were written.
    fn rewrite(&self, config: &mut Value) {
        let width = self.width();
        if width == 0 {
            return;
        }
        if let Some(obj) = config.as_object_mut() {
            for key in ["num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"] {
                if obj.contains_key(key) {
                    obj.insert(key.into(), json!(width));
                }
            }
            for key in ["num_experts_per_tok", "moe_top_k"] {
                if obj.contains_key(key) {
                    obj.insert(key.into(), json!(self.top_k.min(width)));
                }
            }
            // Group-limited routing cannot survive renumbering: the groups were
            // contiguous ranges of the original experts, and are no longer.
            for key in ["n_group", "topk_group"] {
                if obj.contains_key(key) {
                    obj.insert(key.into(), json!(1));
                }
            }
        }
    }
}

/// Find a chat template beside a checkpoint.
///
/// Two conventions: a `chat_template` key in `tokenizer_config.json`, and — the
/// newer one, since embedding Jinja in JSON is miserable — a `chat_template.jinja`
/// file of its own. Some checkpoints ship a *list* of named templates; the one
/// called `default` is the chat one, and the rest are for tool use.
pub fn chat_template_in(dir: &Path) -> Option<String> {
    let dir = if dir.is_dir() { dir.to_path_buf() } else { dir.parent()?.to_path_buf() };
    if let Ok(s) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
        if !s.trim().is_empty() {
            return Some(s);
        }
    }
    let raw = std::fs::read(dir.join("tokenizer_config.json")).ok()?;
    let cfg: Value = serde_json::from_slice(&raw).ok()?;
    match &cfg["chat_template"] {
        Value::String(s) => Some(s.clone()),
        Value::Array(list) => list
            .iter()
            .find(|t| t["name"] == "default")
            .or_else(|| list.first())
            .and_then(|t| t["template"].as_str().map(String::from)),
        _ => None,
    }
}

/// Round a byte range out to page boundaries, as the memory syscalls require.
#[cfg(unix)]
fn pages(bytes: &[u8]) -> Option<(*mut libc::c_void, usize)> {
    if bytes.is_empty() {
        return None;
    }
    let page = (unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).max(1) as usize;
    let start = bytes.as_ptr() as usize;
    let aligned = start & !(page - 1);
    Some((aligned as *mut libc::c_void, bytes.len() + (start - aligned)))
}

/// Ask the kernel to fetch a byte range in the background.
///
/// This is advice, not a read: it returns before the pages arrive, so they land
/// while the caller is busy elsewhere. Advice that turns out to be wrong costs
/// nothing but some page cache — which is what makes guessing at which experts a
/// token will choose safe in a way guessing at its output is not.
#[cfg(unix)]
pub fn advise(bytes: &[u8]) {
    if let Some((addr, len)) = pages(bytes) {
        // Deliberately unchecked: failure means the pages arrive later, which is
        // exactly what would have happened without asking.
        unsafe { libc::madvise(addr, len, libc::MADV_WILLNEED) };
    }
}

/// Keep a byte range resident, so it can never be evicted.
#[cfg(unix)]
pub fn lock(bytes: &[u8]) -> io::Result<()> {
    match pages(bytes) {
        None => Ok(()),
        Some((addr, len)) => match unsafe { libc::mlock(addr, len) } {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        },
    }
}

/// Windows and the rest have no portable equivalent worth the linkage; the
/// engine simply does not prefetch there, and correctness never depended on it.
#[cfg(not(unix))]
pub fn advise(_bytes: &[u8]) {}

#[cfg(not(unix))]
pub fn lock(_bytes: &[u8]) -> io::Result<()> {
    Err(io::Error::other("locking pages is not supported on this platform"))
}

/// Fault a range in by reading one byte per page. Slower than [`advise`] because
/// it blocks, but it works everywhere and needs no privilege, which makes it the
/// fallback when [`lock`] is refused.
pub fn touch(bytes: &[u8]) -> u64 {
    let mut sink = 0u64;
    for i in (0..bytes.len()).step_by(4096) {
        sink = sink.wrapping_add(unsafe { std::ptr::read_volatile(&bytes[i]) } as u64);
    }
    std::hint::black_box(sink);
    bytes.len() as u64
}

/// Expert weights: the bulk of a sparse model and the only tensors read on a
/// per-token, per-route basis.
pub fn is_expert(name: &str) -> bool {
    name.contains(".experts.")
}

/// Big 2-D projections worth quantising, as opposed to gates and norms.
fn is_big_weight(name: &str) -> bool {
    !(name.contains("norm") || name.ends_with(".bias") || name.ends_with("gate.weight") || name.contains("e_score"))
}
