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
        Ok(Store { maps, index, config, tokenizer: None, path: dir.into(), packed: false })
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
    pub fn pack(&self, out: &Path, weight_dt: Dt, expert_dt: Dt, mut log: impl FnMut(&str)) -> io::Result<()> {
        let mut tensors = serde_json::Map::new();
        let mut plan: Vec<(String, Rec, Dt)> = Vec::new();
        let mut off = 0u64;
        for (name, r) in &self.index {
            if name.contains("rotary_emb") || name.ends_with(".inv_freq") {
                continue;
            }
            let target = if r.rows == 1 || !is_big_weight(name) {
                Dt::F32
            } else if is_expert(name) {
                expert_dt
            } else {
                weight_dt
            };
            let dt = if target.fits(r.cols) { target } else { Dt::F32 };
            let len = (r.slabs * r.rows * dt.row_bytes(r.cols)) as u64;
            let shape = if r.slabs > 1 { json!([r.slabs, r.rows, r.cols]) } else { json!([r.rows, r.cols]) };
            tensors.insert(name.clone(), json!({"dt": dt.name(), "shape": shape, "off": off, "len": len}));
            plan.push((name.clone(), *r, dt));
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
        let header = json!({"config": self.config, "tokenizer": tokenizer, "tensors": Value::Object(tensors)});
        let hbytes = serde_json::to_vec(&header)?;

        let mut f = io::BufWriter::with_capacity(1 << 22, File::create(out)?);
        f.write_all(MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(hbytes.len() as u64).to_le_bytes())?;
        f.write_all(&hbytes)?;

        let total = plan.len();
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for (i, (name, r, dt)) in plan.iter().enumerate() {
            let t = self.get(name).unwrap();
            src.resize(r.cols, 0.0);
            dst.resize(dt.row_bytes(r.cols), 0);
            for row in 0..r.slabs * r.rows {
                t.dequant_row(row, &mut src);
                quantize(*dt, &src, &mut dst);
                f.write_all(&dst)?;
            }
            if i % 64 == 0 || i + 1 == total {
                log(&format!("  [{:>5}/{}] {name} -> {}", i + 1, total, dt.name()));
            }
        }
        f.flush()
    }
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
