//! The decoder: weight binding and the forward pass.
//!
//! One code path serves prefill and decode — a step takes `t` tokens, and
//! decode is simply `t == 1`. Everything is computed in f32 from quantised
//! weights; nothing is materialised that a token does not need, so a routed
//! layer only ever touches the experts it selected.

use crate::quant::{matmul, matvec, Dt, QT};
use crate::spec::{Mla, Spec};
use crate::store::Store;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

struct Mlp {
    gate: QT,
    up: QT,
    down: QT,
}

impl Mlp {
    fn inter(&self) -> usize {
        self.gate.rows
    }

    /// SwiGLU: `down(silu(gate(x)) * up(x))` over a batch of `t` rows.
    fn forward(&self, x: &[f32], t: usize, out: &mut [f32]) {
        let n = self.inter();
        let mut g = vec![0.0; t * n];
        let mut u = vec![0.0; t * n];
        rayon::join(|| matmul(&self.gate, x, &mut g), || matmul(&self.up, x, &mut u));
        for (a, b) in g.iter_mut().zip(&u) {
            *a = (*a / (1.0 + (-*a).exp())) * b;
        }
        matmul(&self.down, &g, out);
    }

    fn bytes(&self) -> u64 {
        [self.gate, self.up, self.down].iter().map(|t| t.data.len() as u64).sum()
    }
}

struct Gqa {
    q: QT,
    k: QT,
    v: QT,
    o: QT,
    bias: [Option<Vec<f32>>; 3],
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

struct MlaAttn {
    dims: Mla,
    q_a: Option<QT>,
    q_a_norm: Option<Vec<f32>>,
    q_b: QT,
    kv_a: QT,
    kv_a_norm: Vec<f32>,
    kv_b: QT,
    o: QT,
}

enum Attn {
    Gqa(Gqa),
    Mla(MlaAttn),
}

enum Ffn {
    Dense(Mlp),
    Moe { router: QT, e_bias: Option<Vec<f32>>, experts: Vec<Mlp>, shared: Option<Mlp>, shared_gate: Option<QT> },
}

/// One expert's contribution: the `(token, weight)` pairs it was chosen for,
/// and its output rows in the same order.
type ExpertOutput = (Vec<(usize, f32)>, Vec<f32>);

struct Layer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    attn: Attn,
    ffn: Ffn,
}

/// One layer's routing decision for one token.
#[derive(Clone, Debug)]
pub struct Route {
    pub pos: u32,
    pub layer: u32,
    /// Selected experts and the weight each was given, strongest first.
    pub experts: Vec<(u32, f32)>,
}

/// Recorded routing, when tracing is switched on. Dense layers contribute
/// nothing, so a layer missing from `routes` is a layer with no experts.
#[derive(Default, Debug)]
pub struct Trace {
    /// Token id at each position the trace covers.
    pub tokens: Vec<u32>,
    pub routes: Vec<Route>,
}

/// Per-sequence mutable state: the KV cache and running counters.
pub struct State {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kdim: usize,
    vdim: usize,
    pub pos: usize,
    pub ctx: usize,
    prev: Vec<Vec<u32>>,
    pub stats: Stats,
    /// Set by [`State::trace`] to record every routing decision.
    pub trace: Option<Trace>,
}

#[derive(Default)]
pub struct Stats {
    /// Expert activations (tokens x layers x top_k).
    pub routed: AtomicU64,
    /// Expert activations that repeat the previous token's choice at that layer.
    pub reused: AtomicU64,
    /// Expert weight bytes touched.
    pub expert_bytes: AtomicU64,
}

impl Stats {
    pub fn reuse_rate(&self) -> f32 {
        let r = self.routed.load(Ordering::Relaxed);
        if r == 0 {
            0.0
        } else {
            self.reused.load(Ordering::Relaxed) as f32 / r as f32
        }
    }
}

impl State {
    pub fn new(m: &Model, ctx: usize) -> State {
        let (kdim, vdim) = m.kv_dims();
        State {
            k: (0..m.spec.layers).map(|_| vec![0.0; ctx * kdim]).collect(),
            v: (0..m.spec.layers).map(|_| vec![0.0; ctx * vdim]).collect(),
            kdim,
            vdim,
            pos: 0,
            ctx,
            prev: vec![Vec::new(); m.spec.layers],
            stats: Stats::default(),
            trace: None,
        }
    }

    /// Start recording routing decisions. Off by default: it costs an
    /// allocation per token per routed layer.
    pub fn trace(&mut self) {
        self.trace = Some(Trace::default());
    }

    /// Bytes held by the KV cache.
    pub fn kv_bytes(&self) -> u64 {
        ((self.k.len() * self.ctx * (self.kdim + self.vdim)) * 4) as u64
    }

    pub fn reset(&mut self) {
        self.truncate(0);
    }

    /// Drop everything cached at or after position `n`, keeping the prefix.
    ///
    /// The KV cache is append-only per position, so later positions are simply
    /// overwritten by whatever is forwarded next; nothing needs clearing. The
    /// per-layer previous-selection does get cleared, because the token that
    /// produced it is no longer the one before the cursor.
    pub fn truncate(&mut self, n: usize) {
        self.pos = n.min(self.pos);
        self.prev.iter_mut().for_each(|p| p.clear());
        if let Some(tr) = self.trace.as_mut() {
            tr.tokens.truncate(self.pos);
            tr.routes.retain(|r| (r.pos as usize) < self.pos);
        }
    }
}

pub struct Model {
    pub spec: Spec,
    pub store: Store,
    embed: QT,
    out_norm: Vec<f32>,
    lm_head: QT,
    layers: Vec<Layer>,
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let scale = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = v * scale * g;
    }
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

/// Rotary embedding, half-split (the layout Hugging Face checkpoints use).
fn rope(x: &mut [f32], pos: usize, theta: f32, scale: f32) {
    let half = x.len() / 2;
    let p = pos as f32 / scale;
    for i in 0..half {
        let freq = theta.powf(-2.0 * i as f32 / x.len() as f32);
        let (s, c) = (p * freq).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

impl Model {
    pub fn load(store: Store) -> Result<Model, String> {
        let spec = Spec::derive(&store)?;
        Model::load_with(store, spec)
    }

    pub fn load_with(store: Store, spec: Spec) -> Result<Model, String> {
        let need = |t: Option<QT>, what: &str| t.ok_or_else(|| format!("missing tensor: {what}"));
        let embed = need(
            store.any(&["model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight"]),
            "embedding",
        )?;
        let out_norm =
            need(store.any(&["model.norm.weight", "norm.weight", "model.final_layernorm.weight"]), "final norm")?
                .to_vec();
        let lm_head = store.any(&["lm_head.weight", "output.weight"]).unwrap_or(embed);

        let mut layers = Vec::with_capacity(spec.layers);
        for l in 0..spec.layers {
            let p = format!("{}{l}.", spec.prefix);
            let g = |s: &str| store.get(&format!("{p}{s}"));
            let vec_of = |s: &str| g(s).map(|t| t.to_vec());
            let alt = |a: &str, b: &str| g(a).or_else(|| g(b));

            let attn = match &spec.mla {
                Some(d) => Attn::Mla(MlaAttn {
                    dims: *d,
                    q_a: g("self_attn.q_a_proj.weight"),
                    q_a_norm: vec_of("self_attn.q_a_layernorm.weight"),
                    q_b: need(alt("self_attn.q_b_proj.weight", "self_attn.q_proj.weight"), "q projection")?,
                    kv_a: need(g("self_attn.kv_a_proj_with_mqa.weight"), "kv_a_proj_with_mqa")?,
                    kv_a_norm: need(g("self_attn.kv_a_layernorm.weight"), "kv_a_layernorm")?.to_vec(),
                    kv_b: need(g("self_attn.kv_b_proj.weight"), "kv_b_proj")?,
                    o: need(g("self_attn.o_proj.weight"), "o_proj")?,
                }),
                None => Attn::Gqa(Gqa {
                    q: need(g("self_attn.q_proj.weight"), "q_proj")?,
                    k: need(g("self_attn.k_proj.weight"), "k_proj")?,
                    v: need(g("self_attn.v_proj.weight"), "v_proj")?,
                    o: need(g("self_attn.o_proj.weight"), "o_proj")?,
                    bias: [
                        vec_of("self_attn.q_proj.bias"),
                        vec_of("self_attn.k_proj.bias"),
                        vec_of("self_attn.v_proj.bias"),
                    ],
                    q_norm: vec_of("self_attn.q_norm.weight"),
                    k_norm: vec_of("self_attn.k_norm.weight"),
                }),
            };

            // A layer is routed if it actually carries experts; that covers
            // dense-prefix models and every-Nth-layer sparsity without config.
            // Experts come either one tensor at a time or as a fused stack,
            // `gate_up_proj: [E, 2*inter, hidden]` + `down_proj: [E, hidden, inter]`,
            // where gate is the first half of the rows and up the second.
            let fused = store.shape(&format!("{p}mlp.experts.gate_up_proj"));
            let expert = |e: usize, which: usize| match fused {
                Some((_, gu_rows, _)) => {
                    let inter = gu_rows / 2;
                    match which {
                        0 => store.view(&format!("{p}mlp.experts.gate_up_proj"), e, 0..inter),
                        1 => store.view(&format!("{p}mlp.experts.gate_up_proj"), e, inter..2 * inter),
                        _ => store
                            .shape(&format!("{p}mlp.experts.down_proj"))
                            .and_then(|(_, r, _)| store.view(&format!("{p}mlp.experts.down_proj"), e, 0..r)),
                    }
                }
                None => {
                    let hf = ["gate_proj", "up_proj", "down_proj"][which];
                    let mx = ["w1", "w3", "w2"][which];
                    g(&format!("mlp.experts.{e}.{hf}.weight"))
                        .or_else(|| g(&format!("block_sparse_moe.experts.{e}.{mx}.weight")))
                }
            };
            let ffn = if spec.experts > 0 && expert(0, 0).is_some() {
                let mut experts = Vec::with_capacity(spec.experts);
                for e in 0..spec.experts {
                    experts.push(Mlp {
                        gate: need(expert(e, 0), &format!("expert {e} gate"))?,
                        up: need(expert(e, 1), &format!("expert {e} up"))?,
                        down: need(expert(e, 2), &format!("expert {e} down"))?,
                    });
                }
                let shared = |which: usize| {
                    let n = ["gate_proj", "up_proj", "down_proj"][which];
                    g(&format!("mlp.shared_expert.{n}.weight")).or_else(|| g(&format!("mlp.shared_experts.{n}.weight")))
                };
                Ffn::Moe {
                    router: need(alt("mlp.gate.weight", "block_sparse_moe.gate.weight"), "router")?,
                    e_bias: vec_of("mlp.gate.e_score_correction_bias"),
                    experts,
                    shared: shared(0).map(|gate| Mlp { gate, up: shared(1).unwrap(), down: shared(2).unwrap() }),
                    shared_gate: g("mlp.shared_expert_gate.weight"),
                }
            } else {
                Ffn::Dense(Mlp {
                    gate: need(alt("mlp.gate_proj.weight", "mlp.w1.weight"), "mlp.gate_proj")?,
                    up: need(alt("mlp.up_proj.weight", "mlp.w3.weight"), "mlp.up_proj")?,
                    down: need(alt("mlp.down_proj.weight", "mlp.w2.weight"), "mlp.down_proj")?,
                })
            };

            layers.push(Layer {
                ln1: need(alt("input_layernorm.weight", "ln1.weight"), "input_layernorm")?.to_vec(),
                ln2: need(alt("post_attention_layernorm.weight", "ln2.weight"), "post_attention_layernorm")?.to_vec(),
                attn,
                ffn,
            });
        }
        Ok(Model { spec, store, embed, out_norm, lm_head, layers })
    }

    /// (key dim, value dim) held per cached position, summed over heads.
    pub fn kv_dims(&self) -> (usize, usize) {
        match &self.spec.mla {
            Some(m) => (self.spec.heads * (m.qk_nope + m.qk_rope), self.spec.heads * m.v_head),
            None => (self.spec.kv_heads * self.spec.head_dim, self.spec.kv_heads * self.spec.head_dim),
        }
    }

    /// Bytes of routed expert weights across the whole model.
    pub fn expert_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| match &l.ffn {
                Ffn::Moe { experts, .. } => experts.iter().map(|e| e.bytes()).sum(),
                Ffn::Dense(_) => 0,
            })
            .sum()
    }

    /// Run `tokens` starting at `state.pos` and return the logits of the last one.
    pub fn forward(&self, tokens: &[u32], st: &mut State) -> Vec<f32> {
        let s = &self.spec;
        let t = tokens.len();
        assert!(t > 0 && st.pos + t <= st.ctx, "context window exhausted");
        let h = s.hidden;

        if let Some(tr) = st.trace.as_mut() {
            tr.tokens.extend_from_slice(tokens);
        }

        let mut x = vec![0.0f32; t * h];
        for (i, tok) in tokens.iter().enumerate() {
            self.embed.dequant_row(*tok as usize % s.vocab, &mut x[i * h..(i + 1) * h]);
        }

        let mut norm = vec![0.0f32; t * h];
        let mut delta = vec![0.0f32; t * h];
        for (li, layer) in self.layers.iter().enumerate() {
            for i in 0..t {
                rmsnorm(&x[i * h..(i + 1) * h], &layer.ln1, s.eps, &mut norm[i * h..(i + 1) * h]);
            }
            self.attention(layer, li, &norm, t, st, &mut delta);
            for (a, b) in x.iter_mut().zip(&delta) {
                *a += b;
            }
            for i in 0..t {
                rmsnorm(&x[i * h..(i + 1) * h], &layer.ln2, s.eps, &mut norm[i * h..(i + 1) * h]);
            }
            self.feed_forward(layer, li, &norm, t, st, &mut delta);
            for (a, b) in x.iter_mut().zip(&delta) {
                *a += b;
            }
        }
        st.pos += t;

        let last = &x[(t - 1) * h..];
        let mut nx = vec![0.0f32; h];
        rmsnorm(last, &self.out_norm, s.eps, &mut nx);
        let mut logits = vec![0.0f32; s.vocab];
        matvec(&self.lm_head, &nx, &mut logits);
        logits
    }

    fn attention(&self, layer: &Layer, li: usize, x: &[f32], t: usize, st: &mut State, out: &mut [f32]) {
        let s = &self.spec;
        let base = st.pos;
        // qd: per-head query/key width. vd: per-head value width.
        let (qd, vd, n_kv) = match &s.mla {
            Some(m) => (m.qk_nope + m.qk_rope, m.v_head, s.heads),
            None => (s.head_dim, s.head_dim, s.kv_heads),
        };
        let mut q = vec![0.0f32; t * s.heads * qd];

        match &layer.attn {
            Attn::Gqa(a) => {
                let (kd, vdim) = (n_kv * qd, n_kv * vd);
                let mut k = vec![0.0f32; t * kd];
                let mut v = vec![0.0f32; t * vdim];
                matmul(&a.q, x, &mut q);
                matmul(&a.k, x, &mut k);
                matmul(&a.v, x, &mut v);
                for (buf, b) in [(&mut q, &a.bias[0]), (&mut k, &a.bias[1]), (&mut v, &a.bias[2])] {
                    if let Some(b) = b {
                        for (i, val) in buf.iter_mut().enumerate() {
                            *val += b[i % b.len()];
                        }
                    }
                }
                for i in 0..t {
                    let p = base + i;
                    for (buf, heads, nrm) in [(&mut q, s.heads, &a.q_norm), (&mut k, n_kv, &a.k_norm)] {
                        let row = &mut buf[i * heads * qd..(i + 1) * heads * qd];
                        // The norm's width says which convention the checkpoint
                        // uses: one weight per head (Qwen3-style) or one across
                        // the whole projection (OLMoE-style).
                        if let Some(w) = nrm {
                            let span = if w.len() == row.len() { row.len() } else { qd };
                            let mut tmp = vec![0.0; span];
                            for chunk in row.chunks_mut(span) {
                                rmsnorm(chunk, w, s.eps, &mut tmp);
                                chunk.copy_from_slice(&tmp);
                            }
                        }
                        for hh in row.chunks_mut(qd) {
                            rope(hh, p, s.theta, s.rope_scale);
                        }
                    }
                    st.k[li][p * kd..(p + 1) * kd].copy_from_slice(&k[i * kd..(i + 1) * kd]);
                    st.v[li][p * vdim..(p + 1) * vdim].copy_from_slice(&v[i * vdim..(i + 1) * vdim]);
                }
            }
            Attn::Mla(a) => {
                let m = a.dims;
                // Queries, optionally through a low-rank bottleneck.
                if let (Some(qa), Some(qn)) = (&a.q_a, &a.q_a_norm) {
                    let mut c = vec![0.0f32; t * m.q_lora];
                    matmul(qa, x, &mut c);
                    let mut cn = vec![0.0f32; t * m.q_lora];
                    for i in 0..t {
                        rmsnorm(
                            &c[i * m.q_lora..(i + 1) * m.q_lora],
                            qn,
                            s.eps,
                            &mut cn[i * m.q_lora..(i + 1) * m.q_lora],
                        );
                    }
                    matmul(&a.q_b, &cn, &mut q);
                } else {
                    matmul(&a.q_b, x, &mut q);
                }
                // Compressed KV plus a decoupled rotary key shared by all heads.
                let ca = m.kv_lora + m.qk_rope;
                let mut c = vec![0.0f32; t * ca];
                matmul(&a.kv_a, x, &mut c);
                let mut cn = vec![0.0f32; t * m.kv_lora];
                for i in 0..t {
                    rmsnorm(
                        &c[i * ca..i * ca + m.kv_lora],
                        &a.kv_a_norm,
                        s.eps,
                        &mut cn[i * m.kv_lora..(i + 1) * m.kv_lora],
                    );
                }
                let kvw = m.qk_nope + m.v_head;
                let mut kv = vec![0.0f32; t * s.heads * kvw];
                matmul(&a.kv_b, &cn, &mut kv);
                let (kd, vdim) = (s.heads * qd, s.heads * vd);
                for i in 0..t {
                    let p = base + i;
                    let mut k_pe = c[i * ca + m.kv_lora..(i + 1) * ca].to_vec();
                    rope(&mut k_pe, p, s.theta, s.rope_scale);
                    for hi in 0..s.heads {
                        let qh = &mut q[i * s.heads * qd + hi * qd..i * s.heads * qd + (hi + 1) * qd];
                        rope(&mut qh[m.qk_nope..], p, s.theta, s.rope_scale);
                        let src = &kv[i * s.heads * kvw + hi * kvw..i * s.heads * kvw + (hi + 1) * kvw];
                        let kdst = &mut st.k[li][p * kd + hi * qd..p * kd + (hi + 1) * qd];
                        kdst[..m.qk_nope].copy_from_slice(&src[..m.qk_nope]);
                        kdst[m.qk_nope..].copy_from_slice(&k_pe);
                        st.v[li][p * vdim + hi * vd..p * vdim + (hi + 1) * vd].copy_from_slice(&src[m.qk_nope..]);
                    }
                }
            }
        }

        // Causal scaled dot-product attention, one task per (token, head).
        let (kd, vdim) = (n_kv * qd, n_kv * vd);
        let scale = 1.0 / (qd as f32).sqrt();
        let rep = s.heads / n_kv.max(1);
        let heads = s.heads;
        let ctx: Vec<f32> = (0..t * heads)
            .into_par_iter()
            .flat_map_iter(|idx| {
                let (i, hi) = (idx / heads, idx % heads);
                let p = base + i;
                let kv_h = if rep > 1 { hi / rep } else { hi.min(n_kv - 1) };
                let qv = &q[i * heads * qd + hi * qd..i * heads * qd + (hi + 1) * qd];
                let mut scores = Vec::with_capacity(p + 1);
                for j in 0..=p {
                    let kj = &st.k[li][j * kd + kv_h * qd..j * kd + (kv_h + 1) * qd];
                    scores.push(crate::quant::dot(qv, kj) * scale);
                }
                softmax(&mut scores);
                let mut acc = vec![0.0f32; vd];
                for (j, w) in scores.iter().enumerate() {
                    let vj = &st.v[li][j * vdim + kv_h * vd..j * vdim + (kv_h + 1) * vd];
                    for (a, b) in acc.iter_mut().zip(vj) {
                        *a += w * b;
                    }
                }
                acc.into_iter()
            })
            .collect();

        let o = match &layer.attn {
            Attn::Gqa(a) => &a.o,
            Attn::Mla(a) => &a.o,
        };
        matmul(o, &ctx, out);
    }

    fn feed_forward(&self, layer: &Layer, li: usize, x: &[f32], t: usize, st: &mut State, out: &mut [f32]) {
        let s = &self.spec;
        let h = s.hidden;
        match &layer.ffn {
            Ffn::Dense(mlp) => mlp.forward(x, t, out),
            Ffn::Moe { router, e_bias, experts, shared, shared_gate } => {
                out.iter_mut().for_each(|v| *v = 0.0);
                let n = experts.len();
                let mut logits = vec![0.0f32; t * n];
                matmul(router, x, &mut logits);

                // Route: score, optional group pruning, top-k, renormalise.
                let mut picks: Vec<Vec<(usize, f32)>> = Vec::with_capacity(t);
                for i in 0..t {
                    let sc = &mut logits[i * n..(i + 1) * n];
                    if s.sigmoid {
                        sc.iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp()));
                    } else {
                        softmax(sc);
                    }
                    let mut rank: Vec<f32> = match e_bias {
                        Some(b) => sc.iter().zip(b).map(|(v, b)| v + b).collect(),
                        None => sc.to_vec(),
                    };
                    if s.n_group > 1 && s.topk_group < s.n_group {
                        prune_groups(&mut rank, s.n_group, s.topk_group);
                    }
                    let mut idx: Vec<usize> = (0..n).collect();
                    idx.sort_unstable_by(|a, b| rank[*b].total_cmp(&rank[*a]));
                    idx.truncate(s.top_k.min(n));
                    let mut w: Vec<f32> = idx.iter().map(|e| sc[*e]).collect();
                    if s.norm_topk {
                        let z: f32 = w.iter().sum::<f32>().max(1e-20);
                        w.iter_mut().for_each(|v| *v /= z);
                    }
                    w.iter_mut().for_each(|v| *v *= s.routed_scale);
                    picks.push(idx.iter().copied().zip(w).collect());
                }

                // Invert the routing table so each selected expert is loaded once
                // per step and applied to all of its tokens at once.
                let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
                for (i, p) in picks.iter().enumerate() {
                    for (e, w) in p {
                        by_expert[*e].push((i, *w));
                    }
                }
                let prev = &st.prev[li];
                let (mut routed, mut reused, mut bytes) = (0u64, 0u64, 0u64);
                let mut hits: Vec<u32> = Vec::new();
                let parts: Vec<ExpertOutput> = by_expert
                    .iter()
                    .enumerate()
                    .filter(|(_, toks)| !toks.is_empty())
                    .map(|(e, toks)| {
                        routed += toks.len() as u64;
                        if prev.contains(&(e as u32)) {
                            reused += toks.len() as u64;
                        }
                        bytes += experts[e].bytes();
                        hits.push(e as u32);
                        (e, toks)
                    })
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|(e, toks)| {
                        let mut gathered = vec![0.0f32; toks.len() * h];
                        for (slot, (i, _)) in toks.iter().enumerate() {
                            gathered[slot * h..(slot + 1) * h].copy_from_slice(&x[i * h..(i + 1) * h]);
                        }
                        let mut y = vec![0.0f32; toks.len() * h];
                        experts[e].forward(&gathered, toks.len(), &mut y);
                        (toks.clone(), y)
                    })
                    .collect();
                for (toks, y) in parts {
                    for (slot, (i, w)) in toks.iter().enumerate() {
                        for (a, b) in out[i * h..(i + 1) * h].iter_mut().zip(&y[slot * h..(slot + 1) * h]) {
                            *a += w * b;
                        }
                    }
                }
                if let Some(tr) = st.trace.as_mut() {
                    let base = st.pos;
                    tr.routes.extend(picks.iter().enumerate().map(|(i, p)| Route {
                        pos: (base + i) as u32,
                        layer: li as u32,
                        experts: p.iter().map(|(e, w)| (*e as u32, *w)).collect(),
                    }));
                }
                st.prev[li] = hits;
                st.stats.routed.fetch_add(routed, Ordering::Relaxed);
                st.stats.reused.fetch_add(reused, Ordering::Relaxed);
                st.stats.expert_bytes.fetch_add(bytes, Ordering::Relaxed);

                if let Some(sh) = shared {
                    let mut y = vec![0.0f32; t * h];
                    sh.forward(x, t, &mut y);
                    if let Some(g) = shared_gate {
                        let mut gate = vec![0.0f32; t];
                        matmul(g, x, &mut gate);
                        for i in 0..t {
                            let s = 1.0 / (1.0 + (-gate[i]).exp());
                            y[i * h..(i + 1) * h].iter_mut().for_each(|v| *v *= s);
                        }
                    }
                    for (a, b) in out.iter_mut().zip(&y) {
                        *a += b;
                    }
                }
            }
        }
    }

    /// Weight footprint by role, for `moe info`.
    pub fn footprint(&self) -> (u64, u64) {
        let total = self.store.bytes();
        let experts = self.expert_bytes();
        (total - experts, experts)
    }

    pub fn dtypes(&self) -> Vec<(Dt, u64)> {
        let mut acc: Vec<(Dt, u64)> = Vec::new();
        for n in self.store.names() {
            if let Some(t) = self.store.get(n) {
                match acc.iter_mut().find(|(d, _)| *d == t.dt) {
                    Some((_, b)) => *b += t.data.len() as u64,
                    None => acc.push((t.dt, t.data.len() as u64)),
                }
            }
        }
        acc.sort_by_key(|(_, b)| std::cmp::Reverse(*b));
        acc
    }
}

/// Group-limited routing: keep only the `keep` groups with the strongest pair of
/// experts, masking the rest out of the top-k selection.
fn prune_groups(rank: &mut [f32], groups: usize, keep: usize) {
    let per = rank.len() / groups;
    let mut score: Vec<(usize, f32)> = (0..groups)
        .map(|g| {
            let mut s: Vec<f32> = rank[g * per..(g + 1) * per].to_vec();
            s.sort_unstable_by(|a, b| b.total_cmp(a));
            (g, s.iter().take(2).sum())
        })
        .collect();
    score.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    for (g, _) in score.into_iter().skip(keep) {
        rank[g * per..(g + 1) * per].iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
    }
}
