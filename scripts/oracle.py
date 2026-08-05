#!/usr/bin/env python3
"""Generate tiny checkpoints and reference logits for the engine tests.

Pure standard library on purpose: the reference forward pass below is written
from the model definitions alone, so it is an independent check on the Rust
implementation rather than a rewrite of it. Run:

    python3 scripts/oracle.py <out-dir>

which writes <out-dir>/{gqa,mla}/ each containing config.json,
model.safetensors and expected.json.
"""

import json
import math
import os
import struct
import sys

# ---------------------------------------------------------------- utilities


class Rand:
    """Deterministic LCG, so fixtures are identical everywhere."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 33) / float(1 << 31)) - 1.0  # [-1, 1)

    def mat(self, rows, cols, scale=0.5):
        return [[self.next() * scale for _ in range(cols)] for _ in range(rows)]

    def vec(self, n, scale=0.5, bias=0.0):
        return [self.next() * scale + bias for _ in range(n)]


def matvec(w, x):
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]


def rmsnorm(x, w, eps):
    s = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * s * g for v, g in zip(x, w)]


def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    z = sum(e)
    return [v / z for v in e]


def silu(v):
    return v / (1.0 + math.exp(-v))


def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


def rope(x, pos, inv):
    half = len(x) // 2
    out = list(x)
    for i in range(half):
        a = pos * inv[i]
        c, s = math.cos(a), math.sin(a)
        out[i] = x[i] * c - x[i + half] * s
        out[i + half] = x[i] * s + x[i + half] * c
    return out


def inv_freqs(cfg, dim):
    """Inverse rotary frequencies, written from the scaling papers.

    Position interpolation divides the frequencies; NTK raises the base instead;
    Llama 3 does it piecewise by wavelength; YaRN ramps between interpolation and
    extrapolation across the frequency band whose wavelengths bracket the trained
    context. Each is stated here directly rather than derived from the engine.
    """
    theta = cfg["rope_theta"]
    base = [theta ** (-2.0 * i / dim) for i in range(dim // 2)]
    sc = cfg.get("rope_scaling") or {}
    kind = sc.get("rope_type", sc.get("type", ""))
    s = sc.get("factor", 1.0)
    if not kind or s <= 1.0:
        return base
    ctx = sc.get("original_max_position_embeddings", cfg["max_position_embeddings"])

    if kind == "linear":
        return [f / s for f in base]
    if kind in ("dynamic", "ntk"):
        t2 = theta * s ** (dim / (dim - 2))
        return [t2 ** (-2.0 * i / dim) for i in range(dim // 2)]
    if kind == "llama3":
        lof, hif = sc.get("low_freq_factor", 1.0), sc.get("high_freq_factor", 4.0)
        lo_wave, hi_wave = ctx / lof, ctx / hif
        out = []
        for f in base:
            wave = 2 * math.pi / f
            if wave > lo_wave:
                out.append(f / s)
            elif wave < hi_wave:
                out.append(f)
            else:
                t = (ctx / wave - lof) / (hif - lof)
                out.append((1 - t) * f / s + t * f)
        return out
    if kind == "yarn":
        bf, bs = sc.get("beta_fast", 32.0), sc.get("beta_slow", 1.0)
        ln_base = math.log(theta)
        bound = lambda b: dim * math.log(ctx / (b * 2 * math.pi)) / (2 * ln_base)
        low, high = math.floor(bound(bf)), math.ceil(bound(bs))
        span = max(high - low, 1e-3)
        out = []
        for i, f in enumerate(base):
            ramp = min(1.0, max(0.0, (i - low) / span))
            out.append((1 - ramp) * (f / s) + ramp * f)
        return out
    return base


def attn_scale(cfg):
    """YaRN's correction to the attention scale; 1.0 for everything else."""
    sc = cfg.get("rope_scaling") or {}
    if sc.get("rope_type", sc.get("type", "")) != "yarn":
        return 1.0
    s = sc.get("factor", 1.0)
    if s <= 1.0:
        return 1.0
    g = lambda m: 0.1 * m * math.log(s) + 1.0
    ms, msa = sc.get("mscale", 0.0), sc.get("mscale_all_dim", 0.0)
    if ms or msa:
        return g(ms) / max(g(msa), 1e-6) * sc.get("attention_factor", 1.0)
    return g(1.0) * sc.get("attention_factor", 1.0)


def windows(cfg, layers):
    """Attention window per layer, across the four config conventions."""
    w = cfg.get("sliding_window") or 0
    if not w:
        return [None] * layers
    types = cfg.get("layer_types")
    if types:
        local = ("sliding_attention", "local_attention")
        return [w if (types[l] if l < len(types) else "") in local else None for l in range(layers)]
    pat = cfg.get("sliding_window_pattern", 0)
    if pat and pat > 1:
        return [None if (l + 1) % pat == 0 else w for l in range(layers)]
    if cfg.get("use_sliding_window") is False:
        return [None] * layers
    frm = cfg.get("max_window_layers", 0)
    return [w if l >= frm else None for l in range(layers)]


def mlp(w, prefix, x):
    g = matvec(w[prefix + "gate_proj.weight"], x)
    u = matvec(w[prefix + "up_proj.weight"], x)
    return matvec(w[prefix + "down_proj.weight"], [silu(a) * b for a, b in zip(g, u)])


# ---------------------------------------------------------------- fixtures


def build_gqa(rng, full_norm=False, long=False):
    """Grouped-query attention, qk-norm, softmax routing, no dense layers.

    `full_norm` switches qk-norm from one weight per head (Qwen3-style) to one
    across the whole projection (OLMoE-style). Only the weight's width says
    which is meant, so both have to be exercised.

    `long` adds the two things a stretched context needs: YaRN rope scaling, and
    a sliding window on one layer but not the other, so a checkpoint that mixes
    local and global attention is covered.
    """
    cfg = dict(
        architectures=["MoeForCausalLM"],
        model_type="moe",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=48,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        norm_topk_prob=True,
        tie_word_embeddings=False,
    )
    if long:
        cfg["rope_scaling"] = dict(
            rope_type="yarn", factor=4.0, original_max_position_embeddings=32,
            beta_fast=32.0, beta_slow=1.0,
        )
        cfg["max_position_embeddings"] = 128
        cfg["sliding_window"] = 3
        cfg["layer_types"] = ["sliding_attention", "full_attention"]
    h, v = cfg["hidden_size"], cfg["vocab_size"]
    hd, nh, nkv = cfg["head_dim"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    w = {
        "model.embed_tokens.weight": rng.mat(v, h),
        "model.norm.weight": rng.vec(h, 0.2, 1.0),
        "lm_head.weight": rng.mat(v, h),
    }
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        w[p + "input_layernorm.weight"] = rng.vec(h, 0.2, 1.0)
        w[p + "post_attention_layernorm.weight"] = rng.vec(h, 0.2, 1.0)
        w[p + "self_attn.q_proj.weight"] = rng.mat(nh * hd, h)
        w[p + "self_attn.k_proj.weight"] = rng.mat(nkv * hd, h)
        w[p + "self_attn.v_proj.weight"] = rng.mat(nkv * hd, h)
        w[p + "self_attn.o_proj.weight"] = rng.mat(h, nh * hd)
        w[p + "self_attn.q_norm.weight"] = rng.vec(nh * hd if full_norm else hd, 0.2, 1.0)
        w[p + "self_attn.k_norm.weight"] = rng.vec(nkv * hd if full_norm else hd, 0.2, 1.0)
        w[p + "mlp.gate.weight"] = rng.mat(cfg["num_experts"], h)
        for e in range(cfg["num_experts"]):
            q = p + f"mlp.experts.{e}."
            w[q + "gate_proj.weight"] = rng.mat(cfg["moe_intermediate_size"], h)
            w[q + "up_proj.weight"] = rng.mat(cfg["moe_intermediate_size"], h)
            w[q + "down_proj.weight"] = rng.mat(h, cfg["moe_intermediate_size"])
    return cfg, w


def build_mla(rng):
    """Latent attention, sigmoid + group-limited routing, shared expert,
    dense first layer."""
    cfg = dict(
        architectures=["MoeLatentForCausalLM"],
        model_type="moe_latent",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=48,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=128,
        q_lora_rank=16,
        kv_lora_rank=12,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        intermediate_size=24,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        n_shared_experts=1,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        n_group=2,
        topk_group=1,
        tie_word_embeddings=False,
    )
    h, v, nh = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"]
    qk = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    w = {
        "model.embed_tokens.weight": rng.mat(v, h),
        "model.norm.weight": rng.vec(h, 0.2, 1.0),
        "lm_head.weight": rng.mat(v, h),
    }
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        w[p + "input_layernorm.weight"] = rng.vec(h, 0.2, 1.0)
        w[p + "post_attention_layernorm.weight"] = rng.vec(h, 0.2, 1.0)
        w[p + "self_attn.q_a_proj.weight"] = rng.mat(cfg["q_lora_rank"], h)
        w[p + "self_attn.q_a_layernorm.weight"] = rng.vec(cfg["q_lora_rank"], 0.2, 1.0)
        w[p + "self_attn.q_b_proj.weight"] = rng.mat(nh * qk, cfg["q_lora_rank"])
        w[p + "self_attn.kv_a_proj_with_mqa.weight"] = rng.mat(
            cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"], h
        )
        w[p + "self_attn.kv_a_layernorm.weight"] = rng.vec(cfg["kv_lora_rank"], 0.2, 1.0)
        w[p + "self_attn.kv_b_proj.weight"] = rng.mat(
            nh * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), cfg["kv_lora_rank"]
        )
        w[p + "self_attn.o_proj.weight"] = rng.mat(h, nh * cfg["v_head_dim"])
        if l < cfg["first_k_dense_replace"]:
            w[p + "mlp.gate_proj.weight"] = rng.mat(cfg["intermediate_size"], h)
            w[p + "mlp.up_proj.weight"] = rng.mat(cfg["intermediate_size"], h)
            w[p + "mlp.down_proj.weight"] = rng.mat(h, cfg["intermediate_size"])
            continue
        w[p + "mlp.gate.weight"] = rng.mat(cfg["n_routed_experts"], h)
        w[p + "mlp.gate.e_score_correction_bias"] = rng.vec(cfg["n_routed_experts"], 0.1)
        for e in range(cfg["n_routed_experts"]):
            q = p + f"mlp.experts.{e}."
            w[q + "gate_proj.weight"] = rng.mat(cfg["moe_intermediate_size"], h)
            w[q + "up_proj.weight"] = rng.mat(cfg["moe_intermediate_size"], h)
            w[q + "down_proj.weight"] = rng.mat(h, cfg["moe_intermediate_size"])
        s = p + "mlp.shared_experts."
        w[s + "gate_proj.weight"] = rng.mat(cfg["moe_intermediate_size"], h)
        w[s + "up_proj.weight"] = rng.mat(cfg["moe_intermediate_size"], h)
        w[s + "down_proj.weight"] = rng.mat(h, cfg["moe_intermediate_size"])
    return cfg, w


# ---------------------------------------------------------------- reference


def route(cfg, w, p, x, experts):
    logits = matvec(w[p + "mlp.gate.weight"], x)
    if cfg.get("scoring_func") == "sigmoid":
        score = [sigmoid(v) for v in logits]
    else:
        score = softmax(logits)
    bias = w.get(p + "mlp.gate.e_score_correction_bias")
    rank = [s + b for s, b in zip(score, bias)] if bias else list(score)
    groups, keep = cfg.get("n_group", 1), cfg.get("topk_group", 1)
    if groups > 1 and keep < groups:
        per = experts // groups
        gs = sorted(
            ((g, sum(sorted(rank[g * per:(g + 1) * per], reverse=True)[:2])) for g in range(groups)),
            key=lambda t: -t[1],
        )
        for g, _ in gs[keep:]:
            for i in range(g * per, (g + 1) * per):
                rank[i] = -1e30
    order = sorted(range(experts), key=lambda e: -rank[e])[: cfg["num_experts_per_tok"]]
    weights = [score[e] for e in order]
    if cfg.get("norm_topk_prob", True):
        z = sum(weights) or 1e-20
        weights = [v / z for v in weights]
    scale = cfg.get("routed_scaling_factor", 1.0)
    return order, [v * scale for v in weights]


def forward(cfg, w, tokens):
    """Returns logits after each prefix of `tokens`."""
    eps = cfg["rms_norm_eps"]
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    is_mla = "kv_lora_rank" in cfg
    rot = cfg["qk_rope_head_dim"] if is_mla else cfg["head_dim"]
    inv = inv_freqs(cfg, rot)
    ascale = attn_scale(cfg)
    wins = windows(cfg, cfg["num_hidden_layers"])
    experts = cfg.get("num_experts", cfg.get("n_routed_experts", 0))
    cache = [([], []) for _ in range(cfg["num_hidden_layers"])]
    out = []

    for pos, tok in enumerate(tokens):
        x = list(w["model.embed_tokens.weight"][tok])
        for l in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{l}."
            n = rmsnorm(x, w[p + "input_layernorm.weight"], eps)

            if is_mla:
                qk_n, qk_r = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"]
                vh, kvl = cfg["v_head_dim"], cfg["kv_lora_rank"]
                c = matvec(w[p + "self_attn.q_a_proj.weight"], n)
                c = rmsnorm(c, w[p + "self_attn.q_a_layernorm.weight"], eps)
                qflat = matvec(w[p + "self_attn.q_b_proj.weight"], c)
                ca = matvec(w[p + "self_attn.kv_a_proj_with_mqa.weight"], n)
                c_kv = rmsnorm(ca[:kvl], w[p + "self_attn.kv_a_layernorm.weight"], eps)
                k_pe = rope(ca[kvl:], pos, inv)
                kv = matvec(w[p + "self_attn.kv_b_proj.weight"], c_kv)
                qs, ks, vs = [], [], []
                for hi in range(nh):
                    qh = qflat[hi * (qk_n + qk_r):(hi + 1) * (qk_n + qk_r)]
                    qs.append(qh[:qk_n] + rope(qh[qk_n:], pos, inv))
                    src = kv[hi * (qk_n + vh):(hi + 1) * (qk_n + vh)]
                    ks.append(src[:qk_n] + k_pe)
                    vs.append(src[qk_n:])
                qd, vd, nkv = qk_n + qk_r, vh, nh
            else:
                hd, nkv = cfg["head_dim"], cfg["num_key_value_heads"]
                qflat = matvec(w[p + "self_attn.q_proj.weight"], n)
                kflat = matvec(w[p + "self_attn.k_proj.weight"], n)
                vflat = matvec(w[p + "self_attn.v_proj.weight"], n)
                qn = w.get(p + "self_attn.q_norm.weight")
                kn = w.get(p + "self_attn.k_norm.weight")
                # A norm as wide as the whole projection applies once, before
                # the heads are split apart.
                if qn and len(qn) == len(qflat):
                    qflat, qn = rmsnorm(qflat, qn, eps), None
                if kn and len(kn) == len(kflat):
                    kflat, kn = rmsnorm(kflat, kn, eps), None
                qs, ks, vs = [], [], []
                for hi in range(nh):
                    qh = qflat[hi * hd:(hi + 1) * hd]
                    if qn:
                        qh = rmsnorm(qh, qn, eps)
                    qs.append(rope(qh, pos, inv))
                for hi in range(nkv):
                    kh = kflat[hi * hd:(hi + 1) * hd]
                    if kn:
                        kh = rmsnorm(kh, kn, eps)
                    ks.append(rope(kh, pos, inv))
                    vs.append(vflat[hi * hd:(hi + 1) * hd])
                qd, vd = hd, hd

            cache[l][0].append(ks)
            cache[l][1].append(vs)
            rep = nh // nkv
            ctx = []
            lo = max(0, pos + 1 - wins[l]) if wins[l] else 0
            for hi in range(nh):
                kvh = hi // rep
                scores = [
                    sum(a * b for a, b in zip(qs[hi], cache[l][0][j][kvh])) * ascale / math.sqrt(qd)
                    for j in range(lo, pos + 1)
                ]
                scores = softmax(scores)
                acc = [0.0] * vd
                for k, s in enumerate(scores):
                    for d in range(vd):
                        acc[d] += s * cache[l][1][lo + k][kvh][d]
                ctx += acc
            x = [a + b for a, b in zip(x, matvec(w[p + "self_attn.o_proj.weight"], ctx))]

            n = rmsnorm(x, w[p + "post_attention_layernorm.weight"], eps)
            if p + "mlp.gate.weight" in w:
                order, weights = route(cfg, w, p, n, experts)
                y = [0.0] * h
                for e, wt in zip(order, weights):
                    ye = mlp(w, p + f"mlp.experts.{e}.", n)
                    y = [a + wt * b for a, b in zip(y, ye)]
                if p + "mlp.shared_experts.gate_proj.weight" in w:
                    ys = mlp(w, p + "mlp.shared_experts.", n)
                    y = [a + b for a, b in zip(y, ys)]
            else:
                y = mlp(w, p + "mlp.", n)
            x = [a + b for a, b in zip(x, y)]

        nx = rmsnorm(x, w["model.norm.weight"], eps)
        out.append(matvec(w["lm_head.weight"], nx))
    return out


# ---------------------------------------------------------------- emit


def write_safetensors(path, w):
    header, blob, off = {}, bytearray(), 0
    for name in sorted(w):
        t = w[name]
        flat = [v for row in t for v in row] if isinstance(t[0], list) else list(t)
        shape = [len(t), len(t[0])] if isinstance(t[0], list) else [len(t)]
        data = struct.pack("<%df" % len(flat), *flat)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
    builders = (
        ("gqa", build_gqa),
        ("gqa_fullnorm", lambda rng: build_gqa(rng, full_norm=True)),
        ("gqa_long", lambda rng: build_gqa(rng, long=True)),
        ("mla", build_mla),
    )
    for name, build in builders:
        cfg, w = build(Rand(0x5EED + len(name)))
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        tokens = [3, 11, 5, 40, 7, 1]
        logits = forward(cfg, w, tokens)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        write_safetensors(os.path.join(d, "model.safetensors"), w)
        with open(os.path.join(d, "expected.json"), "w") as f:
            json.dump({"tokens": tokens, "logits": logits}, f)
        print(f"{d}: {len(w)} tensors, {len(tokens)} reference steps")


if __name__ == "__main__":
    main()
