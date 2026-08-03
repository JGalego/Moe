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


def rope(x, pos, theta, scale):
    half = len(x) // 2
    out = list(x)
    for i in range(half):
        freq = theta ** (-2.0 * i / len(x))
        a = pos * freq / scale
        c, s = math.cos(a), math.sin(a)
        out[i] = x[i] * c - x[i + half] * s
        out[i + half] = x[i] * s + x[i + half] * c
    return out


def mlp(w, prefix, x):
    g = matvec(w[prefix + "gate_proj.weight"], x)
    u = matvec(w[prefix + "up_proj.weight"], x)
    return matvec(w[prefix + "down_proj.weight"], [silu(a) * b for a, b in zip(g, u)])


# ---------------------------------------------------------------- fixtures


def build_gqa(rng):
    """Grouped-query attention, qk-norm, softmax routing, no dense layers."""
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
        w[p + "self_attn.q_norm.weight"] = rng.vec(hd, 0.2, 1.0)
        w[p + "self_attn.k_norm.weight"] = rng.vec(hd, 0.2, 1.0)
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
    eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
    h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    is_mla = "kv_lora_rank" in cfg
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
                k_pe = rope(ca[kvl:], pos, theta, 1.0)
                kv = matvec(w[p + "self_attn.kv_b_proj.weight"], c_kv)
                qs, ks, vs = [], [], []
                for hi in range(nh):
                    qh = qflat[hi * (qk_n + qk_r):(hi + 1) * (qk_n + qk_r)]
                    qs.append(qh[:qk_n] + rope(qh[qk_n:], pos, theta, 1.0))
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
                qs, ks, vs = [], [], []
                for hi in range(nh):
                    qh = qflat[hi * hd:(hi + 1) * hd]
                    if qn:
                        qh = rmsnorm(qh, qn, eps)
                    qs.append(rope(qh, pos, theta, 1.0))
                for hi in range(nkv):
                    kh = kflat[hi * hd:(hi + 1) * hd]
                    if kn:
                        kh = rmsnorm(kh, kn, eps)
                    ks.append(rope(kh, pos, theta, 1.0))
                    vs.append(vflat[hi * hd:(hi + 1) * hd])
                qd, vd = hd, hd

            cache[l][0].append(ks)
            cache[l][1].append(vs)
            rep = nh // nkv
            ctx = []
            for hi in range(nh):
                kvh = hi // rep
                scores = [
                    sum(a * b for a, b in zip(qs[hi], cache[l][0][j][kvh])) / math.sqrt(qd)
                    for j in range(pos + 1)
                ]
                scores = softmax(scores)
                acc = [0.0] * vd
                for j, s in enumerate(scores):
                    for d in range(vd):
                        acc[d] += s * cache[l][1][j][kvh][d]
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
    for name, build in (("gqa", build_gqa), ("mla", build_mla)):
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
