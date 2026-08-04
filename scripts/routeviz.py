#!/usr/bin/env python3
"""Render a routing trace as an SVG heatmap.

    moe run <model> -p "..." --trace routes.jsonl
    python3 scripts/routeviz.py routes.jsonl -o routes.svg

Given two traces it renders their difference instead, which is how you see
whether a model routes prose and code to different experts:

    python3 scripts/routeviz.py prose.jsonl code.jsonl -o diff.svg

Standard library only. Colour follows the job: one hue with monotone lightness
for magnitude, two hues either side of a neutral grey for polarity — computed in
OKLab so the steps are perceptually even rather than eyeballed.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

SURFACE = "#0d1117"
INK = "#d6dde8"
INK_DIM = "#8892a2"
GRID = "#1c2430"
AMBER = "#e0973a"  # warm pole / sequential hue
TEAL = "#17b3a3"  # cool pole
NEUTRAL = "#323232"  # diverging midpoint: achromatic by construction, never a hue

# ---------------------------------------------------------------- colour


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def hex_to_oklab(h):
    r, g, b = (_srgb_to_linear(int(h[i : i + 2], 16) / 255) for i in (1, 3, 5))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklab_to_hex(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    out = ""
    for c in (r, g, bb):
        v = round(255 * min(1.0, max(0.0, _linear_to_srgb(min(1.0, max(0.0, c))))))
        out += f"{v:02x}"
    return "#" + out


def ramp(to_hex, steps, floor=0.04):
    """A sequential ramp from near the surface up to `to_hex`.

    On a dark surface the anchor flips: "near zero" is the dark end. Lightness
    rises monotonically by construction, which is the check that matters for a
    sequential scale.
    """
    lo, hi = hex_to_oklab(SURFACE), hex_to_oklab(to_hex)
    out = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        # Ease the low end so the first non-zero cell is already visible.
        e = floor + (1 - floor) * t
        out.append(oklab_to_hex(tuple(lo[c] + (hi[c] - lo[c]) * e for c in range(3))))
    return out


def diverging(steps):
    """Two hues either side of a neutral grey, equal steps per arm.

    Each arm runs from the midpoint outward, not from the surface: starting at
    the surface would make the steps nearest the middle darker than the middle
    itself, so the ramp would dip instead of rising away from zero.
    """
    half = steps // 2
    lo = hex_to_oklab(NEUTRAL)

    def arm(to_hex):
        hi = hex_to_oklab(to_hex)
        return [oklab_to_hex(tuple(lo[c] + (hi[c] - lo[c]) * (i / half) for c in range(3))) for i in range(half + 1)]

    cool, warm = arm(TEAL), arm(AMBER)
    return cool[::-1][:-1] + [NEUTRAL] + warm[1:]


def midpoint_is_neutral(colours):
    """The "nothing" step must be the least chromatic, or zero reads as a value."""
    chroma = [sum(c * c for c in hex_to_oklab(h)[1:]) for h in colours]
    return chroma.index(min(chroma)) == len(colours) // 2


def lightness_is_monotone(colours):
    ls = [hex_to_oklab(c)[0] for c in colours]
    return all(b >= a - 1e-9 for a, b in zip(ls, ls[1:]))


# ---------------------------------------------------------------- data


def load(path):
    """-> (share[(layer, expert)], layers, experts, tokens, records, header)"""
    count = defaultdict(int)
    layers, experts, positions = set(), set(), set()
    header, n = {}, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "pos" not in r:  # the self-describing header line
                header = r
                continue
            layers.add(r["layer"])
            positions.add(r["pos"])
            for e, _w in r["experts"]:
                count[(r["layer"], e)] += 1
                experts.add(e)
            n += 1
    tokens = len(positions)
    share = {k: v / max(1, tokens) for k, v in count.items()}
    n_experts = header.get("experts") or (max(experts) + 1 if experts else 0)
    return share, sorted(layers), n_experts, tokens, n, header


# ---------------------------------------------------------------- render


def svg(share, layers, n_experts, title, subtitle, colours, vmax, vmin=0.0, legend_labels=None):
    cell, gap = 13, 2
    pad_l, pad_t, pad_r, pad_b = 62, 62, 20, 64
    w = pad_l + n_experts * cell + pad_r
    h = pad_t + len(layers) * cell + pad_b
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="\'SF Mono\',Menlo,Consolas,monospace">',
        f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>',
        f'<text x="{pad_l}" y="26" fill="{INK}" font-size="13" font-weight="600">{title}</text>',
        f'<text x="{pad_l}" y="44" fill="{INK_DIM}" font-size="11">{subtitle}</text>',
    ]

    def colour_of(v):
        if vmax == vmin:
            return colours[len(colours) // 2]
        t = (v - vmin) / (vmax - vmin)
        return colours[min(len(colours) - 1, max(0, int(t * (len(colours) - 1) + 0.5)))]

    for row, layer in enumerate(layers):
        y = pad_t + row * cell
        if layer % 4 == 0:
            o.append(
                f'<text x="{pad_l - 8}" y="{y + cell - 4}" fill="{INK_DIM}" font-size="9.5" '
                f'text-anchor="end">L{layer}</text>'
            )
        for e in range(n_experts):
            v = share.get((layer, e), 0.0)
            x = pad_l + e * cell
            # A 2px surface gap keeps adjacent cells from reading as one block.
            o.append(
                f'<rect x="{x}" y="{y}" width="{cell - gap}" height="{cell - gap}" rx="2" '
                f'fill="{colour_of(v)}"><title>layer {layer}, expert {e}: {v:+.1%}</title></rect>'
            )
    for e in range(0, n_experts, 8):
        o.append(
            f'<text x="{pad_l + e * cell + (cell - gap) / 2}" y="{pad_t - 8}" fill="{INK_DIM}" '
            f'font-size="9.5" text-anchor="middle">{e}</text>'
        )
    o.append(
        f'<text x="{pad_l + n_experts * cell / 2}" y="{h - 34}" fill="{INK_DIM}" font-size="10" '
        f'text-anchor="middle">expert</text>'
    )

    # Scale legend: a sequential or diverging ramp is unreadable without one.
    lw, lx, ly = 168, pad_l, h - 22
    step = lw / len(colours)
    for i, c in enumerate(colours):
        o.append(f'<rect x="{lx + i * step:.2f}" y="{ly}" width="{step + 0.6:.2f}" height="9" fill="{c}"/>')
    lo_lab, hi_lab = legend_labels or (f"{vmin:.0%}", f"{vmax:.0%}")
    o.append(f'<text x="{lx - 6}" y="{ly + 8}" fill="{INK_DIM}" font-size="9.5" text-anchor="end">{lo_lab}</text>')
    o.append(f'<text x="{lx + lw + 6}" y="{ly + 8}" fill="{INK_DIM}" font-size="9.5">{hi_lab}</text>')
    o.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="9" fill="none" stroke="{GRID}"/>')
    o.append("</svg>")
    return "\n".join(o)


def summarise(share, layers, n_experts, tokens, label):
    """The table view: the same numbers, readable without the picture."""
    print(f"{label}: {tokens} tokens, {len(layers)} routed layers, {n_experts} experts")
    for layer in layers:
        row = sorted(((v, e) for (l, e), v in share.items() if l == layer), reverse=True)[:5]
        top = "  ".join(f"e{e}:{v:.0%}" for v, e in row)
        print(f"  L{layer:<3} {top}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("other", nargs="?", help="second trace: render the difference")
    ap.add_argument("-o", "--out", default="routes.svg")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    share, layers, n_experts, tokens, records, head = load(args.trace)
    if not records:
        sys.exit(f"{args.trace}: no routing records (a dense model traces nothing)")

    model_note = f" — {head['model']}" if head.get("model") else ""

    if args.other:
        other, l2, e2, tokens2, _, _ = load(args.other)
        layers = sorted(set(layers) | set(l2))
        n_experts = max(n_experts, e2)
        diff = {}
        for layer in layers:
            for e in range(n_experts):
                d = share.get((layer, e), 0.0) - other.get((layer, e), 0.0)
                if d:
                    diff[(layer, e)] = d
        span = max((abs(v) for v in diff.values()), default=1.0)
        colours = diverging(15)
        mid = len(colours) // 2
        assert lightness_is_monotone(colours[: mid + 1][::-1]), "cool arm dips"
        assert lightness_is_monotone(colours[mid:]), "warm arm dips"
        assert midpoint_is_neutral(colours), "diverging midpoint is not the least chromatic step"
        doc = svg(
            diff,
            layers,
            n_experts,
            args.title or f"expert routing — difference{model_note}",
            f"{os.path.basename(args.trace)} ({tokens} tokens) vs "
            f"{os.path.basename(args.other)} ({tokens2} tokens); "
            f"amber favours the first, teal the second",
            colours,
            vmax=span,
            vmin=-span,
            legend_labels=(f"-{span:.0%}", f"+{span:.0%}"),
        )
        summarise(diff, layers, n_experts, tokens, "difference")
    else:
        colours = ramp(AMBER, 12)
        assert lightness_is_monotone(colours), "sequential ramp is not monotone in lightness"
        vmax = max(share.values(), default=1.0)
        doc = svg(
            share,
            layers,
            n_experts,
            args.title or f"expert routing{model_note}",
            f"{tokens} tokens x {len(layers)} layers, top-{head.get('top_k', '?')} of {n_experts}; "
            f"share of tokens selecting each expert",
            colours,
            vmax=vmax,
        )
        summarise(share, layers, n_experts, tokens, args.trace)

    with open(args.out, "w") as f:
        f.write(doc)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
