#!/usr/bin/env python3
"""Compare `moe tokenize` against the reference tokenizers library.

    pip install tokenizers
    python3 scripts/tokcheck.py path/to/tokenizer.json [--cases N]

Exits non-zero on any mismatch. Text is drawn from a fixed set of awkward cases
plus generated ones (digit runs, whitespace, mixed scripts, code) so the
pre-tokenizer boundaries get exercised, not just the merge table.
"""

import argparse
import random
import subprocess
import sys

from tokenizers import Tokenizer

FIXED = [
    "The capital of France is Paris.",
    "hello world",
    "def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n",
    "  leading and trailing   ",
    "Numbers: 1234567 and 42, plus 3.14159",
    "Unicode: café, naïve, 日本語テキスト, emoji 🚀 done",
    "don't can't it's I'll we've they're DON'T IT'S",
    "<|im_start|>user\nHi there<|im_end|>\n",
    "Mixed_CASE-punct!?;: [brackets] {braces} (parens)",
    "a\tb\nc\r\nd",
    "Tab\tand multiple    spaces",
    "trailing newline\n\n\n",
    "\n\nleading newlines",
    "   ",
    "\t\t\t",
    "",
    "a",
    " ",
    "…ellipsis—dashes«guillemets»",
    "https://example.com/path?q=1&r=2#frag",
    "snake_case camelCase PascalCase SCREAMING_SNAKE kebab-case",
    "0x1F 0b1010 1e-9 -273.15 1,000,000",
    "Здравствуй мир! שלום עולם! مرحبا بالعالم!",
    "🚀🌍🎉 back-to-back emoji",
    "x" * 200,
    "word " * 60,
]

WORDS = "the quick brown fox jumps over lazy dog data model token expert route".split()
CHARS = " \t\n.,;:!?'\"()[]{}<>/\\|-_=+*&^%$#@~`abcXYZ0189éüñ日本🚀"


def generated(n, rng):
    out = []
    for _ in range(n):
        kind = rng.randrange(4)
        if kind == 0:
            out.append(" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 12))))
        elif kind == 1:
            out.append("".join(rng.choice(CHARS) for _ in range(rng.randint(1, 60))))
        elif kind == 2:
            out.append("".join(str(rng.randrange(10)) for _ in range(rng.randint(1, 12))))
        else:
            out.append(
                rng.choice(["  ", "\n", "\t", " "]).join(
                    rng.choice(WORDS + ["123", "!!", "_x", "'s"]) for _ in range(rng.randint(2, 8))
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--bin", default="./target/release/moe")
    args = ap.parse_args()

    tk = Tokenizer.from_file(args.tokenizer)
    cases = FIXED + generated(args.cases, random.Random(7))
    bad = 0
    for text in cases:
        want = tk.encode(text, add_special_tokens=False).ids
        proc = subprocess.run(
            [args.bin, "tokenize", args.tokenizer, "-p", text], capture_output=True, text=True
        )
        got = [int(x) for x in proc.stdout.strip().split(",") if x]
        if got != want:
            bad += 1
            if bad <= 10:
                print(f"ENCODE {text!r}\n  want {want}\n  got  {got}")
            continue
        # Decoding matters just as much: added tokens hold raw text and must not
        # be pushed back through the byte-level map.
        want_text = tk.decode(want)
        # Bytes, not text: capturing as text would silently fold \r\n into \n.
        proc = subprocess.run(
            [args.bin, "tokenize", args.tokenizer, "--decode", ",".join(map(str, want))],
            capture_output=True,
        )
        got_text = proc.stdout.decode("utf-8", "replace")
        if got_text != want_text:
            bad += 1
            if bad <= 10:
                print(f"DECODE {text!r}\n  want {want_text!r}\n  got  {got_text!r}")
    print(f"{len(cases) - bad}/{len(cases)} exact, encode and decode  ({args.tokenizer})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
