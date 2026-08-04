//! A small BPE tokenizer that reads Hugging Face `tokenizer.json` directly.
//!
//! Covers the pre-tokenizers these checkpoints actually ship: byte-level
//! (GPT-2 style) and metaspace (sentencepiece-style). Added and special tokens
//! are matched before anything else, so chat control tokens survive.

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(PartialEq, Clone, Copy)]
enum Pre {
    ByteLevel,
    Metaspace,
}

#[derive(PartialEq, Clone, Copy)]
enum Digits {
    /// `\p{N}` — one token per digit.
    Single,
    /// `\p{N}{1,3}` — groups of at most three.
    Upto3,
    /// `\p{N}+` — whole runs.
    Run,
}

/// The byte-level split rule, read off the `Split` regex the tokenizer declares.
///
/// Every checkpoint in this family uses the same alternation, differing only in
/// how digits are grouped, whether a word may absorb any non-alphanumeric
/// character or only a space, and how newlines attach — so the rule is modelled
/// by those flags instead of by a regex engine.
struct Split {
    word_any_prefix: bool,
    digits: Digits,
    digit_space: bool,
    symbol_newline: bool,
    newline_rule: bool,
    ci_contractions: bool,
}

impl Default for Split {
    /// The original GPT-2 alternation.
    fn default() -> Split {
        Split {
            word_any_prefix: false,
            digits: Digits::Run,
            digit_space: true,
            symbol_newline: false,
            newline_rule: false,
            ci_contractions: false,
        }
    }
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    ranks: HashMap<(String, String), u32>,
    added: Vec<(String, u32)>,
    special: Vec<bool>,
    /// Ids whose text is stored verbatim rather than byte-level encoded.
    literal: Vec<bool>,
    pre: Pre,
    prefix_space: bool,
    /// The decoder undoes `Prepend` by stripping one leading space.
    strip_space: bool,
    split: Split,
    byte_dec: HashMap<char, u8>,
    byte_enc: Vec<char>,
}

/// A merge candidate: `(rank, left index, left text, right text)`, ordered so
/// the smallest rank — and on a tie the leftmost pair — comes out first.
type Pair = std::cmp::Reverse<(u32, usize, String, String)>;

/// Find the first `{"Regex": "..."}` anywhere in the pre-tokenizer description.
fn find_regex(v: &Value) -> Option<String> {
    match v {
        Value::Object(o) => {
            if let Some(Value::String(s)) = o.get("Regex") {
                return Some(s.clone());
            }
            o.values().find_map(find_regex)
        }
        Value::Array(a) => a.iter().find_map(find_regex),
        _ => None,
    }
}

/// The GPT-2 byte <-> printable-codepoint bijection.
fn byte_table() -> Vec<char> {
    let mut map = vec!['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
    }
    map
}

impl Tokenizer {
    /// Load `tokenizer.json` from a file or from a model directory.
    pub fn load(path: &Path) -> Result<Tokenizer, String> {
        let file = if path.is_dir() { path.join("tokenizer.json") } else { path.to_path_buf() };
        let raw = std::fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let j: Value = serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", file.display()))?;
        Tokenizer::from_json(&j)
    }

    /// Build from an already-parsed `tokenizer.json`, as embedded in a `.moe`.
    pub fn from_json(j: &Value) -> Result<Tokenizer, String> {
        let model = &j["model"];
        let mut vocab = HashMap::new();
        for (k, v) in model["vocab"].as_object().ok_or("tokenizer.json has no model.vocab")? {
            vocab.insert(k.clone(), v.as_u64().unwrap_or(0) as u32);
        }
        let mut ranks = HashMap::new();
        for (i, m) in model["merges"].as_array().map(|a| a.iter()).into_iter().flatten().enumerate() {
            let pair = match m {
                Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())),
                Value::Array(a) if a.len() == 2 => {
                    Some((a[0].as_str().unwrap_or("").to_string(), a[1].as_str().unwrap_or("").to_string()))
                }
                _ => None,
            };
            if let Some(p) = pair {
                ranks.entry(p).or_insert(i as u32);
            }
        }

        // How text is cut up before BPE is declared in the file; read it rather
        // than assuming a family.
        let norm = serde_json::to_string(&j["normalizer"]).unwrap_or_default();
        let desc = serde_json::to_string(&j["pre_tokenizer"]).unwrap_or_default()
            + &serde_json::to_string(&j["decoder"]).unwrap_or_default();
        let pre = if desc.contains("Metaspace") || desc.contains("\u{2581}") || norm.contains("\u{2581}") {
            Pre::Metaspace
        } else {
            Pre::ByteLevel
        };
        let prefix_space = norm.contains("\"Prepend\"")
            || desc.contains("prepend_scheme\":\"always")
            || (pre == Pre::Metaspace && desc.contains("add_prefix_space\":true"));
        // A `Strip` in the decoder removes the separator `Prepend` added.
        let strip_space = j["decoder"]["decoders"]
            .as_array()
            .map(|a| a.iter())
            .into_iter()
            .flatten()
            .chain(std::iter::once(&j["decoder"]))
            .any(|d| d["type"] == "Strip" && d["content"] == " " && d["start"].as_u64().unwrap_or(0) >= 1);
        let split = match find_regex(&j["pre_tokenizer"]) {
            Some(re) => Split {
                word_any_prefix: re.contains("[^\\r\\n\\p{L}\\p{N}]?"),
                digits: if re.contains("\\p{N}{1,3}") {
                    Digits::Upto3
                } else if re.contains("\\p{N}+") {
                    Digits::Run
                } else {
                    Digits::Single
                },
                digit_space: re.contains(" ?\\p{N}"),
                symbol_newline: re.contains("]+[\\r\\n]*"),
                newline_rule: re.contains("\\s*[\\r\\n]+"),
                ci_contractions: re.contains("(?i"),
            },
            None => Split::default(),
        };

        let n = vocab.values().copied().max().unwrap_or(0) as usize + 1;
        let mut tokens = vec![String::new(); n];
        let mut special = vec![false; n];
        let mut literal = vec![false; n];
        for (t, i) in &vocab {
            tokens[*i as usize] = t.clone();
        }
        let mut added = Vec::new();
        for a in j["added_tokens"].as_array().map(|a| a.iter()).into_iter().flatten() {
            let (Some(c), Some(id)) = (a["content"].as_str(), a["id"].as_u64()) else { continue };
            let id = id as u32;
            if (id as usize) < n {
                tokens[id as usize] = c.to_string();
                special[id as usize] = a["special"].as_bool().unwrap_or(false);
                // Added tokens hold raw text: a vocabulary entry of four spaces
                // is four spaces, not four byte-level codepoints.
                literal[id as usize] = true;
            }
            added.push((c.to_string(), id));
        }
        // Longest first so `<|im_start|>` wins over any prefix of itself.
        added.sort_by_key(|(c, _)| std::cmp::Reverse(c.len()));

        let byte_enc = byte_table();
        let byte_dec = byte_enc.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        Ok(Tokenizer {
            vocab,
            tokens,
            ranks,
            added,
            special,
            literal,
            pre,
            prefix_space,
            strip_space,
            split,
            byte_dec,
            byte_enc,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary carries this exact token, which is how chat
    /// formats are recognised without interpreting a template.
    pub fn has_token(&self, text: &str) -> bool {
        self.vocab.contains_key(text)
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    /// Whether the checkpoint's decoder drops one leading space.
    pub fn strips_leading_space(&self) -> bool {
        self.strip_space
    }

    /// Length in chars of the piece starting at `i`, following the declared
    /// alternation in order: contractions, words, digits, symbols, whitespace.
    fn piece(&self, c: &[char], i: usize) -> usize {
        let n = c.len();
        let (is_l, is_n, is_s) = (char::is_alphabetic, char::is_numeric, char::is_whitespace);
        let sym = |ch: char| !is_l(ch) && !is_n(ch) && !is_s(ch);

        if c[i] == '\'' && i + 1 < n {
            let mut tail: String = c[i..n.min(i + 3)].iter().collect();
            if self.split.ci_contractions {
                tail = tail.to_lowercase();
            }
            for k in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] {
                if tail.starts_with(k) {
                    return k.chars().count();
                }
            }
        }

        // word: an optional leading character, then letters
        let mut j = i;
        let lead_ok = if self.split.word_any_prefix {
            !is_l(c[j]) && !is_n(c[j]) && c[j] != '\r' && c[j] != '\n'
        } else {
            c[j] == ' '
        };
        if lead_ok && j + 1 < n && is_l(c[j + 1]) {
            j += 1;
        }
        if j < n && is_l(c[j]) {
            while j < n && is_l(c[j]) {
                j += 1;
            }
            return j - i;
        }

        // digits
        let mut j = i;
        if self.split.digit_space && c[j] == ' ' && j + 1 < n && is_n(c[j + 1]) {
            j += 1;
        }
        if is_n(c[j]) {
            let max = match self.split.digits {
                Digits::Single => 1,
                Digits::Upto3 => 3,
                Digits::Run => usize::MAX,
            };
            let mut took = 0;
            while j < n && is_n(c[j]) && took < max {
                j += 1;
                took += 1;
            }
            return j - i;
        }

        // symbols, optionally trailing newlines
        let mut j = i;
        if c[j] == ' ' && j + 1 < n && sym(c[j + 1]) {
            j += 1;
        }
        if sym(c[j]) {
            while j < n && sym(c[j]) {
                j += 1;
            }
            if self.split.symbol_newline {
                while j < n && (c[j] == '\r' || c[j] == '\n') {
                    j += 1;
                }
            }
            return j - i;
        }

        // whitespace: a run ending at a newline, a run that reaches the end of
        // the text, or a run minus the space that belongs to the next piece
        if is_s(c[i]) {
            let mut end = i;
            while end < n && is_s(c[end]) {
                end += 1;
            }
            if self.split.newline_rule {
                if let Some(last) = (i..end).rev().find(|k| c[*k] == '\r' || c[*k] == '\n') {
                    return last + 1 - i;
                }
            }
            if end == n || end - 1 == i {
                return end - i;
            }
            return end - 1 - i;
        }
        1
    }

    fn pretokenize(&self, text: &str) -> Vec<String> {
        let c: Vec<char> = text.chars().collect();
        let (mut out, mut i) = (Vec::new(), 0usize);
        while i < c.len() {
            let len = self.piece(&c, i).max(1);
            out.push(c[i..i + len].iter().collect());
            i += len;
        }
        out
    }

    fn pair(&self, parts: &[String], i: usize, j: usize) -> Option<Pair> {
        let key = (parts[i].clone(), parts[j].clone());
        let rank = *self.ranks.get(&key)?;
        Some(std::cmp::Reverse((rank, i, key.0, key.1)))
    }

    /// Merge by ascending rank, leftmost first. A heap over live neighbour pairs
    /// keeps this linear-ish, which matters because sentencepiece-style
    /// checkpoints hand the whole prompt to BPE as a single word.
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if parts.len() > 1 {
            let n = parts.len();
            let mut next: Vec<usize> = (1..=n).collect();
            let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
            let mut alive = vec![true; n];
            // (rank, left index, expected left, expected right)
            let mut heap: std::collections::BinaryHeap<Pair> = std::collections::BinaryHeap::new();
            for i in 0..n - 1 {
                heap.extend(self.pair(&parts, i, i + 1));
            }
            while let Some(std::cmp::Reverse((_, i, li, ri))) = heap.pop() {
                let j = next[i];
                // Skip entries invalidated by an earlier merge.
                if !alive[i] || j >= n || !alive[j] || parts[i] != li || parts[j] != ri {
                    continue;
                }
                parts[i] = li + &ri;
                alive[j] = false;
                next[i] = next[j];
                if next[i] < n {
                    prev[next[i]] = i;
                    heap.extend(self.pair(&parts, i, next[i]));
                }
                if prev[i] < n {
                    heap.extend(self.pair(&parts, prev[i], i));
                }
            }
            let mut merged = Vec::new();
            let mut i = 0;
            while i < n {
                merged.push(std::mem::take(&mut parts[i]));
                i = next[i];
            }
            parts = merged;
        }
        for p in parts {
            match self.vocab.get(&p) {
                Some(id) => out.push(*id),
                None => {
                    // Byte fallback for sentencepiece-style vocabularies.
                    for b in p.bytes() {
                        if let Some(id) = self.vocab.get(&format!("<0x{b:02X}>")) {
                            out.push(*id);
                        }
                    }
                }
            }
        }
    }

    /// Scan left to right, taking the longest added token that matches at each
    /// position. Order matters: an earlier short match beats a later long one.
    pub fn encode(&self, text: &str, bos: Option<u32>) -> Vec<u32> {
        let mut ids = Vec::new();
        ids.extend(bos);
        let (mut start, mut i) = (0usize, 0usize);
        while i < text.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            // `added` is sorted longest first, so the first hit is the longest.
            match self.added.iter().find(|(c, _)| text[i..].starts_with(c.as_str())) {
                Some((content, id)) => {
                    if start < i {
                        self.encode_plain(&text[start..i], &mut ids);
                    }
                    ids.push(*id);
                    i += content.len();
                    start = i;
                }
                None => i += 1,
            }
        }
        if start < text.len() {
            self.encode_plain(&text[start..], &mut ids);
        }
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        match self.pre {
            Pre::ByteLevel => {
                for piece in self.pretokenize(text) {
                    let mapped: String = piece.bytes().map(|b| self.byte_enc[b as usize]).collect();
                    self.bpe(&mapped, ids);
                }
            }
            Pre::Metaspace => {
                let mut s = text.replace(' ', "\u{2581}");
                if self.prefix_space {
                    // The Prepend normaliser is unconditional, even when the
                    // text already begins with a separator.
                    s.insert(0, '\u{2581}');
                }
                self.bpe(&s, ids);
            }
        }
    }

    /// Append the raw bytes a token stands for. Decoding works in bytes because
    /// a multi-byte character can be split across several tokens.
    pub fn decode_bytes(&self, ids: &[u32], out: &mut Vec<u8>) {
        for id in ids {
            let Some(tok) = self.tokens.get(*id as usize) else { continue };
            if self.literal.get(*id as usize).copied().unwrap_or(false) {
                out.extend_from_slice(tok.as_bytes());
                continue;
            }
            match self.pre {
                Pre::ByteLevel => out.extend(tok.chars().map(|c| *self.byte_dec.get(&c).unwrap_or(&b'?'))),
                Pre::Metaspace => match tok
                    .strip_prefix("<0x")
                    .and_then(|s| s.strip_suffix('>'))
                    .and_then(|h| u8::from_str_radix(h, 16).ok())
                {
                    Some(b) => out.push(b),
                    None => out.extend(tok.replace('\u{2581}', " ").into_bytes()),
                },
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        self.decode_bytes(ids, &mut b);
        let text = String::from_utf8_lossy(&b);
        match self.strip_space {
            true => text.strip_prefix(' ').unwrap_or(&text).to_string(),
            false => text.into_owned(),
        }
    }

    /// Text of a single token, on its own. Prefer [`Stream`] when printing
    /// tokens as they are produced.
    pub fn decode_one(&self, id: u32) -> String {
        self.decode(&[id])
    }
}

/// Incremental detokeniser: holds back the tail of a character whose bytes have
/// not all arrived yet, so streaming never prints a broken glyph.
#[derive(Default)]
pub struct Stream {
    buf: Vec<u8>,
    started: bool,
}

impl Stream {
    /// Feed one token, returning whatever text is now complete.
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        tok.decode_bytes(&[id], &mut self.buf);
        let mut out = String::new();
        // The decoder's leading-space strip applies once, at the very start.
        if !self.started && tok.strips_leading_space() && self.buf.first() == Some(&b' ') {
            self.buf.remove(0);
        }
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    self.started |= !out.is_empty();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.buf[..good]).unwrap());
                    match e.error_len() {
                        // Genuinely invalid bytes: emit a replacement and move on.
                        Some(n) => {
                            out.push('\u{fffd}');
                            self.buf.drain(..good + n);
                        }
                        // Truncated tail: keep it until the next token arrives.
                        None => {
                            self.buf.drain(..good);
                            return out;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alternation shipped by current byte-level checkpoints.
    const MODERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    /// The original GPT-2 alternation.
    const CLASSIC: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    fn byte_level(pattern: &str, vocab: Value, merges: Value) -> Tokenizer {
        let j = serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": merges},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated"},
                {"type": "ByteLevel", "add_prefix_space": false},
            ]},
            "decoder": {"type": "ByteLevel"},
        });
        Tokenizer::from_json(&j).unwrap()
    }

    fn split(pattern: &str, text: &str) -> Vec<String> {
        byte_level(pattern, serde_json::json!({}), serde_json::json!([])).pretokenize(text)
    }

    #[test]
    fn modern_split_boundaries() {
        assert_eq!(split(MODERN, "hello world"), ["hello", " world"]);
        // A word may absorb any single non-alphanumeric character, not just a space.
        assert_eq!(split(MODERN, "Mixed_CASE"), ["Mixed", "_CASE"]);
        assert_eq!(split(MODERN, "a\tb\nc\r\nd"), ["a", "\tb", "\n", "c", "\r\n", "d"]);
        // One token per digit under `\p{N}`.
        assert_eq!(split(MODERN, "42"), ["4", "2"]);
        // A whitespace run gives its last space to the next word,
        // but keeps the whole run at end of text.
        assert_eq!(split(MODERN, "  lead"), [" ", " lead"]);
        assert_eq!(split(MODERN, "trail   "), ["trail", "   "]);
        assert_eq!(split(MODERN, "don't DON'T"), ["don", "'t", " DON", "'T"]);
    }

    #[test]
    fn classic_split_boundaries() {
        // Only a space attaches to a word, digits stay in runs,
        // and newlines do not bind to symbols.
        assert_eq!(split(CLASSIC, "Mixed_CASE"), ["Mixed", "_", "CASE"]);
        assert_eq!(split(CLASSIC, "42 7"), ["42", " 7"]);
        assert_eq!(split(CLASSIC, "a\tb"), ["a", "\t", "b"]);
        // Case-sensitive contractions here.
        assert_eq!(split(CLASSIC, "DON'T"), ["DON", "'", "T"]);
    }

    #[test]
    fn bpe_applies_lowest_rank_first() {
        // Ranks: "a b" beats "b c", so "abc" must come from ab + c.
        let vocab = serde_json::json!({"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5});
        let t = byte_level(MODERN, vocab, serde_json::json!(["a b", "b c", "ab c"]));
        let mut ids = Vec::new();
        t.bpe("abc", &mut ids);
        assert_eq!(ids, [5]);

        // With only the "b c" merge available, the left pair cannot form.
        let vocab = serde_json::json!({"a": 0, "b": 1, "c": 2, "bc": 4});
        let t = byte_level(MODERN, vocab, serde_json::json!(["b c"]));
        let mut ids = Vec::new();
        t.bpe("abc", &mut ids);
        assert_eq!(ids, [0, 4]);
    }

    #[test]
    fn byte_level_round_trips_every_byte() {
        let table = byte_table();
        let vocab: serde_json::Map<String, Value> = (0..256).map(|b| (table[b].to_string(), Value::from(b))).collect();
        let t = byte_level(MODERN, Value::Object(vocab), serde_json::json!([]));
        for text in ["héllo wörld", "🚀 tabs\tand\nnewlines", "0x1F/\\|~`"] {
            let ids = t.encode(text, None);
            assert_eq!(t.decode(&ids), text, "round trip failed for {text:?}");
            // Streaming one token at a time must produce the same text, with no
            // broken glyphs where a character spans several tokens.
            let mut st = Stream::default();
            let streamed: String = ids.iter().map(|i| st.push(&t, *i)).collect();
            assert_eq!(streamed, text, "streamed round trip failed for {text:?}");
        }
    }

    #[test]
    fn metaspace_prepends_unconditionally() {
        let j = serde_json::json!({
            "model": {"type": "BPE", "vocab": {"\u{2581}": 0, "a": 1}, "merges": []},
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "\u{2581}"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            ]},
        });
        let t = Tokenizer::from_json(&j).unwrap();
        // "a" -> "_a", and " a" -> "__a": the separator is added even when the
        // text already starts with one.
        assert_eq!(t.encode("a", None), [0, 1]);
        assert_eq!(t.encode(" a", None), [0, 0, 1]);
        assert_eq!(t.decode(&[0, 1]), " a");
    }

    /// An added token holds raw text, so it must bypass the byte-level map on
    /// the way out — a vocabulary entry of four spaces decodes to four spaces.
    #[test]
    fn added_tokens_decode_literally() {
        let vocab = serde_json::json!({"a": 0, "b": 1, "    ": 7, "\n": 8});
        let j = serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": 7, "content": "    ", "special": false},
                {"id": 8, "content": "\n", "special": false},
            ],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        });
        let t = Tokenizer::from_json(&j).unwrap();
        assert_eq!(t.decode(&[0, 7, 1]), "a    b");
        assert_eq!(t.decode(&[8, 7, 0]), "\n    a");
        let mut st = Stream::default();
        let streamed: String = [0u32, 7, 1].iter().map(|i| st.push(&t, *i)).collect();
        assert_eq!(streamed, "a    b");
    }

    /// A `Strip` in the decoder undoes the `Prepend` in the normaliser, so a
    /// round trip does not gain a leading space.
    #[test]
    fn metaspace_decoder_strips_the_prepended_space() {
        let j = serde_json::json!({
            "model": {"type": "BPE", "vocab": {"\u{2581}": 0, "a": 1}, "merges": []},
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "\u{2581}"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            ]},
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ]},
        });
        let t = Tokenizer::from_json(&j).unwrap();
        assert!(t.strips_leading_space());
        assert_eq!(t.decode(&t.encode("a", None)), "a");
        assert_eq!(t.decode(&t.encode(" a", None)), " a");
        let mut st = Stream::default();
        let streamed: String = t.encode("a", None).iter().map(|i| st.push(&t, *i)).collect();
        assert_eq!(streamed, "a");
    }

    #[test]
    fn added_tokens_are_matched_before_text() {
        let vocab = serde_json::json!({"a": 0, "b": 1, "<|end|>": 7});
        let j = serde_json::json!({
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
            "added_tokens": [{"id": 7, "content": "<|end|>", "special": true}],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        });
        let t = Tokenizer::from_json(&j).unwrap();
        assert_eq!(t.encode("a<|end|>b", None), [0, 7, 1]);
        assert!(t.is_special(7));
        assert!(!t.is_special(0));
    }
}
