//! Constrained decoding: making invalid output unrepresentable.
//!
//! A model asked for JSON usually produces JSON, and "usually" is the problem —
//! a caller that parses the answer needs a guarantee, not a tendency. So instead
//! of asking nicely in the prompt and retrying on failure, the sampler is given a
//! mask: at every step, tokens whose bytes could not continue a valid document
//! have their logits set to negative infinity. The model still chooses; it simply
//! cannot choose something malformed.
//!
//! The checker is a pushdown automaton over *bytes*, not characters or tokens,
//! because a token is an arbitrary byte string that may straddle any boundary —
//! `":"`, `"},{"` and `"\"na"` are all single tokens in real vocabularies. Its
//! state is [`Copy`] and shallow by construction, so testing a candidate token is
//! a stack copy of a few dozen bytes plus one pass over the token's bytes; over a
//! 50k vocabulary that is a small fraction of the forward pass it rides along
//! with.
//!
//! The same machine covers free-form JSON and a JSON Schema. Free-form is just
//! the schema that permits anything, so there is one automaton rather than two.

use crate::tokenizer::Tokenizer;
use serde_json::Value;

/// Nesting limit. Deeper than this and the schema is the problem, not the limit.
const MAX_DEPTH: usize = 32;
/// Alternatives at one decision point — object properties, or enum members.
/// Bounded because the live set is tracked as a bitmask.
pub const MAX_OPTIONS: usize = 64;

/// The node id of `Any`, which every grammar carries at index 0 so that a
/// free-form value always has a shape to point at.
const ANY: u32 = 0;

const WORDS: [&str; 3] = ["true", "false", "null"];

// ------------------------------------------------------------------- the shape

#[derive(Debug, Clone)]
enum Node {
    /// Any JSON value.
    Any,
    Str,
    Num {
        integer: bool,
    },
    Bool,
    Null,
    /// Permitted values, as their exact JSON encodings.
    Enum(Vec<String>),
    Object {
        /// Property names, quoted and escaped so they match the wire bytes.
        keys: Vec<String>,
        values: Vec<u32>,
        required: Vec<bool>,
    },
    Array {
        items: u32,
        min: u32,
        max: u32,
    },
}

/// A compiled shape a document must have.
#[derive(Debug, Clone)]
pub struct Grammar {
    nodes: Vec<Node>,
    root: u32,
}

impl Grammar {
    /// Any syntactically valid JSON document.
    pub fn json() -> Grammar {
        Grammar { nodes: vec![Node::Any], root: ANY }
    }

    /// Compile a JSON Schema.
    ///
    /// The supported subset is the part that constrains shape: `type`,
    /// `properties`, `required`, `items`, `minItems`, `maxItems`, `enum` and
    /// `const`. Anything else is permissive rather than an error, so a schema
    /// carrying documentation, `$id`s or validation keywords the engine does not
    /// model still constrains everything it can.
    ///
    /// Properties are emitted in a fixed order — those named in `required`
    /// first, in that order, then the remainder alphabetically — because a
    /// deterministic order is what lets the automaton stay deterministic.
    /// Optional properties may be skipped, but not reordered.
    pub fn from_schema(schema: &Value) -> Result<Grammar, String> {
        let mut g = Grammar { nodes: vec![Node::Any], root: ANY };
        g.root = g.compile(schema, 0)?;
        Ok(g)
    }

    fn add(&mut self, n: Node) -> u32 {
        self.nodes.push(n);
        (self.nodes.len() - 1) as u32
    }

    fn compile(&mut self, s: &Value, depth: usize) -> Result<u32, String> {
        if depth > MAX_DEPTH {
            return Err(format!("schema nests deeper than {MAX_DEPTH} levels"));
        }
        // `const` and `enum` pin the value regardless of any declared type.
        if let Some(c) = s.get("const") {
            return Ok(self.add(Node::Enum(vec![c.to_string()])));
        }
        if let Some(Value::Array(vals)) = s.get("enum") {
            if vals.is_empty() {
                return Err("enum with no members permits nothing".into());
            }
            if vals.len() > MAX_OPTIONS {
                return Err(format!("enum has {} members, the limit is {MAX_OPTIONS}", vals.len()));
            }
            return Ok(self.add(Node::Enum(vals.iter().map(|v| v.to_string()).collect())));
        }
        // A union of types is not modelled; permitting anything is the honest
        // fallback, and still constrains everything nested inside.
        let Some(ty) = s.get("type").and_then(|t| t.as_str()) else { return Ok(ANY) };
        match ty {
            "string" => Ok(self.add(Node::Str)),
            "integer" => Ok(self.add(Node::Num { integer: true })),
            "number" => Ok(self.add(Node::Num { integer: false })),
            "boolean" => Ok(self.add(Node::Bool)),
            "null" => Ok(self.add(Node::Null)),
            "array" => {
                let items = match s.get("items") {
                    Some(i) => self.compile(i, depth + 1)?,
                    None => ANY,
                };
                let min = s["minItems"].as_u64().unwrap_or(0) as u32;
                let max = s["maxItems"].as_u64().map(|v| v as u32).unwrap_or(u32::MAX);
                if max < min {
                    return Err(format!("maxItems {max} is below minItems {min}"));
                }
                Ok(self.add(Node::Array { items, min, max }))
            }
            "object" => {
                let props = s.get("properties").and_then(|p| p.as_object());
                let Some(props) = props else {
                    // An object with no declared properties constrains only that
                    // it is an object; keys and values are free.
                    return Ok(self.add(Node::Object { keys: Vec::new(), values: Vec::new(), required: Vec::new() }));
                };
                let req: Vec<&str> =
                    s["required"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
                // Required first, in the order the schema names them, so the
                // output reads the way the author wrote it; the rest follow in
                // the map's own (sorted) order.
                let mut order: Vec<&String> = Vec::new();
                for name in &req {
                    if let Some((k, _)) = props.iter().find(|(k, _)| k.as_str() == *name) {
                        order.push(k);
                    }
                }
                order.extend(props.keys().filter(|k| !req.contains(&k.as_str())));
                if order.len() > MAX_OPTIONS {
                    return Err(format!("object has {} properties, the limit is {MAX_OPTIONS}", order.len()));
                }
                let mut keys = Vec::new();
                let mut values = Vec::new();
                let mut required = Vec::new();
                for k in order {
                    keys.push(Value::String(k.clone()).to_string());
                    values.push(self.compile(&props[k], depth + 1)?);
                    required.push(req.contains(&k.as_str()));
                }
                Ok(self.add(Node::Object { keys, values, required }))
            }
            other => Err(format!("unsupported schema type: {other}")),
        }
    }

    fn node(&self, i: u32) -> &Node {
        &self.nodes[i as usize]
    }

    /// The alternatives a [`Choice`] frame is choosing between.
    fn options(&self, set: Set) -> Vec<&[u8]> {
        match set {
            Set::Words => WORDS.iter().map(|w| w.as_bytes()).collect(),
            Set::Enum(n) => match self.node(n) {
                Node::Enum(v) => v.iter().map(|s| s.as_bytes()).collect(),
                _ => Vec::new(),
            },
            Set::Keys(n) => match self.node(n) {
                Node::Object { keys, .. } => keys.iter().map(|s| s.as_bytes()).collect(),
                _ => Vec::new(),
            },
        }
    }
}

// ----------------------------------------------------------------- the machine

/// Where a number can be, mid-literal. JSON's number grammar forbids a leading
/// zero followed by digits, a bare `-`, and an exponent with no digits, so the
/// phase has to be tracked rather than just "some digits were seen".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Num {
    /// Nothing yet: `-` or a digit.
    Lead,
    /// After `-`: a digit must follow.
    Sign,
    /// The integer part is a lone `0`, so no further digit may follow.
    Zero,
    /// One or more integer digits, not a lone zero.
    Int,
    /// After `.`: a digit must follow.
    Dot,
    Frac,
    /// After `e`/`E`: a sign or digit must follow.
    Exp,
    /// After an exponent sign: a digit must follow.
    ExpSign,
    ExpDigits,
}

impl Num {
    /// Whether the literal read so far is a complete number.
    fn terminable(self) -> bool {
        matches!(self, Num::Zero | Num::Int | Num::Frac | Num::ExpDigits)
    }

    /// The next phase, or `None` if this byte is not part of the number.
    fn feed(self, b: u8, integer: bool) -> Option<Num> {
        let digit = b.is_ascii_digit();
        Some(match (self, b) {
            (Num::Lead, b'-') => Num::Sign,
            (Num::Lead | Num::Sign, b'0') => Num::Zero,
            (Num::Lead | Num::Sign, _) if digit => Num::Int,
            (Num::Int, _) if digit => Num::Int,
            (Num::Zero | Num::Int, b'.') if !integer => Num::Dot,
            (Num::Dot, _) if digit => Num::Frac,
            (Num::Frac, _) if digit => Num::Frac,
            (Num::Zero | Num::Int | Num::Frac, b'e' | b'E') if !integer => Num::Exp,
            (Num::Exp, b'+' | b'-') => Num::ExpSign,
            (Num::Exp | Num::ExpSign, _) if digit => Num::ExpDigits,
            (Num::ExpDigits, _) if digit => Num::ExpDigits,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ObjPhase {
    /// After `{`: the first key, or `}` for an empty object.
    KeyOrEnd,
    /// After `,`: a key is now mandatory, which is what forbids `{"a":1,}`.
    Key,
    /// A key is being read by the frame above.
    InKey,
    /// After a key: `:`.
    AfterKey,
    /// After `:`: the value must begin.
    BeforeValue,
    /// The value is being read by the frame above.
    InValue,
    /// After a value: `,` or `}`.
    CommaOrEnd,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArrPhase {
    /// After `[`: an element, or `]`.
    ValueOrEnd,
    /// After `,`: an element must follow.
    Value,
    InValue,
    CommaOrEnd,
}

/// Which set of fixed alternatives a [`Frame::Choice`] is narrowing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Set {
    /// `true`, `false`, `null`.
    Words,
    Enum(u32),
    Keys(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Frame {
    Empty,
    /// The document: one value, then nothing but whitespace.
    Root {
        node: u32,
        started: bool,
        done: bool,
    },
    Obj {
        node: u32,
        open: bool,
        next: u32,
        key: u32,
        phase: ObjPhase,
    },
    Arr {
        items: u32,
        count: u32,
        min: u32,
        max: u32,
        phase: ArrPhase,
    },
    /// Inside a string, past the opening quote.
    Str {
        escape: bool,
        hex: u8,
    },
    Num {
        state: Num,
        integer: bool,
    },
    /// Matching one of a fixed set of byte strings; `alive` is a bitmask of the
    /// options still consistent with what has been read.
    Choice {
        set: Set,
        alive: u64,
        at: u32,
    },
}

/// The automaton's state. Copyable on purpose: checking a candidate token means
/// forking this, and that has to be cheap.
#[derive(Clone, Copy)]
pub struct Machine {
    stack: [Frame; MAX_DEPTH],
    len: usize,
    failed: bool,
}

impl Machine {
    fn top(&self) -> Frame {
        if self.len == 0 {
            Frame::Empty
        } else {
            self.stack[self.len - 1]
        }
    }

    fn set_top(&mut self, f: Frame) {
        if self.len > 0 {
            self.stack[self.len - 1] = f;
        }
    }

    fn push(&mut self, f: Frame) -> bool {
        if self.len >= MAX_DEPTH {
            return false;
        }
        self.stack[self.len] = f;
        self.len += 1;
        true
    }

    fn pop(&mut self) {
        self.len = self.len.saturating_sub(1);
    }

    /// Copy only the live part of the stack. The whole point of the fixed array
    /// is that a fork costs `len` frames, not `MAX_DEPTH` of them.
    fn fork_into(&self, dst: &mut Machine) {
        dst.stack[..self.len].copy_from_slice(&self.stack[..self.len]);
        dst.len = self.len;
        dst.failed = self.failed;
    }

    /// A unit — value or key — just finished; tell the frame that wanted it.
    /// `matched` is which alternative a [`Frame::Choice`] settled on.
    fn finish(&mut self, matched: Option<u32>) -> bool {
        match self.top() {
            Frame::Root { node, started, .. } => {
                self.set_top(Frame::Root { node, started, done: true });
                true
            }
            Frame::Obj { node, open, next, key, phase } => match phase {
                ObjPhase::InKey => {
                    self.set_top(Frame::Obj { node, open, next, key: matched.unwrap_or(0), phase: ObjPhase::AfterKey });
                    true
                }
                ObjPhase::InValue => {
                    self.set_top(Frame::Obj { node, open, next: key + 1, key, phase: ObjPhase::CommaOrEnd });
                    true
                }
                _ => false,
            },
            Frame::Arr { items, count, min, max, phase: ArrPhase::InValue } => {
                self.set_top(Frame::Arr { items, count: count + 1, min, max, phase: ArrPhase::CommaOrEnd });
                true
            }
            _ => false,
        }
    }
}

impl Default for Machine {
    fn default() -> Machine {
        Machine { stack: [Frame::Empty; MAX_DEPTH], len: 0, failed: false }
    }
}

impl Grammar {
    pub fn start(&self) -> Machine {
        let mut m = Machine::default();
        m.push(Frame::Root { node: self.root, started: false, done: false });
        m
    }

    /// Feed one byte. A machine that has failed stays failed.
    pub fn feed(&self, m: &mut Machine, b: u8) -> bool {
        if m.failed {
            return false;
        }
        if !self.step(m, b, 0) {
            m.failed = true;
            return false;
        }
        true
    }

    /// Feed a whole token's bytes, stopping at the first that cannot follow.
    pub fn feed_all(&self, m: &mut Machine, bytes: &[u8]) -> bool {
        bytes.iter().all(|b| self.feed(m, *b))
    }

    /// Whether the document is a complete value and could stop here.
    pub fn complete(&self, m: &Machine) -> bool {
        if m.failed {
            return false;
        }
        let mut c = *m;
        if !self.settle(&mut c) {
            return false;
        }
        matches!(c.top(), Frame::Root { done: true, .. })
    }

    /// Close any scalar that a delimiter would have closed. A number or a
    /// numeric enum member is only known to be finished when something that
    /// cannot extend it arrives — including the end of the output.
    fn settle(&self, m: &mut Machine) -> bool {
        loop {
            match m.top() {
                Frame::Num { state, .. } => {
                    if !state.terminable() {
                        return false;
                    }
                    m.pop();
                    if !m.finish(None) {
                        return false;
                    }
                }
                Frame::Choice { set, alive, at } => match self.settled_option(set, alive, at) {
                    Some(i) => {
                        m.pop();
                        if !m.finish(Some(i)) {
                            return false;
                        }
                    }
                    None => return false,
                },
                _ => return true,
            }
        }
    }

    /// The alternative that ends exactly here, if any.
    fn settled_option(&self, set: Set, alive: u64, at: u32) -> Option<u32> {
        self.options(set)
            .iter()
            .enumerate()
            .position(|(i, o)| alive & (1 << i) != 0 && o.len() == at as usize)
            .map(|i| i as u32)
    }

    /// Options still alive after reading `b` at offset `at`.
    fn narrow(&self, set: Set, alive: u64, at: u32, b: u8) -> u64 {
        let mut out = 0u64;
        for (i, o) in self.options(set).iter().enumerate() {
            if alive & (1 << i) != 0 && o.len() > at as usize && o[at as usize] == b {
                out |= 1 << i;
            }
        }
        out
    }

    fn step(&self, m: &mut Machine, b: u8, depth: usize) -> bool {
        // A delimiter can close at most one scalar before being reprocessed, but
        // guard the recursion anyway rather than trusting that.
        if depth > 4 {
            return false;
        }
        // Whatever scalar is in progress takes the byte first.
        match m.top() {
            Frame::Str { escape, hex } => return self.string_byte(m, b, escape, hex),
            Frame::Num { state, integer } => {
                return match state.feed(b, integer) {
                    Some(next) => {
                        m.set_top(Frame::Num { state: next, integer });
                        true
                    }
                    // Not part of the number, so the number ends here — if it
                    // legally can — and this byte is a delimiter.
                    None => {
                        if !state.terminable() {
                            return false;
                        }
                        m.pop();
                        m.finish(None) && self.step(m, b, depth + 1)
                    }
                };
            }
            Frame::Choice { set, alive, at } => {
                let narrowed = self.narrow(set, alive, at, b);
                if narrowed != 0 {
                    m.set_top(Frame::Choice { set, alive: narrowed, at: at + 1 });
                    return true;
                }
                let Some(i) = self.settled_option(set, alive, at) else { return false };
                m.pop();
                return m.finish(Some(i)) && self.step(m, b, depth + 1);
            }
            _ => {}
        }

        // Between tokens, whitespace is always allowed.
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
            return true;
        }

        match m.top() {
            Frame::Root { node, started: false, .. } => {
                m.set_top(Frame::Root { node, started: true, done: false });
                self.begin(m, node, b, depth)
            }
            Frame::Root { .. } => false,
            Frame::Obj { node, open, next, key, phase } => match phase {
                ObjPhase::KeyOrEnd | ObjPhase::Key => match b {
                    b'"' => {
                        m.set_top(Frame::Obj { node, open, next, key, phase: ObjPhase::InKey });
                        if open {
                            // No declared properties: any string is a key.
                            m.push(Frame::Str { escape: false, hex: 0 })
                        } else {
                            let alive = self.keys_from(node, next);
                            alive != 0
                                && m.push(Frame::Choice { set: Set::Keys(node), alive, at: 0 })
                                && self.step(m, b, depth + 1)
                        }
                    }
                    // Only an empty object may close here; after a comma a key
                    // has been promised.
                    b'}' if phase == ObjPhase::KeyOrEnd && self.object_may_end(node, next, open) => {
                        m.pop();
                        m.finish(None)
                    }
                    _ => false,
                },
                ObjPhase::AfterKey => {
                    if b == b':' {
                        m.set_top(Frame::Obj { node, open, next, key, phase: ObjPhase::BeforeValue });
                        true
                    } else {
                        false
                    }
                }
                ObjPhase::BeforeValue => {
                    m.set_top(Frame::Obj { node, open, next, key, phase: ObjPhase::InValue });
                    let shape = if open { ANY } else { self.value_of(node, key) };
                    self.begin(m, shape, b, depth)
                }
                ObjPhase::CommaOrEnd => match b {
                    b',' if open || self.keys_from(node, next) != 0 => {
                        m.set_top(Frame::Obj { node, open, next, key, phase: ObjPhase::Key });
                        true
                    }
                    b'}' if self.object_may_end(node, next, open) => {
                        m.pop();
                        m.finish(None)
                    }
                    _ => false,
                },
                // A frame above is mid-key or mid-value; nothing reaches here.
                ObjPhase::InKey | ObjPhase::InValue => false,
            },
            Frame::Arr { items, count, min, max, phase } => match phase {
                ArrPhase::ValueOrEnd => match b {
                    b']' if count >= min => {
                        m.pop();
                        m.finish(None)
                    }
                    _ if count < max => {
                        m.set_top(Frame::Arr { items, count, min, max, phase: ArrPhase::InValue });
                        self.begin(m, items, b, depth)
                    }
                    _ => false,
                },
                ArrPhase::Value if count < max => {
                    m.set_top(Frame::Arr { items, count, min, max, phase: ArrPhase::InValue });
                    self.begin(m, items, b, depth)
                }
                ArrPhase::CommaOrEnd => match b {
                    b',' if count < max => {
                        m.set_top(Frame::Arr { items, count, min, max, phase: ArrPhase::Value });
                        true
                    }
                    b']' if count >= min => {
                        m.pop();
                        m.finish(None)
                    }
                    _ => false,
                },
                ArrPhase::Value | ArrPhase::InValue => false,
            },
            Frame::Empty | Frame::Str { .. } | Frame::Num { .. } | Frame::Choice { .. } => false,
        }
    }

    /// Keys that may legally come next at property index `from`: the one at
    /// `from`, plus any beyond it reachable by skipping only optional ones.
    fn keys_from(&self, node: u32, from: u32) -> u64 {
        let Node::Object { keys, required, .. } = self.node(node) else { return 0 };
        let mut alive = 0u64;
        for (i, req) in required.iter().enumerate().skip(from as usize).take(keys.len()) {
            alive |= 1 << i;
            if *req {
                break;
            }
        }
        alive
    }

    /// Whether `}` is legal here: every property still to come is optional.
    fn object_may_end(&self, node: u32, from: u32, open: bool) -> bool {
        if open {
            return true;
        }
        let Node::Object { required, .. } = self.node(node) else { return true };
        required.iter().skip(from as usize).all(|r| !r)
    }

    fn value_of(&self, node: u32, key: u32) -> u32 {
        match self.node(node) {
            Node::Object { values, .. } => values.get(key as usize).copied().unwrap_or(ANY),
            _ => ANY,
        }
    }

    /// Begin a value of shape `node`, whose first byte is `b`.
    fn begin(&self, m: &mut Machine, node: u32, b: u8, depth: usize) -> bool {
        let num = |m: &mut Machine, integer: bool| {
            m.push(Frame::Num { state: Num::Lead, integer }) && self.step(m, b, depth + 1)
        };
        let choice = |m: &mut Machine, set: Set, alive: u64| {
            alive != 0 && m.push(Frame::Choice { set, alive, at: 0 }) && self.step(m, b, depth + 1)
        };
        match self.node(node) {
            Node::Any => match b {
                b'{' => m.push(Frame::Obj { node: ANY, open: true, next: 0, key: 0, phase: ObjPhase::KeyOrEnd }),
                b'[' => m.push(Frame::Arr { items: ANY, count: 0, min: 0, max: u32::MAX, phase: ArrPhase::ValueOrEnd }),
                b'"' => m.push(Frame::Str { escape: false, hex: 0 }),
                b'-' | b'0'..=b'9' => num(m, false),
                b't' | b'f' | b'n' => choice(m, Set::Words, 0b111),
                _ => false,
            },
            Node::Object { keys, .. } => {
                b == b'{'
                    && m.push(Frame::Obj { node, open: keys.is_empty(), next: 0, key: 0, phase: ObjPhase::KeyOrEnd })
            }
            Node::Array { items, min, max } => {
                b == b'['
                    && m.push(Frame::Arr { items: *items, count: 0, min: *min, max: *max, phase: ArrPhase::ValueOrEnd })
            }
            Node::Str => b == b'"' && m.push(Frame::Str { escape: false, hex: 0 }),
            Node::Num { integer } => (b == b'-' || b.is_ascii_digit()) && num(m, *integer),
            // `true` and `false` only: bits 0 and 1 of WORDS.
            Node::Bool => choice(m, Set::Words, 0b011),
            Node::Null => choice(m, Set::Words, 0b100),
            Node::Enum(vals) => choice(m, Set::Enum(node), (1u64 << vals.len()) - 1),
        }
    }

    fn string_byte(&self, m: &mut Machine, b: u8, escape: bool, hex: u8) -> bool {
        if hex > 0 {
            if !b.is_ascii_hexdigit() {
                return false;
            }
            m.set_top(Frame::Str { escape: false, hex: hex - 1 });
            return true;
        }
        if escape {
            return match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    m.set_top(Frame::Str { escape: false, hex: 0 });
                    true
                }
                b'u' => {
                    m.set_top(Frame::Str { escape: false, hex: 4 });
                    true
                }
                _ => false,
            };
        }
        match b {
            b'"' => {
                m.pop();
                m.finish(None)
            }
            b'\\' => {
                m.set_top(Frame::Str { escape: true, hex: 0 });
                true
            }
            // Raw control bytes must be escaped in JSON.
            0x00..=0x1f => false,
            _ => true,
        }
    }
}

// ------------------------------------------------------------------- the guide

/// A grammar bound to a vocabulary: masks logits, and tracks what has been said.
pub struct Guide {
    grammar: Grammar,
    /// Raw bytes each token stands for. Empty for control tokens, which carry no
    /// document bytes and therefore can never be emitted under a constraint.
    bytes: Vec<Vec<u8>>,
    machine: Machine,
    scratch: Machine,
}

impl Guide {
    pub fn new(grammar: Grammar, tok: &Tokenizer) -> Guide {
        let n = tok.vocab_size();
        let mut bytes = Vec::with_capacity(n);
        for id in 0..n as u32 {
            if tok.is_special(id) {
                bytes.push(Vec::new());
                continue;
            }
            let mut b = Vec::new();
            tok.decode_bytes(&[id], &mut b);
            bytes.push(b);
        }
        let machine = grammar.start();
        Guide { grammar, bytes, machine, scratch: Machine::default() }
    }

    /// Set the logits of tokens that would break the grammar to `-inf`.
    /// Returns how many tokens survived.
    pub fn mask(&mut self, logits: &mut [f32]) -> usize {
        let mut allowed = 0usize;
        for (id, l) in logits.iter_mut().enumerate() {
            let ok = match self.bytes.get(id) {
                Some(b) if !b.is_empty() => {
                    self.machine.fork_into(&mut self.scratch);
                    self.grammar.feed_all(&mut self.scratch, b)
                }
                _ => false,
            };
            if ok {
                allowed += 1;
            } else {
                *l = f32::NEG_INFINITY;
            }
        }
        allowed
    }

    /// Commit a token. False means it did not fit, which a masked sampler
    /// cannot produce.
    pub fn accept(&mut self, id: u32) -> bool {
        match self.bytes.get(id as usize) {
            Some(b) if !b.is_empty() => {
                let bytes = b.clone();
                self.grammar.feed_all(&mut self.machine, &bytes)
            }
            _ => false,
        }
    }

    /// Whether what has been emitted is a complete document.
    pub fn complete(&self) -> bool {
        self.grammar.complete(&self.machine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Does the grammar accept this text as a complete document?
    fn accepts(g: &Grammar, s: &str) -> bool {
        let mut m = g.start();
        g.feed_all(&mut m, s.as_bytes()) && g.complete(&m)
    }

    /// Could this text still be extended into a valid document?
    fn viable(g: &Grammar, s: &str) -> bool {
        let mut m = g.start();
        g.feed_all(&mut m, s.as_bytes())
    }

    fn schema(s: &str) -> Grammar {
        Grammar::from_schema(&serde_json::from_str(s).unwrap()).unwrap()
    }

    #[test]
    fn free_json_accepts_valid_documents() {
        let g = Grammar::json();
        for s in [
            "null",
            "true",
            "false",
            "0",
            "-0",
            "1",
            "-12",
            "1.5",
            "1e10",
            "1E+10",
            "-2.5e-3",
            "\"\"",
            "\"hi\"",
            "\"a\\\"b\"",
            "\"\\u00e9\"",
            "\"\\n\\t\\\\\"",
            "[]",
            "[1]",
            "[1,2,3]",
            "[[[]]]",
            "{}",
            "{\"a\":1}",
            "{\"a\":1,\"b\":[true,null]}",
            "  {  \"a\" :  [ 1 , 2 ]  }  ",
            "{\"nested\":{\"deep\":{\"x\":\"y\"}}}",
            "\"é\"",
            "\"日本\"",
        ] {
            assert!(accepts(&g, s), "rejected valid JSON: {s}");
        }
    }

    #[test]
    fn free_json_rejects_invalid_documents() {
        let g = Grammar::json();
        for s in [
            "",
            "{",
            "}",
            "[",
            "[,]",
            "[1,]",
            "{,}",
            "{\"a\"}",
            "{\"a\":}",
            "{a:1}",
            "{'a':1}",
            "{\"a\":1,}",
            "01",
            "-",
            "+1",
            ".5",
            "1.",
            "1e",
            "1e+",
            "tru",
            "truex",
            "nul",
            "\"unterminated",
            "\"bad\\escape\"",
            "\"\\u00g0\"",
            "1 2",
            "[1 2]",
            "{\"a\":1}{}",
            "[1]]",
        ] {
            assert!(!accepts(&g, s), "accepted invalid JSON: {s}");
        }
    }

    /// The mask has to be built from prefix viability, not from whole documents,
    /// so a valid prefix must stay viable while an impossible one must not.
    #[test]
    fn prefixes_are_judged_on_what_could_still_follow() {
        let g = Grammar::json();
        for s in ["{", "{\"", "{\"a", "{\"a\"", "{\"a\":", "{\"a\":1", "{\"a\":1,", "[", "[1", "[1,", "1", "1.", "1e"] {
            assert!(viable(&g, s), "a valid prefix was rejected: {s}");
        }
        for s in ["{]", "[}", "{\"a\":,", "{:", "1.e", "--", "\"\\q"] {
            assert!(!viable(&g, s), "an impossible prefix stayed viable: {s}");
        }
        // Viable is not the same as complete.
        assert!(!accepts(&g, "{\"a\":1"));
    }

    #[test]
    fn a_number_is_only_finished_when_nothing_extends_it() {
        let g = Grammar::json();
        // Complete on its own...
        assert!(accepts(&g, "12"));
        // ...and closed by a delimiter, which must then be processed.
        assert!(accepts(&g, "[12,13]"));
        assert!(accepts(&g, "{\"a\":1}"));
        // A number that cannot end where it stops is not a document.
        assert!(!accepts(&g, "1e"));
        assert!(!accepts(&g, "[1e]"));
    }

    #[test]
    fn schema_pins_a_scalar_type() {
        assert!(accepts(&schema(r#"{"type":"string"}"#), "\"x\""));
        assert!(!accepts(&schema(r#"{"type":"string"}"#), "1"));
        assert!(accepts(&schema(r#"{"type":"integer"}"#), "-7"));
        // An integer may not carry a fraction or an exponent.
        assert!(!accepts(&schema(r#"{"type":"integer"}"#), "1.5"));
        assert!(!accepts(&schema(r#"{"type":"integer"}"#), "1e3"));
        assert!(accepts(&schema(r#"{"type":"number"}"#), "1.5"));
        assert!(accepts(&schema(r#"{"type":"boolean"}"#), "true"));
        assert!(!accepts(&schema(r#"{"type":"boolean"}"#), "null"));
        assert!(accepts(&schema(r#"{"type":"null"}"#), "null"));
        assert!(!accepts(&schema(r#"{"type":"null"}"#), "false"));
    }

    #[test]
    fn schema_requires_declared_properties_in_order() {
        let g = schema(
            r#"{"type":"object",
                "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
                "required":["name","age"]}"#,
        );
        assert!(accepts(&g, r#"{"name":"ada","age":36}"#));
        // Order is fixed, so the reverse is not accepted.
        assert!(!accepts(&g, r#"{"age":36,"name":"ada"}"#));
        // Required properties cannot be dropped.
        assert!(!accepts(&g, r#"{"name":"ada"}"#));
        assert!(!accepts(&g, "{}"));
        // Undeclared properties cannot be invented.
        assert!(!accepts(&g, r#"{"name":"ada","age":36,"extra":1}"#));
        // Types are enforced per property.
        assert!(!accepts(&g, r#"{"name":1,"age":36}"#));
        assert!(!accepts(&g, r#"{"name":"ada","age":"36"}"#));
    }

    #[test]
    fn optional_properties_may_be_skipped_but_not_reordered() {
        let g = schema(
            r#"{"type":"object",
                "properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},
                "required":["a","c"]}"#,
        );
        // Required first in declared order, then the rest: a, c, b.
        assert!(accepts(&g, r#"{"a":1,"c":3}"#));
        assert!(accepts(&g, r#"{"a":1,"c":3,"b":2}"#));
        // b is optional, so it may be left out, but it cannot come before c.
        assert!(!accepts(&g, r#"{"a":1,"b":2,"c":3}"#));
        assert!(!accepts(&g, r#"{"c":3}"#));
    }

    #[test]
    fn schema_constrains_arrays_by_shape_and_length() {
        let g = schema(r#"{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3}"#);
        assert!(accepts(&g, "[1,2]"));
        assert!(accepts(&g, "[1,2,3]"));
        assert!(!accepts(&g, "[1]"), "minItems not enforced");
        assert!(!accepts(&g, "[]"));
        assert!(!accepts(&g, "[1,2,3,4]"), "maxItems not enforced");
        assert!(!accepts(&g, "[1,\"x\"]"), "item type not enforced");
        // A full array cannot even start another element.
        assert!(!viable(&g, "[1,2,3,"));
    }

    #[test]
    fn enums_and_consts_admit_exactly_their_members() {
        let g = schema(r#"{"enum":["red","green",7,true,null]}"#);
        for ok in ["\"red\"", "\"green\"", "7", "true", "null"] {
            assert!(accepts(&g, ok), "rejected enum member {ok}");
        }
        for bad in ["\"blue\"", "8", "false", "\"re\"", "\"redx\""] {
            assert!(!accepts(&g, bad), "accepted non-member {bad}");
        }
        let c = schema(r#"{"const":42}"#);
        assert!(accepts(&c, "42"));
        assert!(!accepts(&c, "43"));
        assert!(!accepts(&c, "4"));
    }

    /// Members that prefix one another resolve on the following byte, the same
    /// way a number does.
    #[test]
    fn enum_members_that_share_a_prefix_are_disambiguated() {
        let g = schema(r#"{"type":"array","items":{"enum":[1,12,123]}}"#);
        for s in ["[1]", "[12]", "[123]", "[1,12,123]"] {
            assert!(accepts(&g, s), "rejected {s}");
        }
        assert!(!accepts(&g, "[13]"));
        assert!(!accepts(&g, "[1234]"));
    }

    #[test]
    fn nested_schemas_compose() {
        let g = schema(
            r#"{"type":"object",
                "properties":{
                  "id":{"type":"integer"},
                  "tags":{"type":"array","items":{"type":"string"}},
                  "meta":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}},
                "required":["id","tags","meta"]}"#,
        );
        assert!(accepts(&g, r#"{"id":1,"tags":["a","b"],"meta":{"ok":true}}"#));
        assert!(accepts(&g, r#"{"id":1,"tags":[],"meta":{"ok":false}}"#));
        assert!(!accepts(&g, r#"{"id":1,"tags":["a"],"meta":{}}"#));
        assert!(!accepts(&g, r#"{"id":1,"tags":[1],"meta":{"ok":true}}"#));
    }

    /// An object with no declared properties constrains the type and nothing else.
    #[test]
    fn an_open_object_takes_any_keys() {
        let g = schema(r#"{"type":"object"}"#);
        assert!(accepts(&g, "{}"));
        assert!(accepts(&g, r#"{"whatever":[1,2],"x":null}"#));
        assert!(!accepts(&g, "[]"));
    }

    /// Keywords the engine does not model must not turn into a hard error, and
    /// must not stop it constraining what it does understand.
    #[test]
    fn unmodelled_keywords_stay_permissive() {
        let g = schema(
            r#"{"type":"object","description":"hi","additionalProperties":false,
                           "properties":{"a":{"type":"integer","minimum":3}},"required":["a"]}"#,
        );
        assert!(accepts(&g, r#"{"a":1}"#), "minimum is not modelled, so 1 is allowed");
        assert!(!accepts(&g, r#"{"a":"x"}"#), "the type still binds");
        // A union type falls back to permitting anything.
        let u = schema(r#"{"type":["string","null"]}"#);
        assert!(accepts(&u, "\"x\"") && accepts(&u, "null") && accepts(&u, "1"));
    }

    #[test]
    fn impossible_schemas_are_refused() {
        assert!(Grammar::from_schema(&serde_json::json!({"enum": []})).is_err());
        assert!(Grammar::from_schema(&serde_json::json!({"type": "widget"})).is_err());
        assert!(
            Grammar::from_schema(&serde_json::json!({"type":"array","minItems":5,"maxItems":2})).is_err(),
            "maxItems below minItems"
        );
        let many: Vec<i32> = (0..MAX_OPTIONS as i32 + 1).collect();
        assert!(Grammar::from_schema(&serde_json::json!({"enum": many})).is_err(), "oversized enum");
    }

    /// Nesting past the stack limit must fail cleanly rather than corrupt state.
    #[test]
    fn depth_is_bounded() {
        let g = Grammar::json();
        let deep = "[".repeat(MAX_DEPTH + 4);
        let mut m = g.start();
        assert!(!g.feed_all(&mut m, deep.as_bytes()));
        // And a failed machine stays failed.
        assert!(!g.feed(&mut m, b']'));
        assert!(!g.complete(&m));
    }

    /// Bytes may arrive in any grouping, since a token is an arbitrary slice.
    #[test]
    fn splitting_the_input_anywhere_gives_the_same_verdict() {
        let g = schema(
            r#"{"type":"object","properties":{"k":{"type":"array","items":{"type":"number"}}},"required":["k"]}"#,
        );
        let doc = r#"{"k":[1.5,-2e3]}"#;
        for cut in 0..=doc.len() {
            let mut m = g.start();
            assert!(g.feed_all(&mut m, &doc.as_bytes()[..cut]), "prefix of {cut} rejected");
            assert!(g.feed_all(&mut m, &doc.as_bytes()[cut..]), "suffix from {cut} rejected");
            assert!(g.complete(&m), "split at {cut} left it incomplete");
        }
    }
}
