//! Rendering a checkpoint's own chat template.
//!
//! Guessing a prompt format from the control tokens in a vocabulary works for
//! the three families that have one, and quietly mis-prompts everything else — a
//! model that wants its system message folded a particular way, or a role marker
//! spelled slightly differently, gets no say. But the format is not really a
//! guess: instruct checkpoints ship it, as a Jinja template in
//! `tokenizer_config.json`. So the engine reads it.
//!
//! This is a subset of Jinja, not Jinja: `{{ }}`, `{% if %}`, `{% for %}`,
//! `{% set %}`, whitespace control, the operators and filters chat templates
//! actually use. Anything outside it is a parse error rather than a silent
//! misrendering, and the caller falls back to detecting a format from the
//! vocabulary — which is why the old path is still there.
//!
//! Values are [`serde_json::Value`] throughout, since that is exactly the data
//! model Jinja expects and the messages arrive as JSON anyway.

use serde_json::{Map, Value};

// -------------------------------------------------------------------- lexing

/// A raw template piece, with the whitespace-control marks it carried.
#[derive(Debug, PartialEq)]
enum Piece {
    Text(String),
    /// `{{ ... }}`
    Out(String, bool, bool),
    /// `{% ... %}`
    Tag(String, bool, bool),
}

fn lex(src: &str) -> Result<Vec<Piece>, String> {
    let b = src.as_bytes();
    let mut out: Vec<Piece> = Vec::new();
    let mut text = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'{' && i + 1 < b.len() && matches!(b[i + 1], b'{' | b'%' | b'#') {
            let kind = b[i + 1];
            let close: &[u8] = match kind {
                b'{' => b"}}",
                b'%' => b"%}",
                _ => b"#}",
            };
            let body_start = i + 2;
            let end = find(b, close, body_start).ok_or_else(|| format!("unclosed {{{}", kind as char))?;
            let raw = &src[body_start..end];
            i = end + 2;
            if kind == b'#' {
                continue; // a comment contributes nothing
            }
            // `{{-` and `-}}` strip the whitespace on that side.
            let left = raw.starts_with('-');
            let right = raw.ends_with('-');
            let body = raw.trim_start_matches('-').trim_end_matches('-').trim().to_string();
            if !text.is_empty() {
                out.push(Piece::Text(std::mem::take(&mut text)));
            }
            out.push(if kind == b'{' { Piece::Out(body, left, right) } else { Piece::Tag(body, left, right) });
        } else {
            text.push(src[i..].chars().next().unwrap());
            i += src[i..].chars().next().unwrap().len_utf8();
        }
    }
    if !text.is_empty() {
        out.push(Piece::Text(text));
    }
    // Apply whitespace control now, while the neighbours are still adjacent.
    let marks: Vec<(bool, bool)> = out
        .iter()
        .map(|p| match p {
            Piece::Out(_, l, r) | Piece::Tag(_, l, r) => (*l, *r),
            Piece::Text(_) => (false, false),
        })
        .collect();
    for (idx, (left, right)) in marks.iter().copied().enumerate() {
        if left && idx > 0 {
            if let Some(Piece::Text(t)) = out.get_mut(idx - 1) {
                let trimmed = t.trim_end().to_string();
                *t = trimmed;
            }
        }
        if right {
            if let Some(Piece::Text(t)) = out.get_mut(idx + 1) {
                let trimmed = t.trim_start().to_string();
                *t = trimmed;
            }
        }
    }
    Ok(out)
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    (from..hay.len().saturating_sub(needle.len() - 1)).find(|i| &hay[*i..*i + needle.len()] == needle)
}

// ------------------------------------------------------------------- the tree

#[derive(Debug)]
enum Node {
    Text(String),
    Out(Expr),
    /// `(condition, body)` per branch, plus the `else` body.
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
    For {
        var: String,
        iter: Expr,
        body: Vec<Node>,
    },
    Set(String, Expr),
}

#[derive(Debug, Clone)]
enum Expr {
    Lit(Value),
    Var(String),
    /// `a.b` or `a['b']` — the same operation either way.
    Index(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Bin(&'static str, Box<Expr>, Box<Expr>),
    /// `value | name(args)`
    Filter(Box<Expr>, String, Vec<Expr>),
    /// `a if c else b`
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    List(Vec<Expr>),
}

/// A parsed chat template.
#[derive(Debug)]
pub struct Template {
    nodes: Vec<Node>,
    source: String,
}

impl Template {
    /// Parse a Jinja chat template, or say why it cannot be handled.
    pub fn parse(src: &str) -> Result<Template, String> {
        let pieces = lex(src)?;
        let mut at = 0usize;
        let nodes = parse_block(&pieces, &mut at, &[])?;
        if at < pieces.len() {
            return Err(format!("unexpected {:?}", pieces[at]));
        }
        Ok(Template { nodes, source: src.to_string() })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render with `messages`, and the flags templates expect to find.
    pub fn render(
        &self,
        messages: &[Value],
        add_generation_prompt: bool,
        bos: &str,
        eos: &str,
    ) -> Result<String, String> {
        let mut scope = Map::new();
        scope.insert("messages".into(), Value::Array(messages.to_vec()));
        scope.insert("add_generation_prompt".into(), Value::Bool(add_generation_prompt));
        scope.insert("bos_token".into(), Value::String(bos.into()));
        scope.insert("eos_token".into(), Value::String(eos.into()));
        // Templates that branch on tools must take the no-tools path.
        scope.insert("tools".into(), Value::Null);
        scope.insert("tools_in_user_message".into(), Value::Bool(false));
        let mut out = String::new();
        exec(&self.nodes, &mut scope, &mut out, 0)?;
        Ok(out)
    }
}

/// Parse nodes until one of `until` (a `{% ... %}` keyword) is reached.
fn parse_block(pieces: &[Piece], at: &mut usize, until: &[&str]) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    while *at < pieces.len() {
        match &pieces[*at] {
            Piece::Text(t) => {
                if !t.is_empty() {
                    nodes.push(Node::Text(t.clone()));
                }
                *at += 1;
            }
            Piece::Out(src, ..) => {
                nodes.push(Node::Out(parse_expr(src)?));
                *at += 1;
            }
            Piece::Tag(src, ..) => {
                let word = src.split_whitespace().next().unwrap_or("");
                if until.contains(&word) {
                    return Ok(nodes);
                }
                *at += 1;
                match word {
                    "if" => {
                        let mut branches = vec![(parse_expr(rest(src, "if"))?, Vec::new())];
                        let mut otherwise = Vec::new();
                        // Collect branches until endif, following elif/else.
                        loop {
                            let body = parse_block(pieces, at, &["elif", "else", "endif"])?;
                            let closing = match pieces.get(*at) {
                                Some(Piece::Tag(s, ..)) => s.clone(),
                                _ => return Err("unclosed {% if %}".into()),
                            };
                            let word = closing.split_whitespace().next().unwrap_or("");
                            match word {
                                "elif" => {
                                    branches.last_mut().unwrap().1 = body;
                                    branches.push((parse_expr(rest(&closing, "elif"))?, Vec::new()));
                                    *at += 1;
                                }
                                "else" => {
                                    branches.last_mut().unwrap().1 = body;
                                    *at += 1;
                                    otherwise = parse_block(pieces, at, &["endif"])?;
                                    *at += 1;
                                    break;
                                }
                                _ => {
                                    branches.last_mut().unwrap().1 = body;
                                    *at += 1;
                                    break;
                                }
                            }
                        }
                        nodes.push(Node::If(branches, otherwise));
                    }
                    "for" => {
                        let body_src = rest(src, "for");
                        let (var, iter) = body_src.split_once(" in ").ok_or("{% for %} without `in`")?;
                        let var = var.trim();
                        if var.contains(',') {
                            return Err("{% for %} over pairs is not supported".into());
                        }
                        let iter = parse_expr(iter)?;
                        let body = parse_block(pieces, at, &["endfor", "else"])?;
                        match pieces.get(*at) {
                            Some(Piece::Tag(s, ..)) if s.starts_with("endfor") => *at += 1,
                            _ => return Err("unclosed {% for %}".into()),
                        }
                        nodes.push(Node::For { var: var.to_string(), iter, body });
                    }
                    "set" => {
                        let body = rest(src, "set");
                        let (name, value) = body.split_once('=').ok_or("{% set %} without `=`")?;
                        nodes.push(Node::Set(name.trim().to_string(), parse_expr(value)?));
                    }
                    // Harmless in a chat template: they affect nothing we model.
                    "generation" | "endgeneration" => {}
                    other => return Err(format!("unsupported tag: {{% {other} %}}")),
                }
            }
        }
    }
    Ok(nodes)
}

fn rest<'a>(src: &'a str, word: &str) -> &'a str {
    src[word.len()..].trim()
}

// --------------------------------------------------------------- expressions

/// How deep an expression may nest. Recursive descent costs stack per level, and
/// the source is a downloaded file, so `((((((...` has to be a parse error well
/// before it is a segfault. No real template comes close.
const MAX_EXPR_DEPTH: usize = 64;

struct Lexer<'a> {
    s: &'a [u8],
    i: usize,
    src: &'a str,
    depth: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Lexer<'a> {
        Lexer { s: src.as_bytes(), i: 0, src, depth: 0 }
    }

    /// Descend one level, refusing to go deeper than the stack can take.
    fn deeper(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            return Err("expression nests too deeply".into());
        }
        Ok(())
    }

    fn space(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek_word(&mut self) -> String {
        self.space();
        let start = self.i;
        let mut j = self.i;
        while j < self.s.len() && (self.s[j].is_ascii_alphanumeric() || self.s[j] == b'_') {
            j += 1;
        }
        self.text(start..j)
    }

    /// A byte range of the source as text. Every caller here selects ASCII, but
    /// this goes through the lossy conversion rather than `str` indexing on
    /// principle: the offsets are driven by untrusted input, and an index that
    /// lands inside a multi-byte character must not be a panic.
    fn text(&self, r: std::ops::Range<usize>) -> String {
        String::from_utf8_lossy(&self.s[r]).into_owned()
    }

    /// Whatever is left, for an error message.
    fn rest(&self) -> String {
        self.text(self.i.min(self.s.len())..self.s.len())
    }

    fn eat_word(&mut self, w: &str) -> bool {
        if self.peek_word() == w {
            self.space();
            self.i += w.len();
            true
        } else {
            false
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        self.space();
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    /// The character at the cursor, or `None` if the cursor is not on a
    /// character boundary — `str::get` returns `None` there rather than
    /// panicking, which is the whole reason it is used.
    fn here(&self) -> Option<char> {
        self.src.get(self.i..).and_then(|r| r.chars().next())
    }

    fn done(&mut self) -> bool {
        self.space();
        self.i >= self.s.len()
    }
}

fn parse_expr(src: &str) -> Result<Expr, String> {
    let mut lx = Lexer::new(src);
    let e = p_cond(&mut lx)?;
    if !lx.done() {
        return Err(format!("cannot parse expression: {src:?} (stopped at {:?})", lx.rest()));
    }
    Ok(e)
}

/// `a if c else b`, the inline conditional Llama's template leans on.
///
/// Every nested sub-expression — parentheses, list items, an index, a filter's
/// arguments — is parsed by a recursive call to this function, and the level is
/// held for as long as that call runs. That makes this the one place the depth
/// has to be counted (`not` and unary minus recurse without passing through it,
/// and count themselves).
fn p_cond(lx: &mut Lexer) -> Result<Expr, String> {
    lx.deeper()?;
    let e = p_cond_inner(lx);
    lx.depth -= 1;
    e
}

fn p_cond_inner(lx: &mut Lexer) -> Result<Expr, String> {
    let a = p_or(lx)?;
    if lx.eat_word("if") {
        let c = p_or(lx)?;
        // A bare `x if c` with no else yields nothing when false.
        let b = if lx.eat_word("else") { p_cond(lx)? } else { Expr::Lit(Value::String(String::new())) };
        return Ok(Expr::Cond(Box::new(c), Box::new(a), Box::new(b)));
    }
    Ok(a)
}

fn p_or(lx: &mut Lexer) -> Result<Expr, String> {
    let mut a = p_and(lx)?;
    while lx.eat_word("or") {
        a = Expr::Bin("or", Box::new(a), Box::new(p_and(lx)?));
    }
    Ok(a)
}

fn p_and(lx: &mut Lexer) -> Result<Expr, String> {
    let mut a = p_not(lx)?;
    while lx.eat_word("and") {
        a = Expr::Bin("and", Box::new(a), Box::new(p_not(lx)?));
    }
    Ok(a)
}

fn p_not(lx: &mut Lexer) -> Result<Expr, String> {
    if lx.eat_word("not") {
        // `not not not ...` recurses here without reaching `p_primary`, so this
        // is the second place the depth has to be counted.
        lx.deeper()?;
        let e = Expr::Not(Box::new(p_not(lx)?));
        lx.depth -= 1;
        return Ok(e);
    }
    p_cmp(lx)
}

fn p_cmp(lx: &mut Lexer) -> Result<Expr, String> {
    let a = p_add(lx)?;
    // `is defined` / `is not none` and friends, as a comparison of one operand.
    if lx.eat_word("is") {
        let negate = lx.eat_word("not");
        let what = lx.peek_word();
        lx.i += what.len();
        let test = match what.as_str() {
            "defined" | "none" | "string" | "mapping" | "sequence" | "iterable" => what,
            other => return Err(format!("unsupported test: is {other}")),
        };
        let e = Expr::Bin("is", Box::new(a), Box::new(Expr::Lit(Value::String(test))));
        return Ok(if negate { Expr::Not(Box::new(e)) } else { e });
    }
    for op in ["==", "!=", ">=", "<=", ">", "<"] {
        if lx.eat(op) {
            let b = p_add(lx)?;
            let op: &'static str = ["==", "!=", ">=", "<=", ">", "<"].iter().find(|o| **o == op).unwrap();
            return Ok(Expr::Bin(op, Box::new(a), Box::new(b)));
        }
    }
    if lx.eat_word("not") {
        // `x not in y`
        if lx.eat_word("in") {
            return Ok(Expr::Not(Box::new(Expr::Bin("in", Box::new(a), Box::new(p_add(lx)?)))));
        }
        return Err("stray `not`".into());
    }
    if lx.eat_word("in") {
        return Ok(Expr::Bin("in", Box::new(a), Box::new(p_add(lx)?)));
    }
    Ok(a)
}

fn p_add(lx: &mut Lexer) -> Result<Expr, String> {
    let mut a = p_postfix(lx)?;
    loop {
        // `~` is Jinja's explicit string concatenation; `+` does double duty as
        // concatenation and addition, so both land on the same node.
        if lx.eat("~") || lx.eat("+") {
            a = Expr::Bin("+", Box::new(a), Box::new(p_postfix(lx)?));
        } else if lx.eat("-") {
            a = Expr::Bin("-", Box::new(a), Box::new(p_postfix(lx)?));
        } else {
            return Ok(a);
        }
    }
}

fn p_postfix(lx: &mut Lexer) -> Result<Expr, String> {
    let mut a = p_primary(lx)?;
    loop {
        lx.space();
        if lx.eat("[") {
            let idx = p_cond(lx)?;
            if !lx.eat("]") {
                return Err("unclosed `[`".into());
            }
            a = Expr::Index(Box::new(a), Box::new(idx));
        } else if lx.eat(".") {
            let name = lx.peek_word();
            if name.is_empty() {
                return Err("`.` with no field".into());
            }
            lx.i += name.len();
            // `x.items()` and friends: a call with no arguments reads as a filter.
            if lx.eat("(") {
                let args = p_args(lx)?;
                a = Expr::Filter(Box::new(a), name, args);
            } else {
                a = Expr::Index(Box::new(a), Box::new(Expr::Lit(Value::String(name))));
            }
        } else if lx.eat("|") {
            let name = lx.peek_word();
            if name.is_empty() {
                return Err("`|` with no filter".into());
            }
            lx.i += name.len();
            let args = if lx.eat("(") { p_args(lx)? } else { Vec::new() };
            a = Expr::Filter(Box::new(a), name, args);
        } else {
            return Ok(a);
        }
    }
}

fn p_args(lx: &mut Lexer) -> Result<Vec<Expr>, String> {
    let mut args = Vec::new();
    if lx.eat(")") {
        return Ok(args);
    }
    loop {
        args.push(p_cond(lx)?);
        if lx.eat(",") {
            continue;
        }
        if lx.eat(")") {
            return Ok(args);
        }
        return Err("unclosed `(`".into());
    }
}

fn p_primary(lx: &mut Lexer) -> Result<Expr, String> {
    lx.space();
    if lx.i >= lx.s.len() {
        return Err("expression ended early".into());
    }
    let c = lx.s[lx.i];
    if c == b'\'' || c == b'"' {
        let quote = c;
        lx.i += 1;
        let mut val = String::new();
        while lx.i < lx.s.len() && lx.s[lx.i] != quote {
            if lx.s[lx.i] == b'\\' && lx.i + 1 < lx.s.len() {
                lx.i += 1;
                // The escaped thing is a whole character, not a byte: stepping
                // one byte past `\` in front of a multi-byte character would
                // leave the cursor inside it.
                let Some(ch) = lx.here() else { return Err("string is not valid text".into()) };
                val.push(match ch {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    other => other,
                });
                lx.i += ch.len_utf8();
                continue;
            }
            let Some(ch) = lx.here() else { return Err("string is not valid text".into()) };
            val.push(ch);
            lx.i += ch.len_utf8();
        }
        if lx.i >= lx.s.len() {
            return Err("unterminated string".into());
        }
        lx.i += 1;
        return Ok(Expr::Lit(Value::String(val)));
    }
    if lx.eat("(") {
        let e = p_cond(lx)?;
        if !lx.eat(")") {
            return Err("unclosed `(`".into());
        }
        return Ok(e);
    }
    if lx.eat("[") {
        let mut items = Vec::new();
        if !lx.eat("]") {
            loop {
                items.push(p_cond(lx)?);
                if lx.eat(",") {
                    continue;
                }
                if lx.eat("]") {
                    break;
                }
                return Err("unclosed `[`".into());
            }
        }
        return Ok(Expr::List(items));
    }
    // Unary minus, which is how a template writes `messages[-1]`. `-----1`
    // recurses back here without reaching `p_cond`, so it counts itself.
    if c == b'-' {
        lx.i += 1;
        lx.deeper()?;
        let inner = p_postfix(lx)?;
        lx.depth -= 1;
        return Ok(match inner {
            Expr::Lit(Value::Number(n)) => match n.as_i64() {
                Some(i) => Expr::Lit(serde_json::json!(-i)),
                None => Expr::Lit(serde_json::json!(-n.as_f64().unwrap_or(0.0))),
            },
            other => Expr::Bin("-", Box::new(Expr::Lit(serde_json::json!(0))), Box::new(other)),
        });
    }
    if c.is_ascii_digit() {
        let start = lx.i;
        while lx.i < lx.s.len() && (lx.s[lx.i].is_ascii_digit() || lx.s[lx.i] == b'.') {
            lx.i += 1;
        }
        let text = lx.text(start..lx.i);
        // An integer literal stays an integer: Jinja renders `1` as "1", and a
        // template that concatenates one into a prompt must not get "1.0".
        return Ok(Expr::Lit(if text.contains('.') {
            serde_json::json!(text.parse::<f64>().map_err(|_| "bad number".to_string())?)
        } else {
            serde_json::json!(text.parse::<i64>().map_err(|_| "bad number".to_string())?)
        }));
    }
    let word = lx.peek_word();
    if word.is_empty() {
        return Err(format!("unexpected {:?}", lx.rest()));
    }
    lx.i += word.len();
    Ok(match word.as_str() {
        "true" | "True" => Expr::Lit(Value::Bool(true)),
        "false" | "False" => Expr::Lit(Value::Bool(false)),
        "none" | "None" => Expr::Lit(Value::Null),
        _ => Expr::Var(word),
    })
}

// ------------------------------------------------------------------ execution

/// Jinja truthiness: empty is false, and so is zero.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// How a value renders inside `{{ }}`.
fn stringify(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        other => other.to_string(),
    }
}

fn exec(nodes: &[Node], scope: &mut Map<String, Value>, out: &mut String, depth: usize) -> Result<(), String> {
    if depth > 32 {
        return Err("template nests too deeply".into());
    }
    for n in nodes {
        match n {
            Node::Text(t) => out.push_str(t),
            Node::Out(e) => out.push_str(&stringify(&eval(e, scope)?)),
            Node::If(branches, otherwise) => {
                let mut taken = false;
                for (cond, body) in branches {
                    if truthy(&eval(cond, scope)?) {
                        exec(body, scope, out, depth + 1)?;
                        taken = true;
                        break;
                    }
                }
                if !taken {
                    exec(otherwise, scope, out, depth + 1)?;
                }
            }
            Node::For { var, iter, body } => {
                let seq = eval(iter, scope)?;
                let items: Vec<Value> = match seq {
                    Value::Array(a) => a,
                    Value::Object(o) => o.into_iter().map(|(k, v)| serde_json::json!([k, v])).collect(),
                    Value::String(s) => s.chars().map(|c| Value::String(c.into())).collect(),
                    Value::Null => Vec::new(),
                    other => return Err(format!("cannot iterate {other}")),
                };
                let n = items.len();
                // Shadow rather than clobber: the loop variable is scoped.
                let saved = scope.get(var).cloned();
                let saved_loop = scope.get("loop").cloned();
                for (i, item) in items.into_iter().enumerate() {
                    scope.insert(var.clone(), item);
                    scope.insert(
                        "loop".into(),
                        serde_json::json!({
                            "index0": i, "index": i + 1, "first": i == 0, "last": i + 1 == n,
                            "length": n, "revindex0": n - 1 - i, "revindex": n - i,
                        }),
                    );
                    exec(body, scope, out, depth + 1)?;
                }
                match saved {
                    Some(v) => scope.insert(var.clone(), v),
                    None => scope.remove(var),
                };
                match saved_loop {
                    Some(v) => scope.insert("loop".into(), v),
                    None => scope.remove("loop"),
                };
            }
            Node::Set(name, e) => {
                let v = eval(e, scope)?;
                scope.insert(name.clone(), v);
            }
        }
    }
    Ok(())
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

fn eval(e: &Expr, scope: &Map<String, Value>) -> Result<Value, String> {
    Ok(match e {
        Expr::Lit(v) => v.clone(),
        // An undefined name is null, as Jinja's default undefined behaves in a
        // boolean test — which is the only way chat templates use it.
        Expr::Var(name) => scope.get(name).cloned().unwrap_or(Value::Null),
        Expr::List(items) => Value::Array(items.iter().map(|i| eval(i, scope)).collect::<Result<_, _>>()?),
        Expr::Index(base, idx) => {
            let b = eval(base, scope)?;
            let i = eval(idx, scope)?;
            match (&b, &i) {
                (Value::Object(o), Value::String(k)) => o.get(k).cloned().unwrap_or(Value::Null),
                (Value::Array(a), Value::Number(n)) => {
                    let n = n.as_i64().unwrap_or(0);
                    // Negative indices count from the end, as in Python.
                    let at = if n < 0 { a.len() as i64 + n } else { n };
                    a.get(at.max(0) as usize).cloned().unwrap_or(Value::Null)
                }
                (Value::String(s), Value::Number(n)) => {
                    let n = n.as_i64().unwrap_or(0);
                    let at = if n < 0 { s.chars().count() as i64 + n } else { n };
                    s.chars().nth(at.max(0) as usize).map(|c| Value::String(c.into())).unwrap_or(Value::Null)
                }
                _ => Value::Null,
            }
        }
        Expr::Not(a) => Value::Bool(!truthy(&eval(a, scope)?)),
        Expr::Cond(c, a, b) => {
            if truthy(&eval(c, scope)?) {
                eval(a, scope)?
            } else {
                eval(b, scope)?
            }
        }
        Expr::Bin(op, l, r) => {
            // `and`/`or` short-circuit and yield the operand, not a bool.
            if *op == "and" {
                let a = eval(l, scope)?;
                return Ok(if truthy(&a) { eval(r, scope)? } else { a });
            }
            if *op == "or" {
                let a = eval(l, scope)?;
                return Ok(if truthy(&a) { a } else { eval(r, scope)? });
            }
            let a = eval(l, scope)?;
            let b = eval(r, scope)?;
            match *op {
                "==" => Value::Bool(a == b),
                "!=" => Value::Bool(a != b),
                ">" | "<" | ">=" | "<=" => {
                    let (x, y) = (as_f64(&a).unwrap_or(0.0), as_f64(&b).unwrap_or(0.0));
                    Value::Bool(match *op {
                        ">" => x > y,
                        "<" => x < y,
                        ">=" => x >= y,
                        _ => x <= y,
                    })
                }
                "in" => Value::Bool(match &b {
                    Value::Array(items) => items.contains(&a),
                    Value::Object(o) => matches!(&a, Value::String(k) if o.contains_key(k)),
                    Value::String(s) => matches!(&a, Value::String(n) if s.contains(n.as_str())),
                    _ => false,
                }),
                "is" => {
                    let test = match &b {
                        Value::String(s) => s.as_str(),
                        _ => "",
                    };
                    Value::Bool(match test {
                        "defined" => !a.is_null(),
                        "none" => a.is_null(),
                        "string" => a.is_string(),
                        "mapping" => a.is_object(),
                        "sequence" | "iterable" => a.is_array() || a.is_string(),
                        _ => false,
                    })
                }
                "+" => match (&a, &b) {
                    (Value::String(x), _) => Value::String(format!("{x}{}", stringify(&b))),
                    (_, Value::String(y)) => Value::String(format!("{}{y}", stringify(&a))),
                    (Value::Array(x), Value::Array(y)) => Value::Array(x.iter().chain(y).cloned().collect()),
                    _ => serde_json::json!(as_f64(&a).unwrap_or(0.0) + as_f64(&b).unwrap_or(0.0)),
                },
                "-" => serde_json::json!(as_f64(&a).unwrap_or(0.0) - as_f64(&b).unwrap_or(0.0)),
                other => return Err(format!("unsupported operator: {other}")),
            }
        }
        Expr::Filter(base, name, args) => {
            let v = eval(base, scope)?;
            let arg =
                |i: usize| -> Result<Value, String> { args.get(i).map(|a| eval(a, scope)).unwrap_or(Ok(Value::Null)) };
            match name.as_str() {
                "trim" => Value::String(stringify(&v).trim().to_string()),
                "lower" => Value::String(stringify(&v).to_lowercase()),
                "upper" => Value::String(stringify(&v).to_uppercase()),
                "capitalize" => {
                    let s = stringify(&v);
                    let mut c = s.chars();
                    Value::String(match c.next() {
                        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                        None => String::new(),
                    })
                }
                "string" => Value::String(stringify(&v)),
                "int" => serde_json::json!(as_f64(&v).unwrap_or_else(|| stringify(&v).parse().unwrap_or(0.0)) as i64),
                "length" | "count" => serde_json::json!(match &v {
                    Value::Array(a) => a.len(),
                    Value::Object(o) => o.len(),
                    Value::String(s) => s.chars().count(),
                    _ => 0,
                }),
                "first" => match &v {
                    Value::Array(a) => a.first().cloned().unwrap_or(Value::Null),
                    _ => Value::Null,
                },
                "last" => match &v {
                    Value::Array(a) => a.last().cloned().unwrap_or(Value::Null),
                    _ => Value::Null,
                },
                "list" | "items" => v,
                "reverse" => match v {
                    Value::Array(mut a) => {
                        a.reverse();
                        Value::Array(a)
                    }
                    other => other,
                },
                "join" => {
                    let sep = match arg(0)? {
                        Value::String(s) => s,
                        _ => String::new(),
                    };
                    match &v {
                        Value::Array(a) => Value::String(a.iter().map(stringify).collect::<Vec<_>>().join(&sep)),
                        other => Value::String(stringify(other)),
                    }
                }
                "default" => {
                    if v.is_null() {
                        arg(0)?
                    } else {
                        v
                    }
                }
                "tojson" => Value::String(v.to_string()),
                // `selectattr`, `map`, `rejectattr` and the rest are used only by
                // tool-calling templates, which this does not claim to render.
                other => return Err(format!("unsupported filter: {other}")),
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<Value> {
        pairs.iter().map(|(r, c)| serde_json::json!({"role": r, "content": c})).collect()
    }

    fn render(src: &str, m: &[(&str, &str)], gen: bool) -> String {
        Template::parse(src).expect("parse").render(&msgs(m), gen, "<s>", "</s>").expect("render")
    }

    #[test]
    fn text_passes_through_untouched() {
        assert_eq!(render("hello", &[], false), "hello");
        assert_eq!(render("a{# comment #}b", &[], false), "ab");
    }

    #[test]
    fn expressions_interpolate() {
        assert_eq!(render("{{ 'x' }}", &[], false), "x");
        assert_eq!(render("{{ bos_token }}", &[], false), "<s>");
        assert_eq!(render("{{ 'a' + 'b' }}", &[], false), "ab");
        assert_eq!(render("{{ 'a' ~ 1 }}", &[], false), "a1");
        assert_eq!(render("{{ messages | length }}", &[("user", "hi"), ("user", "yo")], false), "2");
        assert_eq!(render("{{ messages[0]['role'] }}", &[("user", "hi")], false), "user");
        assert_eq!(render("{{ messages[0].content }}", &[("user", "hi")], false), "hi");
        assert_eq!(render("{{ messages[-1].content }}", &[("a", "1"), ("b", "2")], false), "2");
        assert_eq!(render("{{ '  pad  ' | trim }}", &[], false), "pad");
    }

    #[test]
    fn conditionals_pick_a_branch() {
        assert_eq!(render("{% if true %}y{% else %}n{% endif %}", &[], false), "y");
        assert_eq!(render("{% if false %}y{% else %}n{% endif %}", &[], false), "n");
        assert_eq!(render("{% if 0 %}y{% elif 1 %}e{% else %}n{% endif %}", &[], false), "e");
        assert_eq!(render("{% if not false %}y{% endif %}", &[], false), "y");
        assert_eq!(render("{{ 'y' if add_generation_prompt else 'n' }}", &[], true), "y");
        assert_eq!(render("{{ 'y' if add_generation_prompt else 'n' }}", &[], false), "n");
        // Undefined names are falsy rather than an error.
        assert_eq!(render("{% if tools %}t{% else %}none{% endif %}", &[], false), "none");
        assert_eq!(render("{% if nothing is defined %}d{% else %}u{% endif %}", &[], false), "u");
    }

    #[test]
    fn loops_expose_position() {
        let src = "{% for m in messages %}{{ loop.index0 }}:{{ m.role }}{% if not loop.last %},{% endif %}{% endfor %}";
        assert_eq!(render(src, &[("a", "1"), ("b", "2"), ("c", "3")], false), "0:a,1:b,2:c");
        assert_eq!(render("{% for m in messages %}x{% endfor %}", &[], false), "");
        // `loop` and the variable are scoped to the loop.
        assert_eq!(render("{% for m in messages %}{% endfor %}{{ m }}{{ loop }}", &[("a", "1")], false), "");
    }

    #[test]
    fn set_binds_a_name() {
        assert_eq!(render("{% set x = 'v' %}{{ x }}", &[], false), "v");
        assert_eq!(render("{% set n = messages | length %}{{ n }}", &[("a", "1")], false), "1");
    }

    /// `{%-` and `-%}` are load-bearing in real templates: without them the
    /// prompt gains newlines the model was never trained on.
    #[test]
    fn whitespace_control_strips_the_right_side() {
        assert_eq!(render("a\n  {{- 'x' }}", &[], false), "ax");
        assert_eq!(render("{{ 'x' -}}\n  b", &[], false), "xb");
        assert_eq!(render("a\n{% if true %}\nb{% endif %}", &[], false), "a\n\nb");
        assert_eq!(render("a\n{%- if true %}b{% endif %}", &[], false), "ab");
    }

    /// The three formats the engine used to hard-code, now expressed as the
    /// templates their checkpoints actually ship. Matching the built-in output
    /// exactly is the point: the template path must be a strict improvement.
    #[test]
    fn chatml_renders_as_the_builtin_did() {
        let src = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + \
                   message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}\
                   {% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
        let got = render(src, &[("system", "S"), ("user", "U")], true);
        assert_eq!(got, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn a_llama3_shaped_template_renders() {
        let src = "{{ bos_token }}{% for message in messages %}\
                   {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + \
                   message['content'] | trim + '<|eot_id|>' }}{% endfor %}\
                   {% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";
        let got = render(src, &[("user", " hi ")], true);
        assert_eq!(
            got,
            "<s><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    /// Mistral folds the system message into the first user turn, which is
    /// exactly the sort of per-checkpoint detail a three-way guess cannot know.
    #[test]
    fn a_mistral_shaped_template_folds_the_system_turn() {
        let src = "{{ bos_token }}{% for message in messages %}\
                   {% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}\
                   {% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}\
                   {% endif %}{% endfor %}";
        let got = render(src, &[("user", "U"), ("assistant", "A"), ("user", "V")], false);
        assert_eq!(got, "<s>[INST] U [/INST]A</s>[INST] V [/INST]");
    }

    /// A template that loops with an index and a lookahead — the shape Qwen and
    /// several others use to decide whether to open an assistant turn.
    #[test]
    fn indexed_lookahead_works() {
        let src = "{% for i in messages %}{{ i.role }}{% endfor %}\
                   {% if messages[-1]['role'] != 'assistant' %}|open{% endif %}";
        assert_eq!(render(src, &[("user", "a")], false), "user|open");
        assert_eq!(render(src, &[("user", "a"), ("assistant", "b")], false), "userassistant");
    }

    #[test]
    fn nested_structures_compose() {
        let src = "{% for m in messages %}{% if m.role == 'user' %}[{% for c in m.content %}{{ c }}{% endfor %}]\
                   {% endif %}{% endfor %}";
        assert_eq!(render(src, &[("user", "ab")], false), "[ab]");
    }

    /// What the engine cannot render must be an error, so the caller falls back
    /// to detection rather than emitting a mangled prompt.
    #[test]
    fn the_unsupported_is_refused_not_guessed() {
        for src in [
            "{% macro x() %}{% endmacro %}",
            "{{ messages | selectattr('role') }}",
            "{% for a, b in items %}{% endfor %}",
            "{{ unclosed ",
            "{% if true %}no end",
            "{% for m in messages %}no end",
            "{{ 'a' ** 'b' }}",
            "{{ x is weird }}",
        ] {
            let r = Template::parse(src).and_then(|t| t.render(&msgs(&[("user", "u")]), true, "", ""));
            assert!(r.is_err(), "should have refused: {src:?} -> {r:?}");
        }
    }

    /// A template must not be able to spin forever or recurse without bound.
    #[test]
    fn depth_is_bounded() {
        let deep = "{% if true %}".repeat(40) + &"{% endif %}".repeat(40);
        let t = Template::parse(&deep).expect("parse");
        assert!(t.render(&[], false, "", "").is_err(), "deep nesting was not refused");
    }
}
