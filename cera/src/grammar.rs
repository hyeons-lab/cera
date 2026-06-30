//! GBNF grammar-constrained decoding (structured output).
//!
//! Parses a [GBNF grammar](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)
//! and, during generation, masks the sampler's logits so only tokens the grammar can
//! accept are sampled — guaranteeing the output conforms (e.g. valid JSON).
//!
//! The engine mirrors llama.cpp's `llama_grammar`: a grammar compiles to a flat element
//! vector per rule, and a [`GrammarState`] holds a set of "stacks" (parse frontiers) that
//! advance byte-by-byte. Matching is **byte-level** — character ranges are interpreted
//! over bytes, which is exact for ASCII/JSON grammars (the headline use case);
//! Unicode-codepoint ranges are a documented v2 limitation.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail, ensure};

// ── Compiled grammar ────────────────────────────────────────────────────────

/// A single compiled grammar element. Rules are `Vec<Elem>` where alternates are
/// separated by [`Elem::Alt`] and terminated by [`Elem::End`]. Mirrors the llama.cpp
/// element taxonomy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Elem {
    /// End of a rule (and of its final alternate).
    End,
    /// Separates alternates within a rule.
    Alt,
    /// Reference to another rule by id.
    RuleRef(u32),
    /// Start of a positive char set; the matched byte.
    Char(u8),
    /// Upper bound of a range, modifying the immediately preceding `Char`/`CharAlt`.
    CharRngUpper(u8),
    /// Start of a negated char set `[^...]`.
    CharNot(u8),
    /// Another member of the current char set.
    CharAlt(u8),
}

impl Elem {
    /// True if this element terminates a sequence/alternate (`End` or `Alt`).
    fn is_end_of_seq(self) -> bool {
        matches!(self, Elem::End | Elem::Alt)
    }
    fn char_value(self) -> u8 {
        match self {
            Elem::Char(c) | Elem::CharRngUpper(c) | Elem::CharNot(c) | Elem::CharAlt(c) => c,
            _ => 0,
        }
    }
}

/// A compiled GBNF grammar. Immutable and cheap to share via [`Arc`].
#[derive(Debug, Clone)]
pub struct Grammar {
    rules: Vec<Vec<Elem>>,
    root: u32,
}

impl Grammar {
    /// Parse a GBNF grammar from text.
    pub fn parse(src: &str) -> Result<Grammar> {
        Parser::new(src).parse_grammar()
    }

    /// Position into a rule's element list.
    #[inline]
    fn elem(&self, p: Pos) -> Elem {
        self.rules[p.rule as usize][p.idx as usize]
    }

    /// Is the element at `p` the end of its sequence (`End`/`Alt` or past the end)?
    #[inline]
    fn is_end_of_seq(&self, p: Pos) -> bool {
        let rule = &self.rules[p.rule as usize];
        p.idx as usize >= rule.len() || rule[p.idx as usize].is_end_of_seq()
    }
}

/// A pointer into a rule's element list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pos {
    rule: u32,
    idx: u32,
}

// ── Runtime matcher ─────────────────────────────────────────────────────────

/// Mutable per-generation grammar state: the set of active parse frontiers.
///
/// A stack is a list of [`Pos`] whose top points at the next terminal to match. An
/// **empty** stack means a complete derivation was reached (→ generation may stop).
#[derive(Clone)]
pub struct GrammarState {
    grammar: Arc<Grammar>,
    stacks: Vec<Vec<Pos>>,
}

impl GrammarState {
    /// Initialize from the grammar's `root` rule.
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let mut stacks = Vec::new();
        let root = grammar.root;
        // Iterate the root rule's alternates, advancing an initial stack for each.
        let mut i = 0u32;
        loop {
            let pos = Pos { rule: root, idx: i };
            let mut stack = Vec::new();
            if !grammar.is_end_of_seq(pos) {
                stack.push(pos);
            }
            advance_stack(&grammar, stack, &mut stacks);
            // Skip to the next alternate.
            while !grammar.is_end_of_seq(Pos { rule: root, idx: i }) {
                i += 1;
            }
            if grammar.elem(Pos { rule: root, idx: i }) == Elem::Alt {
                i += 1; // step past the ALT into the next alternate
            } else {
                break; // hit End
            }
        }
        GrammarState { grammar, stacks }
    }

    /// True when a complete derivation has been reached (an empty stack is present),
    /// i.e. the grammar permits termination (EOS) here.
    pub fn is_complete(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    /// True when no parse frontier survives — nothing more can be generated.
    pub fn is_dead(&self) -> bool {
        self.stacks.is_empty()
    }

    /// Would the grammar accept `bytes` from the current state (without committing)?
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            // A zero-byte token contributes nothing; never let it satisfy the grammar.
            return false;
        }
        let mut stacks = self.stacks.clone();
        for &b in bytes {
            stacks = step(&self.grammar, &stacks, b);
            if stacks.is_empty() {
                return false;
            }
        }
        true
    }

    /// Commit `bytes` to the state, advancing the frontiers. If a byte dead-ends the
    /// grammar, the state is left dead (`is_dead()` becomes true) and the remaining bytes
    /// are not applied. Callers should only `accept` tokens that [`accepts`](Self::accepts)
    /// returned true for.
    pub fn accept(&mut self, bytes: &[u8]) {
        for &b in bytes {
            let next = step(&self.grammar, &self.stacks, b);
            if next.is_empty() {
                // Caller should only `accept` tokens that `accepts()` returned true for;
                // guard anyway so a misuse leaves the state dead rather than corrupt.
                self.stacks = next;
                return;
            }
            self.stacks = next;
        }
    }
}

/// Expand `stack` until its top is a terminal (or it is empty), appending the resulting
/// ready-to-match stacks to `out` (deduplicated).
fn advance_stack(g: &Grammar, stack: Vec<Pos>, out: &mut Vec<Vec<Pos>>) {
    let Some(&top) = stack.last() else {
        // Empty stack = complete derivation. Keep one as the "accepting" marker.
        if !out.contains(&stack) {
            out.push(stack);
        }
        return;
    };
    match g.elem(top) {
        Elem::RuleRef(rule_id) => {
            let mut i = 0u32;
            loop {
                let alt_start = Pos {
                    rule: rule_id,
                    idx: i,
                };
                let mut new_stack = stack[..stack.len() - 1].to_vec();
                // Continuation: the element after the rule-ref in the current rule.
                let cont = Pos {
                    rule: top.rule,
                    idx: top.idx + 1,
                };
                if !g.is_end_of_seq(cont) {
                    new_stack.push(cont);
                }
                // The referenced alternate's first element, if non-empty.
                if !g.is_end_of_seq(alt_start) {
                    new_stack.push(alt_start);
                }
                advance_stack(g, new_stack, out);
                // Advance `i` to the next alternate of the referenced rule.
                while !g.is_end_of_seq(Pos {
                    rule: rule_id,
                    idx: i,
                }) {
                    i += 1;
                }
                if g.elem(Pos {
                    rule: rule_id,
                    idx: i,
                }) == Elem::Alt
                {
                    i += 1;
                } else {
                    break;
                }
            }
        }
        // Terminal — ready to match a byte (dedup so equal frontiers don't multiply).
        Elem::Char(_) | Elem::CharNot(_) if !out.contains(&stack) => out.push(stack),
        // Already-present terminal, or `End`/`Alt` (continuations skip them) — ignore.
        _ => {}
    }
}

/// Advance every stack in `stacks` by one byte `b`, returning the new stack set.
fn step(g: &Grammar, stacks: &[Vec<Pos>], b: u8) -> Vec<Vec<Pos>> {
    let mut out = Vec::new();
    for stack in stacks {
        let Some(&top) = stack.last() else {
            continue; // empty (accepting) stack can't consume more bytes
        };
        if let Some(next_idx) = match_char(&g.rules[top.rule as usize], top.idx, b) {
            let mut new_stack = stack[..stack.len() - 1].to_vec();
            let next = Pos {
                rule: top.rule,
                idx: next_idx,
            };
            if !g.is_end_of_seq(next) {
                new_stack.push(next);
            }
            advance_stack(g, new_stack, &mut out);
        }
    }
    out
}

/// Try to match byte `b` against the char set beginning at `rule[idx]`. On success,
/// returns the index of the element just past the whole set.
fn match_char(rule: &[Elem], idx: u32, b: u8) -> Option<u32> {
    let mut i = idx as usize;
    let negated = matches!(rule[i], Elem::CharNot(_));
    let mut found = false;
    loop {
        let lo = rule[i].char_value();
        // Optional range upper bound.
        let hi = if i + 1 < rule.len()
            && let Elem::CharRngUpper(h) = rule[i + 1]
        {
            i += 1;
            h
        } else {
            lo
        };
        if b >= lo && b <= hi {
            found = true;
        }
        // Another set member?
        if i + 1 < rule.len() && matches!(rule[i + 1], Elem::CharAlt(_)) {
            i += 1;
            continue;
        }
        break;
    }
    let matched = found != negated;
    if matched { Some(i as u32 + 1) } else { None }
}

// ── Token mask ──────────────────────────────────────────────────────────────

/// Per-vocabulary lookup that turns a [`GrammarState`] into a logit mask. Holds each
/// token's raw output bytes plus the EOS / special-token ids to handle specially.
pub struct GrammarMask {
    /// `token_bytes[id]` = the raw bytes token `id` emits (see `BpeTokenizer::token_output_bytes`).
    token_bytes: Vec<Vec<u8>>,
    eos: Option<u32>,
    special: Vec<bool>,
}

impl GrammarMask {
    /// Build from per-token output bytes, the EOS id, and a per-token "is special" flag.
    ///
    /// Panics if `token_bytes` and `special` differ in length (they must both equal the
    /// vocab size) — a real `assert` so a caller mistake fails loudly here rather than as
    /// an opaque index panic inside [`apply`](Self::apply) in release builds.
    pub fn new(token_bytes: Vec<Vec<u8>>, eos: Option<u32>, special: Vec<bool>) -> Self {
        assert_eq!(
            token_bytes.len(),
            special.len(),
            "GrammarMask: token_bytes and special must have the same (vocab) length"
        );
        GrammarMask {
            token_bytes,
            eos,
            special,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    /// The raw output bytes of token `id` (empty for out-of-range ids).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.token_bytes
            .get(id as usize)
            .map_or(&[], |v| v.as_slice())
    }

    /// Set `logits[id] = -inf` for every token the grammar can't accept in `state`.
    /// EOS is allowed iff the grammar is complete; special/control and empty-byte tokens
    /// are always masked. Returns the number of tokens left allowed (0 ⇒ dead end).
    pub fn apply(&self, state: &GrammarState, logits: &mut [f32]) -> usize {
        let n = self.token_bytes.len().min(logits.len());
        let mut allowed = 0usize;
        let complete = state.is_complete();
        for (id, logit) in logits.iter_mut().take(n).enumerate() {
            let ok = if Some(id as u32) == self.eos {
                complete
            } else if self.special[id] {
                false
            } else {
                state.accepts(&self.token_bytes[id])
            };
            if ok {
                allowed += 1;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }
        // Any logits beyond the vocab (shouldn't happen) are masked.
        for l in logits.iter_mut().skip(n) {
            *l = f32::NEG_INFINITY;
        }
        allowed
    }
}

// ── GBNF parser ─────────────────────────────────────────────────────────────

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    rules: Vec<Vec<Elem>>,
    symbol_ids: HashMap<String, u32>,
    defined: Vec<bool>,
}

impl<'a> Parser<'a> {
    /// Upper bound on a `{n,m}` repetition count. The desugaring inlines up to
    /// `count` copies of the repeated unit (each copy spans the unit's elements),
    /// so the worst-case expansion is `count × unit_size` — cap `count` so a
    /// grammar can't amplify a large unit into an OOM. 1024 is far more than any
    /// real fixed-width field needs; use `*`/`+` for genuinely unbounded counts.
    const MAX_REPETITION: usize = 1024;

    fn new(src: &'a str) -> Self {
        Parser {
            src: src.as_bytes(),
            pos: 0,
            rules: Vec::new(),
            symbol_ids: HashMap::new(),
            defined: Vec::new(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    /// Reserve (or fetch) a rule id for a named rule.
    fn symbol_id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.symbol_ids.get(name) {
            return id;
        }
        let id = self.rules.len() as u32;
        self.rules.push(Vec::new());
        self.defined.push(false);
        self.symbol_ids.insert(name.to_string(), id);
        id
    }

    /// Reserve a fresh anonymous rule id (for groups / repetitions).
    fn anon_rule(&mut self) -> u32 {
        let id = self.rules.len() as u32;
        self.rules.push(Vec::new());
        self.defined.push(true);
        id
    }

    /// Consume whitespace and `#` comments (newlines included).
    fn skip_ws(&mut self) {
        loop {
            match self.peek() {
                Some(b' ' | b'\t' | b'\r' | b'\n') => {
                    self.pos += 1;
                }
                Some(b'#') => {
                    while let Some(c) = self.peek() {
                        self.pos += 1;
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn is_word_byte(c: u8) -> bool {
        c.is_ascii_alphanumeric() || c == b'-' || c == b'_'
    }

    fn parse_name(&mut self) -> Result<String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if Self::is_word_byte(c) {
                self.pos += 1;
            } else {
                break;
            }
        }
        ensure!(self.pos > start, "expected a rule name at byte {}", start);
        Ok(String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
    }

    /// Lookahead: does the upcoming input start a new `name ::=` rule definition?
    fn looks_like_rule_start(&self) -> bool {
        let mut i = self.pos;
        let s = self.src;
        if i >= s.len() || !Self::is_word_byte(s[i]) {
            return false;
        }
        while i < s.len() && Self::is_word_byte(s[i]) {
            i += 1;
        }
        while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\r' | b'\n') {
            i += 1;
        }
        s[i..].starts_with(b"::=")
    }

    fn parse_grammar(mut self) -> Result<Grammar> {
        self.skip_ws();
        while self.peek().is_some() {
            self.parse_rule()?;
            self.skip_ws();
        }
        let root = *self
            .symbol_ids
            .get("root")
            .ok_or_else(|| anyhow::anyhow!("grammar has no `root` rule"))?;
        // Every referenced rule must be defined.
        for (name, &id) in &self.symbol_ids {
            ensure!(self.defined[id as usize], "undefined rule: `{name}`");
        }
        // Reject left recursion (incl. ε-cycles), which would infinite-loop the matcher.
        self.check_no_left_recursion()?;
        Ok(Grammar {
            rules: self.rules,
            root,
        })
    }

    /// Split a rule's element list into its alternates (the slices between `Alt`/`End`).
    fn alternates(&self, rule: usize) -> Vec<&[Elem]> {
        let elems = &self.rules[rule];
        let mut alts = Vec::new();
        let mut start = 0usize;
        for (i, e) in elems.iter().enumerate() {
            match e {
                Elem::Alt => {
                    alts.push(&elems[start..i]);
                    start = i + 1;
                }
                Elem::End => {
                    alts.push(&elems[start..i]);
                    break;
                }
                _ => {}
            }
        }
        alts
    }

    /// Which rules can derive the empty string (fixpoint over the alternates).
    fn compute_nullable(&self) -> Vec<bool> {
        let n = self.rules.len();
        let mut nullable = vec![false; n];
        loop {
            let mut changed = false;
            for r in 0..n {
                if nullable[r] {
                    continue;
                }
                // A rule is nullable if any alternate is entirely nullable rule-refs
                // (an empty alternate vacuously qualifies); a terminal makes it not.
                let any = self.alternates(r).iter().any(|alt| {
                    alt.iter().all(|e| match e {
                        Elem::RuleRef(b) => nullable[*b as usize],
                        _ => false,
                    })
                });
                if any {
                    nullable[r] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        nullable
    }

    fn rule_label(&self, id: u32) -> String {
        match self.symbol_ids.iter().find(|&(_, &v)| v == id) {
            Some((name, _)) => format!("rule `{name}`"),
            None => "an anonymous (…)/repetition subrule".to_string(),
        }
    }

    /// Reject left recursion: this stack matcher (like GBNF generally) requires that no
    /// rule can recurse without first consuming a byte. Left recursion — including
    /// ε-cycles such as `""*` or `A ::= B; B ::= A` — would infinite-loop `advance_stack`.
    fn check_no_left_recursion(&self) -> Result<()> {
        let n = self.rules.len();
        let nullable = self.compute_nullable();
        // Left-corner graph: A → B if some alternate of A reaches `RuleRef(B)` through a
        // prefix of nullable rule-refs (B can be entered without consuming a byte).
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (a, adj_a) in adj.iter_mut().enumerate() {
            for alt in self.alternates(a) {
                for &e in alt {
                    match e {
                        Elem::RuleRef(b) => {
                            adj_a.push(b);
                            if !nullable[b as usize] {
                                break; // non-nullable: nothing past it is a left corner
                            }
                        }
                        // A terminal consumes a byte → left corner ends here.
                        _ => break,
                    }
                }
            }
        }
        // Iterative 3-colour DFS for a back edge (a cycle in the left-corner graph).
        #[derive(Clone, Copy, PartialEq)]
        enum Color {
            White,
            Gray,
            Black,
        }
        let mut color = vec![Color::White; n];
        for s in 0..n {
            if color[s] != Color::White {
                continue;
            }
            color[s] = Color::Gray;
            let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
            while let Some(&(node, ci)) = stack.last() {
                if ci < adj[node].len() {
                    stack.last_mut().unwrap().1 += 1;
                    let b = adj[node][ci] as usize;
                    match color[b] {
                        Color::Gray => bail!(
                            "grammar is left-recursive: {} can recurse without consuming input",
                            self.rule_label(b as u32)
                        ),
                        Color::White => {
                            color[b] = Color::Gray;
                            stack.push((b, 0));
                        }
                        Color::Black => {}
                    }
                } else {
                    color[node] = Color::Black;
                    stack.pop();
                }
            }
        }
        Ok(())
    }

    fn parse_rule(&mut self) -> Result<()> {
        let name = self.parse_name()?;
        self.skip_ws();
        ensure!(
            self.src[self.pos..].starts_with(b"::="),
            "expected `::=` after rule `{name}`"
        );
        self.pos += 3;
        self.skip_ws();
        let id = self.symbol_id(&name);
        ensure!(!self.defined[id as usize], "rule `{name}` defined twice");
        let elems = self.parse_alternates(&name)?;
        self.rules[id as usize] = elems;
        self.defined[id as usize] = true;
        Ok(())
    }

    fn parse_alternates(&mut self, rule_name: &str) -> Result<Vec<Elem>> {
        let mut out = Vec::new();
        self.parse_sequence(&mut out, rule_name)?;
        self.skip_ws();
        while self.peek() == Some(b'|') {
            self.pos += 1;
            self.skip_ws();
            out.push(Elem::Alt);
            self.parse_sequence(&mut out, rule_name)?;
            self.skip_ws();
        }
        out.push(Elem::End);
        Ok(out)
    }

    fn parse_sequence(&mut self, out: &mut Vec<Elem>, rule_name: &str) -> Result<()> {
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b'|') | Some(b')') => break,
                _ if self.looks_like_rule_start() => break,
                _ => {}
            }
            let last_start = out.len();
            match self.peek().unwrap() {
                b'"' => self.parse_string_literal(out)?,
                b'[' => self.parse_char_class(out)?,
                b'(' => {
                    self.pos += 1; // '('
                    let sub = self.anon_rule();
                    let body = self.parse_alternates(rule_name)?;
                    self.rules[sub as usize] = body;
                    self.skip_ws();
                    ensure!(
                        self.peek() == Some(b')'),
                        "unclosed `(` in rule `{rule_name}`"
                    );
                    self.pos += 1; // ')'
                    out.push(Elem::RuleRef(sub));
                }
                c if Self::is_word_byte(c) => {
                    let name = self.parse_name()?;
                    let id = self.symbol_id(&name);
                    out.push(Elem::RuleRef(id));
                }
                c => bail!("unexpected `{}` in rule `{rule_name}`", c as char),
            }
            // Postfix repetition binds to the element just parsed (no
            // intervening whitespace, matching `* + ?`).
            match self.peek() {
                Some(op @ (b'*' | b'+' | b'?')) => {
                    self.pos += 1;
                    self.apply_repetition(out, last_start, op)?;
                }
                Some(b'{') => {
                    let (min, max) = self.parse_repetition_bound()?;
                    self.apply_bounded_repetition(out, last_start, min, max)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Rewrite `out[start..]` (the just-parsed unit) into a fresh recursive rule per the
    /// repetition operator, replacing it with a single `RuleRef`.
    fn apply_repetition(&mut self, out: &mut Vec<Elem>, start: usize, op: u8) -> Result<()> {
        // An empty unit (e.g. `""*`) would desugar to `S ::= S | ε` — a left-recursive
        // ε-cycle that infinite-loops the matcher. Reject it up front with a clear error.
        ensure!(
            start < out.len(),
            "repetition operator `{}` has nothing to repeat",
            op as char
        );
        let unit: Vec<Elem> = out.split_off(start);
        let sub = self.anon_rule();
        let mut body = Vec::new();
        match op {
            b'*' => {
                // S ::= unit S | ε
                body.extend_from_slice(&unit);
                body.push(Elem::RuleRef(sub));
                body.push(Elem::Alt);
                body.push(Elem::End);
            }
            b'+' => {
                // S ::= unit S | unit
                body.extend_from_slice(&unit);
                body.push(Elem::RuleRef(sub));
                body.push(Elem::Alt);
                body.extend_from_slice(&unit);
                body.push(Elem::End);
            }
            b'?' => {
                // S ::= unit | ε
                body.extend_from_slice(&unit);
                body.push(Elem::Alt);
                body.push(Elem::End);
            }
            _ => unreachable!(),
        }
        self.rules[sub as usize] = body;
        out.push(Elem::RuleRef(sub));
        Ok(())
    }

    /// Parse a bounded-repetition suffix `{n}`, `{n,}`, `{n,m}`, or `{,m}` (the
    /// opening `{` is at the cursor). Returns `(min, max)` where `max == None`
    /// means unbounded (`{n,}`). A missing min defaults to 0 (`{,m}`); a missing
    /// max in `{n}` form defaults to `min` (exact count). Counts are capped at
    /// [`Self::MAX_REPETITION`] so a pathological grammar can't blow the parser's
    /// memory up via the inline expansion.
    fn parse_repetition_bound(&mut self) -> Result<(usize, Option<usize>)> {
        self.pos += 1; // '{'
        let min_digits = self.parse_count_digits()?;
        let (min, max) = if self.peek() == Some(b',') {
            self.pos += 1; // ','
            let max_digits = self.parse_count_digits()?;
            // At least one of `{n,}` / `{,m}` / `{n,m}` must carry a bound.
            ensure!(
                min_digits.is_some() || max_digits.is_some(),
                "empty repetition bound `{{,}}`"
            );
            (min_digits.unwrap_or(0), max_digits)
        } else {
            // `{n}` — exact count; the min digits are required here.
            let n = min_digits.ok_or_else(|| anyhow::anyhow!("empty repetition bound `{{}}`"))?;
            (n, Some(n))
        };
        ensure!(
            self.peek() == Some(b'}'),
            "unterminated repetition bound (expected `}}`)"
        );
        self.pos += 1; // '}'
        if let Some(max) = max {
            ensure!(
                max >= min,
                "repetition bound `{{{min},{max}}}` has max < min"
            );
            ensure!(
                max <= Self::MAX_REPETITION,
                "repetition count {max} exceeds maximum {} (use `*`/`+` for unbounded)",
                Self::MAX_REPETITION
            );
        }
        ensure!(
            min <= Self::MAX_REPETITION,
            "repetition count {min} exceeds maximum {} (use `*`/`+` for unbounded)",
            Self::MAX_REPETITION
        );
        Ok((min, max))
    }

    /// Parse a run of ASCII digits as a count, or `None` if there are no digits
    /// at the cursor (so a caller can distinguish `{,m}`/`{n,}` from `{n,m}`).
    fn parse_count_digits(&mut self) -> Result<Option<usize>> {
        let start = self.pos;
        let mut value: usize = 0;
        while let Some(c @ b'0'..=b'9') = self.peek() {
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add((c - b'0') as usize))
                .ok_or_else(|| anyhow::anyhow!("repetition count overflows"))?;
            self.pos += 1;
        }
        Ok(if self.pos == start { None } else { Some(value) })
    }

    /// Desugar `unit{min,max}` into anonymous rules, mirroring [`apply_repetition`].
    /// `min` mandatory copies are inlined, then either a `*`-style recursive tail
    /// (`max == None`) or `max - min` nested-optional copies are appended.
    fn apply_bounded_repetition(
        &mut self,
        out: &mut Vec<Elem>,
        start: usize,
        min: usize,
        max: Option<usize>,
    ) -> Result<()> {
        // An empty unit makes the `{n,}` tail a left-recursive ε-cycle (and the
        // mandatory/optional copies vacuous). Reject it, as `* + ?` do.
        ensure!(start < out.len(), "repetition bound has nothing to repeat");
        let unit: Vec<Elem> = out.split_off(start);

        // `min` mandatory copies, inlined.
        for _ in 0..min {
            out.extend_from_slice(&unit);
        }

        match max {
            // `{min,}` — unbounded tail: S ::= unit S | ε (same shape as `*`).
            None => {
                let sub = self.anon_rule();
                let mut body = Vec::new();
                body.extend_from_slice(&unit);
                body.push(Elem::RuleRef(sub));
                body.push(Elem::Alt);
                body.push(Elem::End);
                self.rules[sub as usize] = body;
                out.push(Elem::RuleRef(sub));
            }
            // `{min,max}` — at most `extra` more copies, as a nested-optional
            // chain built inside-out: T_k ::= unit [T_{k+1}] | ε.
            Some(max) => {
                let mut tail: Option<u32> = None;
                for _ in 0..(max - min) {
                    let sub = self.anon_rule();
                    let mut body = Vec::new();
                    body.extend_from_slice(&unit);
                    if let Some(inner) = tail {
                        body.push(Elem::RuleRef(inner));
                    }
                    body.push(Elem::Alt);
                    body.push(Elem::End);
                    self.rules[sub as usize] = body;
                    tail = Some(sub);
                }
                if let Some(sub) = tail {
                    out.push(Elem::RuleRef(sub));
                }
            }
        }
        Ok(())
    }

    fn parse_string_literal(&mut self, out: &mut Vec<Elem>) -> Result<()> {
        self.pos += 1; // opening quote
        loop {
            match self.peek() {
                None => bail!("unterminated string literal"),
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    self.pos += 1;
                    for b in self.parse_escape()? {
                        out.push(Elem::Char(b));
                    }
                }
                Some(_) => {
                    // Push the byte directly (UTF-8 bytes become a sequence — exact for
                    // ASCII; multi-byte chars match byte-by-byte).
                    let b = self.bump().unwrap();
                    out.push(Elem::Char(b));
                }
            }
        }
        Ok(())
    }

    fn parse_char_class(&mut self, out: &mut Vec<Elem>) -> Result<()> {
        self.pos += 1; // '['
        let negated = self.peek() == Some(b'^');
        if negated {
            self.pos += 1;
        }
        let mut first = true;
        loop {
            match self.peek() {
                None => bail!("unterminated char class"),
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => {}
            }
            let lo = self.parse_class_char()?;
            if first {
                out.push(if negated {
                    Elem::CharNot(lo)
                } else {
                    Elem::Char(lo)
                });
                first = false;
            } else {
                out.push(Elem::CharAlt(lo));
            }
            // Range `a-z` (a trailing `-` before `]` is a literal, handled above).
            if self.peek() == Some(b'-') && self.src.get(self.pos + 1) != Some(&b']') {
                self.pos += 1; // '-'
                let hi = self.parse_class_char()?;
                out.push(Elem::CharRngUpper(hi));
            }
        }
        ensure!(!first, "empty char class `[]`");
        Ok(())
    }

    /// Parse one byte inside a char class (handles escapes). Byte-level: non-ASCII is
    /// rejected (Unicode ranges are a v2 limitation).
    fn parse_class_char(&mut self) -> Result<u8> {
        match self.peek() {
            Some(b'\\') => {
                self.pos += 1;
                let bytes = self.parse_escape()?;
                ensure!(
                    bytes.len() == 1,
                    "multi-byte escape not allowed inside a char class (byte-level v1)"
                );
                Ok(bytes[0])
            }
            Some(c) if c < 0x80 => {
                self.pos += 1;
                Ok(c)
            }
            Some(c) => {
                bail!("non-ASCII byte 0x{c:02x} in char class not supported (byte-level v1)")
            }
            None => bail!("unterminated char class"),
        }
    }

    /// Parse the body of an escape sequence (the backslash is already consumed).
    fn parse_escape(&mut self) -> Result<Vec<u8>> {
        let c = self
            .bump()
            .ok_or_else(|| anyhow::anyhow!("dangling escape `\\`"))?;
        Ok(match c {
            b'n' => vec![b'\n'],
            b'r' => vec![b'\r'],
            b't' => vec![b'\t'],
            b'\\' => vec![b'\\'],
            b'"' => vec![b'"'],
            b'\'' => vec![b'\''],
            b']' => vec![b']'],
            b'[' => vec![b'['],
            b'-' => vec![b'-'],
            b'x' => {
                let h = self.take_hex(2)?;
                vec![h as u8]
            }
            b'u' => {
                let cp = self.take_hex(4)?;
                let ch = char::from_u32(cp)
                    .ok_or_else(|| anyhow::anyhow!("invalid \\u escape: {cp:#x}"))?;
                ch.to_string().into_bytes()
            }
            other => bail!("unknown escape `\\{}`", other as char),
        })
    }

    fn take_hex(&mut self, n: usize) -> Result<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let c = self
                .bump()
                .ok_or_else(|| anyhow::anyhow!("short hex escape"))?;
            let d = (c as char)
                .to_digit(16)
                .ok_or_else(|| anyhow::anyhow!("bad hex digit `{}`", c as char))?;
            v = v * 16 + d;
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grammar(src: &str) -> Arc<Grammar> {
        Arc::new(Grammar::parse(src).expect("grammar should parse"))
    }

    /// Walk a full byte string through a grammar; returns the final state if every byte
    /// was accepted, else None.
    fn run(g: &Arc<Grammar>, input: &[u8]) -> Option<GrammarState> {
        let mut st = GrammarState::new(g.clone());
        for &b in input {
            if !st.accepts(&[b]) {
                return None;
            }
            st.accept(&[b]);
        }
        Some(st)
    }

    #[test]
    fn literal_sequence() {
        let g = grammar(r#"root ::= "ab""#);
        let st = run(&g, b"ab").unwrap();
        assert!(st.is_complete());
        assert!(run(&g, b"ac").is_none());
        // Incomplete prefix is accepted but not complete.
        let part = run(&g, b"a").unwrap();
        assert!(!part.is_complete());
    }

    #[test]
    fn alternation_and_class() {
        let g = grammar(r#"root ::= "yes" | "no" | [0-9]"#);
        assert!(run(&g, b"yes").unwrap().is_complete());
        assert!(run(&g, b"no").unwrap().is_complete());
        assert!(run(&g, b"7").unwrap().is_complete());
        assert!(run(&g, b"maybe").is_none());
    }

    #[test]
    fn repetition_star_plus_opt() {
        let star = grammar(r#"root ::= "a"*"#);
        assert!(GrammarState::new(star.clone()).is_complete()); // zero reps ok
        assert!(run(&star, b"aaaa").unwrap().is_complete());

        let plus = grammar(r#"root ::= "a"+"#);
        assert!(!GrammarState::new(plus.clone()).is_complete()); // needs >=1
        assert!(run(&plus, b"a").unwrap().is_complete());
        assert!(run(&plus, b"aaa").unwrap().is_complete());

        let opt = grammar(r#"root ::= "a"? "b""#);
        assert!(run(&opt, b"ab").unwrap().is_complete());
        assert!(run(&opt, b"b").unwrap().is_complete());
        assert!(run(&opt, b"aab").is_none());
    }

    #[test]
    fn negated_class_and_groups() {
        let g = grammar(r#"root ::= "\"" ([^"\\])* "\"""#);
        assert!(run(&g, br#""hello""#).unwrap().is_complete());
        assert!(run(&g, br#""""#).unwrap().is_complete());
        // A bare quote inside closes the string early; the trailing bytes can't be
        // consumed, so the whole input is rejected.
        assert!(run(&g, br#""a"b""#).is_none());
    }

    #[test]
    fn rule_references_and_recursion() {
        let g = grammar(
            r#"
            root  ::= list
            list  ::= "[" items "]"
            items ::= digit ("," digit)*
            digit ::= [0-9]
            "#,
        );
        assert!(run(&g, b"[1,2,3]").unwrap().is_complete());
        assert!(run(&g, b"[5]").unwrap().is_complete());
        assert!(run(&g, b"[1,]").is_none());
        assert!(run(&g, b"[]").is_none());
    }

    #[test]
    fn errors_not_panics() {
        assert!(Grammar::parse("foo ::= bar").is_err()); // no root, undefined bar
        assert!(Grammar::parse(r#"root ::= "unterminated"#).is_err());
        assert!(Grammar::parse("root ::= ").is_ok()); // empty rule = matches empty input
        assert!(Grammar::parse(r#"root ::= ("a""#).is_err()); // unclosed group
    }

    #[test]
    fn rejects_left_recursion_no_stack_overflow() {
        // Empty repetition unit (`""*`) — caught at the postfix site.
        assert!(Grammar::parse(r#"root ::= ""*"#).is_err());
        assert!(Grammar::parse(r#"root ::= ""+"#).is_err());
        // Direct left recursion.
        assert!(Grammar::parse(r#"root ::= root "x" | "y""#).is_err());
        // Indirect left recursion (A → B → A).
        assert!(Grammar::parse("root ::= a\na ::= b\nb ::= a").is_err());
        // Repetition of a nullable rule is an ε-cycle.
        assert!(Grammar::parse("root ::= n*\nn ::= \"a\"?").is_err());
        // Right recursion is fine (consumes before recursing).
        assert!(Grammar::parse(r#"root ::= "a" root | "a""#).is_ok());
        // A nullable rule used in sequence (not repeated) is fine.
        assert!(Grammar::parse("root ::= ws \"x\"\nws ::= \" \"*").is_ok());
    }

    #[test]
    fn mask_gates_eos_and_specials() {
        let g = grammar(r#"root ::= "hi""#);
        // Stub vocab: 0="h", 1="i", 2="x", 3=EOS(special), 4=special control.
        let token_bytes = vec![b"h".to_vec(), b"i".to_vec(), b"x".to_vec(), vec![], vec![]];
        let special = vec![false, false, false, true, true];
        let mask = GrammarMask::new(token_bytes, Some(3), special);

        let mut st = GrammarState::new(g.clone());
        let mut logits = vec![0.0f32; 5];
        let allowed = mask.apply(&st, &mut logits);
        assert_eq!(allowed, 1); // only "h"
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], f32::NEG_INFINITY); // "i" not yet
        assert_eq!(logits[3], f32::NEG_INFINITY); // EOS masked (not complete)

        st.accept(b"h");
        st.accept(b"i");
        let mut logits = vec![0.0f32; 5];
        let allowed = mask.apply(&st, &mut logits);
        assert_eq!(allowed, 1); // only EOS now
        assert_eq!(logits[3], 0.0); // EOS allowed (complete)
        assert_eq!(logits[0], f32::NEG_INFINITY);
    }

    #[test]
    fn bounded_repetition_exact() {
        // `{n}` — exactly n copies.
        let g = grammar(r#"root ::= [0-9]{4}"#);
        assert!(run(&g, b"2026").unwrap().is_complete());
        assert!(!run(&g, b"202").unwrap().is_complete()); // 3 is a valid prefix, not complete
        assert!(run(&g, b"20267").is_none()); // 5th digit rejected
        assert!(run(&g, b"20a6").is_none()); // non-digit rejected
        // `{1}` is a plain single copy.
        let one = grammar(r#"root ::= "a"{1}"#);
        assert!(run(&one, b"a").unwrap().is_complete());
        assert!(run(&one, b"aa").is_none());
    }

    #[test]
    fn bounded_repetition_range() {
        // `{m,n}` — between m and n copies inclusive.
        let g = grammar(r#"root ::= "a"{2,4}"#);
        assert!(!run(&g, b"a").unwrap().is_complete()); // 1 < min
        assert!(run(&g, b"aa").unwrap().is_complete()); // min
        assert!(run(&g, b"aaa").unwrap().is_complete());
        assert!(run(&g, b"aaaa").unwrap().is_complete()); // max
        assert!(run(&g, b"aaaaa").is_none()); // > max
    }

    #[test]
    fn bounded_repetition_optional_and_unbounded() {
        // `{0,m}` — at most m (zero allowed).
        let opt = grammar(r#"root ::= "a"{0,2}"#);
        assert!(GrammarState::new(opt.clone()).is_complete()); // zero ok
        assert!(run(&opt, b"aa").unwrap().is_complete());
        assert!(run(&opt, b"aaa").is_none());
        // `{,m}` — missing min defaults to 0.
        let comma = grammar(r#"root ::= "a"{,2}"#);
        assert!(GrammarState::new(comma.clone()).is_complete());
        assert!(run(&comma, b"aa").unwrap().is_complete());
        assert!(run(&comma, b"aaa").is_none());
        // `{n,}` — n or more, unbounded above.
        let unb = grammar(r#"root ::= "a"{2,}"#);
        assert!(!run(&unb, b"a").unwrap().is_complete()); // < min
        assert!(run(&unb, b"aa").unwrap().is_complete());
        assert!(run(&unb, b"aaaaaaaa").unwrap().is_complete());
    }

    #[test]
    fn bounded_repetition_composes() {
        // Bound binds to the immediately-preceding unit (a group here), and the
        // rest of the sequence still applies.
        let g = grammar(r#"root ::= "[" ([0-9]{1,3} ",")* [0-9]{1,3} "]""#);
        assert!(run(&g, b"[1]").unwrap().is_complete());
        assert!(run(&g, b"[12,345,6]").unwrap().is_complete());
        assert!(run(&g, b"[1234]").is_none()); // 4 digits > max
    }

    #[test]
    fn bounded_repetition_errors() {
        assert!(Grammar::parse(r#"root ::= "a"{2,1}"#).is_err()); // max < min
        assert!(Grammar::parse(r#"root ::= "a"{}"#).is_err()); // no bound
        assert!(Grammar::parse(r#"root ::= "a"{,}"#).is_err()); // no bound
        assert!(Grammar::parse(r#"root ::= "a"{2"#).is_err()); // unterminated
        assert!(Grammar::parse(r#"root ::= "a"{1,2,3}"#).is_err()); // extra field
        assert!(Grammar::parse(r#"root ::= ""{2}"#).is_err()); // empty unit
        assert!(Grammar::parse(r#"root ::= {3}"#).is_err()); // `{` not a valid unit start
        // Over the cap.
        assert!(Grammar::parse(r#"root ::= "a"{99999999}"#).is_err());
    }
}
