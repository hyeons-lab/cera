//! A small order-preserving JSON value.
//!
//! The d1 prompt depends on JSON in two ways `serde_json` does not give by default: an object
//! keeps the order its keys were written in (a question's options are read in that order), and a
//! state that is not a string is written the way Python's `json.dumps(state,
//! ensure_ascii=False)` writes it, because the model was trained on that text: `", "` and `": "`
//! separators, `1e-05` for small floats, `Infinity` for an overflowing one, control characters
//! escaped and everything else left as it is.

use std::fmt::Write as _;

use anyhow::{Result, bail, ensure};

/// A JSON value with ordered objects.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer, kept as written (without a sign on zero) so no digit is ever lost.
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Keys in the order they were first written; a repeated key keeps the last value.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Parse a JSON document.
    ///
    /// # Errors
    ///
    /// Fails on anything that is not one complete JSON value.
    pub fn parse(text: &str) -> Result<Self> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
            depth: 0,
        };
        parser.skip_space();
        let value = parser.value()?;
        parser.skip_space();
        ensure!(
            parser.pos == parser.bytes.len(),
            "unexpected data after the JSON value at byte {}",
            parser.pos
        );
        Ok(value)
    }

    /// The value of `key` when this is an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The text of a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Python's `json.dumps(self, ensure_ascii=False)`.
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write_to(&mut out);
        out
    }

    fn write_to(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(digits) => out.push_str(digits),
            Json::Float(f) => out.push_str(&float_repr(*f)),
            Json::Str(s) => write_string(out, s),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write_to(out);
                }
                out.push(']');
            }
            Json::Object(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_string(out, key);
                    out.push_str(": ");
                    value.write_to(out);
                }
                out.push('}');
            }
        }
    }
}

/// A string as `json.dumps(ensure_ascii=False)` writes it.
fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that read back the same value, fixed notation
/// for a decimal exponent from -4 up to 15 and scientific (`1e-05`, `1.5e+16`) outside it.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` is the shortest round-trip digit string: `d.ddde<exp>`
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if (point as usize) >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{body}")
    } else {
        let frac = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{frac}e{exp_sign}{:02}", exp.abs())
    }
}

/// Deepest nesting the parser follows.
const MAX_DEPTH: usize = 256;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<()> {
        ensure!(
            self.peek() == Some(byte),
            "expected `{}` at byte {}",
            byte as char,
            self.pos
        );
        self.pos += 1;
        Ok(())
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json> {
        ensure!(
            self.bytes[self.pos..].starts_with(word.as_bytes()),
            "invalid JSON literal at byte {}",
            self.pos
        );
        self.pos += word.len();
        Ok(value)
    }

    fn value(&mut self) -> Result<Json> {
        match self.peek() {
            Some(b'n') => self.literal("null", Json::Null),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'N') => self.literal("NaN", Json::Float(f64::NAN)),
            Some(b'I') => self.literal("Infinity", Json::Float(f64::INFINITY)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => self.array(),
            Some(b'{') => self.object(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => bail!("unexpected input at byte {}", self.pos),
        }
    }

    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.depth += 1;
        ensure!(self.depth <= MAX_DEPTH, "JSON is nested too deeply");
        let out = f(self);
        self.depth -= 1;
        out
    }

    fn array(&mut self) -> Result<Json> {
        self.nested(|p| {
            p.expect(b'[')?;
            let mut items = Vec::new();
            p.skip_space();
            if p.peek() == Some(b']') {
                p.pos += 1;
                return Ok(Json::Array(items));
            }
            loop {
                p.skip_space();
                items.push(p.value()?);
                p.skip_space();
                match p.peek() {
                    Some(b',') => p.pos += 1,
                    Some(b']') => {
                        p.pos += 1;
                        return Ok(Json::Array(items));
                    }
                    _ => bail!("expected `,` or `]` at byte {}", p.pos),
                }
            }
        })
    }

    fn object(&mut self) -> Result<Json> {
        self.nested(|p| {
            p.expect(b'{')?;
            let mut entries: Vec<(String, Json)> = Vec::new();
            p.skip_space();
            if p.peek() == Some(b'}') {
                p.pos += 1;
                return Ok(Json::Object(entries));
            }
            loop {
                p.skip_space();
                ensure!(p.peek() == Some(b'"'), "expected a key at byte {}", p.pos);
                let key = p.string()?;
                p.skip_space();
                p.expect(b':')?;
                p.skip_space();
                let value = p.value()?;
                // Python's dict: a repeated key keeps its first position and its last value
                match entries.iter_mut().find(|(k, _)| *k == key) {
                    Some(slot) => slot.1 = value,
                    None => entries.push((key, value)),
                }
                p.skip_space();
                match p.peek() {
                    Some(b',') => p.pos += 1,
                    Some(b'}') => {
                        p.pos += 1;
                        return Ok(Json::Object(entries));
                    }
                    _ => bail!("expected `,` or `}}` at byte {}", p.pos),
                }
            }
        })
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
            if self.peek() == Some(b'I') {
                self.literal("Infinity", Json::Null)?;
                return Ok(Json::Float(f64::NEG_INFINITY));
            }
        }
        let digits = |p: &mut Self| {
            let from = p.pos;
            while matches!(p.peek(), Some(b'0'..=b'9')) {
                p.pos += 1;
            }
            p.pos - from
        };
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => bail!("invalid number at byte {start}"),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            self.pos += 1;
            ensure!(digits(self) > 0, "invalid number at byte {start}");
            is_float = true;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            ensure!(digits(self) > 0, "invalid number at byte {start}");
            is_float = true;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).expect("ASCII digits");
        if is_float {
            Ok(Json::Float(text.parse()?))
        } else if text == "-0" {
            Ok(Json::Int("0".into()))
        } else {
            Ok(Json::Int(text.to_string()))
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        ensure!(
            self.pos + 4 <= self.bytes.len(),
            "truncated \\u escape at byte {}",
            self.pos
        );
        let text = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])?;
        self.pos += 4;
        Ok(u32::from_str_radix(text, 16)?)
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let Some(byte) = self.peek() else {
                bail!("unterminated string");
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let Some(esc) = self.peek() else {
                        bail!("unterminated escape");
                    };
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let scalar = if (0xD800..0xDC00).contains(&first)
                                && self.bytes[self.pos..].starts_with(b"\\u")
                            {
                                self.pos += 2;
                                let second = self.hex4()?;
                                ensure!(
                                    (0xDC00..0xE000).contains(&second),
                                    "invalid surrogate pair"
                                );
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else {
                                first
                            };
                            // a lone surrogate has no `char`: Python keeps it, the prompt
                            // cannot, so it becomes the replacement character
                            out.push(char::from_u32(scalar).unwrap_or('\u{FFFD}'));
                        }
                        _ => bail!("invalid escape at byte {}", self.pos - 1),
                    }
                }
                0..=0x1f => bail!("control character in a string at byte {}", self.pos),
                _ => {
                    let rest = std::str::from_utf8(&self.bytes[self.pos..])?;
                    let c = rest.chars().next().expect("non-empty");
                    out.push(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_keep_their_order_and_a_repeated_key_keeps_the_last_value() {
        let v = Json::parse(r#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
        assert_eq!(v.dumps(), r#"{"b": 3, "a": 2}"#);
    }

    #[test]
    fn dumps_matches_python_defaults() {
        let v = Json::parse(r#"{"k":[1,2.5,"é☕",null,true],"n":{}}"#).unwrap();
        assert_eq!(v.dumps(), r#"{"k": [1, 2.5, "é☕", null, true], "n": {}}"#);
    }

    #[test]
    fn strings_escape_what_python_escapes() {
        let v = Json::Str("a\"b\\c\nd\u{1}e\u{7f}".into());
        assert_eq!(v.dumps(), "\"a\\\"b\\\\c\\nd\\u0001e\u{7f}\"");
    }

    #[test]
    fn float_repr_follows_python() {
        for (f, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (100.0, "100.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e20, "1.2345e+20"),
            (-2.5, "-2.5"),
            (123456.789, "123456.789"),
            (f64::INFINITY, "Infinity"),
        ] {
            assert_eq!(float_repr(f), want, "{f:?}");
        }
    }

    #[test]
    fn numbers_keep_their_kind() {
        assert_eq!(Json::parse("-0").unwrap().dumps(), "0");
        assert_eq!(
            Json::parse("12345678901234567890123").unwrap().dumps(),
            "12345678901234567890123"
        );
        assert_eq!(Json::parse("1E2").unwrap().dumps(), "100.0");
        assert_eq!(Json::parse("-1.50").unwrap().dumps(), "-1.5");
    }

    #[test]
    fn surrogate_pairs_decode() {
        assert_eq!(Json::parse(r#""😀""#).unwrap(), Json::Str("😀".into()));
    }

    #[test]
    fn malformed_documents_are_refused() {
        for bad in [
            "",
            "{",
            "[1,]",
            r#"{"a" 1}"#,
            "01",
            "1.",
            "\"x",
            "nul",
            "[1] 2",
        ] {
            assert!(Json::parse(bad).is_err(), "{bad:?}");
        }
        let deep = "[".repeat(300) + &"]".repeat(300);
        assert!(Json::parse(&deep).is_err());
    }
}
