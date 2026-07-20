use std::collections::HashMap;

use anyhow::{Context, Result};
use regex::Regex;

use crate::gguf::{GgufFile, GgufValue};

/// A segment of text split at special token boundaries.
enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

/// A minimal byte-level BPE tokenizer.
///
/// Loads vocabulary and merges from GGUF metadata. Implements the same
/// byte-level BPE algorithm used by LLaMA, LFM2, GPT-NeoX, etc.
pub struct BpeTokenizer {
    /// Token ID → token string (may contain raw bytes as escaped sequences).
    vocab: Vec<Vec<u8>>,
    /// Token string → token ID.
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Merge pairs in priority order (highest priority = lowest index).
    /// Maps (token_a, token_b) → priority rank.
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// Special token name → token ID.
    special_tokens: HashMap<String, u32>,
    /// BOS token ID.
    bos_id: Option<u32>,
    /// EOS token ID.
    eos_id: Option<u32>,
    /// GGUF `tokenizer.ggml.add_bos_token`: whether `encode_special` prepends BOS.
    add_bos: bool,
    /// GGUF `tokenizer.ggml.add_eos_token`: whether `encode_special` appends EOS.
    add_eos: bool,
    /// Chat template (Jinja2 format) if present.
    chat_template: Option<String>,
    /// Pre-compiled pretokenizer regex.
    pretokenize_re: Regex,
    /// True when the pretokenizer splits digits as a bare `\p{N}` with no leading
    /// space (REFACT family — Granite/Refact/CodeShell/SmolLM). The whitespace
    /// `\s+(?!\S)` emulation in `pretokenize` must then NOT donate a trailing
    /// space to a following digit, since llama.cpp keeps such digits bare. False
    /// for GPT-2/LLAMA3-style pretokenizers, whose ` ?\p{N}+` absorbs the space.
    digits_split_bare: bool,
    /// GPT-2 byte→unicode mapping (computed once, used in encode).
    byte_to_unicode: [char; 256],
    /// GPT-2 unicode→byte mapping (computed once, used in decode).
    unicode_to_byte: HashMap<char, u8>,
}

impl BpeTokenizer {
    /// Load a tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Extract vocabulary tokens
        let tokens = gguf
            .get_string_array("tokenizer.ggml.tokens")
            .context("missing tokenizer.ggml.tokens")?;

        let vocab_size = tokens.len();

        // Build vocab and reverse mapping
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vocab_size);

        for (id, token_str) in tokens.iter().enumerate() {
            let token_bytes = unescape_token(token_str);
            token_to_id.insert(token_bytes.clone(), id as u32);
            vocab.push(token_bytes);
        }

        // Extract merge rules
        let mut merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();

        if let Some(merges) = gguf.get_string_array("tokenizer.ggml.merges") {
            for (rank, merge_str) in merges.iter().enumerate() {
                // Each merge is "token_a token_b" separated by a space
                if let Some((a, b)) = merge_str.split_once(' ') {
                    let a_bytes = unescape_token(a);
                    let b_bytes = unescape_token(b);
                    merge_ranks.insert((a_bytes, b_bytes), rank);
                }
            }
        }

        // Extract special tokens
        let mut special_tokens = HashMap::new();

        // Check for token type array to identify special tokens
        if let Some(GgufValue::Array(types)) = gguf.metadata.get("tokenizer.ggml.token_type") {
            for (id, type_val) in types.iter().enumerate() {
                if let GgufValue::I32(t) = type_val {
                    // Type 3 = control token, Type 4 = user-defined special
                    if (*t == 3 || *t == 4) && id < vocab_size {
                        let token_str = &tokens[id];
                        special_tokens.insert(token_str.to_string(), id as u32);
                    }
                }
            }
        }

        // Extract BOS/EOS token IDs
        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id");

        // Whether `encode_special` should add BOS/EOS. Absent keys default to
        // `false`, matching llama.cpp's default for BPE-family tokenizers (the
        // only family cera implements).
        let add_bos = gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false);
        let add_eos = gguf
            .get_bool("tokenizer.ggml.add_eos_token")
            .unwrap_or(false);

        // Extract chat template. Pre-strip Jinja2 `{% generation %}`
        // / `{% endgeneration %}` block markers at load time so the
        // stored template is already minijinja-compatible — saves
        // re-running the regex on every render. The block is a
        // training-time loss-masking marker that's a no-op at
        // inference; see `strip_generation_markers` for details.
        let chat_template = gguf
            .get_str("tokenizer.chat_template")
            .map(|raw| strip_generation_markers(raw).into_owned());

        // Select pretokenizer based on model type
        let pre_type = gguf.get_str("tokenizer.ggml.pre").unwrap_or("gpt2");
        let pretokenize_re = build_pretokenize_regex(pre_type);
        let digits_split_bare = pre_type == "refact";
        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte = build_unicode_to_byte();

        Ok(BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens,
            bos_id,
            eos_id,
            add_bos,
            add_eos,
            chat_template,
            pretokenize_re,
            digits_split_bare,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    /// Encode text into token IDs using byte-level BPE with pretokenization.
    ///
    /// The text is first split into chunks using a regex pattern (matching
    /// llama.cpp's LLAMA3 pretokenizer, which is used for LFM2 models).
    /// BPE merges are then applied within each chunk independently.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // First, split text at special token boundaries.
        // Special tokens (e.g., <|im_start|>, <|im_end|>) are emitted as single
        // token IDs; the text segments between them are BPE-encoded normally.
        let segments = self.split_special_tokens(text);
        let mut result = Vec::new();
        for segment in &segments {
            match segment {
                Segment::Special(id) => result.push(*id),
                Segment::Text(s) => {
                    for chunk in self.pretokenize(s) {
                        result.extend(self.bpe_encode_chunk(chunk));
                    }
                }
            }
        }
        result
    }

    /// Encode text, optionally adding the model's special BOS/EOS markers —
    /// the analog of llama.cpp's `llama_tokenize(..., add_special)`.
    ///
    /// With `add_special == true`, BOS is prepended when the GGUF declares
    /// `tokenizer.ggml.add_bos_token` (and a BOS id exists), and EOS is
    /// appended when it declares `tokenizer.ggml.add_eos_token` — so token
    /// counts match llama.cpp for the same text. With `add_special == false`
    /// this is exactly [`Self::encode`].
    ///
    /// Note this is orthogonal to chat templating: templates render their own
    /// special markers in the text, so template-rendered prompts should keep
    /// using plain `encode` (or pass `add_special = false`).
    pub fn encode_special(&self, text: &str, add_special: bool) -> Vec<u32> {
        let bos = if add_special && self.add_bos {
            self.bos_id
        } else {
            None
        };
        let eos = if add_special && self.add_eos {
            self.eos_id
        } else {
            None
        };
        // Common case (and `add_special == false`): no markers to add, so
        // return `encode`'s Vec by move — no extra allocation or copy.
        if bos.is_none() && eos.is_none() {
            return self.encode(text);
        }
        // Build with exact capacity and append the encoded tokens once —
        // avoids the O(n) `insert(0, bos)` memmove. `Option` iterates as 0/1
        // items, so `extend` prepends/appends the marker when present.
        let encoded = self.encode(text);
        let mut result =
            Vec::with_capacity(encoded.len() + bos.is_some() as usize + eos.is_some() as usize);
        result.extend(bos);
        result.extend_from_slice(&encoded);
        result.extend(eos);
        result
    }

    /// Pretokenize a text segment into chunks, applying the regex split plus the
    /// `\s+(?!\S)` whitespace-donation the `regex` crate can't express directly.
    ///
    /// In the GPT-2/LLAMA3/Qwen2 pretokenizers a whitespace run *followed by* a
    /// non-whitespace char yields its LAST whitespace char to the following
    /// chunk — e.g. `"\n    return"` → `"\n   "` + `" return"`, not `"\n    "` +
    /// `"return"`. Without this, indented/multi-space text (notably code)
    /// tokenizes differently from llama.cpp.
    ///
    /// REFACT-family pretokenizers (`digits_split_bare`) split digits as a bare
    /// `\p{N}` with no leading space, so a whitespace run before a digit keeps
    /// its trailing space (llama.cpp leaves the digit bare); every other
    /// pretokenizer uses ` ?\p{N}+`, where the digit does absorb the space.
    fn pretokenize<'a>(&self, s: &'a str) -> Vec<&'a str> {
        let mut ranges: Vec<(usize, usize)> = self
            .pretokenize_re
            .find_iter(s)
            .map(|m| (m.start(), m.end()))
            .collect();
        for i in 0..ranges.len() {
            let (a, b) = ranges[i];
            // Whitespace runs are ASCII, so the last char is one byte — `b - 1`
            // is a valid boundary.
            let is_ws_run = b - a >= 2 && s.as_bytes()[a..b].iter().all(u8::is_ascii_whitespace);
            // Only hand a trailing *space/tab* to the next chunk. A run ending in
            // a newline is normally claimed by the earlier `\s*[\r\n]+`
            // alternative, but guard anyway: the lookahead applies to a space
            // before a word, never to a newline.
            let last_is_spacetab = matches!(s.as_bytes()[b - 1], b' ' | b'\t');
            let next_char = s[b..].chars().next();
            let next_non_ws = next_char.is_some_and(|c| !c.is_whitespace());
            // REFACT keeps digits bare, so don't donate a space to a following
            // digit; other pretokenizers' ` ?\p{N}+` does take it.
            let next_takes_space =
                !(self.digits_split_bare && next_char.is_some_and(char::is_numeric));
            if is_ws_run
                && last_is_spacetab
                && next_non_ws
                && next_takes_space
                && i + 1 < ranges.len()
            {
                ranges[i].1 = b - 1;
                ranges[i + 1].0 = b - 1;
            }
        }
        ranges.iter().map(|&(a, b)| &s[a..b]).collect()
    }

    /// Split text at special token boundaries, returning alternating
    /// text segments and special token IDs.
    fn split_special_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Text(text)];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest special token in the remaining text.
            let mut best: Option<(usize, usize, u32)> = None; // (start, end, id)
            for (tok_str, &tok_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok_str.as_str()) {
                    let end = pos + tok_str.len();
                    if best.is_none()
                        || pos < best.unwrap().0
                        || (pos == best.unwrap().0 && end > best.unwrap().1)
                    {
                        best = Some((pos, end, tok_id));
                    }
                }
            }

            match best {
                Some((start, end, id)) => {
                    if start > 0 {
                        segments.push(Segment::Text(&remaining[..start]));
                    }
                    segments.push(Segment::Special(id));
                    remaining = &remaining[end..];
                }
                None => {
                    segments.push(Segment::Text(remaining));
                    break;
                }
            }
        }

        segments
    }

    /// Apply BPE to a single pretokenized chunk.
    fn bpe_encode_chunk(&self, chunk: &str) -> Vec<u32> {
        if chunk.is_empty() {
            return vec![];
        }

        // Convert each raw byte through the GPT-2 byte-to-unicode mapping,
        // then split into individual unicode characters. This ensures space (0x20)
        // becomes Ġ (U+0120), matching how vocab tokens are stored.
        let unicode_str = bytes_to_gpt2_unicode(chunk.as_bytes(), &self.byte_to_unicode);

        // Each mapped unicode character becomes an initial BPE symbol.
        // Characters may be multi-byte in UTF-8 (e.g., Ġ = \xC4\xA0).
        let mut tokens: Vec<Vec<u8>> = unicode_str
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.as_bytes().to_vec()
            })
            .collect();

        // Repeatedly merge the highest-priority pair
        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_rank = usize::MAX;
            let mut best_idx = 0;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            let merged = [tokens[best_idx].as_slice(), tokens[best_idx + 1].as_slice()].concat();
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }

        // Convert token byte sequences to IDs
        tokens
            .iter()
            .map(|t| self.token_to_id.get(t).copied().unwrap_or(0))
            .collect()
    }

    /// Decode token IDs back to a string.
    ///
    /// Reverses the GPT-2 byte-to-unicode mapping: collects the unicode chars
    /// from each token's vocab entry, maps them back to raw bytes, then
    /// interprets the result as UTF-8.
    pub fn decode(&self, token_ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(token_ids)).into_owned()
    }

    /// Decode `token_ids` to their raw byte sequence, without the lossy UTF-8
    /// conversion that [`Self::decode`] applies. A single multi-byte character
    /// can span several byte-fallback tokens (e.g. an emoji as `<0xE2><0x80>…`),
    /// so streaming callers should accumulate these bytes and only convert
    /// complete UTF-8 prefixes — converting one token at a time would turn each
    /// fragment into a U+FFFD replacement char.
    pub fn decode_bytes(&self, token_ids: &[u32]) -> Vec<u8> {
        let mut raw_bytes = Vec::new();
        for &id in token_ids {
            if let Some(token_bytes) = self.vocab.get(id as usize) {
                // Try to parse as UTF-8 and reverse the GPT-2 byte-to-unicode mapping.
                // For raw byte tokens (e.g., <0x80> → [0x80], not valid UTF-8),
                // emit the bytes directly — they're already the raw values we want.
                match std::str::from_utf8(token_bytes) {
                    Ok(s) => {
                        for ch in s.chars() {
                            if let Some(&b) = self.unicode_to_byte.get(&ch) {
                                raw_bytes.push(b);
                            } else {
                                let mut buf = [0u8; 4];
                                let encoded = ch.encode_utf8(&mut buf);
                                raw_bytes.extend_from_slice(encoded.as_bytes());
                            }
                        }
                    }
                    Err(_) => {
                        // Non-UTF-8 token (raw byte token) — emit as-is
                        raw_bytes.extend_from_slice(token_bytes);
                    }
                }
            }
        }
        raw_bytes
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// BOS token ID.
    pub fn bos_token(&self) -> Option<u32> {
        self.bos_id
    }

    /// EOS token ID.
    pub fn eos_token(&self) -> Option<u32> {
        self.eos_id
    }

    /// GGUF `tokenizer.ggml.add_bos_token` — whether [`Self::encode_special`]
    /// prepends BOS. `false` when the key is absent.
    pub fn add_bos_token(&self) -> bool {
        self.add_bos
    }

    /// GGUF `tokenizer.ggml.add_eos_token` — whether [`Self::encode_special`]
    /// appends EOS. `false` when the key is absent.
    pub fn add_eos_token(&self) -> bool {
        self.add_eos
    }

    /// Get the chat template string if present.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    /// Look up a special token by name.
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Check if a token ID is a special/control token.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_tokens.values().any(|&v| v == id)
    }

    /// Raw output bytes a single token contributes to the decoded stream.
    ///
    /// This is the per-token slice of [`decode`](Self::decode): valid-UTF-8 tokens have
    /// the GPT-2 `unicode_to_byte` mapping reversed (e.g. `Ġ` → space). Byte-fallback
    /// tokens — written in GGUF as the escape `<0xHH>` and already decoded to a single
    /// raw byte at load time by `unescape_token` — are stored in `vocab` as that raw
    /// byte (which on its own may not be valid UTF-8) and emitted as-is. Used by
    /// grammar-constrained decoding to test a candidate token against the grammar at the
    /// byte level — do NOT use the raw `vocab` entry, which for GPT-2/Qwen vocabs stores
    /// the *remapped* bytes. Returns an empty vec for an out-of-range id.
    pub fn token_output_bytes(&self, id: u32) -> Vec<u8> {
        let Some(token_bytes) = self.vocab.get(id as usize) else {
            return Vec::new();
        };
        match std::str::from_utf8(token_bytes) {
            Ok(s) => {
                let mut out = Vec::with_capacity(token_bytes.len());
                for ch in s.chars() {
                    if let Some(&b) = self.unicode_to_byte.get(&ch) {
                        out.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                out
            }
            Err(_) => token_bytes.clone(),
        }
    }
}

// ── Chat template rendering ─────────────────────────────────────────────────

/// A chat message with role and string content. The bread-and-
/// butter shape used by every text-only call site. Multimodal
/// callers (image / audio in chat templates) use
/// [`ChatMessageMultimodal`] instead — both flow through
/// [`apply_chat_template`] which is generic over `Serialize`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// One item in a multimodal chat message's content list. Serializes
/// to the JSON shape LFM2-VL's chat template expects:
/// `{"type": "text", "text": "..."}` / `{"type": "image"}`. The
/// template's `parse_content` macro dispatches on `item.type`.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentItem {
    /// A plain text fragment. Rendered verbatim into the surrounding
    /// chat template position.
    Text { text: String },
    /// An image placeholder. The chat template emits the literal
    /// string `<image>` here — cera's
    /// [`crate::session::Session::append_chat_with_images`] helper
    /// later swaps each `<image>` for the
    /// `<|image_start|> + image embeddings + <|image_end|>`
    /// envelope at append time.
    Image,
}

/// A chat message whose content is a list of typed items (text +
/// image). Use this shape when at least one message in the
/// conversation includes an image; for text-only conversations
/// [`ChatMessage`] is simpler. Both serialize to a Jinja2 context
/// the same chat template understands — the template's
/// `parse_content` macro branches on whether `content is string`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessageMultimodal {
    pub role: String,
    pub content: Vec<ContentItem>,
}

/// Render a chat template using minijinja.
///
/// The template is a Jinja2 template from the GGUF metadata.
/// Variables available: `messages` (array of {role, content}),
/// `bos_token`, `eos_token`, `add_generation_prompt`.
///
/// Generic over the message element type so both
/// [`ChatMessage`] (string content) and [`ChatMessageMultimodal`]
/// (list-of-items content) work without per-shape branching here —
/// the chat template itself dispatches on whether `content` is a
/// string or a list.
///
/// For tool calling, use [`apply_chat_template_with_tools`], which also passes
/// a `tools` array so a template's `{% if tools %}` branch renders.
pub fn apply_chat_template<M: serde::Serialize>(
    tokenizer: &BpeTokenizer,
    messages: &[M],
    add_generation_prompt: bool,
) -> Result<String> {
    apply_chat_template_with_tools(tokenizer, messages, &[], add_generation_prompt)
}

/// Like [`apply_chat_template`], but also exposes a `tools` array to the
/// template so tool-trained models render their tool-definition block.
///
/// `tools` serializes to the OpenAI "function" JSON shape
/// (`{name, description, parameters}`); tool-trained templates iterate it
/// under `{% if tools %} … {% for tool in tools %}`. An **empty** `tools`
/// slice is rendered as an empty array, which is falsy in Jinja — so passing
/// no tools is exactly equivalent to the plain [`apply_chat_template`] and
/// leaves non-tool templates untouched.
pub fn apply_chat_template_with_tools<M: serde::Serialize>(
    tokenizer: &BpeTokenizer,
    messages: &[M],
    tools: &[crate::tools::ToolDef],
    add_generation_prompt: bool,
) -> Result<String> {
    let template_str = tokenizer
        .chat_template()
        .context("model has no chat template")?;

    // Template is already cleaned of `{% generation %}` /
    // `{% endgeneration %}` block markers at `BpeTokenizer::from_gguf`
    // load time, so feed it directly to minijinja.
    let mut env = minijinja::Environment::new();
    env.add_template("chat", template_str)
        .context("invalid chat template")?;

    let tmpl = env.get_template("chat").unwrap();

    // Build BOS/EOS token strings for the template
    let bos_token = tokenizer
        .bos_token()
        .and_then(|id| tokenizer.vocab.get(id as usize))
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .unwrap_or_default();

    let eos_token = tokenizer
        .eos_token()
        .and_then(|id| tokenizer.vocab.get(id as usize))
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .unwrap_or_default();

    let ctx = minijinja::context! {
        messages => messages,
        tools => tools,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => add_generation_prompt,
    };

    tmpl.render(ctx).context("rendering chat template")
}

/// Remove `{% generation %}` / `{% endgeneration %}` markers (and
/// the `{%- … -%}` whitespace-stripping variants) from a chat
/// template. The block is a no-op at inference time — it exists so
/// trainers can mask the assistant's reply for loss computation —
/// but the tag isn't a built-in minijinja construct so it has to be
/// stripped before `add_template` will accept the template.
///
/// Called once at `BpeTokenizer::from_gguf` load time; the cleaned
/// string is stored as the canonical chat template so renders skip
/// this work. Returns `Cow::Borrowed` (no allocation) when the
/// template has no markers.
fn strip_generation_markers(template: &str) -> std::borrow::Cow<'_, str> {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        // `\{%-?\s*(?:end)?generation\b\s*-?%\}` — match optional
        // `-` for whitespace control on either side, allow inner
        // whitespace, and capture both the opening and closing
        // forms. The `\b` after `generation` is explicit about the
        // word boundary so a hypothetical `{% generation_count %}`
        // (a variable name in a future template) wouldn't match —
        // today `\s*` would already block it but `\b` is the
        // honest expression of the constraint.
        Regex::new(r"\{%-?\s*(?:end)?generation\b\s*-?%\}").expect("static regex must compile")
    });
    re.replace_all(template, "")
}

// ── GPT-2 byte-to-unicode mapping ──────────────────────────────────────────

/// Build the GPT-2 byte-to-unicode mapping table.
///
/// GPT-2 BPE uses a reversible mapping from bytes (0-255) to Unicode characters
/// so that every byte can be represented as a valid UTF-8 token. Printable ASCII
/// and Latin-1 bytes map to themselves; other bytes map to Unicode codepoints
/// starting at U+0100. Space (0x20) → Ġ (U+0120).
fn build_byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32; // counter for non-printable bytes

    for b in 0u16..256 {
        let ch = match b {
            // Printable ASCII subset + Latin-1 supplement (these map to themselves)
            0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => b as u32,
            // Everything else maps to U+0100 + n (sequential assignment)
            _ => {
                let c = 256 + n;
                n += 1;
                c
            }
        };
        table[b as usize] = char::from_u32(ch).unwrap();
    }
    table
}

/// Convert raw bytes to a GPT-2 unicode string using the byte-to-unicode mapping.
fn bytes_to_gpt2_unicode(bytes: &[u8], table: &[char; 256]) -> String {
    bytes.iter().map(|&b| table[b as usize]).collect()
}

/// Build the reverse mapping: GPT-2 unicode char → original byte.
fn build_unicode_to_byte() -> HashMap<char, u8> {
    let table = build_byte_to_unicode();
    table
        .iter()
        .enumerate()
        .map(|(b, &ch)| (ch, b as u8))
        .collect()
}

// ── Pretokenization ────────────────────────────────────────────────────────

/// Build the pretokenizer regex based on the `tokenizer.ggml.pre` type.
///
/// Different model families use different pretokenizer patterns:
/// - "lfm2", "llama3", "llama-v3", "llama-bpe" → LLAMA3 pattern (case-insensitive
///   contractions, 1-3 digit groups, newline handling)
/// - "gpt2" → GPT-2 pattern (simpler, case-sensitive contractions)
/// - Others → defaults to LLAMA3 with a warning
fn build_pretokenize_regex(pre_type: &str) -> Regex {
    let pattern = match pre_type {
        // LLAMA3 pattern — used by LFM2, LLaMA 3, and similar models.
        // The original's `\s+(?!\S)|\s+` needs lookahead (unsupported by the
        // `regex` crate); the trailing-whitespace-before-word behaviour is
        // emulated in `encode`'s split loop instead, so this arm just uses `\s+`.
        // `llama-bpe` is llama.cpp's canonical spelling for this pre type — it
        // groups "llama3" | "llama-v3" | "llama-bpe" onto LLAMA_VOCAB_PRE_TYPE_LLAMA3.
        // cera had the first two, so every Llama-3 GGUF warned "unknown ... type
        // 'llama-bpe'" and then picked LLAMA3 anyway: right pattern, false alarm.
        "lfm2" | "llama3" | "llama-v3" | "llama-bpe" => concat!(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
            r"|[^\r\n\p{L}\p{N}]?\p{L}+",
            r"|\p{N}{1,3}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"|\s*[\r\n]+",
            r"|\s+",
        ),
        // Qwen2 pattern — GPT-2 family with single-digit number splitting and
        // case-insensitive contractions. Matches llama.cpp's "qwen2" regex
        // (LLM_CHAT pre type), which differs from LLAMA3 only in splitting
        // numbers one digit at a time (`\p{N}` rather than `\p{N}{1,3}`). The
        // upstream `\s+(?!\S)` lookahead is emulated in `encode`'s split loop
        // (see the LLAMA3 arm). Used by both Qwen2 and Qwen3 GGUFs.
        "qwen2" => concat!(
            r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
            r"|[^\r\n\p{L}\p{N}]?\p{L}+",
            r"|\p{N}",
            r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
            r"|\s*[\r\n]+",
            r"|\s+",
        ),
        // GPT-2 pattern — simpler, case-sensitive contractions.
        "gpt2" => concat!(
            r"(?:'s|'t|'re|'ve|'m|'ll|'d)",
            r"| ?\p{L}+",
            r"| ?\p{N}+",
            r"| ?[^\s\p{L}\p{N}]+",
            r"|\s+",
        ),
        // Refact pattern — used by Granite (and Refact/CodeShell/SmolLM). Matches
        // llama.cpp's `LLAMA_VOCAB_PRE_TYPE_REFACT`: the GPT-2 pattern, but numbers
        // are split one digit at a time (a leading `\p{N}` expr) — so `\p{N}`
        // replaces GPT-2's ` ?\p{N}+`. Whitespace is GPT-2-style (`\s+`, with the
        // `\s+(?!\S)` trailing-space lookahead emulated in `encode`), NOT the
        // LLAMA3 `\s*[\r\n]+` newline handling — that distinction is what makes
        // indentation (`\n    `) tokenize correctly for code.
        "refact" => concat!(
            r"(?:'s|'t|'re|'ve|'m|'ll|'d)",
            r"| ?\p{L}+",
            r"|\p{N}",
            r"| ?[^\s\p{L}\p{N}]+",
            r"|\s+",
        ),
        // Default to LLAMA3 for unknown types (most general pattern)
        other => {
            tracing::warn!(
                "unknown tokenizer.ggml.pre type '{other}', defaulting to LLAMA3 pretokenizer"
            );
            concat!(
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                r"|\p{N}{1,3}",
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                r"|\s*[\r\n]+",
                r"|\s+",
            )
        }
    };

    Regex::new(pattern).expect("invalid pretokenizer regex")
}

// ── Token unescaping ────────────────────────────────────────────────────────

/// Convert a token string from GGUF vocabulary to raw bytes.
///
/// GGUF vocabularies use various escape conventions:
/// - `<0xHH>` for raw byte values
/// - `▁` (U+2581) for space (common in sentencepiece)
/// - Regular UTF-8 strings as-is
fn unescape_token(s: &str) -> Vec<u8> {
    // Handle byte tokens: <0xHH>
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }

    // Handle sentencepiece space marker
    let s = s.replace('▁', " ");

    s.into_bytes()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> BpeTokenizer {
        // Build a minimal tokenizer for testing
        // Vocab: individual bytes + some merged tokens
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::new();

        // First 256 entries: single bytes
        for b in 0u8..=255 {
            vocab.push(vec![b]);
            token_to_id.insert(vec![b], b as u32);
        }

        // Add some merged tokens
        let merged_tokens = vec![
            (256u32, b"he".to_vec()),
            (257, b"ll".to_vec()),
            (258, b"lo".to_vec()),
            (259, b"hell".to_vec()),
            (260, b"hello".to_vec()),
        ];
        for (id, bytes) in &merged_tokens {
            vocab.push(bytes.clone());
            token_to_id.insert(bytes.clone(), *id);
        }

        // Merges in priority order
        let mut merge_ranks = HashMap::new();
        merge_ranks.insert((b"h".to_vec(), b"e".to_vec()), 0); // h+e -> he
        merge_ranks.insert((b"l".to_vec(), b"l".to_vec()), 1); // l+l -> ll
        merge_ranks.insert((b"l".to_vec(), b"o".to_vec()), 2); // l+o -> lo
        merge_ranks.insert((b"he".to_vec(), b"ll".to_vec()), 3); // he+ll -> hell
        merge_ranks.insert((b"hell".to_vec(), b"o".to_vec()), 4); // hell+o -> hello

        BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens: HashMap::new(),
            bos_id: None,
            eos_id: None,
            add_bos: false,
            add_eos: false,
            chat_template: None,
            pretokenize_re: build_pretokenize_regex("lfm2"),
            digits_split_bare: false,
            byte_to_unicode: build_byte_to_unicode(),
            unicode_to_byte: build_unicode_to_byte(),
        }
    }

    #[test]
    fn test_byte_to_unicode_space() {
        let table = build_byte_to_unicode();
        // Space (0x20) should map to Ġ (U+0120)
        assert_eq!(table[0x20], '\u{0120}');
        // Printable ASCII should map to itself
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b'z' as usize], 'z');
        assert_eq!(table[b'0' as usize], '0');
        // Newline (0x0A) should NOT map to itself (it's a control char)
        assert_ne!(table[0x0A], '\n');
    }

    #[test]
    fn encode_special_prepends_bos_when_declared() {
        let mut tok = make_test_tokenizer();
        tok.bos_id = Some(1000);
        tok.add_bos = true;
        let plain = tok.encode("hello");
        let special = tok.encode_special("hello", true);
        assert_eq!(special[0], 1000);
        assert_eq!(&special[1..], &plain[..]);
        // add_special = false is exactly `encode`.
        assert_eq!(tok.encode_special("hello", false), plain);
    }

    #[test]
    fn encode_special_without_metadata_flag_is_plain_encode() {
        let mut tok = make_test_tokenizer();
        // BOS id exists, but the GGUF didn't declare add_bos_token.
        tok.bos_id = Some(1000);
        assert_eq!(tok.encode_special("hello", true), tok.encode("hello"));
    }

    #[test]
    fn encode_special_flag_without_bos_id_is_noop() {
        let mut tok = make_test_tokenizer();
        // Flag declared, but the vocab has no BOS token to prepend.
        tok.add_bos = true;
        assert_eq!(tok.encode_special("hello", true), tok.encode("hello"));
    }

    #[test]
    fn encode_special_appends_eos_and_handles_empty_text() {
        let mut tok = make_test_tokenizer();
        tok.eos_id = Some(1001);
        tok.add_eos = true;
        let special = tok.encode_special("hello", true);
        assert_eq!(special.last(), Some(&1001));
        assert_eq!(&special[..special.len() - 1], &tok.encode("hello")[..]);
        // Empty text still emits the declared markers.
        tok.bos_id = Some(1000);
        tok.add_bos = true;
        assert_eq!(tok.encode_special("", true), vec![1000, 1001]);
    }

    #[test]
    fn test_byte_unicode_roundtrip() {
        let table = build_byte_to_unicode();
        let reverse = build_unicode_to_byte();
        for b in 0u8..=255 {
            let ch = table[b as usize];
            assert_eq!(reverse[&ch], b, "roundtrip failed for byte {b:#04x}");
        }
    }

    #[test]
    fn test_pretokenize_splits_words() {
        let re = build_pretokenize_regex("lfm2");
        let chunks: Vec<&str> = re
            .find_iter("The meaning of life")
            .map(|m| m.as_str())
            .collect();
        assert_eq!(chunks, vec!["The", " meaning", " of", " life"]);
    }

    #[test]
    fn test_pretokenize_contractions() {
        let re = build_pretokenize_regex("lfm2");
        let chunks: Vec<&str> = re.find_iter("I'm don't").map(|m| m.as_str()).collect();
        assert!(chunks.contains(&"'m"));
        assert!(chunks.contains(&"'t"));
    }

    #[test]
    fn test_pretokenize_numbers() {
        let re = build_pretokenize_regex("lfm2");
        let chunks: Vec<&str> = re.find_iter("test 12345").map(|m| m.as_str()).collect();
        // Numbers split into 1-3 digit groups
        assert_eq!(chunks, vec!["test", " ", "123", "45"]);
    }

    #[test]
    fn test_pretokenize_refact_splits_each_digit() {
        // REFACT (Granite) splits numbers one digit at a time via a bare `\p{N}`,
        // unlike LLAMA3's ` ?\p{N}+` which groups 1-3 digits.
        let re = build_pretokenize_regex("refact");
        let chunks: Vec<&str> = re.find_iter("12345").map(|m| m.as_str()).collect();
        assert_eq!(chunks, vec!["1", "2", "3", "4", "5"]);
    }

    #[test]
    fn test_pretokenize_refact_contractions_and_words() {
        // The non-digit arms still match GPT-2 behavior (leading-space words,
        // contraction suffixes).
        let re = build_pretokenize_regex("refact");
        let chunks: Vec<&str> = re.find_iter("I'm ok").map(|m| m.as_str()).collect();
        assert_eq!(chunks, vec!["I", "'m", " ok"]);
    }

    #[test]
    fn test_refact_keeps_whitespace_run_before_digit() {
        // llama.cpp REFACT is a two-pass split (`\p{N}` first), so a digit is
        // always bare and a preceding whitespace run keeps its trailing space.
        // `pretokenize` must not donate a space to a digit when `digits_split_bare`.
        let tok = make_pretok_tokenizer("refact");
        assert_eq!(tok.pretokenize("  3"), vec!["  ", "3"]);
        assert_eq!(tok.pretokenize("x  9"), vec!["x", "  ", "9"]);
        // Indented digit (the common code case): "\n    3" → ["\n    ", "3"].
        assert_eq!(tok.pretokenize("\n    3"), vec!["\n    ", "3"]);
        // A single space before a digit is unaffected (emulation needs a run ≥2).
        assert_eq!(tok.pretokenize(" 3"), vec![" ", "3"]);
    }

    #[test]
    fn test_refact_still_donates_space_before_word() {
        // Donation MUST still happen before a letter/symbol (those arms carry the
        // ` ?` leading space in REFACT too), so only digits are special-cased.
        let tok = make_pretok_tokenizer("refact");
        assert_eq!(tok.pretokenize("  x"), vec![" ", " x"]);
        assert_eq!(tok.pretokenize("\n    return"), vec!["\n   ", " return"]);
    }

    #[test]
    fn test_gpt2_digit_absorbs_donated_space() {
        // The GPT-2 pretokenizer's number arm is ` ?\p{N}+`, so a digit genuinely
        // takes the donated leading space (matching llama.cpp's GPT-2 split) —
        // only the bare-`\p{N}` REFACT family suppresses the donation. This guards
        // the gating: the fix must not change non-REFACT pretokenizers.
        let tok = make_pretok_tokenizer("gpt2");
        assert_eq!(tok.pretokenize("  3"), vec![" ", " 3"]);
    }

    /// Minimal tokenizer for exercising `pretokenize` (regex + whitespace
    /// emulation) without a full vocab.
    fn make_pretok_tokenizer(pre: &str) -> BpeTokenizer {
        BpeTokenizer {
            vocab: Vec::new(),
            token_to_id: HashMap::new(),
            merge_ranks: HashMap::new(),
            special_tokens: HashMap::new(),
            bos_id: None,
            eos_id: None,
            add_bos: false,
            add_eos: false,
            chat_template: None,
            pretokenize_re: build_pretokenize_regex(pre),
            digits_split_bare: pre == "refact",
            byte_to_unicode: build_byte_to_unicode(),
            unicode_to_byte: build_unicode_to_byte(),
        }
    }

    /// Build a tokenizer with GPT-2-style vocab (Ġ-prefixed tokens for spaces).
    fn make_gpt2_style_tokenizer() -> BpeTokenizer {
        let table = build_byte_to_unicode();
        let mut vocab: Vec<Vec<u8>> = Vec::new();
        let mut token_to_id: HashMap<Vec<u8>, u32> = HashMap::new();

        // Token 0: pad
        vocab.push(b"<pad>".to_vec());
        token_to_id.insert(b"<pad>".to_vec(), 0);

        // Individual GPT-2 unicode chars for common bytes
        for b in 0u8..=127 {
            let ch = table[b as usize];
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            let bytes = s.as_bytes().to_vec();
            let id = vocab.len() as u32;
            vocab.push(bytes.clone());
            token_to_id.insert(bytes, id);
        }

        // Add merged tokens with Ġ prefix (space = U+0120 in GPT-2 encoding)
        let space_char = table[b' ' as usize]; // Ġ
        let add_token =
            |vocab: &mut Vec<Vec<u8>>, map: &mut HashMap<Vec<u8>, u32>, s: &str| -> u32 {
                let bytes = s.as_bytes().to_vec();
                let id = vocab.len() as u32;
                vocab.push(bytes.clone());
                map.insert(bytes, id);
                id
            };

        // "Hi" merged token
        let _hi_id = add_token(&mut vocab, &mut token_to_id, "Hi");
        // "Ġworld" merged token (space-prefixed "world")
        let space_world = format!("{space_char}world");
        let _world_id = add_token(&mut vocab, &mut token_to_id, &space_world);

        // Merges
        let mut merge_ranks = HashMap::new();
        merge_ranks.insert((b"H".to_vec(), b"i".to_vec()), 0);
        // Ġ + w merge
        let g_bytes = {
            let mut buf = [0u8; 4];
            space_char.encode_utf8(&mut buf).as_bytes().to_vec()
        };
        merge_ranks.insert((g_bytes.clone(), b"w".to_vec()), 1);
        // Ġw + o
        let gw_bytes = [g_bytes.as_slice(), b"w"].concat();
        merge_ranks.insert((gw_bytes.clone(), b"o".to_vec()), 2);
        // Ġwo + r
        let gwor = [gw_bytes.as_slice(), b"o"].concat();
        merge_ranks.insert((gwor.clone(), b"r".to_vec()), 3);
        // Ġwor + l
        let gworl = [gwor.as_slice(), b"r"].concat();
        merge_ranks.insert((gworl.clone(), b"l".to_vec()), 4);
        // Ġworl + d
        let gworld = [gworl.as_slice(), b"l"].concat();
        merge_ranks.insert((gworld.clone(), b"d".to_vec()), 5);

        BpeTokenizer {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens: HashMap::new(),
            bos_id: None,
            eos_id: None,
            add_bos: false,
            add_eos: false,
            chat_template: None,
            pretokenize_re: build_pretokenize_regex("lfm2"),
            digits_split_bare: false,
            byte_to_unicode: build_byte_to_unicode(),
            unicode_to_byte: build_unicode_to_byte(),
        }
    }

    #[test]
    fn test_gpt2_encode_space_prefix() {
        let tok = make_gpt2_style_tokenizer();
        let ids = tok.encode("Hi world");
        // "Hi world" → pretokenize → ["Hi", " world"]
        // "Hi" → BPE merges to "Hi" token
        // " world" → byte-to-unicode → "Ġworld" → BPE merges to "Ġworld" token
        let hi_id = *tok.token_to_id.get(b"Hi".as_slice()).unwrap();
        let table = build_byte_to_unicode();
        let space_world = format!("{}world", table[b' ' as usize]);
        let world_id = *tok.token_to_id.get(space_world.as_bytes()).unwrap();
        assert_eq!(ids, vec![hi_id, world_id]);
    }

    #[test]
    fn test_gpt2_decode_reverses_encode() {
        let tok = make_gpt2_style_tokenizer();
        let ids = tok.encode("Hi world");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "Hi world");
    }

    #[test]
    fn test_encode_single_bytes() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![b'a' as u32, b'b' as u32]);
    }

    #[test]
    fn test_encode_with_merges() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hello");
        // Should merge: h+e->he, l+l->ll, he+ll->hell, hell+o->hello
        assert_eq!(ids, vec![260]); // "hello" as single token
    }

    #[test]
    fn test_encode_partial_merges() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("hell");
        // h+e->he, l+l->ll, he+ll->hell
        assert_eq!(ids, vec![259]); // "hell" as single token
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = make_test_tokenizer();
        let text = "hello";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_single_bytes() {
        let tok = make_test_tokenizer();
        let decoded = tok.decode(&[72, 105]); // 'H', 'i'
        assert_eq!(decoded, "Hi");
    }

    #[test]
    fn test_encode_empty() {
        let tok = make_test_tokenizer();
        // Explicit type on the RHS: once `serde_json` is in the lib-test
        // dep graph (pulled in by `cera::manifest`) the compiler sees an
        // `impl PartialEq<serde_json::Value> for u32` alongside the
        // reflexive `impl PartialEq for u32`, which makes `vec![]`'s
        // element type ambiguous here.
        assert_eq!(tok.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn test_unescape_byte_token() {
        assert_eq!(unescape_token("<0x0A>"), vec![0x0A]); // newline
        assert_eq!(unescape_token("<0xFF>"), vec![0xFF]);
        assert_eq!(unescape_token("<0x00>"), vec![0x00]);
    }

    #[test]
    fn test_unescape_space_marker() {
        assert_eq!(unescape_token("▁hello"), b" hello");
        assert_eq!(unescape_token("▁"), b" ");
    }

    #[test]
    fn test_unescape_regular() {
        assert_eq!(unescape_token("hello"), b"hello");
    }

    #[test]
    fn test_chat_template_rendering() {
        let mut tok = make_test_tokenizer();
        tok.chat_template = Some(
            "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
                .to_string(),
        );

        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let result = apply_chat_template(&tok, &messages, true).unwrap();
        assert_eq!(result, "user: Hello!\nassistant: ");
    }

    /// `strip_generation_markers` removes `{% generation %}` /
    /// `{% endgeneration %}` block tags (with all whitespace-control
    /// variants) and leaves the contained content untouched. Same
    /// shape as the LFM2.5-VL chat template.
    #[test]
    fn strip_generation_markers_removes_all_variants() {
        let cases = [
            ("{% generation %}", ""),
            ("{%- generation -%}", ""),
            ("{%- generation %}", ""),
            ("{% generation -%}", ""),
            ("{% endgeneration %}", ""),
            ("{%- endgeneration -%}", ""),
            ("a{% generation %}b{% endgeneration %}c", "abc"),
            (
                "before{%- generation -%}inside{%- endgeneration -%}after",
                "beforeinsideafter",
            ),
            ("no markers here", "no markers here"),
        ];
        for (input, want) in cases {
            let got = strip_generation_markers(input);
            assert_eq!(&*got, want, "input: {input:?}");
        }
    }

    /// LFM2.5-VL chat template (carrying `{% generation %}` blocks)
    /// must render without erroring once the markers are stripped.
    /// `BpeTokenizer::from_gguf` runs the strip at load time and
    /// stores the cleaned template; this test mirrors that step on
    /// a hand-built tokenizer so we cover the marker shape end-to-
    /// end without needing a real GGUF.
    #[test]
    fn chat_template_with_generation_block_renders() {
        let mut tok = make_test_tokenizer();
        let raw = "{% for msg in messages %}\
             {%- if msg.role == 'assistant' -%}\
             {%- generation -%}{{ msg.role }}: {{ msg.content }}\n{%- endgeneration -%}\
             {%- else -%}{{ msg.role }}: {{ msg.content }}\n{%- endif -%}\
             {% endfor %}";
        // Mirror the `from_gguf` load-time clean step.
        tok.chat_template = Some(strip_generation_markers(raw).into_owned());
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
            },
        ];
        let got = apply_chat_template(&tok, &messages, false).unwrap();
        // The `{%- … -%}` whitespace-stripping variants eat the
        // surrounding newlines; the assertion checks that rendering
        // succeeded AND the user/assistant content lands in order
        // (i.e. the `{% generation %}` block was passthrough, not a
        // syntax error). Newline behavior is governed by the
        // template's strip markers, not by the generation block.
        assert!(
            got.contains("user: Hi") && got.contains("assistant: Hello!"),
            "expected user+assistant content; got {got:?}"
        );
    }

    /// `apply_chat_template` renders multimodal `ChatMessageMultimodal`
    /// messages through a Jinja2 template that mirrors LFM2-VL's
    /// `parse_content` macro: dispatches on `content is string` vs
    /// list-of-items, and emits literal `<image>` for items with
    /// `type == "image"`. Both shapes flow through the same generic
    /// render function so callers don't have to pick a branch
    /// upstream.
    #[test]
    fn chat_template_multimodal_emits_image_marker() {
        let mut tok = make_test_tokenizer();
        // Minimal `parse_content`-shape template: handle string OR
        // list-of-items in the same loop.
        tok.chat_template = Some(
            "{% for msg in messages %}\
             {%- if msg.content is string -%}\
                 {{ msg.role }}: {{ msg.content }}\n\
             {%- else -%}\
                 {{ msg.role }}: \
                 {%- for item in msg.content -%}\
                     {%- if item.type == 'image' -%}<image>\
                     {%- elif item.type == 'text' -%}{{ item.text }}\
                     {%- endif -%}\
                 {%- endfor -%}\n\
             {%- endif -%}\
             {% endfor %}"
                .to_string(),
        );
        // String content path: zero `<image>` markers, content
        // rendered verbatim.
        let text_only = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let rendered = apply_chat_template(&tok, &text_only, false).unwrap();
        assert!(
            rendered.contains("user: Hi"),
            "string-content render lost the body: {rendered:?}"
        );
        assert!(
            !rendered.contains("<image>"),
            "string-content render shouldn't emit <image>: {rendered:?}"
        );

        // List-of-items content path: image item emits literal
        // `<image>`; text item emits the text.
        let multimodal = vec![ChatMessageMultimodal {
            role: "user".to_string(),
            content: vec![
                ContentItem::Image,
                ContentItem::Text {
                    text: " Describe.".to_string(),
                },
            ],
        }];
        let rendered = apply_chat_template(&tok, &multimodal, false).unwrap();
        // `{%- for ... -%}` whitespace-strip eats the space after `:`
        // — assert on the marker + text combination, which is what
        // matters for the helper's downstream tokenize+walk.
        assert!(
            rendered.contains("<image> Describe."),
            "list-content render didn't combine items in order: {rendered:?}"
        );
        // Exactly one `<image>` marker — caller will pair it with one
        // image's bytes.
        assert_eq!(
            rendered.matches("<image>").count(),
            1,
            "expected exactly one <image> marker; got {rendered:?}"
        );
    }
}
