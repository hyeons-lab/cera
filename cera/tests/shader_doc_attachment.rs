//! Assert every shader constant's doc comment describes *that* constant.
//!
//! # Why this test exists
//!
//! A Rust doc comment binds to the next item below it, so inserting an item
//! above a `pub const` puts the newcomer under the existing doc, and deleting an
//! item leaves its doc attached to whatever follows. Both are pure position
//! bugs: the text never changes, the diff reads as "added N lines", and nothing
//! in the toolchain objects. `rustdoc -D warnings` passes because the intra-doc
//! links still *resolve*, they just point somewhere irrelevant. There is no
//! `missing_docs` lint here, so the item that lost its doc is silent too.
//!
//! It has happened repeatedly. The worst instance was the Slang migration
//! (#356), where deleting twelve `*_SLANG` constants per backend slid 26 doc
//! blocks onto the *next* constant each: `QK_NORM_ROPE` documented as RoPE's
//! Slang port, `ATTENTION` carrying the conv1d doc, `SOFTMAX` losing its doc
//! entirely. That shipped through a green doc build and was caught by review.
//!
//! # What makes it checkable here
//!
//! These docs name their shader source (`shaders/slang/rmsnorm.slang`) and the
//! constant contains that same path (`include_str!(.. "/rmsnorm.metal")`). So
//! the association a human would verify by reading is, for this module, an exact
//! string relation a machine can verify. Where a doc names no shader file this
//! says nothing, which is the right amount to say.
//!
//! Deliberately not `#[cfg(feature = ...)]`: it reads the backend sources as
//! *text*, so it runs on any host in the default test job, including a Linux
//! runner with neither Metal nor a GPU.

/// Shader stems named by a doc block, e.g. `rmsnorm` from
/// `shaders/slang/rmsnorm.slang` or from a bare `gelu.wgsl`.
fn stems_named_in_doc(doc: &str) -> Vec<String> {
    doc.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '/'))
        .filter_map(|tok| {
            let (stem, ext) = tok.rsplit_once('.')?;
            if !matches!(ext, "slang" | "wgsl" | "metal") {
                return None;
            }
            Some(stem.rsplit('/').next()?.to_string())
        })
        .collect()
}

/// The shader stem a constant actually embeds, from its `include_str!` path.
fn stem_included_by(decl: &str) -> Option<String> {
    let after = decl.split("include_str!").nth(1)?;
    // Both spellings appear: `include_str!("shaders/x.metal")` for the
    // handwritten kernels, and `include_str!(concat!(env!("OUT_DIR"), "/x.metal"))`
    // for the generated ones. The last string literal is the path either way.
    let path = after.rsplit('"').nth(1)?;
    let (stem, ext) = path.rsplit_once('.')?;
    if !matches!(ext, "slang" | "wgsl" | "metal") {
        return None;
    }
    Some(stem.rsplit('/').next()?.to_string())
}

/// `(const_name, doc_text, decl_text)` for every `pub const .. = include_str!(..)`.
fn shader_consts(src: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut doc = String::new();
    let mut lines = src.lines();
    while let Some(line) = lines.next() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("///") {
            doc.push_str(rest);
            doc.push('\n');
            continue;
        }
        if let Some(rest) = t.strip_prefix("pub const ") {
            // A declaration may wrap, so keep pulling lines until the `;`.
            let mut decl = rest.to_string();
            while !decl.contains(';') {
                match lines.next() {
                    Some(next) => {
                        decl.push(' ');
                        decl.push_str(next.trim());
                    }
                    None => break,
                }
            }
            // `split_once`, not `split(..).next()`: the latter is always `Some`,
            // so it reads as a check on the `NAME: Type = ..` shape while
            // accepting anything.
            if decl.contains("include_str!")
                && let Some((name, _)) = decl.split_once(':')
            {
                out.push((name.trim().to_string(), doc.clone(), decl));
            }
            doc.clear();
            continue;
        }
        // Anything else (attribute, blank line, comment, other item) ends the
        // block. Attributes are kept so `#[cfg]`-gated consts still pair up.
        if !t.starts_with("#[") {
            doc.clear();
        }
    }
    out
}

#[test]
fn shader_const_docs_describe_their_own_constant() {
    let sources: &[(&str, &str)] = &[
        ("metal.rs", include_str!("../src/backend/metal.rs")),
        ("wgpu.rs", include_str!("../src/backend/wgpu.rs")),
    ];

    let mut checked = 0usize;
    let failures: Vec<String> = sources
        .iter()
        .flat_map(|&(file, src)| {
            shader_consts(src)
                .into_iter()
                .filter_map(|(name, doc, decl)| {
                    let included = stem_included_by(&decl)?;
                    let named = stems_named_in_doc(&doc);
                    if named.is_empty() {
                        return None; // doc names no shader; nothing to check
                    }
                    checked += 1;
                    // The *first* shader a doc names is the one it is about:
                    // "Generated from `shaders/slang/rmsnorm.slang` ...". Later
                    // mentions are sibling references (the other backend's twin,
                    // a kernel it contrasts with), so matching on any of them is
                    // too lenient. Deleting a constant merges its doc into the
                    // next one's, and the merged block still names the survivor
                    // somewhere; only the leading mention catches that.
                    if named[0] == included {
                        return None;
                    }
                    Some(format!(
                        "{file}: `{name}` embeds `{included}` but its doc leads with \
                         `{}` (names {named:?}). Either the doc drifted onto the wrong \
                         constant, which is what an insertion or deletion above it does, \
                         or it needs updating.",
                        named[0]
                    ))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(
        failures.is_empty(),
        "shader doc/constant mismatch:\n  {}",
        failures.join("\n  ")
    );
    // A parser that silently matched nothing would make the assert above
    // vacuously green, which is the failure mode this whole file is about.
    assert!(
        checked >= 20,
        "only {checked} shader constants had a doc naming a shader file; the \
         parser has probably stopped matching the source"
    );
}
