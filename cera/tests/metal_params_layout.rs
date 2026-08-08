//! Assert every Rust param mirror in `backend::metal::params` still matches the MSL
//! struct it mirrors, by parsing the shader source.
//!
//! # Why this test exists
//!
//! The `const _: () = assert!(size_of::<T>() == N)` next to each mirror only catches
//! someone editing the *Rust* struct. It cannot see the `.metal` file, so it is blind to
//! the direction that actually caused a NaN bug in production: a field added to the
//! *shader's* struct, leaving the Rust upload short. The kernel then reads past the end
//! of the upload — undefined behaviour, which that time surfaced as NaN and silent audio,
//! but just as easily returns plausible-but-wrong numbers with every test green.
//!
//! This test closes that direction. It parses each `struct` out of the embedded MSL
//! source, counts its scalar fields, and asserts the Rust mirror is exactly as wide.
//!
//! It dispatches nothing and needs no GPU, so unlike the parity/oracle suites it is
//! meaningful on any machine the `metal` feature compiles on — including a CI runner with
//! no Metal device.

#![cfg(feature = "metal")]

use cera::backend::metal::params::*;
use cera::backend::metal::shaders;

/// Byte size of `struct <name>` in `src`, counting 4-byte scalar fields.
///
/// Returns `None` if the struct isn't found — the caller turns that into a failure, so a
/// renamed or deleted MSL struct fails loudly instead of silently passing.
fn msl_struct_bytes(src: &str, name: &str) -> Option<usize> {
    let start = src.find(&format!("struct {name} "))?;
    let open = start + src[start..].find('{')?;
    let close = open + src[open..].find("};")?;
    let body = &src[open + 1..close];

    // Strip `// comments` first, then split on `;` rather than on newlines: several of
    // these structs are declared on a single line (`struct Params { uint n; uint _pad; };`),
    // so one-field-per-line would undercount them.
    let code: String = body
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ");

    let mut bytes = 0usize;
    for decl in code.split(';') {
        let ty = decl.split_whitespace().next().unwrap_or("");
        match ty {
            // The trailing split after the final `;` (and any blank line) is empty.
            "" => {}
            // Every scalar these params structs use is 4 bytes wide.
            "uint" | "int" | "float" => bytes += 4,
            "half" => bytes += 2,
            // Anything else — a vector (`uint2`), array (`uint x[2]`), or byte
            // (`uchar`) — would be silently counted as 0 and could make a genuine
            // width mismatch cancel out to a false pass. Fail loudly instead: a new
            // field type means this parser (and the `#[repr(C)]` mirror) needs a
            // deliberate look, not a silent skip.
            other => panic!(
                "struct {name}: unrecognized MSL field type `{other}` — teach \
                 msl_struct_bytes its width before trusting this test"
            ),
        }
    }
    Some(bytes)
}

/// `(rust_size, shader_source, msl_struct_name, rust_type_name)`
fn cases() -> Vec<(usize, &'static str, &'static str, &'static str)> {
    vec![
        (
            size_of::<QkNormRopeParams>(),
            shaders::QK_NORM_ROPE,
            "Params",
            "QkNormRopeParams",
        ),
        (
            size_of::<QkNormRopeBatchParams>(),
            shaders::QK_NORM_ROPE_BATCH,
            "BatchParams",
            "QkNormRopeBatchParams",
        ),
        (
            size_of::<KvShiftKParams>(),
            shaders::KV_SHIFT,
            "KParams",
            "KvShiftKParams",
        ),
        (
            size_of::<KvCopyParams>(),
            shaders::KV_SHIFT,
            "CopyParams",
            "KvCopyParams",
        ),
        (
            size_of::<GemmF32Params>(),
            shaders::GEMM_F32,
            "GemmParams",
            "GemmF32Params",
        ),
        (
            size_of::<QuantGemmParams>(),
            shaders::GEMM_Q4_0,
            "GemmParams",
            "QuantGemmParams (q4_0)",
        ),
        (
            size_of::<QuantGemmParams>(),
            shaders::GEMM_Q8_0,
            "GemmParams",
            "QuantGemmParams (q8_0)",
        ),
        (
            size_of::<QuantGemmParams>(),
            shaders::GEMM_Q4_K,
            "GemmParams",
            "QuantGemmParams (q4_k)",
        ),
        (
            size_of::<QuantGemmParams>(),
            shaders::GEMM_Q5_K,
            "GemmParams",
            "QuantGemmParams (q5_k)",
        ),
        (
            size_of::<QuantGemmParams>(),
            shaders::GEMM_Q6_K,
            "GemmParams",
            "QuantGemmParams (q6_k)",
        ),
        (
            size_of::<GemvBatchParams>(),
            shaders::GEMV_Q4_0_BATCH,
            "BatchParams",
            "GemvBatchParams (q4_0)",
        ),
        (
            size_of::<GemvBatchParams>(),
            shaders::GEMV_Q8_0_BATCH,
            "BatchParams",
            "GemvBatchParams (q8_0)",
        ),
        (
            size_of::<GemvQkvParams>(),
            shaders::GEMV_Q4_0_FAST,
            "ParamsQKV",
            "GemvQkvParams",
        ),
        (
            size_of::<GemvRmsParams>(),
            shaders::GEMV_Q4_0_FAST,
            "RMSParams",
            "GemvRmsParams",
        ),
        (
            size_of::<GemvSplitKParams>(),
            shaders::GEMV_Q4_0_FAST,
            "SplitKParams",
            "GemvSplitKParams",
        ),
        (
            size_of::<FlashAttnParams>(),
            shaders::FLASH_ATTENTION,
            "Params",
            "FlashAttnParams",
        ),
        // `FlashAttnParams` is also uploaded to the classic decode kernels — the
        // default path for seq_len <= 4096 (`attention.metal`) and the CERA_ATTN=gqa
        // path (`attention_gqa.metal`). Their `Params` must stay identical to
        // flash_attention's, so guard all three, not just flash.
        (
            size_of::<FlashAttnParams>(),
            shaders::ATTENTION,
            "Params",
            "FlashAttnParams (classic)",
        ),
        (
            size_of::<FlashAttnParams>(),
            shaders::ATTENTION_GQA,
            "Params",
            "FlashAttnParams (gqa)",
        ),
        (
            size_of::<SplitAttnParams>(),
            shaders::ATTENTION_SPLITK,
            "SplitParams",
            "SplitAttnParams",
        ),
        (
            size_of::<PrefillAttnParams>(),
            shaders::ATTENTION_PREFILL,
            "PrefillAttnParams",
            "PrefillAttnParams",
        ),
        (
            size_of::<ElementwiseParams>(),
            shaders::ELEMENTWISE,
            "Params",
            "ElementwiseParams",
        ),
        (
            size_of::<ScaleParams>(),
            shaders::ELEMENTWISE,
            "ScaleParams",
            "ScaleParams",
        ),
        // ViT vision encoder (`MetalVitOps`).
        (
            size_of::<VitLinearParams>(),
            shaders::VIT_LINEAR,
            "Params",
            "VitLinearParams",
        ),
        // One `VitAttnParams` mirrors both ViT attention kernels — guard each shader.
        (
            size_of::<VitAttnParams>(),
            shaders::VIT_ATTENTION,
            "Params",
            "VitAttnParams (scalar)",
        ),
        (
            size_of::<VitAttnParams>(),
            shaders::VIT_ATTENTION_MMA,
            "VitAttnParams",
            "VitAttnParams (mma)",
        ),
        (
            size_of::<TqParams>(),
            shaders::TURBOQUANT,
            "TqParams",
            "TqParams",
        ),
        (
            size_of::<TqAttnParams>(),
            shaders::FLASH_ATTENTION_TQ,
            "TqAttnParams",
            "TqAttnParams",
        ),
    ]
}

/// Width, in bytes, of the params buffer a **Slang** kernel expects.
///
/// The Slang ports do not declare a `struct Params` for `msl_struct_bytes` to
/// find. They take a typed buffer instead, and not uniformly: `rope.slang` uses
/// `StructuredBuffer<uint>` indexed to `params[6]`, `bias_add.slang` a
/// `StructuredBuffer<uint2>` swizzled `par_buf[0].x/.y`. One rule covers both:
///
///     u32 slots = components(element type) * (highest element index + 1)
///
/// This closes a gap that predates the migration. While the generated kernels
/// were reachable only from tests and benches, *nothing* checked their params
/// layout against the Rust mirrors, so the guard that exists because a shader
/// struct once outgrew its upload (NaN and silent audio in production) never
/// covered them at all.
fn slang_params_bytes(src: &str) -> Option<usize> {
    // Declaration: `StructuredBuffer<uintN> name : register(tN);`
    let decl = src
        .lines()
        .find(|l| l.contains("StructuredBuffer<uint") && !l.trim_start().starts_with("//"))?;
    let after = decl.split("StructuredBuffer<uint").nth(1)?;
    let (comp_txt, rest) = after.split_once('>')?;
    let components: usize = if comp_txt.is_empty() {
        1
    } else {
        comp_txt.parse().ok()?
    };
    let name = rest.split_whitespace().next()?;

    // A `__target_switch` port carries both backends' bodies in one source and
    // they can read different numbers of params: `rope.slang`'s metal branch
    // stops at `params[4]` while its wgsl branch reaches `params[6]`. This is
    // the *Metal* layout test, so the wgsl body must not count.
    //
    // Drop the `default:` arm rather than keeping only the `case metal:` one.
    // Most reads sit outside the switch entirely (in `rmsnorm_batch.slang` the
    // switch is a reduction detail inside a helper, and every `par_buf` read is
    // after it), so keeping only the metal arm would see almost nothing and
    // silently under-report the width.
    // Anchor on the statement, not the word: `rope.slang` names
    // `__target_switch` in a doc comment on line 3, and matching that put the
    // brace scan in the middle of prose and silently kept the wgsl arm.
    let switch_at = src.match_indices("__target_switch").find(|(i, _)| {
        src[i + "__target_switch".len()..]
            .trim_start()
            .starts_with('{')
    });
    let scope: String = match switch_at.map(|(i, _)| i) {
        None => src.to_string(),
        Some(sw) => {
            // Brace-match from the switch to find its extent.
            let bytes = src.as_bytes();
            let open = sw + src[sw..].find('{').unwrap_or(0);
            let (mut depth, mut end) = (0usize, src.len());
            for (i, &b) in bytes.iter().enumerate().skip(open) {
                match b {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            match src[open..end].find("default:") {
                // Everything but the default arm: before the switch, the metal
                // arm, and everything after it.
                Some(d) => format!("{}{}", &src[..open + d], &src[end..]),
                None => src.to_string(),
            }
        }
    };
    let scope = scope.as_str();

    // Highest element index actually read, over `name[N]` (with or without a swizzle).
    let mut max_idx = 0usize;
    for (i, _) in scope.match_indices(name) {
        let tail = &scope[i + name.len()..];
        if let Some(inner) = tail
            .strip_prefix('[')
            .and_then(|t| t.split_once(']'))
            .map(|(a, _)| a)
            && let Ok(n) = inner.trim().parse::<usize>()
        {
            max_idx = max_idx.max(n);
        }
    }
    Some(components * (max_idx + 1) * 4)
}

/// Rust mirrors whose kernel is now a Slang port, paired with that source.
fn slang_cases() -> Vec<(usize, &'static str, &'static str)> {
    vec![
        (
            size_of::<RopeParams>(),
            include_str!("../src/backend/shaders/slang/rope.slang"),
            "RopeParams",
        ),
        (
            size_of::<BiasAddParams>(),
            include_str!("../src/backend/shaders/slang/bias_add.slang"),
            "BiasAddParams",
        ),
        (
            size_of::<ElementwiseParams>(),
            include_str!("../src/backend/shaders/slang/gelu.slang"),
            "ElementwiseParams (gelu)",
        ),
        (
            size_of::<LayerNormBatchParams>(),
            include_str!("../src/backend/shaders/slang/layernorm_batch.slang"),
            "LayerNormBatchParams",
        ),
        (
            size_of::<RmsNormBatchParams>(),
            include_str!("../src/backend/shaders/slang/rmsnorm_batch.slang"),
            "RmsNormBatchParams",
        ),
        (
            size_of::<Conv1dBatchParams>(),
            include_str!("../src/backend/shaders/slang/conv1d_fused_batch.slang"),
            "Conv1dBatchParams",
        ),
    ]
}

#[test]
fn rust_param_mirrors_match_msl_structs() {
    let mut failures = Vec::new();
    for (rust_bytes, src, msl_name, rust_name) in cases() {
        match msl_struct_bytes(src, msl_name) {
            None => failures.push(format!(
                "{rust_name}: MSL `struct {msl_name}` not found — renamed or deleted?"
            )),
            Some(msl_bytes) if msl_bytes != rust_bytes => failures.push(format!(
                "{rust_name}: Rust mirror is {rust_bytes} B but MSL `struct {msl_name}` is \
                 {msl_bytes} B. A kernel reading a struct wider than the upload reads past \
                 the end of it — undefined behaviour, not a crash. Update the Rust mirror \
                 so its width matches, keeping the fields in the same order as the shader."
            )),
            Some(_) => {}
        }
    }
    for (rust_bytes, slang_src, rust_name) in slang_cases() {
        match slang_params_bytes(slang_src) {
            None => failures.push(format!(
                "{rust_name}: no `StructuredBuffer<uint*>` params binding found in the Slang \
                 source — renamed, or the port changed how it takes parameters?"
            )),
            Some(slang_bytes) if slang_bytes != rust_bytes => failures.push(format!(
                "{rust_name}: Rust mirror is {rust_bytes} B but the Slang kernel reads \
                 {slang_bytes} B of params. Same hazard as the MSL case: the kernel reads \
                 past the end of the upload."
            )),
            Some(_) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "MSL/Rust params layout drift:\n  {}",
        failures.join("\n  ")
    );
}

/// The parser has to actually parse — a `msl_struct_bytes` that silently returned 0 for
/// everything would make the test above vacuously green.
#[test]
fn parser_counts_fields_and_ignores_comments() {
    let src = "
struct Foo {
    uint a;
    int  b;    // int c; <- a decoy inside a comment
    float d;
    half e;
};
struct OneLiner { uint n; uint _pad; };
";
    assert_eq!(msl_struct_bytes(src, "Foo"), Some(4 + 4 + 4 + 2));
    assert_eq!(msl_struct_bytes(src, "Missing"), None);
    // Single-line structs are real (elementwise.metal); one-field-per-line undercounts.
    assert_eq!(msl_struct_bytes(src, "OneLiner"), Some(8));

    // And it must agree with a struct we know the size of by hand.
    assert_eq!(
        msl_struct_bytes(shaders::QK_NORM_ROPE, "Params"),
        Some(36),
        "qk_norm_rope Params is 9 uints"
    );
}
