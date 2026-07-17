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
            size_of::<RopeParams>(),
            shaders::ROPE,
            "Params",
            "RopeParams",
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
        (
            size_of::<BiasAddParams>(),
            shaders::BIAS_ADD,
            "Params",
            "BiasAddParams",
        ),
        (
            size_of::<RmsNormBatchParams>(),
            shaders::RMSNORM_BATCH,
            "Params",
            "RmsNormBatchParams",
        ),
        (
            size_of::<Conv1dBatchParams>(),
            shaders::CONV1D_FUSED_BATCH,
            "Params",
            "Conv1dBatchParams",
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
            size_of::<LayerNormBatchParams>(),
            shaders::LAYERNORM_BATCH,
            "Params",
            "LayerNormBatchParams",
        ),
        // `ElementwiseParams` is reused for `gelu.metal`, which has its own `Params`.
        (
            size_of::<ElementwiseParams>(),
            shaders::GELU,
            "Params",
            "ElementwiseParams (gelu)",
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
