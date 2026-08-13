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
///
/// Two things this is deliberately weaker at than [`msl_struct_bytes`], both
/// worth knowing before extending the table:
///
/// 1. It reads the `.slang` *input*, not the `.metal` the GPU is handed. The
///    emitted MSL has no `struct Params` to parse, so there is nothing stricter
///    to point at, but a change in how Slang lowers a params buffer would not be
///    caught here.
/// 2. It is a text scan, so it can only be trusted where its assumptions hold.
///    Every assumption below therefore returns `None` when violated rather than
///    guessing, and `None` is a test failure: under-reporting a width is exactly
///    the silent pass this guard exists to prevent.
fn slang_params_bytes(src: &str) -> Option<usize> {
    // Declaration: `StructuredBuffer<uintN> name : register(tN);`
    //
    // Require exactly one, and skip `RWStructuredBuffer` (whose name contains
    // this needle): an unqualified "first match" picks the output buffer in
    // `argmax_f32.slang`, and the wgsl-side params buffer in the two-binding
    // sources, both of which report a plausible but wrong width.
    let candidates: Vec<&str> = src
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("//")
                && t.contains("StructuredBuffer<uint")
                && !t.contains("RWStructuredBuffer<uint")
        })
        .collect();
    // Only `uint`-typed bindings are recognized, which every params buffer here
    // is. That is not a silent assumption: a params binding of some other
    // element type leaves `candidates` empty, which falls through to `None` and
    // fails the test, rather than being skipped. Widen the needle when a kernel
    // actually needs it.
    //
    // A `__target_switch` source can declare one params binding per target, as
    // `rmsnorm.slang` does with `p_wgsl` and `p_metal`, because the two branches
    // want different layouts. This is the Metal layout test, so take the
    // `_metal` one; refusing outright (as an earlier revision did) left every
    // such kernel uncovered, which is worse than the ambiguity it avoided.
    // Anything else with more than one binding is still ambiguous.
    let decl = match candidates.as_slice() {
        [only] => only,
        many => *many
            .iter()
            .find(|l| l.contains("_metal") || l.contains("metal_"))?,
    };
    let after = decl.split("StructuredBuffer<uint").nth(1)?;
    let (comp_txt, rest) = after.split_once('>')?;
    let components: usize = if comp_txt.is_empty() {
        1
    } else {
        comp_txt.parse().ok()?
    };
    let name = rest.split_whitespace().next()?;

    // Strip `//` comments before anything else looks at the text, as
    // `msl_struct_bytes` does. Prose is not code: `rope.slang` documents its two
    // arms as `params[0..4]` and `params[0..6]`, which are neither real indices
    // nor parseable ones, and a comment mentioning `case metal:` or `default:`
    // would move the arm boundary found below. Safe here because no `.slang`
    // source contains `//` inside a string literal or a URL.
    let code: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let code = code.as_str();

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
    // Every statement-level switch is handled, not just the first. `softmax.slang`
    // has three and `rmsnorm.slang` two, so bailing on the second (as an earlier
    // revision did) left exactly the multi-switch kernels uncovered.
    let switch_starts: Vec<usize> = code
        .match_indices("__target_switch")
        .filter(|(i, _)| {
            code[i + "__target_switch".len()..]
                .trim_start()
                .starts_with('{')
        })
        .map(|(i, _)| i)
        .collect();

    // The byte range of each switch's `default:` arm, i.e. what to delete.
    let bytes = code.as_bytes();
    let mut cuts: Vec<(usize, usize)> = Vec::new();
    for sw in switch_starts {
        let open = sw + code[sw..].find('{')?;
        let (mut depth, mut end) = (0usize, 0usize);
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
        // Unbalanced braces: the extent is a guess, so do not make one.
        if end == 0 {
            return None;
        }
        let body = &code[open..end];
        let Some(d) = body.find("default:") else {
            continue; // no portable arm to drop
        };
        // Slang allows `default:` before `case metal:`, which would drop the
        // metal arm instead. Refuse rather than report the smaller width.
        match body.find("case metal:") {
            Some(m) if m < d => cuts.push((open + d, end - 1)),
            _ => return None,
        }
    }

    // Delete back to front so earlier offsets stay valid.
    let mut scope = code.to_string();
    for (from, to) in cuts.into_iter().rev() {
        scope.replace_range(from..to, "");
    }
    let scope = scope.as_str();

    // Highest element index actually read, over `name[N]` (with or without a
    // swizzle). Two ways this refuses to guess rather than under-report:
    // a non-literal index (`par_buf[i]`, `par_buf[base + 1]`) cannot be resolved
    // by a text scan, and *no* indexed read at all means the binding was renamed
    // out from under this scan, since a params buffer nothing reads would not be
    // declared.
    let mut seen = false;
    let max_idx = scope
        .match_indices(name)
        .filter_map(|(i, _)| {
            scope[i + name.len()..]
                .strip_prefix('[')
                .and_then(|t| t.split_once(']'))
                .map(|(inner, _)| inner.trim().parse::<usize>())
        })
        .try_fold(0usize, |acc, parsed| {
            seen = true;
            parsed.ok().map(|n| acc.max(n))
        })?;
    seen.then(|| components * (max_idx + 1) * 4)
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
        (
            size_of::<ArgmaxParams>(),
            include_str!("../src/backend/shaders/slang/argmax_f32.slang"),
            "ArgmaxParams",
        ),
        (
            size_of::<ElementwiseParams>(),
            include_str!("../src/backend/shaders/slang/activations.slang"),
            "ElementwiseParams (activations)",
        ),
        (
            size_of::<Conv2dDirectParams>(),
            include_str!("../src/backend/shaders/slang/conv2d_direct.slang"),
            "Conv2dDirectParams",
        ),
        (
            size_of::<TransposeBlockedParams>(),
            include_str!("../src/backend/shaders/slang/transpose_blocked.slang"),
            "TransposeBlockedParams",
        ),
        (
            size_of::<Batch2dParams>(),
            include_str!("../src/backend/shaders/slang/glu_split.slang"),
            "Batch2dParams (glu_split)",
        ),
        (
            size_of::<Batch2dParams>(),
            include_str!("../src/backend/shaders/slang/chan_affine_silu.slang"),
            "Batch2dParams (chan_affine_silu)",
        ),
        (
            size_of::<AudioXlAttnParams>(),
            include_str!("../src/backend/shaders/slang/audio_xl_attention.slang"),
            "AudioXlAttnParams",
        ),
        (
            size_of::<StftFrameParams>(),
            include_str!("../src/backend/shaders/slang/stft_frame.slang"),
            "StftFrameParams",
        ),
        (
            size_of::<PowerSpecParams>(),
            include_str!("../src/backend/shaders/slang/power_spec.slang"),
            "PowerSpecParams",
        ),
        (
            size_of::<MelProjectParams>(),
            include_str!("../src/backend/shaders/slang/mel_project.slang"),
            "MelProjectParams",
        ),
        (
            size_of::<MelNormParams>(),
            include_str!("../src/backend/shaders/slang/mel_norm.slang"),
            "MelNormParams",
        ),
    ]
}

/// The host upload must cover everything the kernel reads.
///
/// This is the half `slang_cases` cannot express. That table asserts *equality*
/// against a mirror, which only holds where the upload is exactly the params
/// struct; several flipped kernels upload more than they read (`conv1d` sends 16
/// bytes for 12) and would fail an equality check while being perfectly safe.
/// The property that actually matters is the inequality, and it is the one that
/// was violated: `argmax_f32` uploaded 4 bytes for a kernel that reads 8.
///
/// Pairs a Rust mirror with the `.slang` it is uploaded to. Every persistent
/// params `Buffer` in `MetalLfm2Model` now goes through one of these rather than
/// a literal `cast_slice(&[..])`, which is what makes the width a thing a test
/// can see.
#[test]
fn metal_uploads_cover_what_the_slang_kernels_read() {
    let cases: &[(usize, &str, &str)] = &[
        (
            size_of::<ArgmaxParams>(),
            include_str!("../src/backend/shaders/slang/argmax_f32.slang"),
            "ArgmaxParams -> argmax_f32.slang",
        ),
        (
            size_of::<NormParams>(),
            include_str!("../src/backend/shaders/slang/rmsnorm.slang"),
            "NormParams -> rmsnorm.slang",
        ),
        (
            size_of::<NormParams>(),
            include_str!("../src/backend/shaders/slang/per_head_rmsnorm.slang"),
            "NormParams -> per_head_rmsnorm.slang",
        ),
        (
            size_of::<Conv1dParams>(),
            include_str!("../src/backend/shaders/slang/conv1d.slang"),
            "Conv1dParams -> conv1d.slang",
        ),
        (
            size_of::<ElementwiseParams>(),
            include_str!("../src/backend/shaders/slang/elementwise.slang"),
            "ElementwiseParams -> elementwise.slang",
        ),
    ];
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|&(upload, src, label)| match slang_params_bytes(src) {
            None => Some(format!("{label}: params binding not resolvable")),
            Some(reads) if upload < reads => Some(format!(
                "{label}: host uploads {upload} B but the kernel reads {reads} B,                  so it reads past the end of the buffer"
            )),
            Some(_) => None,
        })
        .collect();
    assert!(
        failures.is_empty(),
        "Metal params upload too small:\n  {}",
        failures.join("\n  ")
    );
}

/// Every `.slang` params binding this test could ever be pointed at, so a
/// rename or a lowering change fails here rather than going unnoticed.
///
/// `slang_cases` can only hold kernels whose host upload is *exactly* a Rust
/// mirror. Several are not: `conv1d` reads 12 bytes from a 16-byte upload, and
/// the ones dispatched from a literal `cast_slice(&[..])` have no mirror to name
/// at all. Those still deserve a guard against the parser silently losing track
/// of their binding, which is the failure that let the `argmax_f32` width
/// mismatch through: it was not in the table, so nothing compared it to
/// anything.
///
/// `None` here means the binding could not be found or resolved. It does not
/// assert the width is *right*, only that it is still knowable.
#[test]
fn every_slang_params_binding_stays_parseable() {
    // (source, name, expected bytes) for every flipped kernel with a params
    // binding. The expected values are what the kernel reads, which is not
    // always what the host uploads.
    let sources: &[(&str, &str, usize)] = &[
        (
            include_str!("../src/backend/shaders/slang/argmax_f32.slang"),
            "argmax_f32",
            8,
        ),
        (
            include_str!("../src/backend/shaders/slang/bias_add.slang"),
            "bias_add",
            8,
        ),
        (
            include_str!("../src/backend/shaders/slang/conv1d.slang"),
            "conv1d",
            12,
        ),
        (
            include_str!("../src/backend/shaders/slang/conv1d_fused.slang"),
            "conv1d_fused",
            12,
        ),
        (
            include_str!("../src/backend/shaders/slang/conv1d_fused_batch.slang"),
            "conv1d_fused_batch",
            24,
        ),
        (
            include_str!("../src/backend/shaders/slang/elementwise.slang"),
            "elementwise",
            8,
        ),
        (
            include_str!("../src/backend/shaders/slang/gelu.slang"),
            "gelu",
            8,
        ),
        (
            include_str!("../src/backend/shaders/slang/layernorm_batch.slang"),
            "layernorm_batch",
            16,
        ),
        (
            include_str!("../src/backend/shaders/slang/per_head_rmsnorm.slang"),
            "per_head_rmsnorm",
            16,
        ),
        // Two params bindings, one per target; the metal arm is the 16-byte one.
        (
            include_str!("../src/backend/shaders/slang/rmsnorm.slang"),
            "rmsnorm",
            16,
        ),
        (
            include_str!("../src/backend/shaders/slang/rmsnorm_batch.slang"),
            "rmsnorm_batch",
            20,
        ),
        (
            include_str!("../src/backend/shaders/slang/rope.slang"),
            "rope",
            20,
        ),
        (
            include_str!("../src/backend/shaders/slang/softmax.slang"),
            "softmax",
            8,
        ),
    ];
    let failures: Vec<String> = sources
        .iter()
        .filter_map(|&(src, name, want)| match slang_params_bytes(src) {
            None => Some(format!(
                "{name}.slang: params binding not found or not resolvable"
            )),
            Some(got) if got != want => Some(format!(
                "{name}.slang: kernel reads {got} B of params, expected {want} B"
            )),
            Some(_) => None,
        })
        .collect();
    assert!(
        failures.is_empty(),
        "Slang params drift:\n  {}",
        failures.join("\n  ")
    );
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
                 source: renamed, or the port changed how it takes parameters?"
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

/// The same obligation for `slang_params_bytes`, which needs it more.
///
/// It is a text scan standing in for a parse, so its failure mode is reporting a
/// plausible-but-small width, which makes `rust_param_mirrors_match_msl_structs`
/// pass while the kernel reads past the end of the upload. Each `None` below is
/// a real way that could happen; without these, nothing but the six live sources
/// exercises any of them, and those six are exactly the cases that work.
#[test]
fn slang_parser_refuses_to_guess() {
    let one = |body: &str| {
        format!("[[vk::binding(0)]] StructuredBuffer<uint> par_buf : register(t0);\n{body}")
    };

    // Baseline: highest literal index wins, and `uintN` scales the width.
    assert_eq!(slang_params_bytes(&one("x = par_buf[2];")), Some(3 * 4));
    assert_eq!(
        slang_params_bytes(
            "[[vk::binding(0)]] StructuredBuffer<uint2> par_buf : register(t0);\nx = par_buf[1].y;"
        ),
        Some(2 * 2 * 4)
    );

    // Comments are prose, not reads: neither the index nor the arm keywords in
    // them may count. `rope.slang` really does document `params[0..6]`.
    assert_eq!(
        slang_params_bytes(&one("x = par_buf[1];  // par_buf[9] is not a read")),
        Some(2 * 4)
    );

    // A non-literal index cannot be resolved, so the width is unknown.
    assert_eq!(slang_params_bytes(&one("x = par_buf[i];")), None);
    assert_eq!(slang_params_bytes(&one("x = par_buf[base + 1];")), None);

    // Never indexed: the binding was renamed out from under the scan.
    assert_eq!(slang_params_bytes(&one("x = 1;")), None);

    // `RWStructuredBuffer<uint>` is an output buffer, not the params binding.
    // Taking the first match anchors on `out_buf` in `argmax_f32.slang`.
    assert_eq!(
        slang_params_bytes(
            "[[vk::binding(0)]] RWStructuredBuffer<uint> out_buf : register(u0);\n\
             [[vk::binding(1)]] StructuredBuffer<uint> par_buf : register(t1);\n\
             out_buf[0] = par_buf[3];"
        ),
        Some(4 * 4)
    );

    // Two read-only params bindings, the wgsl/metal split `rmsnorm.slang` uses.
    // Take the metal one: this is the Metal layout test, and refusing outright
    // left every such kernel uncovered.
    assert_eq!(
        slang_params_bytes(
            "[[vk::binding(0)]] StructuredBuffer<uint> p_wgsl : register(t0);\n\
             [[vk::binding(1)]] StructuredBuffer<uint4> p_metal : register(t1);\n\
             x = p_wgsl[7]; y = p_metal[0];"
        ),
        Some(4 * 4)
    );
    // A params binding of a non-`uint` element type is not silently skipped.
    assert_eq!(
        slang_params_bytes(
            "[[vk::binding(0)]] StructuredBuffer<float> par_buf : register(t0);\n\
             x = par_buf[1];"
        ),
        None
    );

    // Two bindings with no metal-specific name is still ambiguous.
    assert_eq!(
        slang_params_bytes(
            "[[vk::binding(0)]] StructuredBuffer<uint> pa : register(t0);\n\
             [[vk::binding(1)]] StructuredBuffer<uint> pb : register(t1);\n\
             x = pa[1];"
        ),
        None
    );

    // The wgsl arm must not count: only `params[1]` in the metal arm does.
    let switched = "[[vk::binding(0)]] StructuredBuffer<uint> params : register(t0);\n\
         void f() {\n  __target_switch {\n  case metal:\n    x = params[1];\n    break;\n\
         \n  default:\n    x = params[7];\n    break;\n  }\n}";
    assert_eq!(slang_params_bytes(switched), Some(2 * 4));

    // `default:` before `case metal:` would drop the metal arm instead.
    let inverted = "[[vk::binding(0)]] StructuredBuffer<uint> params : register(t0);\n\
         void f() {\n  __target_switch {\n  default:\n    x = params[7];\n    break;\n\
         \n  case metal:\n    x = params[1];\n    break;\n  }\n}";
    assert_eq!(slang_params_bytes(inverted), None);

    // Every switch gets its default arm dropped, not just the first. Here the
    // second switch's portable arm reads a higher index than the metal one, and
    // it must not count. `softmax.slang` really has three switches.
    let two = format!(
        "{switched}\nvoid g() {{\n  __target_switch {{\n  case metal:\n    y = params[2];\n    break;\n\n  default:\n    y = params[9];\n    break;\n  }}\n}}"
    );
    assert_eq!(slang_params_bytes(&two), Some(3 * 4));

    // And it must agree with a source whose width we know independently:
    // `RopeParams` is 5 uints, asserted against `size_of` in `slang_cases`.
    assert_eq!(
        slang_params_bytes(include_str!("../src/backend/shaders/slang/rope.slang")),
        Some(20),
        "rope.slang's metal arm reads params[0..4]"
    );
}
