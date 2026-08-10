//! Tests for the generated-MSL post-pass that `build.rs` runs over `gemm_q8_0`.
//!
//! The pass itself lives in `build_support/msl_postpass.rs` and is `include!`d by both
//! `build.rs` and this file, so what is tested here is exactly what builds.
//! These are pure text transforms, so none of this needs a GPU or the `metal`
//! feature; `tests/slang_multitarget_parity.rs` covers whether the rewritten
//! kernel still computes the right answer.

include!("../build_support/msl_postpass.rs");

/// Real slangc output, the same bytes `build.rs` feeds the pass when no slangc
/// is installed. Committed output is byte-compared against slangc in CI, so
/// this fixture cannot drift from what the toolchain emits.
const GENERATED: &str = include_str!("../src/backend/shaders/slang/gemm_q8_0.metal");

fn patched() -> String {
    postpass_gemm_msl(GENERATED).expect("post-pass should apply to committed slangc output")
}

/// Guards the whole point of the pass: all three conditions must be present,
/// because any proper subset is not a partial speedup. Forcing the unroll alone
/// measured *slower* than leaving the shader alone.
#[test]
fn applies_all_three_conditions() {
    let out = patched();

    // 1. scratch is a caller-supplied allocation, not static groupshared
    assert!(
        out.contains("threadgroup char* shmem_p_0 [[threadgroup(0)]]"),
        "kernel should take a threadgroup scratch parameter"
    );
    assert!(
        !out.contains("threadgroup array<half, int("),
        "static half scratch should be gone"
    );
    assert!(
        !out.contains("threadgroup array<float, int("),
        "static float scratch should be gone"
    );

    // 2. the k-loop walks pointers
    assert!(
        out.contains("threadgroup const half* pa_1 = pa_0 + "),
        "half scratch should advance by pointer"
    );
    assert!(
        out.contains("threadgroup const float* pb_1 = pb_0 + "),
        "float scratch should advance by pointer"
    );
    assert!(
        !out.contains("(_sa)[0]) + (lsma"),
        "no matrix load should still address through the uint index"
    );

    // 3. the k-loop has an explicit unroll count
    assert!(
        out.contains("#pragma unroll(4)"),
        "k-loop should carry its trip count as an unroll factor"
    );
}

/// The sb slice must start exactly where sa ends. Asserting the literal 4096
/// would pass just as happily against a hardcoded offset, so this retiles the
/// fixture and checks the offset tracks: 1024 halves must put sb at byte 2048.
#[test]
fn derives_the_scratch_split_from_the_declared_extent() {
    assert!(
        patched().contains("(threadgroup float*)(shmem_p_0 + 4096)"),
        "2048 halves should put the float scratch at byte 4096"
    );

    // Same 8 KB total, different split.
    let retiled = GENERATED
        .replace("array<half, int(2048)>", "array<half, int(1024)>")
        .replace("array<float, int(1024)>", "array<float, int(1536)>");
    assert_ne!(retiled, GENERATED, "fixture should contain both extents");
    let out = postpass_gemm_msl(&retiled).expect("retiled fixture should still apply");
    assert!(
        out.contains("(threadgroup float*)(shmem_p_0 + 2048)"),
        "1024 halves should move the float scratch to byte 2048"
    );
}

/// Call sites bind a fixed 8 KB and nothing propagates a new size to them, so
/// scratch that outgrows the budget has to decline. Accepting it would move the
/// arrays into an allocation too small to hold them.
#[test]
fn declines_when_scratch_exceeds_what_callers_bind() {
    let big = GENERATED.replace("array<half, int(2048)>", "array<half, int(4096)>");
    assert_ne!(big, GENERATED, "fixture should contain the extent");
    let err = postpass_gemm_msl(&big).expect_err("should decline on oversized scratch");
    assert!(err.contains("callers bind"), "unexpected error: {err}");
}

/// The float slice is cast from a byte pointer and written through a
/// `float2x4*`, so a split that lands off a 16-byte boundary is undefined
/// behaviour rather than a slowdown.
#[test]
fn declines_on_a_misaligned_scratch_split() {
    // 2050 halves + 1023 floats is still exactly 8 KB, but starts sb at 4100.
    let odd = GENERATED
        .replace("array<half, int(2048)>", "array<half, int(2050)>")
        .replace("array<float, int(1024)>", "array<float, int(1023)>");
    assert_ne!(odd, GENERATED, "fixture should contain both extents");
    let err = postpass_gemm_msl(&odd).expect_err("should decline on a misaligned split");
    assert!(err.contains("multiple of 16"), "unexpected error: {err}");
}

/// The pointer walk only tracks the uint index while that index is written
/// exactly twice: once to seed it, once to advance it. Any other write moves the
/// index without moving the pointer, and since the loads now read the pointer,
/// the kernel would quietly read the wrong tile. Every shape below produced
/// fully patched, silently wrong MSL against an earlier version of this guard.
#[test]
fn declines_when_the_k_loop_index_is_written_again() {
    let step = "            uint lsma_1 = lsma_0 + 512U;";
    let after_step = "            lsmb_0 = lsmb_1;";
    let header = "        for(;;)";

    let cases = [
        // before the step, inside the loop
        (step, format!("            lsma_0 = lsma_0 ^ 64U;\n{step}")),
        // after the last step but still inside the loop
        (
            after_step,
            format!("{after_step}\n            lsma_0 = lsma_0 + 8U;"),
        ),
        // between the seed and the loop header, on a different indentation
        (
            header,
            format!("            lsma_0 = lsma_0 + 8U;\n{header}"),
        ),
        // a compound form the guard must not read as a mere mention
        (step, format!("            lsma_0 += 8U;\n{step}")),
        // and an increment
        (step, format!("            lsma_0++;\n{step}")),
    ];

    for (anchor, replacement) in cases {
        let mutated = GENERATED.replacen(anchor, &replacement, 1);
        assert_ne!(mutated, GENERATED, "fixture should contain {anchor:?}");
        let err =
            postpass_gemm_msl(&mutated).expect_err(&format!("should decline for {replacement:?}"));
        assert!(
            err.contains("to be written 2 time(s)"),
            "unexpected error for {replacement:?}: {err}"
        );
    }
}

/// The seed is read back off `lsma_0` rather than re-spliced, so the seed
/// expression is evaluated exactly once however it is spelled. Splicing the text
/// a second time would double any side effect in it, leaving the pointer off the
/// index with no compile error, since the loads read the pointer.
#[test]
fn seeds_the_pointer_from_the_index_not_a_second_copy() {
    let out = patched();
    assert!(
        out.contains("pa_0 = _sa + lsma_0;") && out.contains("pb_0 = _sb + lsmb_0;"),
        "pointers should be seeded from the index variables"
    );

    // A side-effecting seed must be evaluated once, not once per splice.
    let seed = "        lsma_0 = _S26;";
    let odd = GENERATED.replace(seed, "        lsma_0 = _S26++;");
    assert_ne!(odd, GENERATED, "fixture should contain the seed");
    let out = postpass_gemm_msl(&odd).expect("a side-effecting seed should still apply");
    assert_eq!(
        out.matches("_S26++").count(),
        1,
        "the seed expression must not be duplicated onto the pointer"
    );

    // A comma seed is likewise harmless now that nothing is re-spliced.
    let odd = GENERATED.replace(seed, "        lsma_0 = _S26, zz_0 = 1U;");
    assert_ne!(odd, GENERATED, "fixture should contain the seed");
    postpass_gemm_msl(&odd).expect("a comma seed should apply, since it is not re-spliced");
}

/// The step is spliced after an existing `+`, so parentheses cannot save it:
/// `lsma_0 + 256U << 1U` steps the index by `(lsma_0 + 256) << 1` while
/// `pa_0 + (256U << 1U)` steps the pointer by 512. That divergence produces MSL
/// that compiles and reads the wrong tile, so only a bare literal is accepted.
#[test]
fn declines_on_a_non_literal_index_step() {
    let step = "            uint lsma_1 = lsma_0 + 512U;";
    for spelling in [
        "            uint lsma_1 = lsma_0 + 256U << 1U;",
        "            uint lsma_1 = lsma_0 + 512U | 1U;",
        "            uint lsma_1 = lsma_0 + kStride;",
    ] {
        let odd = GENERATED.replacen(step, spelling, 1);
        assert_ne!(odd, GENERATED, "fixture should contain the step");
        let err = postpass_gemm_msl(&odd).expect_err(&format!("should decline for {spelling:?}"));
        assert!(
            err.contains("not a plain integer literal"),
            "unexpected error for {spelling:?}: {err}"
        );
    }

    // The literal slangc actually emits still applies.
    let out = patched();
    assert!(
        out.contains("pa_1 = pa_0 + (512U);"),
        "the plain literal step should still be mirrored onto the pointer"
    );
}

/// A load left addressing through the uint index because it is spelled slightly
/// differently from the ones the rewrite knows. That shape compiles and computes
/// the right answer while silently keeping the slow addressing, so it has to be
/// caught by a guard rather than by the rewrite.
///
/// The scan for the index form cannot be that guard: it is anchored on the same
/// spelling the rewrite matches, so whatever defeats one defeats the other and
/// it passes vacuously. It would see the first spelling below but not the other
/// two, which is why all three are checked here and why all three decline on
/// reading the address back out of the load, a check that runs first and does
/// not care how the address is spelled. All three are plausible slangc output:
/// `int(0)` is how it already writes array extents and `params_1[int(0)]` in
/// this very file.
#[test]
fn declines_when_a_load_respelling_defeats_the_rewrite() {
    let spaced = "&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0 + 64U)";
    for respelled in [
        "&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0+64U)",
        "&((*(((&kernelContext_0)->sa_0)))[int(0)]) + (lsma_0 + 64U)",
        "&((*(((&kernelContext_0)->sa_0)))[0])+(lsma_0 + 64U)",
    ] {
        let odd = GENERATED.replacen(spaced, respelled, 1);
        assert_ne!(odd, GENERATED, "fixture should contain the load");
        let Err(err) = postpass_gemm_msl(&odd) else {
            panic!("should decline on the respelled load {respelled}");
        };
        assert!(
            err.contains("not the pointer walk"),
            "unexpected error for {respelled}: {err}"
        );
    }
}

/// The hardest shape to catch: the base address hoisted out of the loop into a
/// temporary. That is textbook licm, and slangc already commons identical
/// subexpressions into `_Sxx` temporaries elsewhere in this same function, so it
/// is a realistic upgrade away. It defeats the load rewrite (the address no
/// longer names the scratch), the index-form scan (the hoist carries no
/// `+ (lsma`), and the check that the body does not mention `_sa` (the only
/// mention is now outside the loop). Only reading the address back out of each
/// load sees it.
///
/// Also asserted for a partial hoist, where the pass would otherwise rewrite the
/// loads it still recognises and leave the rest slow.
#[test]
fn declines_when_the_load_base_is_hoisted_out_of_the_loop() {
    let seed = "        uint lsmb_0 = ";
    let hoist = format!(
        "        threadgroup half* base_a_0 = &((*(((&kernelContext_0)->sa_0)))[0]);\n{seed}"
    );
    let hoisted = GENERATED.replacen(seed, &hoist, 1);
    assert_ne!(hoisted, GENERATED, "fixture should contain the index seed");

    let all = hoisted.replace(
        "&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0",
        "base_a_0 + (lsma_0",
    );
    let one = hoisted.replacen(
        "&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0 + 64U)",
        "base_a_0 + (lsma_0 + 64U)",
        1,
    );
    for (label, odd) in [("every load", all), ("one load", one)] {
        assert_ne!(odd, hoisted, "fixture should contain the loads");
        let Err(err) = postpass_gemm_msl(&odd) else {
            panic!("should decline when {label} reads a hoisted base");
        };
        assert!(
            err.contains("not the pointer walk"),
            "unexpected error for {label}: {err}"
        );
    }
}

/// The address is read from the start of the call's argument list to the first
/// `,` or `)` that belongs to the call itself, so the error quotes the argument
/// and nothing else. Both bounds matter and neither shows up as a decline that
/// should have been an accept, so they are pinned on the quoted text:
///
/// - stopping only at `,` runs past a single-argument call and reads the address
///   as a blob spanning later statements;
/// - testing the terminator after updating the depth instead of before makes the
///   `)` closing a parenthesized argument end it, truncating the address.
#[test]
fn reads_the_load_address_up_to_the_call_boundary() {
    let load = "&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0), (ulong)(8U)";
    for (spelling, expected) in [
        // Single argument: only the call's own paren ends it.
        ("zz_0", "\"zz_0\""),
        // Parenthesized argument: its paren must not end it.
        ("(zz_0), (ulong)(8U)", "\"(zz_0)\""),
    ] {
        let odd = GENERATED.replacen(load, spelling, 1);
        assert_ne!(odd, GENERATED, "fixture should contain the load");
        let Err(err) = postpass_gemm_msl(&odd) else {
            panic!("should decline on a load reading {spelling}");
        };
        assert!(
            err.contains(&format!("a matrix load reads {expected},")),
            "address read for {spelling} is not bounded by the call: {err}"
        );
    }
}

/// Recognising no load at all must decline rather than pass. A check that walks
/// the loads proves nothing if the walk finds none, and slangc has more than one
/// spelling to reach for here: the generated file already declares a transposing
/// wrapper alongside the plain one, and the `.slang` is a single
/// `CoopMatMatrixLayout` token away from asking for it. Combined with the hoist
/// above, that would otherwise be an accepted half-patch.
#[test]
fn declines_when_no_matrix_load_is_recognized() {
    let renamed = GENERATED.replace("_slang_simdgroup_load<", "_slang_coopmat_read<");
    assert_ne!(
        renamed, GENERATED,
        "fixture should contain the load wrapper"
    );
    let err = postpass_gemm_msl(&renamed).expect_err("should decline when no load is recognized");
    assert!(
        err.contains("nothing proves the loads were repointed"),
        "unexpected error: {err}"
    );
}

/// The offset-form load rewrite is scoped to the k-loop, because `lsma_0` is
/// live again in the ragged epilogue where `pa_0` has already walked past the
/// tile. A load left in index form out there must decline, not get repointed at
/// the stale pointer: that version compiles and reads the wrong address.
#[test]
fn declines_on_an_index_form_load_outside_the_k_loop() {
    let epilogue = "            lsma_0 = sv_groupindex_0;";
    let injected = format!(
        "{epilogue}\n            (void)(&((*(((&kernelContext_0)->sa_0)))[0]) + (lsma_0 + 32U));"
    );
    let odd = GENERATED.replacen(epilogue, &injected, 1);
    assert_ne!(odd, GENERATED, "fixture should contain the epilogue anchor");
    let err = postpass_gemm_msl(&odd).expect_err("should decline on an epilogue index load");
    assert!(
        err.contains("still uses the index form"),
        "unexpected error: {err}"
    );
}

/// MSL rejects two unroll directives on one loop, so a shader that already
/// carries one has to decline. Appending a second turns a slowdown into a
/// shader-compile failure at pipeline creation. The check is whole-file rather
/// than "is it adjacent to the k-loop", because deciding adjacency would mean
/// tracking comments and `#line` runs backwards from the header; every spelling
/// and position below has to decline.
#[test]
fn declines_when_the_shader_already_has_a_directive() {
    let header = "        for(;;)\n";
    for injected in [
        format!("        #pragma unroll(2)\n{header}"),
        format!("        #pragma unroll(2)\n#line 278\n{header}"),
        format!("        #pragma unroll(2)\n        // slang: k loop\n{header}"),
        format!("        #pragma unroll(2)\n        /* k */\n{header}"),
        format!("        _Pragma(\"unroll(2)\")\n{header}"),
        // Whitespace between `#` and the directive is legal C preprocessing.
        format!("        #  pragma unroll(2)\n{header}"),
    ] {
        let odd = GENERATED.replacen(header, &injected, 1);
        assert_ne!(odd, GENERATED, "fixture should contain the loop header");
        let err = postpass_gemm_msl(&odd).expect_err("should decline on an existing directive");
        assert!(
            err.contains("which this pass would collide with"),
            "unexpected error for {injected:?}: {err}"
        );
    }

    // The check is whole-file, so a directive nowhere near the k-loop declines
    // too. This is what it buys over the adjacency scan it replaced.
    let far = GENERATED.replacen("[[kernel]]", "#pragma clang diagnostic push\n[[kernel]]", 1);
    assert_ne!(far, GENERATED, "fixture should contain the entry point");
    postpass_gemm_msl(&far).expect_err("should decline on a directive far from the k-loop");
}

/// The k-loop is found by walking back from the pointer step to the nearest
/// `for(;;)`. If slangc ever spells that header differently, the walk runs on to
/// the enclosing k-tile loop, which would attach the pragma to the wrong loop
/// and widen the load rewrite across the epilogue. It must decline instead.
#[test]
fn declines_when_the_k_loop_header_is_spelled_differently() {
    let header = "        for(;;)\n";
    let odd = GENERATED.replacen(header, "        while(true)\n", 1);
    assert_ne!(odd, GENERATED, "fixture should contain the loop header");
    let err = postpass_gemm_msl(&odd).expect_err("should decline on an unrecognized header");
    assert!(
        err.contains("not spelled for(;;)"),
        "unexpected error: {err}"
    );
}

/// The pass splices in its own identifiers. If slangc ever emits one of them,
/// the result is a redefinition that fails the shader compile at pipeline
/// creation, which is neither a decline nor a build-time error.
#[test]
fn declines_when_an_introduced_name_is_already_taken() {
    for taken in ["pa_0", "_sa", "shmem_p_0"] {
        let odd = GENERATED.replacen(
            "    uint lsma_0;\n",
            &format!("    uint {taken};\n    uint lsma_0;\n"),
            1,
        );
        assert_ne!(odd, GENERATED, "fixture should contain the anchor");
        let err = postpass_gemm_msl(&odd).expect_err(&format!("should decline for {taken}"));
        assert!(
            err.contains(&format!("already defines {taken}")),
            "unexpected error for {taken}: {err}"
        );
    }
}

/// The step temporaries feed the pointer step at a single initializer, so a
/// later write to one advances the index without advancing the pointer. That is
/// the same divergence as writing the index itself, one variable over.
#[test]
fn declines_when_a_step_temporary_is_written_again() {
    let step = "            uint lsmb_1 = lsmb_0 + 256U;";
    let odd = GENERATED.replacen(
        step,
        &format!("{step}\n            lsma_1 = lsma_1 + 8U;"),
        1,
    );
    assert_ne!(odd, GENERATED, "fixture should contain the step");
    let err = postpass_gemm_msl(&odd).expect_err("should decline on a second step write");
    assert!(
        err.contains("lsma_1 to be written 1 time(s)"),
        "unexpected error: {err}"
    );
}

/// The static scratch arrays are deleted outright, so any other reference to
/// them dangles. That fails the shader compile at pipeline creation rather than
/// declining, so the pass has to catch it.
#[test]
fn declines_when_a_deleted_scratch_array_is_still_referenced() {
    let bind = "    (&kernelContext_0)->sa_0 = &sa_1;";
    let odd = GENERATED.replacen(bind, &format!("{bind}\n    sa_1[0] = (half)(0.0);"), 1);
    assert_ne!(odd, GENERATED, "fixture should contain the binding");
    let err = postpass_gemm_msl(&odd).expect_err("should decline on a dangling array reference");
    assert!(
        err.contains("sa_1 is still referenced"),
        "unexpected error: {err}"
    );
}

/// The scratch parameter goes in at threadgroup index 0, so a shader that
/// already binds one must decline. This is the shape slangc would emit if
/// shader-slang/slang#8173 lands, which is exactly when this pass should step
/// aside rather than produce two parameters at the same index.
#[test]
fn declines_when_a_threadgroup_slot_is_already_bound() {
    let sig = "float device* dst_1 [[buffer(2)]])";
    let odd = GENERATED.replacen(
        sig,
        "float device* dst_1 [[buffer(2)]], threadgroup half* scratch_0 [[threadgroup(0)]])",
        1,
    );
    assert_ne!(odd, GENERATED, "fixture should contain the signature");
    let err = postpass_gemm_msl(&odd).expect_err("should decline on an occupied threadgroup slot");
    assert!(
        err.contains("which this pass would collide with"),
        "unexpected error: {err}"
    );

    // The attribute is whitespace-tolerant, so the check must not be anchored on
    // the `[[` spelling.
    let spaced = GENERATED.replacen(
        sig,
        "float device* dst_1 [[buffer(2)]], threadgroup half* scratch_0 [[ threadgroup(0) ]])",
        1,
    );
    assert_ne!(spaced, GENERATED, "fixture should contain the signature");
    postpass_gemm_msl(&spaced).expect_err("should decline on the spaced attribute spelling");

    let inner = GENERATED.replacen(
        sig,
        "float device* dst_1 [[buffer(2)]], threadgroup half* s_0 [[ threadgroup (0) ]])",
        1,
    );
    assert_ne!(inner, GENERATED, "fixture should contain the signature");
    postpass_gemm_msl(&inner).expect_err("should decline on space before the attribute paren");
}

/// The threadgroup precondition must not fire on unrelated attributes.
/// `[[max_total_threads_per_threadgroup(N)]]` is the natural MSL rendering of
/// the `[numthreads]` already on this entry point, so matching a bare
/// `threadgroup(` would cost the whole speedup the day slangc emits it.
#[test]
fn tolerates_an_unrelated_threadgroup_attribute() {
    // Placed before `[[kernel]]` so the signature anchor is untouched and the
    // precondition is the only thing under test.
    let odd = GENERATED.replacen(
        "[[kernel]] void gemm_q8_0(",
        "[[max_total_threads_per_threadgroup(128)]] [[kernel]] void gemm_q8_0(",
        1,
    );
    assert_ne!(odd, GENERATED, "fixture should contain the entry point");
    let out =
        postpass_gemm_msl(&odd).expect("an unrelated threadgroup attribute should not decline");
    assert!(
        out.contains("threadgroup char* shmem_p_0 [[threadgroup(0)]]"),
        "the scratch parameter should still have been spliced in"
    );
}

/// The post-condition that guards against a slangc version introducing a *new*
/// way to spell the scratch access. That failure mode compiles and computes the
/// right answer while silently keeping the slow addressing, so it has to be
/// caught here rather than by a parity test.
#[test]
fn declines_on_an_unrecognized_access_spelling() {
    let extra = GENERATED.replace(
        "    (&kernelContext_0)->sa_0 = &sa_1;\n",
        "    (&kernelContext_0)->sa_0 = &sa_1;\n    (void)((&kernelContext_0)->sa_0);\n",
    );
    assert_ne!(extra, GENERATED, "fixture should contain the binding");
    let err = postpass_gemm_msl(&extra).expect_err("should decline on an unknown access form");
    assert!(err.contains("surviving sa_0"), "unexpected error: {err}");
}

/// `build.rs` may feed the pass its own output: when slangc is absent it copies
/// the committed `.metal`, and a patched file committed by mistake would come
/// straight back through. Re-running must be an exact no-op, not a second set
/// of edits applied to already-edited text.
#[test]
fn is_idempotent() {
    let once = patched();
    let twice = postpass_gemm_msl(&once).expect("re-running should succeed");
    assert_eq!(once, twice, "post-pass should be idempotent");

    // Specifically, nothing should have been inserted a second time.
    assert_eq!(once.matches("#pragma unroll(4)").count(), 1);
    assert_eq!(once.matches("pa_0 = pa_1;").count(), 1);
    assert_eq!(once.matches(POSTPASS_MARKER).count(), 1);
}

/// Declining must be total. A slangc upgrade that renames one temporary has to
/// yield unpatched-but-correct MSL, never a shader with two of three conditions.
#[test]
fn declines_rather_than_half_applying() {
    // Rename the induction variable the way a slangc version bump might.
    let moved = GENERATED.replace("lsma_1", "lsmaPrime_1");
    assert_ne!(moved, GENERATED, "fixture should contain the anchor");
    let err = postpass_gemm_msl(&moved).expect_err("should decline on a moved anchor");
    assert!(
        err.contains("lsma_1"),
        "error should name the anchor: {err}"
    );
}

/// An anchor that appears an unexpected number of times is also a decline: the
/// pass must not guess which occurrence was meant.
#[test]
fn declines_on_a_duplicated_anchor() {
    let dup = GENERATED.replace("    uint lsma_0;\n", "    uint lsma_0;\n    uint lsma_0;\n");
    assert_ne!(dup, GENERATED, "fixture should contain the anchor");
    let err = postpass_gemm_msl(&dup).expect_err("should decline when an anchor is ambiguous");
    assert!(
        err.contains("uint lsma_0;") && err.contains("found 2"),
        "should decline on the duplicated anchor itself: {err}"
    );
}

/// Unrolling is bounded. A loop shape that no longer matches the tile is a
/// reason to decline, not to emit an enormous unrolled body.
#[test]
fn declines_on_an_implausible_trip_count() {
    let wide = GENERATED.replace("if(ik_0 < 4U)", "if(ik_0 < 4096U)");
    assert_ne!(wide, GENERATED, "fixture should contain the trip test");
    let err = postpass_gemm_msl(&wide).expect_err("should decline on a huge trip count");
    assert!(
        err.contains("refusing to unroll"),
        "unexpected error: {err}"
    );
}
