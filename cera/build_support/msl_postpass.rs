// Rewrites the Metal shader Slang emits for `gemm_q8_0` into a form the native
// AGX compiler can generate folded addressing for.
//
// # Why
//
// Slang's MSL for this kernel runs at ~0.93x the handwritten `gemm_q8_0.metal`
// on an M1 Max. Disassembling both to native AGX shows why: the handwritten
// kernel issues its 26 `threadgroup_load`s from 4 base registers with the
// displacement folded into the instruction's immediate field (21 distinct
// immediates), while Slang's issues its 13 from 11 bases with 2 immediates and
// computes each address instead.
//
// Read those two rows carefully: they are not like for like. Slang's loop sits
// at 16 MMAs against the handwritten kernel's 32, so it has half the loads to
// place, and a base-register count only means anything at a matched unroll
// factor. Comparing them as if they were comparable is what delayed this
// diagnosis by a session: the Slang variants sat at 10 to 11 bases against 4,
// which reads as a 2.5x gap when it is really 5 address computations per
// iteration against 1. The matched pair is this: rebuilt, the generated kernel
// lands on 4 bases and 21 immediates at 32 MMAs, the same signature as the
// handwritten kernel, in 391 instructions against its 417. Unrolled to match
// but not rebuilt, it needs 82 more `iadd` and 18 more `or`.
//
// Three properties of the emitted MSL are each individually necessary for that
// folding, and only their conjunction is sufficient:
//
// 1. scratch arrives as a `[[threadgroup(0)]]` parameter, not static groupshared
// 2. the k-loop walks pointer-typed induction variables, not `uint` indices
// 3. the k-loop carries `#pragma unroll(N)`
//
// Each was confirmed by a single-variable ablation on the handwritten kernel:
// breaking 1 collapses it to 25 base registers, 2 to 24, and 3 to 13. Their
// conjunction was confirmed by rebuilding all three onto the generated MSL,
// which reproduces the handwritten kernel's exact 4-base/21-immediate signature.
// Applying only some is not a partial win: condition 3 alone measures *worse*
// than doing nothing (0.908x vs 0.927x), which is why this pass is all-or-nothing.
//
// Slang cannot express any of the three today, and fixing that upstream takes
// three changes, not one. Condition 3 is the most tractable: `#pragma unroll`
// does not unroll in AIR, it attaches `llvm.loop.unroll.count` metadata that the
// native translator honors, so Slang's `[ForceUnroll]` (which unrolls in Slang
// IR and emits straight-line MSL carrying no metadata) is not a substitute.
// Condition 1 needs a threadgroup pointer type, tracked upstream as
// shader-slang/slang#8173. Condition 2 needs a `CoopMat.Load` overload taking a
// pointer rather than an array and an index, which is a separate ask and has no
// issue filed yet.
//
// # Contract
//
// Pure text in, pure text out, so it is testable without a build. It runs
// *after* the compile-or-fallback step in `build.rs` and writes only to
// `OUT_DIR`, which means it covers both the slangc and the committed-fallback
// paths while leaving the committed `.metal` byte-identical to slangc output.
// CI byte-compares that file, so patching it in place would break the drift
// check.
//
// Two safety properties matter more than the speedup:
//
// - **All or nothing.** Every rewrite is proven to have covered its targets,
//   either by an exact occurrence count or by a post-condition whose spelling
//   is independent of the pattern it guards, and returns `Err` otherwise. The
//   independence matters: a post-condition anchored on the same text as its
//   rewrite is defeated by whatever respelling defeated the rewrite, so it
//   would pass vacuously. A slangc upgrade that renames temporaries makes the
//   pass decline, and the caller ships unpatched MSL: correct, just slower.
//   Half-patched output is never produced.
// - **Idempotent.** `postpass_gemm_msl(postpass_gemm_msl(x)) == postpass_gemm_msl(x)`,
//   enforced by a version marker checked on entry and written on exit. This is
//   not hypothetical: if a patched `.metal` were ever committed as the
//   fallback, the build would feed it straight back through this function.
//
// Note that condition 1 changes the kernel's ABI, so callers must supply
// threadgroup memory at index 0. Setting it for a kernel that does not declare
// the parameter is harmless, so call sites can do so unconditionally and stay
// correct whether or not this pass applied.

/// Marks MSL this pass has already rewritten.
///
/// Bumping the version makes an artifact patched by an older version decline
/// rather than be silently accepted as current: the freshness check below sees
/// the names that version already introduced. Declining is the safe outcome, but
/// note it leaves the old patched text in place, so a bump wants the committed
/// fallback regenerated too.
const POSTPASS_MARKER: &str = "// cera:msl-postpass=v1";

/// Largest k-loop trip count worth fully unrolling. The measured-good value is
/// 4; a much larger one would mean the loop shape changed under us, and
/// unrolling it would cost instruction cache rather than save address math.
const POSTPASS_MAX_UNROLL: u32 = 8;

/// Threadgroup scratch the rewritten kernel requires its caller to bind, in
/// bytes: 4 KB of half weights plus 4 KB of float input.
///
/// Callers pass this size literally (`examples/slang_gemm_bench.rs`,
/// `tests/slang_multitarget_parity.rs`), and nothing propagates a new value to
/// them. Retiling the `.slang` past this budget must therefore decline here:
/// moving the arrays into an allocation smaller than they need would read and
/// write past it, which is a wrong-answer bug rather than a slow one.
const POSTPASS_SCRATCH_BYTES: usize = 8192;

/// Alignment `_sb` must satisfy. The staging store through it is a
/// `threadgroup float2x4*`, so the float slice has to start on a 16-byte
/// boundary; an odd half count would otherwise produce a misaligned vector
/// access, which is undefined behaviour rather than a decline.
const POSTPASS_SCRATCH_ALIGN: usize = 16;

/// Bytes in an MSL `half`. Named rather than `size_of::<u16>()` so it is clear
/// this sizes the shader's type, not a Rust array.
const POSTPASS_HALF_BYTES: usize = 2;

/// Bytes in an MSL `float`.
const POSTPASS_FLOAT_BYTES: usize = 4;

/// Writes each step temporary must receive: just its initializer.
const POSTPASS_STEP_WRITES: usize = 1;

/// Identifiers this pass splices into the shader. None may already appear in
/// slangc's output; see the freshness check at the top of the pass.
const POSTPASS_INTRODUCED: &[&str] = &["shmem_p_0", "_sa", "_sb", "pa_0", "pa_1", "pb_0", "pb_1"];

/// The pointer walk itself: condition 2 holds when, and only when, every matrix
/// load in the k-loop addresses through one of these. A subset of
/// `POSTPASS_INTRODUCED`.
const POSTPASS_INDUCTION: &[&str] = &["pa_0", "pa_1", "pb_0", "pb_1"];

/// Substring of the slangc-emitted wrapper every simdgroup matrix load goes
/// through. Deliberately not the whole identifier: the generated file spells the
/// wrapper `_slang_simdgroup_load` today, but also declares a transposing
/// variant whose name merely adds a suffix. Matching on the substring keeps a
/// decorated name counted as a load rather than silently skipped.
const POSTPASS_LOAD_CALL: &str = "simdgroup_load";

/// Writes each k-loop index must receive across the seed-plus-loop span: one to
/// seed it before the loop, one to advance it inside. See the post-condition
/// that uses it.
const POSTPASS_INDEX_WRITES: usize = 2;

/// Rewrite Slang's `gemm_q8_0` MSL for folded addressing, or explain why not.
///
/// Returns the input unchanged when it has already been rewritten.
fn postpass_gemm_msl(src: &str) -> Result<String, String> {
    // First-line equality rather than `starts_with`, so a later `v10` marker is
    // not read as an already-current `v1`.
    if src.lines().next() == Some(POSTPASS_MARKER) {
        return Ok(src.to_string());
    }

    // The rewrite introduces these names. If slangc ever emits one of its own,
    // splicing ours in produces a redefinition that fails the shader compile at
    // pipeline creation, which is neither a decline nor a build-time error.
    for name in POSTPASS_INTRODUCED {
        if pp_contains_word(src, name) {
            return Err(format!(
                "generated MSL already defines {name}, which this pass introduces"
            ));
        }
    }

    // Two whole-file preconditions, matched against a whitespace-stripped copy so
    // that `[[ threadgroup(0) ]]` and `#  pragma` cannot slip past on spelling.
    //
    // `[[threadgroup(`: the scratch parameter is spliced in at threadgroup index
    // 0, so the slot has to be free. This is the shape slangc would emit if
    // shader-slang/slang#8173 lands and it gains threadgroup pointers, which is
    // precisely the future where this pass should step aside. The `[[` is part of
    // the pattern on purpose: `[[max_total_threads_per_threadgroup(N)]]` is an
    // unrelated entry-point attribute, and it is the natural MSL rendering of the
    // `[numthreads]` already on this kernel, so matching a bare `threadgroup(`
    // would decline for nothing the day slangc starts emitting it.
    //
    // `#pragma` / `_Pragma`: MSL rejects two unroll directives on one loop, and
    // deciding whether an existing directive applies to the k-loop would mean
    // tracking comments and `#line` runs backwards from the header. Declining on
    // any directive anywhere is both simpler and stricter.
    //
    // slangc emits none of these in any shader in this tree today, so the checks
    // cost nothing now; they exist so a future slangc declines rather than
    // producing MSL that fails to compile at pipeline creation.
    let squeezed: String = src.chars().filter(|c| !c.is_whitespace()).collect();
    for token in ["[[threadgroup(", "#pragma", "_Pragma"] {
        if squeezed.contains(token) {
            return Err(format!(
                "generated MSL already contains {token:?}, which this pass would collide with"
            ));
        }
    }

    // --- condition 1: scratch becomes a [[threadgroup(0)]] parameter --------
    //
    // Slang declares two `threadgroup array<T, N>` locals and points context
    // fields at them. Both become slices of one caller-supplied allocation, so
    // the sb offset is derived from sa's element count rather than hardcoded:
    // editing the tile sizes in the .slang must not silently mis-offset sb.
    let sa_field = pp_line_containing(src, "threadgroup* sa_0;")?;
    let sb_field = pp_line_containing(src, "threadgroup* sb_0;")?;
    let sa_elems: usize = pp_between(&sa_field, "array<half, int(", ")>")?
        .parse()
        .map_err(|_| format!("unparsable sa extent in {sa_field:?}"))?;
    let sb_elems: usize = pp_between(&sb_field, "array<float, int(", ")>")?
        .parse()
        .map_err(|_| format!("unparsable sb extent in {sb_field:?}"))?;

    // Both extents are checked against what callers actually bind, because the
    // rewrite is what makes the sizes a shared contract rather than a private
    // detail of the shader. The arithmetic is checked so an absurd extent
    // declines like everything else: unchecked, a debug build script would panic
    // on overflow (a hard build failure, not the promised fallback) and a
    // release one would wrap into a plausible size.
    let overflow = || "scratch extents overflow a usize".to_string();
    let sa_bytes = sa_elems
        .checked_mul(POSTPASS_HALF_BYTES)
        .ok_or_else(overflow)?;
    let total = sb_elems
        .checked_mul(POSTPASS_FLOAT_BYTES)
        .and_then(|sb| sa_bytes.checked_add(sb))
        .ok_or_else(overflow)?;
    if total != POSTPASS_SCRATCH_BYTES {
        return Err(format!(
            "scratch is {total} bytes but callers bind {POSTPASS_SCRATCH_BYTES}; \
             update the call sites before retiling"
        ));
    }
    if !sa_bytes.is_multiple_of(POSTPASS_SCRATCH_ALIGN) {
        return Err(format!(
            "sb would start at byte {sa_bytes}, not a multiple of {POSTPASS_SCRATCH_ALIGN}"
        ));
    }

    let t = pp_swap(src, &sa_field, "    threadgroup half* sa_0;")?;
    let t = pp_swap(&t, &sb_field, "    threadgroup float* sb_0;")?;

    let sig = pp_line_containing(&t, "[[kernel]] void gemm_q8_0(")?;
    let head = sig
        .strip_suffix(')')
        .ok_or_else(|| format!("kernel signature does not end in ')': {sig:?}"))?;
    let t = pp_swap(
        &t,
        &sig,
        &format!("{head}, threadgroup char* shmem_p_0 [[threadgroup(0)]])"),
    )?;

    let sa_decl = pp_line_containing(&t, "threadgroup array<half, int(")?;
    let sb_decl = pp_line_containing(&t, "threadgroup array<float, int(")?;
    let t = pp_swap(&t, &format!("{sa_decl}\n"), "")?;
    let t = pp_swap(&t, &format!("{sb_decl}\n"), "")?;

    // The `_sa` / `_sb` locals also serve condition 2: every access goes through
    // a plain pointer instead of a load from the context struct.
    let sa_bind = pp_line_containing(&t, "->sa_0 = &sa_1;")?;
    let sb_bind = pp_line_containing(&t, "->sb_0 = &sb_1;")?;
    let t = pp_swap(
        &t,
        &sa_bind,
        "    threadgroup half* _sa = (threadgroup half*)(shmem_p_0);\n    (&kernelContext_0)->sa_0 = _sa;",
    )?;
    let t = pp_swap(
        &t,
        &sb_bind,
        &format!(
            "    threadgroup float* _sb = (threadgroup float*)(shmem_p_0 + {});\n    (&kernelContext_0)->sb_0 = _sb;",
            sa_bytes
        ),
    )?;

    // Slang spells the same access two ways depending on whether it is indexing
    // or taking an address, so both forms are rewritten. Neither is required to
    // occur: a slangc that normalized to one spelling would still be fully
    // rewritten, and the survivor count below is what actually proves coverage,
    // so demanding both here would decline for nothing.
    let t = t
        .replace("(*(((&kernelContext_0)->sa_0)))", "(_sa)")
        .replace("(*(&kernelContext_0)->sa_0)", "(_sa)")
        .replace("(*(((&kernelContext_0)->sb_0)))", "(_sb)")
        .replace("(*(&kernelContext_0)->sb_0)", "(_sb)");

    // Post-condition: the only surviving mentions of each field are its
    // declaration in the context struct and the binding above. Anything else is
    // an access form this pass does not know about, which would silently keep
    // the slow addressing for those loads.
    //
    // Counted as a word rather than as `->sa_0`, so that a spelling reaching the
    // field some other way (`kernelContext_0.sa_0`, say) still trips it. Both
    // fields were retyped above, from `array<T, N> threadgroup*` to
    // `threadgroup T*`, so a surviving `(*(...sa_0))[i]` no longer subscripts
    // a pointer-to-array: it would fail the shader compile at pipeline creation
    // instead of declining here, which is the outcome this check exists to
    // prevent.
    for field in ["sa_0", "sb_0"] {
        let n = pp_count_words(&t, field);
        if n != 2 {
            return Err(format!(
                "expected exactly 2 surviving {field} (its declaration and its binding), found {n}"
            ));
        }
    }

    // The static arrays were deleted, so any other reference to them is now
    // dangling. That fails the shader compile at pipeline creation rather than
    // declining here, so it has to be caught now.
    for arr in ["sa_1", "sb_1"] {
        if pp_contains_word(&t, arr) {
            return Err(format!(
                "{arr} is still referenced after its declaration was removed"
            ));
        }
    }

    // --- condition 2: pointer-typed induction variables ---------------------
    //
    // The `uint` indices are left alone rather than retyped: Slang reuses
    // `lsma_0` for the epilogue's ragged copy loop, where it really is an index.
    let t = pp_swap(
        &t,
        "    uint lsma_0;\n",
        "    uint lsma_0;\n    threadgroup const half* pa_0;\n    threadgroup const float* pb_0;\n",
    )?;

    // Seed and step are captured from the uint arithmetic rather than hardcoded,
    // so retiling the .slang cannot leave the two walks disagreeing.
    //
    // The two captures need different treatment.
    //
    // A seed is read back off the variable rather than re-spliced: `pa_0` is
    // seeded from `lsma_0` immediately after `lsma_0` is assigned, so the seed
    // expression is evaluated exactly once no matter what it contains. Splicing
    // the text a second time would double any side effect in it: `lsma_0 = _S26++`
    // would increment twice and leave the pointer one element off the index,
    // silently, since the loads read the pointer.
    //
    // A step has no variable to read back, so it is spliced, and parentheses are
    // not enough there: `lsma_0 + 256U << 1U` steps the index by
    // `(lsma_0 + 256) << 1` while `pa_0 + (256U << 1U)` steps the pointer by 512.
    // So a step must be a bare integer literal, which is what slangc emits and
    // which has neither precedence nor side-effect hazards; anything else
    // declines.
    let a_seed = pp_after(&t, "\n        lsma_0 = ", ';')?;
    let t = pp_swap(
        &t,
        &format!("\n        lsma_0 = {a_seed};"),
        &format!("\n        lsma_0 = {a_seed};\n        pa_0 = _sa + lsma_0;"),
    )?;
    let b_seed = pp_after(&t, "\n        uint lsmb_0 = ", ';')?;
    let t = pp_swap(
        &t,
        &format!("\n        uint lsmb_0 = {b_seed};"),
        &format!("\n        uint lsmb_0 = {b_seed};\n        pb_0 = _sb + lsmb_0;"),
    )?;

    let a_step = pp_after(&t, "\n            uint lsma_1 = lsma_0 + ", ';')?;
    pp_check_step(&a_step)?;
    let t = pp_swap(
        &t,
        &format!("\n            uint lsma_1 = lsma_0 + {a_step};"),
        &format!(
            "\n            uint lsma_1 = lsma_0 + {a_step};\n            threadgroup const half* pa_1 = pa_0 + ({a_step});"
        ),
    )?;
    let b_step = pp_after(&t, "\n            uint lsmb_1 = lsmb_0 + ", ';')?;
    pp_check_step(&b_step)?;
    let t = pp_swap(
        &t,
        &format!("\n            uint lsmb_1 = lsmb_0 + {b_step};"),
        &format!(
            "\n            uint lsmb_1 = lsmb_0 + {b_step};\n            threadgroup const float* pb_1 = pb_0 + ({b_step});"
        ),
    )?;
    let t = pp_swap(
        &t,
        "\n            lsma_0 = lsma_1;",
        "\n            lsma_0 = lsma_1;\n            pa_0 = pa_1;",
    )?;
    let t = pp_swap(
        &t,
        "\n            lsmb_0 = lsmb_1;",
        "\n            lsmb_0 = lsmb_1;\n            pb_0 = pb_1;",
    )?;

    // Locate the k-loop before touching the loads. It is anchored on the pointer
    // step this pass just introduced, so it cannot resolve to the epilogue loop,
    // which also tests `ik_0`.
    let (_, line_at, for_at, loop_end) = pp_locate_k_loop(&t)?;
    let indent = t[line_at..for_at].to_string();

    // Point the matrix loads at the pointer walk. The `+ (` in the offset form
    // reuses the closing paren already in the text.
    //
    // Scoped to the loop body: `lsma_0` is live again in the ragged epilogue,
    // where it is a genuine index and `pa_0` has already walked past the tile.
    // Rewriting there would repoint a load at a stale pointer, which compiles
    // and reads the wrong address. Anything left in index form outside the loop
    // trips the post-condition below and declines instead.
    //
    // Note the final iteration advances `pa_0` / `pb_0` past the end of the
    // scratch allocation, because the step happens before the loop test. The
    // uint indices this mirrors could overshoot harmlessly; a pointer doing so
    // is only safe because it is never dereferenced afterwards.
    let body = t
        .get(for_at..loop_end)
        .ok_or("k-loop body is not a byte range of the shader")?;
    let body = body
        .replace("&((_sa)[0]) + (lsma_0)", "pa_0")
        .replace("&((_sa)[0]) + (lsma_0 + ", "pa_0 + (")
        .replace("&((_sb)[0]) + (lsmb_0)", "pb_0")
        .replace("&((_sb)[0]) + (lsmb_0 + ", "pb_0 + (");

    // Post-condition: every matrix load in the body now reads the pointer walk.
    // This is what proves the rewrite covered them, and it holds whatever the
    // address expression looks like, which a count on the patterns above would
    // not: a slangc that renders an address differently (`[int(0)]` for `[0]`,
    // different spacing around the `+`, or a base hoisted into a `_Sxx`
    // temporary by the CSE it already applies elsewhere in this function)
    // defeats the `.replace()` calls, and any guard sharing their spelling is
    // defeated by the same respelling and passes vacuously.
    //
    // Accepting one would emit MSL with conditions 1 and 3 but not 2, which the
    // ablations measured at 21 base registers: unfolded, the exact state this
    // pass exists to get out of. That combination was never separately timed,
    // but condition 3 on its own measured slower than doing nothing (0.908x
    // against 0.927x), so declining is the better of the two.
    //
    // Checked per load rather than as a total, so a partial hoist (some loads
    // rewritten, others not) declines too.
    pp_check_loads_use_pointers(&body)?;

    // The scratch arrays are addressed only by those loads, so a fully rewritten
    // body no longer names them. A surviving mention is some other access shape,
    // e.g. a future slangc software-pipelining the tile copy into the loop. That
    // is a shape this pass has not been validated against, so unpatched MSL is
    // the right answer until someone extends it.
    for name in ["_sa", "_sb"] {
        if pp_contains_word(&body, name) {
            return Err(format!(
                "the k-loop body still addresses {name} directly, which this pass does not rewrite"
            ));
        }
    }

    let t = format!("{}{body}{}", &t[..for_at], &t[loop_end..]);

    // Post-condition: no matrix load anywhere still addresses through the uint
    // index. Scoping the rewrite above means an epilogue load in that form
    // survives to here, and declining is the right answer for it. This one is
    // spelling-bound, so it is a supplement to the body check above rather than
    // the proof of coverage.
    for stale in ["(_sa)[0]) + (lsma", "(_sb)[0]) + (lsmb"] {
        if t.contains(stale) {
            return Err(format!("a matrix load still uses the index form: {stale}"));
        }
    }

    // --- condition 3: an explicit unroll count on the k-loop ----------------
    //
    // The rewrite above changed the body's length, so re-derive the span.
    let (seed_at, line_at, for_at, loop_end) = pp_locate_k_loop(&t)?;

    // Post-condition: the pointer walk mirrors the uint indices, so it is only
    // correct while each index is written exactly twice, once to seed it and
    // once to advance it. Any other write would move the index without moving
    // the pointer, and the loads (which now read the pointer) would silently
    // address the wrong tile. Every other rewrite is guarded by an occurrence
    // count; this invariant is about the loop's shape, so it is checked against
    // the text spanning the seeds and the whole loop body.
    //
    // The span runs from the first seed to the loop's matching brace rather than
    // from the loop header, so a write sitting between the seed and the header
    // cannot slip through on a different indentation.
    let span = t
        .get(seed_at..loop_end)
        .ok_or("the k-loop index is seeded after the loop that steps it")?;
    //
    // The step temporaries are checked too: `pa_1`/`pb_1` mirror `lsma_1`/
    // `lsmb_1` at their single initializer, so a later write to one of those
    // advances the index without advancing the pointer, exactly like a write to
    // the index itself.
    for (idx, writes) in [
        ("lsma_0", POSTPASS_INDEX_WRITES),
        ("lsmb_0", POSTPASS_INDEX_WRITES),
        ("lsma_1", POSTPASS_STEP_WRITES),
        ("lsmb_1", POSTPASS_STEP_WRITES),
    ] {
        let n = pp_write_count(span, idx);
        if n != writes {
            return Err(format!(
                "expected {idx} to be written {writes} time(s) in the k-loop span, \
                 found {n}; the pointer walk would no longer track the index"
            ));
        }
    }

    // Read the trip count from between this loop's header and its body, so the
    // uniqueness requirement applies to the k-loop rather than to the whole
    // shader; the epilogue tests `ik_0` too.
    let trip = pp_after(&t[for_at..loop_end], "if(ik_0 < ", ')')?;
    let n: u32 = trip
        .trim_end_matches('U')
        .parse()
        .map_err(|_| format!("unparsable k-loop trip count {trip:?}"))?;
    if n == 0 || n > POSTPASS_MAX_UNROLL {
        return Err(format!(
            "k-loop trip count {n} outside 1..={POSTPASS_MAX_UNROLL}; refusing to unroll"
        ));
    }

    let (before, after) = t.split_at(line_at);
    Ok(format!(
        "{POSTPASS_MARKER}\n{before}{indent}#pragma unroll({n})\n{after}"
    ))
}

/// Reject a step capture that is not a bare unsigned integer literal.
///
/// See the comment at the capture site: a step is spliced after an existing `+`,
/// so anything binding looser than `+` advances the pointer and the index by
/// different amounts without any compile error to reveal it.
fn pp_check_step(expr: &str) -> Result<(), String> {
    let digits = expr.strip_suffix('U').unwrap_or(expr);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "k-loop index step {expr:?} is not a plain integer literal; \
             the pointer walk could not mirror it safely"
        ));
    }
    Ok(())
}

/// Every matrix load in `body` must take its address from the pointer walk.
///
/// The address is read out of the call rather than matched against a fixed
/// spelling, so this proves coverage of the load rewrite without inheriting its
/// assumptions about how slangc renders the address. A load still reaching the
/// scratch some other way keeps the slow addressing, and applying the rest of
/// the pass without it is worse than applying none of it.
///
/// Finding no load at all is a decline, not a pass. The check would otherwise
/// prove nothing the moment slangc decorates the wrapper's name, and it has more
/// than one name to decorate: the same generated file already declares a
/// transposing variant, one `CoopMatMatrixLayout` token away in the `.slang`.
/// Matching the calls on a substring of the name rather than on the whole
/// identifier is the other half of that, so a decorated name still counts.
///
/// What neither covers is a *second* wrapper, named without `simdgroup_load`,
/// carrying only some of the loads: the ones it carries go unseen while the
/// count stays nonzero. Reaching that needs slangc both to add the wrapper and
/// to hoist those loads' base address out of the loop, since otherwise the
/// scratch name left behind in the body is caught below. It is worth knowing
/// the check stops there rather than believing it total.
fn pp_check_loads_use_pointers(body: &str) -> Result<(), String> {
    let bytes = body.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut seen = 0usize;
    for (at, _) in body.match_indices(POSTPASS_LOAD_CALL) {
        seen += 1;
        // Skip the rest of the identifier, then the template arguments before
        // the call's parentheses. Those nest (`simdgroup_matrix<half, int(8),
        // int(8)>`) and contain parens of their own, so the argument list cannot
        // be found by scanning for `(` alone.
        let after_name = (at + POSTPASS_LOAD_CALL.len()..)
            .find(|&i| !bytes.get(i).copied().is_some_and(&is_word))
            .unwrap_or(bytes.len());
        let mut i = after_name;
        let mut depth = 0usize;
        while let Some(&c) = bytes.get(i) {
            match c {
                b'<' => depth += 1,
                // Guarded, so a `>` that is not closing a template argument list
                // falls to the arm below rather than underflowing. That would
                // panic the build script, and this pass declines instead.
                b'>' if depth > 0 => {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                _ if depth == 0 => break,
                _ => {}
            }
            i += 1;
        }
        let args = body
            .get(i..)
            .and_then(|rest| rest.strip_prefix('('))
            .ok_or("a matrix load is not a call this pass can read")?;
        // The first argument ends at the top-level comma before the stride, or
        // at the call's own closing paren if it is the only argument. Testing
        // for the terminator before updating the depth is what distinguishes
        // that paren from one closing a parenthesized argument: `args` starts
        // inside the call, so a `)` seen at depth 0 can only be the call's.
        // Without it the scan runs past the end of the call to the next comma in
        // the shader, and the text it then reads as an address spans unrelated
        // statements.
        let mut depth = 0usize;
        let end = args
            .char_indices()
            .find(|&(_, c)| {
                let terminates = depth == 0 && (c == ',' || c == ')');
                match c {
                    '(' => depth += 1,
                    ')' => depth = depth.saturating_sub(1),
                    _ => {}
                }
                terminates
            })
            .map(|(i, _)| i)
            .ok_or("a matrix load's argument list is unterminated")?;
        let addr = args[..end].trim();
        if !POSTPASS_INDUCTION.iter().any(|p| pp_contains_word(addr, p)) {
            return Err(format!(
                "a matrix load reads {addr:?}, not the pointer walk, so it kept the slow addressing"
            ));
        }
    }
    if seen == 0 {
        return Err(format!(
            "the k-loop body has no {POSTPASS_LOAD_CALL} call, so nothing proves the loads were repointed"
        ));
    }
    Ok(())
}

/// Offsets at which `needle` appears in `s` as a whole identifier.
fn pp_word_matches<'a>(s: &'a str, needle: &'a str) -> impl Iterator<Item = usize> + 'a {
    let bytes = s.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    s.match_indices(needle)
        .filter(move |(i, _)| {
            let after = i + needle.len();
            (*i == 0 || !is_word(bytes[i - 1])) && (after >= bytes.len() || !is_word(bytes[after]))
        })
        .map(|(i, _)| i)
}

/// True when `needle` appears in `s` as a whole identifier.
fn pp_contains_word(s: &str, needle: &str) -> bool {
    pp_word_matches(s, needle).next().is_some()
}

/// How many times `needle` appears in `s` as a whole identifier.
fn pp_count_words(s: &str, needle: &str) -> usize {
    pp_word_matches(s, needle).count()
}

/// Locate the k-loop as `(seed, line start, `for(;;)` offset, end of block)`.
///
/// Anchored on the pointer step this pass introduces, so it cannot resolve to
/// the epilogue loop, which also tests `ik_0`. Rejecting `for_at < seed_at` is
/// what makes a miss decline rather than mis-resolve: were the k-loop header
/// spelled anything but `for(;;)`, `rfind` would walk past it to the enclosing
/// k-tile loop, and the pass would attach the pragma to the wrong loop and widen
/// the load rewrite across the epilogue.
fn pp_locate_k_loop(t: &str) -> Result<(usize, usize, usize, usize), String> {
    let seed_at = t
        .find("\n        lsma_0 = ")
        .ok_or("k-loop index seed missing")?;
    let anchor = t
        .find("\n            pa_0 = pa_1;")
        .ok_or("k-loop anchor missing")?;
    let for_at = t[..anchor]
        .rfind("for(;;)")
        .ok_or("no for(;;) precedes the k-loop step")?;
    if for_at < seed_at {
        return Err(
            "the nearest for(;;) before the k-loop step sits before the index \
                    seed, so the k-loop header is not spelled for(;;)"
                .to_string(),
        );
    }
    let line_at = t[..for_at].rfind('\n').map_or(0, |i| i + 1);
    let indent = &t[line_at..for_at];
    if !indent.chars().all(char::is_whitespace) {
        return Err(format!("for(;;) is not first on its line: {indent:?}"));
    }
    Ok((seed_at, line_at, for_at, pp_block_end(t, for_at)?))
}

/// End of the block that opens at the first `{` at or after `from`, as a byte
/// index just past its matching `}`.
fn pp_block_end(s: &str, from: usize) -> Result<usize, String> {
    let open = s[from..]
        .find('{')
        .ok_or("no block after the k-loop header")?
        + from;
    let mut depth = 0usize;
    for (i, c) in s[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(open + i + 1);
                }
            }
            _ => {}
        }
    }
    Err("unbalanced braces in the k-loop".to_string())
}

/// Count writes to `ident` in `region`: plain `=`, any compound assignment, and
/// pre- or post-increment and decrement.
///
/// Matching only `"ident = "` would miss `+=` and `++`. Slang emits neither
/// today, but this backs a guard whose entire job is to survive a future slangc,
/// so it recognises every assignment and increment *syntax* rather than the one
/// currently in use. A write through a taken address or an out-parameter is not
/// visible to it; Slang's flat, SSA-shaped output has neither.
fn pp_write_count(region: &str, ident: &str) -> usize {
    let bytes = region.as_bytes();
    let is_word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    region
        .match_indices(ident)
        .filter(|(i, _)| {
            // Whole-word only, so `lsma_0` does not match inside `lsma_01`.
            let after = i + ident.len();
            (*i == 0 || !is_word(bytes[i - 1])) && (after >= bytes.len() || !is_word(bytes[after]))
        })
        .filter(|(i, _)| {
            let before = region[..*i].trim_end();
            if before.ends_with("++") || before.ends_with("--") {
                return true;
            }
            let after = region[i + ident.len()..].trim_start();
            if after.starts_with("++") || after.starts_with("--") {
                return true;
            }
            let mut c = after.chars();
            match (c.next(), c.next()) {
                // `==` is a comparison, a lone `=` is a write.
                (Some('='), Some('=')) => false,
                (Some('='), _) => true,
                (Some(op), Some('=')) if "+-*/%&|^".contains(op) => true,
                (Some('<'), Some('<')) | (Some('>'), Some('>')) => after[2..].starts_with('='),
                _ => false,
            }
        })
        .count()
}

/// The single line containing `needle`, or an error naming how many matched.
fn pp_line_containing(s: &str, needle: &str) -> Result<String, String> {
    let hits: Vec<&str> = s.lines().filter(|l| l.contains(needle)).collect();
    match hits.as_slice() {
        [one] => Ok((*one).to_string()),
        other => Err(format!(
            "expected exactly 1 line containing {needle:?}, found {}",
            other.len()
        )),
    }
}

/// The text between the sole `open` and the following `close`.
fn pp_between(s: &str, open: &str, close: &str) -> Result<String, String> {
    let n = s.matches(open).count();
    if n != 1 {
        return Err(format!("expected 1 occurrence of {open:?}, found {n}"));
    }
    let from = s.find(open).ok_or_else(|| format!("{open:?} vanished"))? + open.len();
    let len = s[from..]
        .find(close)
        .ok_or_else(|| format!("no {close:?} after {open:?}"))?;
    Ok(s[from..from + len].to_string())
}

/// The text between the sole `open` and the following `stop` character.
fn pp_after(s: &str, open: &str, stop: char) -> Result<String, String> {
    pp_between(s, open, &stop.to_string())
}

/// Replace the sole occurrence of `from` with `to`.
///
/// Anything other than exactly one match is an error rather than a guess: these
/// anchors identify a specific site in the generated MSL, so two of them means
/// slangc changed shape and the pass can no longer tell which was meant.
fn pp_swap(s: &str, from: &str, to: &str) -> Result<String, String> {
    let got = s.matches(from).count();
    if got != 1 {
        return Err(format!("expected 1 occurrence of {from:?}, found {got}"));
    }
    Ok(s.replace(from, to))
}
