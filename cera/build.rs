use std::process::Command;

// Rewrites Slang's `gemm_q8_0` MSL into the shape the native AGX compiler folds
// addressing for. Kept in its own file, and free of I/O, so `tests/msl_postpass.rs`
// can `include!` the same source and exercise it directly.
include!("build_support/msl_postpass.rs");

// Embeds a best-effort short git SHA into the build for provenance
// (`cera::build_info()` / `cera::GIT_SHA`). Consumers such as the Pipette
// benchmark app report it alongside results, the way llama.cpp surfaces its
// build commit. Falls back to "unknown" when git is unavailable (e.g. a
// packaged source build with no repository), and can be overridden explicitly
// by setting `CERA_GIT_SHA` in the build environment (used by release CI for a
// deterministic value).
fn main() {
    // An explicit env value wins — lets release pipelines pin the sha without
    // depending on a `.git` directory being present at build time.
    let sha = std::env::var("CERA_GIT_SHA").ok().unwrap_or_else(|| {
        Command::new("git")
            .args(["rev-parse", "--short=12", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    });
    println!("cargo:rustc-env=CERA_GIT_SHA={sha}");
    println!("cargo:rerun-if-env-changed=CERA_GIT_SHA");
    // Cargo picks this up through the build script's dep-info, but the
    // dependency is declared explicitly so that editing the post-pass visibly
    // regenerates the shader rather than relying on that inference.
    println!("cargo:rerun-if-changed=build_support/msl_postpass.rs");

    // Compile the Slang SPIR-V passthrough kernels (gpu feature only). Each is
    // written to OUT_DIR and `include_spirv_raw!`d from there, so slangc is the
    // source of truth. When slangc is unavailable (e.g. CI without a Slang
    // toolchain) we fall back to the committed `.spv` next to the `.slang`, which
    // a developer with slangc regenerates via `just slang`; a cargo:warning
    // flags the fallback so drift is visible.
    if std::env::var_os("CARGO_FEATURE_GPU").is_some() {
        compile_slang_kernels();
    }

    // Multi-target Slang kernels: one `.slang` emitted to WGSL *and* MSL, so the
    // two GPU backends share a source instead of two hand-maintained twins.
    // Unlike the SPIR-V passthrough kernels above this is not Vulkan-specific,
    // so it runs for either GPU feature and emits only the targets that feature
    // needs.
    let want_wgsl = std::env::var_os("CARGO_FEATURE_GPU").is_some();
    // The Metal backend is itself cfg-gated to macOS/iOS, so gate MSL generation
    // on the target OS too. Without this, `--all-features` on a non-Apple target
    // would compile `.metal` and emit fallback `cargo:warning`s for shaders that
    // nothing on that target can build or use.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_apple = target_os == "macos" || target_os == "ios";
    let want_msl = std::env::var_os("CARGO_FEATURE_METAL").is_some() && is_apple;
    if want_wgsl || want_msl {
        compile_slang_multitarget(want_wgsl, want_msl);
    }
}

/// Slang passthrough kernels, by basename under `src/backend/shaders/spirv/`.
/// Extend as loaders are ported; each needs a `<name>.slang` and a committed
/// `<name>.spv` fallback.
const SLANG_KERNELS: &[&str] = &[
    "mul_mat_reg_tile_q4_0",
    "mul_mat_reg_tile_q8_0",
    "mul_mat_reg_tile_q4_k",
    "mul_mat_reg_tile_q5_k",
    "mul_mat_reg_tile_q6_k",
];

fn compile_slang_kernels() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR unset");
    let dir = "src/backend/shaders/spirv";
    let slangc = find_slangc();

    for name in SLANG_KERNELS {
        let src = format!("{dir}/{name}.slang");
        let committed = format!("{dir}/{name}.spv");
        let out = format!("{out_dir}/{name}.spv");
        println!("cargo:rerun-if-changed={src}");
        println!("cargo:rerun-if-changed={committed}");

        let compiled = slangc.as_ref().is_some_and(|sc| {
            Command::new(sc)
                .args([
                    src.as_str(),
                    "-target",
                    "spirv",
                    "-O3",
                    "-entry",
                    "main",
                    "-stage",
                    "compute",
                    "-o",
                    out.as_str(),
                ])
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        });

        if !compiled {
            if slangc.is_none() {
                println!(
                    "cargo:warning=slangc not found; using committed {name}.spv. Set SLANGC or add ~/.local/slang/bin to PATH to recompile from {name}.slang."
                );
            } else {
                println!(
                    "cargo:warning=slangc failed on {name}.slang; using committed {name}.spv."
                );
            }
            std::fs::copy(&committed, &out)
                .unwrap_or_else(|e| panic!("no compiled or committed SPIR-V for {name}: {e}"));
        }
    }
    println!("cargo:rerun-if-env-changed=SLANGC");
}

/// Multi-target Slang kernels, by basename under `src/backend/shaders/slang/`.
/// Each needs a `<name>.slang` plus committed `<name>.wgsl` / `<name>.metal`
/// fallbacks for build hosts without slangc, regenerated by `just slang`.
const SLANG_MULTI_KERNELS: &[&str] = &[
    "softmax",
    "coopmat_probe",
    "gemm_q8_0",
    "bias_add",
    "gelu",
    "elementwise",
    "rope",
    "per_head_rmsnorm",
    "layernorm_batch",
    "rmsnorm_batch",
    "argmax_f32",
    "rmsnorm",
    "conv1d",
    "conv1d_fused",
    "conv1d_fused_batch",
    "exp_polar",
    "overlap_add",
    "activations",
    "conv2d_direct",
    "transpose_blocked",
    "glu_split",
    "chan_affine_silu",
    "audio_xl_attention",
];

/// Compile each multi-target kernel to the shader languages the enabled
/// features actually need.
///
/// Same fallback contract as [`compile_slang_kernels`]: slangc is the source of
/// truth, the committed output is copied when it is unavailable, and a
/// `cargo:warning` makes the fallback visible so drift cannot pass silently.
/// The emitted text is `include_str!`'d from OUT_DIR, so a stale committed file
/// never wins over a successful compile.
fn compile_slang_multitarget(want_wgsl: bool, want_msl: bool) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR unset");
    let dir = "src/backend/shaders/slang";
    let slangc = find_slangc();

    // (slangc target name, file extension). Extension doubles as the committed
    // fallback's suffix, so the two can never disagree.
    let mut targets: Vec<(&str, &str)> = Vec::with_capacity(2);
    if want_wgsl {
        targets.push(("wgsl", "wgsl"));
    }
    if want_msl {
        targets.push(("metal", "metal"));
    }

    for name in SLANG_MULTI_KERNELS {
        let src = format!("{dir}/{name}.slang");
        println!("cargo:rerun-if-changed={src}");

        let entries = slang_entry_points(&src, name);

        for (target, ext) in &targets {
            let committed = format!("{dir}/{name}.{ext}");
            let out = format!("{out_dir}/{name}.{ext}");
            println!("cargo:rerun-if-changed={committed}");

            // slangc has no entry-point auto-discovery: it needs one `-entry`
            // per `[shader]` function. Single-entry kernels default to the
            // basename; multi-entry ones (and any whose entry name differs from
            // the file, like gelu's `gelu_inplace`) declare their entries in a
            // `// slang-entries:` header line.
            let mut args: Vec<&str> = vec![src.as_str(), "-target", target, "-O3"];
            for e in &entries {
                args.push("-entry");
                args.push(e);
            }
            args.extend(["-stage", "compute", "-o", out.as_str()]);

            let compiled = slangc.as_ref().is_some_and(|sc| {
                Command::new(sc)
                    .args(&args)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
            });

            if !compiled {
                if slangc.is_none() {
                    println!(
                        "cargo:warning=slangc not found; using committed {name}.{ext}. Set SLANGC or add ~/.local/slang/bin to PATH to recompile from {name}.slang."
                    );
                } else {
                    println!(
                        "cargo:warning=slangc failed on {name}.slang for target {target}; using committed {name}.{ext}."
                    );
                }
                std::fs::copy(&committed, &out)
                    .unwrap_or_else(|e| panic!("no compiled or committed {ext} for {name}: {e}"));
            }

            // Deliberately after the compile-or-fallback write, so it covers
            // both paths, and against OUT_DIR only, so the committed artifact
            // stays byte-identical to slangc output for the CI drift check.
            if *name == "gemm_q8_0" && *target == "metal" {
                apply_msl_postpass(&out, name);
            }
        }
    }
    println!("cargo:rerun-if-env-changed=SLANGC");
}

/// Rewrite a generated MSL file in place for folded addressing.
///
/// Declining is not an error: the unpatched shader is correct, just slower, so a
/// slangc upgrade that moves the anchors costs performance and prints a warning
/// rather than breaking the build. See `build_support/msl_postpass.rs` for the
/// contract.
fn apply_msl_postpass(path: &str, name: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            println!("cargo:warning=cannot read generated {name}.metal to post-process: {e}");
            return;
        }
    };
    match postpass_gemm_msl(&src) {
        Ok(patched) => {
            // Write beside the target and rename, so a failure part-way through
            // leaves the readable unpatched file rather than a truncated one.
            // `fs::write` truncates first, which would produce exactly the
            // half-patched artifact the pass promises never to emit.
            let tmp = format!("{path}.tmp");
            let staged = std::fs::write(&tmp, patched).and_then(|()| std::fs::rename(&tmp, path));
            if let Err(e) = staged {
                let _ = std::fs::remove_file(&tmp);
                println!("cargo:warning=cannot write post-processed {name}.metal: {e}");
            }
        }
        Err(why) => println!(
            "cargo:warning=MSL post-pass declined on {name}.metal ({why}); shipping unpatched Slang output, which is correct but ~5% slower than the patched shader on the simdgroup GEMM."
        ),
    }
}

/// Entry-point names to pass to slangc for a multi-target kernel.
///
/// slangc has no entry-point auto-discovery (no `-entry` compiles the default
/// `main`), so every `[shader]` function needs its own `-entry`. A kernel whose
/// single entry matches its basename needs nothing; anything else declares its
/// entries in a `// slang-entries: a b c` header line, which both `just slang`
/// and the CI drift check parse the same way. Entries are collected from every
/// matching header line (mirroring the shell `sed ... p` sites, which print all
/// matches), so all three stay identical and the committed output can never
/// drift from what CI regenerates. Files in practice carry exactly one header.
fn slang_entry_points(src_path: &str, basename: &str) -> Vec<String> {
    // Fail fast rather than defaulting to the basename: an unreadable `.slang`
    // is a repo-integrity problem, and silently falling through to the committed
    // artifact would hide it.
    let text = std::fs::read_to_string(src_path)
        .unwrap_or_else(|e| panic!("failed to read Slang source {src_path}: {e}"));
    let mut names: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("//")
            && let Some(list) = rest.trim_start().strip_prefix("slang-entries:")
        {
            names.extend(list.split_whitespace().map(str::to_string));
        }
    }
    if names.is_empty() {
        vec![basename.to_string()]
    } else {
        names
    }
}

/// Locate slangc: `SLANGC` env, then PATH, then the default local install.
/// Every candidate (including `SLANGC`) is probed with `-v`, so a bad `SLANGC`
/// value falls through to PATH instead of being reported as a compile failure.
fn find_slangc() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut candidates: Vec<String> = Vec::new();
    if let Some(p) = std::env::var_os("SLANGC") {
        candidates.push(p.to_string_lossy().into_owned());
    }
    candidates.push("slangc".to_string());
    candidates.push(format!("{home}/.local/slang/bin/slangc"));
    candidates
        .into_iter()
        .find(|cand| Command::new(cand).arg("-v").output().is_ok())
}
