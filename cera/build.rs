use std::process::Command;

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

    // Compile the Slang SPIR-V passthrough kernels (gpu feature only). Each is
    // written to OUT_DIR and `include_spirv_raw!`d from there, so slangc is the
    // source of truth. When slangc is unavailable (e.g. CI without a Slang
    // toolchain) we fall back to the committed `.spv` next to the `.slang`, which
    // a developer with slangc regenerates via `just slang`; a cargo:warning
    // flags the fallback so drift is visible.
    if std::env::var_os("CARGO_FEATURE_GPU").is_some() {
        compile_slang_kernels();
    }
}

/// Slang passthrough kernels, by basename under `src/backend/shaders/spirv/`.
/// Extend as loaders are ported; each needs a `<name>.slang` and a committed
/// `<name>.spv` fallback.
const SLANG_KERNELS: &[&str] = &["mul_mat_reg_tile_q4_0", "mul_mat_reg_tile_q8_0"];

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
