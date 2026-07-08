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
}
