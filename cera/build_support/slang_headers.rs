// Shared `// slang-*` header readers for `build.rs`, free of build-script
// I/O beyond reading the `.slang` file, so `tests/slang_headers.rs` can
// `include!` the same source and exercise it directly. `scripts/slang-headers.sh`
// is the shell copy the same test pins; both parsers must agree (see the
// agreement test there), and the committed output can never drift from
// what CI regenerates.

/// Values of one `// slang-<key>: ...` header (`key` includes the colon),
/// unioned across every matching header line: `None` when no header line
/// exists, `Some` (possibly empty) when at least one does. The union
/// mirrors `scripts/slang-headers.sh` (shared by `just slang` and CI),
/// which prints all matches, so both parsers stay identical and the
/// committed output can never drift from what CI regenerates. Files in
/// practice carry exactly one header.
///
/// Fail fast on an unreadable `.slang` rather than defaulting: it is a
/// repo-integrity problem, and silently falling through to the committed
/// artifact would hide it.
fn slang_header_values(src_path: &str, key: &str) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(src_path)
        .unwrap_or_else(|e| panic!("failed to read Slang source {src_path}: {e}"));
    let mut values: Vec<String> = Vec::new();
    let mut matched = false;
    for line in text.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("//")
            && let Some(list) = rest.trim_start().strip_prefix(key)
        {
            matched = true;
            values.extend(list.split_whitespace().map(str::to_string));
        }
    }
    matched.then_some(values)
}

/// Entry points for a multi-entry kernel: the `// slang-entries: a b c`
/// header, defaulting to the basename when absent (a kernel whose single
/// entry matches its basename needs nothing). Every `[shader]` function
/// needs its own `-entry`, since slangc emits only `main` otherwise.
fn slang_entry_points(src_path: &str, basename: &str) -> Vec<String> {
    match slang_header_values(src_path, "slang-entries:") {
        Some(names) if !names.is_empty() => names,
        _ => vec![basename.to_string()],
    }
}

/// Target allowlist for a multi-target kernel: the `// slang-targets: ...`
/// header (`wgsl`/`metal`, space-separated), defaulting to both targets
/// when absent. A kernel whose twin nothing compiles or dispatches
/// restricts to the live target, so the dead twin is neither generated
/// nor expected as a committed fallback. Unknown or empty lists fail fast
/// (a typo'd header silently dropping a shipped twin would be worse than
/// no allowlist at all).
fn slang_targets(src_path: &str) -> Vec<String> {
    match slang_header_values(src_path, "slang-targets:") {
        None => vec!["wgsl".to_string(), "metal".to_string()],
        Some(targets) => {
            assert!(
                !targets.is_empty(),
                "empty slang-targets list in {src_path}"
            );
            for t in &targets {
                assert!(
                    t == "wgsl" || t == "metal",
                    "unknown slang-targets entry {t:?} in {src_path} (want wgsl/metal)"
                );
            }
            targets
        }
    }
}
