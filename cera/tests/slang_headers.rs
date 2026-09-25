//! Pins the `// slang-*` header readers on both sides: the shell copy
//! (`scripts/slang-headers.sh`, shared by `just slang` and the CI drift
//! check) and the Rust copy (`build_support/slang_headers.rs`, included by
//! both `build.rs` and this file, so what is tested is exactly what
//! builds). Covers the header match, the defaults, union-across-lines,
//! every fail-fast leg (empty targets, unknown target, unreadable file),
//! and shell↔Rust agreement; the real-shader legs pin both against the
//! repo's actual headers.
//!
//! Unix-only: the script is bash.

#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

include!("../build_support/slang_headers.rs");

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scripts/slang-headers.sh")
}

fn fixture(name: &str, body: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("slang-headers-test-{}-{name}", std::process::id()));
    std::fs::write(&p, body).unwrap();
    p
}

fn run(key: &str, file: &std::path::Path) -> std::process::Output {
    Command::new(script())
        .arg(key)
        .arg(file)
        .output()
        .expect("run slang-headers.sh")
}

fn stdout_lines(out: &std::process::Output) -> Vec<String> {
    assert!(out.status.success(), "script failed: {out:?}");
    String::from_utf8(out.stdout.clone())
        .unwrap()
        .lines()
        .map(|l| l.to_string())
        .collect()
}

#[test]
fn entries_default_union_and_absent() {
    // Absent header → the basename (minus `.slang`).
    let f = fixture("noheader.slang", "// no headers here\n[shader]\n");
    let stem = f.file_stem().unwrap().to_string_lossy().into_owned();
    assert_eq!(stdout_lines(&run("entries", &f)), vec![stem]);
    // Present headers union across lines (same as build.rs).
    let f = fixture(
        "doubled.slang",
        "// slang-entries: foo bar\n// slang-entries: baz\n",
    );
    assert_eq!(stdout_lines(&run("entries", &f)), vec!["foo bar", "baz"]);
    // Indented `//` still matches (same match as build.rs).
    let f = fixture("indented.slang", "   //   slang-entries: qux\n");
    assert_eq!(stdout_lines(&run("entries", &f)), vec!["qux"]);
    // Empty list falls back to the basename (unlike targets, which fails).
    let f = fixture("emptyentries.slang", "// slang-entries:\n");
    let stem = f.file_stem().unwrap().to_string_lossy().into_owned();
    assert_eq!(stdout_lines(&run("entries", &f)), vec![stem]);
}

#[test]
fn targets_default_union_and_fail_fast() {
    // Absent header → both targets.
    let f = fixture("notargets.slang", "// no headers here\n");
    assert_eq!(stdout_lines(&run("targets", &f)), vec!["wgsl metal"]);
    // Present + doubled headers union.
    let f = fixture(
        "split.slang",
        "// slang-targets: wgsl\n// slang-targets: metal\n",
    );
    assert_eq!(stdout_lines(&run("targets", &f)), vec!["wgsl", "metal"]);
    // Empty list fails fast (must not silently default to both).
    let f = fixture("empty.slang", "// slang-targets:\n");
    let out = run("targets", &f);
    assert!(!out.status.success(), "empty targets must fail: {out:?}");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("empty"), "stderr: {stderr}");
    // Unknown target fails fast (the build.rs allowlist, mirrored).
    let f = fixture("typo.slang", "// slang-targets: wgsl glsl\n");
    let out = run("targets", &f);
    assert!(!out.status.success(), "unknown target must fail: {out:?}");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("unknown"), "stderr: {stderr}");
}

#[test]
fn bare_or_bad_key_reaches_usage() {
    // No `set -u` unbound-variable crash: a bare invocation (or an
    // unknown key) names the contract on stderr and exits nonzero.
    for args in [vec![], vec!["bogus".to_string()]] {
        let out = Command::new(script())
            .args(&args)
            .output()
            .expect("run slang-headers.sh");
        assert!(!out.status.success(), "args {args:?} must fail: {out:?}");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(stderr.contains("usage:"), "args {args:?} stderr: {stderr}");
        assert!(
            !stderr.contains("unbound variable"),
            "args {args:?} stderr: {stderr}"
        );
    }
}

#[test]
fn unreadable_file_fails_both_modes() {
    // Neither mode may mistake a read error for a missing header: `targets`
    // must not silently default to "wgsl metal" (like build.rs, fail fast).
    let missing = std::env::temp_dir().join(format!(
        "slang-headers-test-{}-does-not-exist.slang",
        std::process::id()
    ));
    for key in ["entries", "targets"] {
        let out = run(key, &missing);
        assert!(
            !out.status.success(),
            "{key} on missing file must fail: {out:?}"
        );
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(stderr.contains("cannot read"), "{key} stderr: {stderr}");
    }
}

#[test]
fn directory_input_fails_both_modes() {
    // Directories are readable but neither parser accepts them: without
    // the `-f` conjunct `targets` prints the default and `entries` prints
    // the basename, both exit 0. The Rust side panics on the same input.
    let dir = std::env::temp_dir();
    for key in ["entries", "targets"] {
        let out = run(key, &dir);
        assert!(
            !out.status.success(),
            "{key} on directory must fail: {out:?}"
        );
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(stderr.contains("cannot read"), "{key} stderr: {stderr}");
    }
}

#[test]
fn rust_targets_default_union_and_fail_fast() {
    let f = fixture("rs-notargets.slang", "// no headers here\n");
    assert_eq!(slang_targets(f.to_str().unwrap()), vec!["wgsl", "metal"]);
    let f = fixture(
        "rs-split.slang",
        "// slang-targets: wgsl\n// slang-targets: metal\n",
    );
    assert_eq!(slang_targets(f.to_str().unwrap()), vec!["wgsl", "metal"]);
}

#[test]
#[should_panic(expected = "empty slang-targets")]
fn rust_targets_empty_list_panics() {
    let f = fixture("rs-empty.slang", "// slang-targets:\n");
    let _ = slang_targets(f.to_str().unwrap());
}

#[test]
#[should_panic(expected = "unknown slang-targets")]
fn rust_targets_unknown_entry_panics() {
    let f = fixture("rs-typo.slang", "// slang-targets: wgsl glsl\n");
    let _ = slang_targets(f.to_str().unwrap());
}

#[test]
#[should_panic(expected = "failed to read Slang source")]
fn rust_unreadable_file_panics() {
    let missing = std::env::temp_dir().join(format!(
        "slang-headers-test-{}-does-not-exist.slang",
        std::process::id()
    ));
    let _ = slang_targets(missing.to_str().unwrap());
}

#[test]
#[should_panic(expected = "failed to read Slang source")]
fn rust_entries_unreadable_file_panics() {
    // Mirror of the targets leg above: without it, an entries parser that
    // returned the basename default on read error would stay green.
    let missing = std::env::temp_dir().join(format!(
        "slang-headers-test-{}-does-not-exist.slang",
        std::process::id()
    ));
    let _ = slang_entry_points(missing.to_str().unwrap(), "stem");
}

#[test]
#[should_panic(expected = "failed to read Slang source")]
fn rust_targets_directory_panics() {
    // Rust half of `directory_input_fails_both_modes`: directories must
    // panic here exactly as they fail closed in the shell.
    let dir = std::env::temp_dir();
    let _ = slang_targets(dir.to_str().unwrap());
}

#[test]
#[should_panic(expected = "failed to read Slang source")]
fn rust_entries_directory_panics() {
    let dir = std::env::temp_dir();
    let _ = slang_entry_points(dir.to_str().unwrap(), "stem");
}

#[test]
fn rust_entries_default_when_absent_or_empty() {
    for (name, body) in [
        ("rs-noentries.slang", "// no headers here\n"),
        ("rs-emptyentries.slang", "// slang-entries:\n"),
    ] {
        let f = fixture(name, body);
        let stem = f.file_stem().unwrap().to_string_lossy().into_owned();
        assert_eq!(
            slang_entry_points(f.to_str().unwrap(), &stem),
            vec![stem],
            "{name}"
        );
    }
    let f = fixture(
        "rs-doubled.slang",
        "// slang-entries: foo bar\n// slang-entries: baz\n",
    );
    assert_eq!(
        slang_entry_points(f.to_str().unwrap(), "ignored"),
        vec!["foo", "bar", "baz"]
    );
}

#[test]
fn shell_and_rust_parsers_agree() {
    // The shell prints matching lines; the Rust side returns flat words
    // (callers split on all whitespace either way). Compare word streams.
    for (name, body, key) in [
        (
            "agree-split.slang",
            "// slang-targets: wgsl\n// slang-targets: metal\n",
            "targets",
        ),
        (
            "agree-entries.slang",
            "// slang-entries: foo bar\n// slang-entries: baz\n",
            "entries",
        ),
        (
            "agree-indented.slang",
            "   //   slang-targets: metal\n",
            "targets",
        ),
    ] {
        let f = fixture(name, body);
        let shell_words: Vec<String> = stdout_lines(&run(key, &f))
            .iter()
            .flat_map(|l| l.split_whitespace().map(str::to_string))
            .collect();
        let rust_words = slang_header_values(f.to_str().unwrap(), &format!("slang-{key}:"))
            .expect("header must match");
        assert_eq!(shell_words, rust_words, "{name}");
    }
    // Absent header: shell defaults equal Rust defaults.
    let f = fixture("agree-noheader.slang", "// nothing\n");
    let stem = f.file_stem().unwrap().to_string_lossy().into_owned();
    assert_eq!(stdout_lines(&run("entries", &f)), vec![stem.clone()]);
    assert_eq!(slang_entry_points(f.to_str().unwrap(), &stem), vec![stem]);
    assert_eq!(
        stdout_lines(&run("targets", &f)),
        vec!["wgsl metal".to_string()]
    );
    assert_eq!(slang_targets(f.to_str().unwrap()), vec!["wgsl", "metal"]);
}

#[test]
fn real_shader_headers_parse() {
    // The repo's own headers, through the same script both callers use.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/backend/shaders/slang");
    let out = run("targets", &dir.join("kv_append.slang"));
    assert_eq!(stdout_lines(&out), vec!["wgsl"]);
    let out = run("entries", &dir.join("activations.slang"));
    assert_eq!(
        stdout_lines(&out),
        vec!["relu_inplace silu_inplace gelu_erf_inplace"]
    );
    let out = run("targets", &dir.join("activations.slang"));
    assert_eq!(stdout_lines(&out), vec!["wgsl metal"]);
    // Same files through the Rust parser: a real-header form the two
    // parsers disagree on must fail here, not slip past fixture-only
    // agreement.
    assert_eq!(
        slang_targets(dir.join("kv_append.slang").to_str().unwrap()),
        vec!["wgsl"]
    );
    assert_eq!(
        slang_entry_points(
            dir.join("activations.slang").to_str().unwrap(),
            "activations"
        ),
        vec!["relu_inplace", "silu_inplace", "gelu_erf_inplace"]
    );
    assert_eq!(
        slang_targets(dir.join("activations.slang").to_str().unwrap()),
        vec!["wgsl", "metal"]
    );
}
