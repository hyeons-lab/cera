"""Execute generated candidate bindings against the public core loading API."""

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import tomllib
from commands import Commands, cargo_artifact
from prepare import (
    REPO,
    ROOT,
    binding_inventory,
    cargo_configs,
    digest,
    inventory,
    once,
    prepare,
    verify_inventory,
)


def probe_environment(inherited):
    """Keep build selectors but remove model and runtime-library overrides."""
    return {
        key: value
        for key, value in inherited.items()
        if not key.startswith(("CERA_", "DYLD_"))
        and key
        not in (
            "JAVA_TOOL_OPTIONS",
            "_JAVA_OPTIONS",
            "JDK_JAVA_OPTIONS",
            "JAVA_OPTS",
            "KOTLIN_OPTS",
            "NODE_OPTIONS",
            "NODE_PATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--target", type=Path, default=ROOT / "build/target")
    parser.add_argument(
        "--jna",
        type=Path,
        default=Path("/private/tmp/cera-leap-api-baseline/jna-5.16.0.jar"),
    )
    parser.add_argument(
        "--java-home",
        type=Path,
        default=None,
        help="Path to Java 21 home directory",
    )
    parser.add_argument(
        "--wasm-bindgen",
        type=Path,
        default=Path.home()
        / "Library/Caches/.wasm-pack/wasm-bindgen-cargo-install-0.2.117/wasm-bindgen",
    )
    args = parser.parse_args()
    args.wasm_bindgen = args.wasm_bindgen.resolve()
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=build))
    workspace, report = prepare(output)
    report.update(
        status="incomplete",
        scope="public Rust loading API with isolated candidate bindings; macOS native and Node WASM",
        commands=[],
    )
    excluded = ("build", "__pycache__")
    report["probe_sha256"] = inventory(ROOT, excluded)
    print(output, flush=True)
    environment = probe_environment(os.environ)
    report["removed_environment_keys"] = sorted(set(os.environ) - set(environment))
    target_root = args.target.resolve()
    target = target_root / output.name
    target.mkdir(parents=True, exist_ok=False)
    environment["CARGO_TARGET_DIR"] = str(target)
    environment["CERA_GIT_SHA"] = "loading-probe"
    if args.java_home:
        environment["JAVA_HOME"] = str(args.java_home)
    elif "JAVA_HOME" not in environment:
        sdkman_path = Path.home() / ".sdkman/candidates/java/21.0.9-zulu"
        if sdkman_path.exists():
            environment["JAVA_HOME"] = str(sdkman_path)
    environment["LOADING_JNA"] = str(args.jna.resolve())
    commands = Commands(output, environment, workspace)
    report["commands"] = commands.results
    report["target"] = str(target)
    report["target_root"] = str(target_root)
    try:
        if args.prepare_only:
            report["status"] = "prepared-only"
            return
        cargo_home = Path(environment.get("CARGO_HOME", Path.home() / ".cargo"))
        if not cargo_home.is_absolute():
            cargo_home = (workspace / cargo_home).resolve()
        report["cargo_config_sha256"] = cargo_configs(workspace, cargo_home)
        report["rust_environment"] = {
            name: environment.get(name)
            for name in (
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_BUILD_RUSTFLAGS",
                "CARGO_BUILD_TARGET",
                "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
                "CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS",
            )
        }
        # Resolve only the mirror's local package changes before locked builds.
        commands.run(
            "lock-prepare", ["cargo", "metadata", "--offline", "--format-version", "1"]
        )
        root_lock = tomllib.loads((REPO / "Cargo.lock").read_text())
        lock = tomllib.loads((workspace / "Cargo.lock").read_text())
        root_packages = {
            (p["name"], p["version"], p.get("checksum"))
            for p in root_lock["package"]
            if "source" in p
        }
        packages = {
            (p["name"], p["version"], p.get("checksum"))
            for p in lock["package"]
            if "source" in p
        }
        if not packages <= root_packages:
            raise RuntimeError("Mirror resolved dependencies outside the root lock")
        report["lock_sha256"] = digest(workspace / "Cargo.lock")
        report["mirror_sha256"] = inventory(workspace)
        report["tools"] = {
            "rust": commands.run("rust-version", ["rustc", "--version"]),
            "swift": commands.run("swift-version", ["xcrun", "swiftc", "--version"]),
            "node": commands.run("node-version", ["node", "--version"]),
            "wasm_bindgen": commands.run(
                "wasm-bindgen-version", [args.wasm_bindgen, "--version"]
            ),
            "kotlin": commands.run(
                "kotlin-version", ["kotlinc", "-version"], combined=True
            ),
            "java": commands.run(
                "java-version",
                [Path(environment["JAVA_HOME"]) / "bin/java", "-version"],
                combined=True,
            ),
        }
        report["wasm_bindgen_sha256"] = digest(args.wasm_bindgen)
        if report["tools"]["wasm_bindgen"] != "wasm-bindgen 0.2.117":
            raise RuntimeError("Expected wasm-bindgen 0.2.117")
        cargo = ["--locked", "--offline"]
        report["native_target"] = "aarch64-apple-darwin"
        native_cargo = [*cargo, "--target", report["native_target"]]
        commands.run(
            "rust-wildcard",
            [
                "cargo",
                "check",
                "-p",
                "loading-consumer",
                "--bin",
                "wildcard",
                *native_cargo,
            ],
        )
        negative = [
            "cargo",
            "check",
            "-p",
            "loading-consumer",
            "--bin",
            "exhaustive",
            "--message-format=json",
            *native_cargo,
        ]
        commands.run("rust-exhaustive-rejected", negative, exhaustive=True)
        prototype = workspace / "cera/src/engine/loading_prototype.rs"
        original = prototype.read_text()
        try:
            prototype.write_text(
                once(
                    original,
                    "#[non_exhaustive]\npub enum ModelHandle",
                    "pub enum ModelHandle",
                )
            )
            commands.run("rust-exhaustive-attribute-removed", negative)
        finally:
            prototype.write_text(original)
        commands.run("rust-exhaustive-restored", negative, exhaustive=True)
        native_output = commands.run(
            "native-build",
            [
                "cargo",
                "build",
                "-p",
                "loading-native",
                "--message-format=json",
                *native_cargo,
            ],
        )
        native = cargo_artifact(
            native_output, workspace / "native/src/lib.rs", suffix=".dylib"
        )
        generator = cargo_artifact(
            native_output, workspace / "native/src/bin/generate.rs", executable=True
        )
        report["artifact_sha256"] = {str(p): digest(p) for p in (native, generator)}
        generated = output / "native"
        generated.mkdir()
        commands.run(
            "native-generate",
            [
                generator,
                "generate",
                "--no-format",
                "--library",
                native,
                "--language",
                "swift",
                "--language",
                "kotlin",
                "--out-dir",
                generated,
            ],
        )
        report["artifact_sha256"].update(
            {str(p): digest(p) for p in generated.rglob("*") if p.is_file()}
        )
        wasm_output = commands.run(
            "wasm-build",
            [
                "cargo",
                "build",
                "-p",
                "loading-web",
                "--message-format=json",
                "--target",
                "wasm32-unknown-unknown",
                *cargo,
            ],
        )
        wasm = cargo_artifact(
            wasm_output, workspace / "wasm/src/lib.rs", suffix=".wasm"
        )
        report["artifact_sha256"][str(wasm)] = digest(wasm)
        web = output / "web"
        commands.run(
            "wasm-generate",
            [
                args.wasm_bindgen,
                wasm,
                "--target",
                "nodejs",
                "--out-dir",
                web,
            ],
        )
        report["artifact_sha256"].update(
            {str(p): digest(p) for p in web.rglob("*") if p.is_file()}
        )
        if not args.build_only:
            fixture_module = REPO / "tests/leap_compat/native_fixture.py"
            report["fixture_builder_sha256"] = digest(fixture_module)
            spec = importlib.util.spec_from_file_location(
                "native_fixture", fixture_module
            )
            fixture = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fixture)
            model = output / "model.gguf"
            model.write_bytes(fixture.tiny_model())
            report["fixture_sha256"] = digest(model)
            from consumers import run_consumers

            report["runtime"] = run_consumers(commands, generated, web, native, model)
            if digest(fixture_module) != report["fixture_builder_sha256"]:
                raise RuntimeError("Fixture builder changed during execution")
        verify_inventory(ROOT, report["probe_sha256"], excluded)
        verify_inventory(workspace, report["mirror_sha256"])
        if cargo_configs(workspace, cargo_home) != report["cargo_config_sha256"]:
            raise RuntimeError("Cargo configuration changed during execution")
        verify_inventory(
            REPO / "cera",
            {
                str(Path(p).relative_to("cera")): h
                for p, h in report["source_sha256"].items()
            },
            ("target",),
        )
        for crate, key in (
            ("cera-ffi", "ffi_source_sha256"),
            ("cera-wasm", "wasm_source_sha256"),
        ):
            if binding_inventory(REPO / crate) != report[key]:
                raise RuntimeError(f"Production binding source drift: {crate}")
        for path, expected in {
            **report["artifact_sha256"],
            **{str(REPO / p): h for p, h in report["workspace_inputs_sha256"].items()},
        }.items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"Input/artifact drift: {path}")
        report["status"] = "build-only-passed" if args.build_only else "passed"
    except BaseException as error:
        report.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
