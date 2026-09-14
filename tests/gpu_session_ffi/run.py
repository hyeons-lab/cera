"""Exercise real Swift/Kotlin GPU session lifetimes on macOS arm64."""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "tests/api_loading"))
from commands import Commands, cargo_artifact
from native_runtime import run_consumers, stage_library
from prepare import cargo_configs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_cases():
    lifecycle = [
        "busy",
        "busy-before-config",
        "extraction-retains",
        "cancel-retains",
        "generation-error-retains",
        "continuation",
        "reset-retains",
        "release",
        "constructor-failure-release",
        "successor",
        "parent-release",
    ]
    cases = ["cpu/sharing"]
    for backend in ("metal", "wgpu"):
        for mode in ("none", "turboquant"):
            cases += [f"{backend}/{mode}/{case}" for case in lifecycle]
        cases += [
            f"{backend}/{case}"
            for case in (
                "async-retains",
                "async-cancel-release",
                "scoped-example",
            )
        ]
    return sorted(cases)


def result(text):
    records = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    if len(records) != 1 or not isinstance(records[0], dict):
        raise RuntimeError("Expected one consumer result")
    record = records[0]
    if (
        set(record) != {"cases"}
        or not isinstance(record["cases"], list)
        or not all(isinstance(case, str) for case in record["cases"])
        or sorted(record["cases"]) != expected_cases()
    ):
        raise RuntimeError("Missing, duplicated or unexpected ownership cases")
    return {"cases": sorted(record["cases"])}


def source_inputs(repo=REPO):
    root = repo / "tests/gpu_session_ffi"
    bindings = repo / "cera-ffi/bindings"
    # Include embedded shaders and build-time generators, not just Rust modules.
    native = {
        path
        for directory in ("cera/src", "cera/build_support", "cera-ffi/src")
        for path in (repo / directory).rglob("*")
        if path.is_file()
    }
    return sorted(
        native
        | {
            *root.glob("*.py"),
            *root.glob("*.swift"),
            *root.glob("*.kt"),
            repo / "tests/api_loading/commands.py",
            repo / "tests/api_loading/native_runtime.py",
            repo / "tests/api_loading/prepare.py",
            repo / "Cargo.toml",
            repo / "Cargo.lock",
            repo / "cera/Cargo.toml",
            repo / "cera-ffi/Cargo.toml",
            repo / "cera/build.rs",
            repo / "cera-ffi/build.rs",
            repo / ".cargo/config.toml",
            repo / "tests/leap_compat/artifacts.json",
            bindings / "swift/cera_ffi.swift",
            bindings / "swift/CeraFFI.h",
            bindings / "swift/CeraFFI.modulemap",
            bindings / "kotlin/uniffi/cera_ffi/cera_ffi.kt",
            repo / "cera-ffi/apple/Sources/Cera/cera_ffi.swift",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Cached pinned JNA and coroutines jars",
    )
    parser.add_argument(
        "--output-parent", type=Path, default=Path(tempfile.gettempdir())
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error(
            "Requires macOS arm64, Metal/wgpu devices, Swift and Kotlin/JDK 21"
        )
    output = Path(
        tempfile.mkdtemp(prefix="cera-gpu-session-", dir=args.output_parent)
    ).resolve()
    print(output, flush=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("DYLD_", "LD_")) or key in {
            "JAVA_TOOL_OPTIONS",
            "_JAVA_OPTIONS",
            "JDK_JAVA_OPTIONS",
            "CLASSPATH",
        }:
            del env[key]
    if not env.get("JAVA_HOME"):
        parser.error("Set JAVA_HOME to JDK 21")
    commands = Commands(output, env, REPO)
    report = {
        "scope": "macOS arm64 Metal/wgpu plus CPU sharing; synthetic correctness only"
    }
    bindings = REPO / "cera-ffi/bindings"
    pins_path = REPO / "tests/leap_compat/artifacts.json"
    before = {str(p): digest(p) for p in source_inputs()}
    cargo_home = Path(env.get("CARGO_HOME", Path.home() / ".cargo"))
    if not cargo_home.is_absolute():
        cargo_home = REPO / cargo_home
    report["cargo_config_sha256"] = cargo_configs(REPO, cargo_home)
    try:
        pins = json.loads(pins_path.read_text())
        dependencies = [
            (args.artifacts / pins[key]["file"]).resolve()
            for key in ("jna", "coroutines")
        ]
        for key, path in zip(("jna", "coroutines"), dependencies):
            if digest(path) != pins[key]["sha256"]:
                raise RuntimeError(f"{key} SHA256 mismatch")
        if digest(REPO / "cera-ffi/apple/Sources/Cera/cera_ffi.swift") != digest(
            bindings / "swift/cera_ffi.swift"
        ):
            raise RuntimeError("SwiftPM wrapper differs from generated Swift")
        build = commands.run(
            "build",
            [
                "cargo",
                "build",
                "-p",
                "cera-ffi",
                "--lib",
                "--features",
                "gpu,metal,ffi-buffer",
                "--locked",
                "--offline",
                "--message-format=json",
            ],
            timeout=1200,
        )
        source = cargo_artifact(build, REPO / "cera-ffi/src/lib.rs", suffix=".dylib")
        native = stage_library(commands, source, output)
        exported = commands.run(
            "fixtures",
            [
                "cargo",
                "test",
                "-p",
                "cera",
                "--features",
                "gpu,metal",
                "--lib",
                "export_gpu_ownership_fixtures",
                "--locked",
                "--offline",
                "--",
                "--ignored",
                "--nocapture",
            ],
            timeout=1200,
        )
        directories = re.findall(
            r"^GPU_OWNERSHIP_FIXTURES=(.+)$", exported, re.MULTILINE
        )
        if len(directories) != 1 or "1 passed; 0 failed" not in exported:
            raise RuntimeError("Fixture export did not run exactly once")
        fixture = Path(directories[0]) / "conversation.gguf"
        fixture_hash = digest(fixture)
        native_hash = digest(native)
        outputs = run_consumers(
            commands,
            bindings,
            output,
            native,
            ROOT / "SessionProbe.swift",
            ROOT / "SessionProbe.kt",
            "sessionprobe.SessionProbeKt",
            dependencies,
            [fixture],
        )
        report["swift"] = result(outputs["swift"])
        report["kotlin"] = result(outputs["kotlin"])
        if report["swift"] != report["kotlin"]:
            raise RuntimeError("Swift/Kotlin ownership results differ")
        if before != {str(p): digest(p) for p in source_inputs()}:
            raise RuntimeError("Sources changed during execution")
        if report["cargo_config_sha256"] != cargo_configs(REPO, cargo_home):
            raise RuntimeError("Cargo configuration changed during execution")
        if fixture_hash != digest(fixture) or native_hash != digest(native):
            raise RuntimeError("Fixture or native library changed during execution")
        for key, path in zip(("jna", "coroutines"), dependencies):
            if digest(path) != pins[key]["sha256"]:
                raise RuntimeError(f"{key} changed during execution")
        report.update(
            passed=True,
            source_sha256=before,
            fixture=str(fixture),
            fixture_sha256=fixture_hash,
            swift_native_library=outputs["native"],
            artifacts_sha256={
                p.name: digest(p)
                for p in [
                    native,
                    output / "libCera.dylib",
                    output / "swift-probe",
                    output / "consumer-probe.jar",
                ]
            },
            dependencies={key: pins[key] for key in ("jna", "coroutines")},
        )
    finally:
        report["commands"] = commands.results
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: {len(expected_cases())} ownership cases each in Swift and Kotlin",
        flush=True,
    )


if __name__ == "__main__":
    main()
