"""Execute Whisper through the real generated Swift and Kotlin bindings on macOS."""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "tests/api_loading"))
from commands import Commands, cargo_artifact
from native_runtime import run_consumers, stage_library


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(text):
    records = [json.loads(line) for line in text.splitlines() if line.startswith('{"')]
    cases = {
        "bytes-owned",
        "file-owned",
        "languages",
        "defaults",
        "sync",
        "async-shared",
        "async-distinct",
        "empty",
        "malformed",
        "missing",
        "recording-example",
    }
    if len(records) != 1:
        raise RuntimeError("Expected one consumer result")
    record = records[0]
    if sorted(record.get("cases", [])) != sorted(cases) or record.get("text") != [
        "aaa",
        "aa",
        "bb",
    ]:
        raise RuntimeError("Missing cases or incorrect transcription")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Cached JNA and coroutines jars from tests/leap_compat/artifacts.json",
    )
    parser.add_argument(
        "--output-parent", type=Path, default=Path(tempfile.gettempdir())
    )
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error(
            "This native probe requires macOS arm64 with Swift and Kotlin/JDK 21"
        )
    output = Path(
        tempfile.mkdtemp(prefix="cera-whisper-", dir=args.output_parent)
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
        parser.error("Set JAVA_HOME to a JDK 21 installation")
    commands = Commands(output, env, REPO)
    report = {
        "scope": "macOS arm64 CPU; synthetic decoding, not speech quality or mobile devices"
    }
    bindings = REPO / "cera-ffi/bindings"
    pins_path = REPO / "tests/leap_compat/artifacts.json"
    inputs = [
        ROOT / "run.py",
        ROOT / "WhisperProbe.swift",
        ROOT / "WhisperProbe.kt",
        REPO / "tests/api_loading/commands.py",
        REPO / "tests/api_loading/native_runtime.py",
        pins_path,
        REPO / "cera-ffi/src/lib.rs",
        REPO / "cera/src/model/whisper.rs",
        REPO / "cera-ffi/examples/whisper_fixture.rs",
        REPO / "cera-ffi/tests/common/whisper_fixture.rs",
        bindings / "swift/cera_ffi.swift",
        bindings / "swift/CeraFFI.h",
        bindings / "swift/CeraFFI.modulemap",
        bindings / "kotlin/uniffi/cera_ffi/cera_ffi.kt",
        REPO / "cera-ffi/apple/Sources/Cera/cera_ffi.swift",
    ]
    before = {str(p): digest(p) for p in inputs}
    try:
        pins = json.loads(pins_path.read_text())
        dependencies = [
            (args.artifacts / pins[key]["file"]).resolve()
            for key in ("jna", "coroutines")
        ]
        for key, path in zip(("jna", "coroutines"), dependencies):
            if digest(path) != pins[key]["sha256"]:
                raise RuntimeError(f"{key} SHA256 mismatch")
        if digest(inputs[-1]) != digest(bindings / "swift/cera_ffi.swift"):
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
                "ffi-buffer",
                "--locked",
                "--offline",
                "--message-format=json",
            ],
            timeout=1200,
        )
        source = cargo_artifact(build, REPO / "cera-ffi/src/lib.rs", suffix=".dylib")
        native = stage_library(commands, source, output)
        fixtures = output / "fixtures"
        commands.run(
            "fixtures",
            [
                "cargo",
                "run",
                "-p",
                "cera-ffi",
                "--example",
                "whisper_fixture",
                "--locked",
                "--offline",
                "--",
                fixtures,
            ],
            timeout=1200,
        )
        fixture_hashes = {p.name: digest(p) for p in fixtures.iterdir()}
        native_hash = digest(native)
        outputs = run_consumers(
            commands,
            bindings,
            output,
            native,
            ROOT / "WhisperProbe.swift",
            ROOT / "WhisperProbe.kt",
            "whisperprobe.WhisperProbeKt",
            dependencies,
            [fixtures],
            jar_name="whisper-probe.jar",
        )
        report["swift"] = result(outputs["swift"])
        report["kotlin"] = result(outputs["kotlin"])
        report["swift_native_library"] = outputs["native"]
        jar = output / "whisper-probe.jar"
        if report["swift"] != report["kotlin"]:
            raise RuntimeError("Swift/Kotlin results differ")
        if before != {str(p): digest(p) for p in inputs}:
            raise RuntimeError("Inputs changed during execution")
        if fixture_hashes != {
            p.name: digest(p) for p in fixtures.iterdir()
        } or native_hash != digest(native):
            raise RuntimeError("Fixtures or native library changed during execution")
        for key, path in zip(("jna", "coroutines"), dependencies):
            if digest(path) != pins[key]["sha256"]:
                raise RuntimeError(f"{key} changed during execution")
        report.update(
            passed=True,
            source_sha256=before,
            fixtures_sha256=fixture_hashes,
            artifacts_sha256={
                p.name: digest(p)
                for p in [
                    native,
                    output / "libCera.dylib",
                    output / "swift-probe",
                    jar,
                ]
            },
            dependencies={key: pins[key] for key in ("jna", "coroutines")},
        )
    finally:
        report["commands"] = commands.results
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: Swift and Kotlin decoded aaa/aa/bb through the current Whisper FFI",
        flush=True,
    )


if __name__ == "__main__":
    main()
