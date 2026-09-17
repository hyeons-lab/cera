"""Build unchanged consumers, require complete case sets and compare generation."""

import json
import os
import shutil
import struct
from pathlib import Path

from prepare import REPO, ROOT, digest
from remote import run_remote

CASES = {
    "bytes-lifetime",
    "parts-defaults",
    "parts-defaults-absent",
    "parts-defaults-text-empty",
    "parts-defaults-audio",
    "parts-defaults-audio-empty",
    "parts-defaults-other",
    "parts-defaults-invalid-json",
    "kind-bert",
    "kind-modernbert",
    "kind-whisper",
    "kind-silero_vad",
    "kind-kws",
    "unknown",
    "malformed",
    "backend",
    "invalid-backend",
    "assembly",
    "inference",
    "future-kind",
}
NATIVE_CASES = {
    "production-kind-bert",
    "production-kind-modernbert",
    "production-kind-whisper",
    "production-kind-silero_vad",
    "production-kind-kws",
    "production-unknown",
    "production-malformed",
    "production-backend",
    "production-assembly",
    "production-inference",
    "production-parts-defaults-absent",
    "production-parts-defaults-text-empty",
    "production-parts-defaults-audio",
    "production-parts-defaults-audio-empty",
    "production-parts-defaults-other",
    "production-parts-defaults-invalid-json",
    "production-backend-transport",
    "production-remote-sources",
    "production-defaults",
    "production-sources",
    "production-session",
    "production-shared-engine",
    "production-kv-config",
    "production-stream-cancel",
    "native-path",
    "native-remote-hf",
    "native-remote-bundle",
    "native-remote-manifest",
    "native-remote-directory",
    "native-remote-no-repo",
    "native-remote-hf-source",
    "native-remote-hf-kind",
    "native-remote-hf-assembly",
    "native-remote-bundle-invalid",
    "native-files",
    "native-files-kind",
    "native-files-missing",
    "native-files-inference",
    "native-config-default",
    "native-config-zero",
    "native-config-cap",
    "native-config-wide",
    "native-config-small",
}
WEB_CASES = {
    "native-context-32",
    "production-defaults",
    "production-session",
    "production-shared-engine",
    "production-kv-config",
    "production-stream-cancel",
}


def result(text, native):
    records = [json.loads(line) for line in text.splitlines() if line.startswith('{"')]
    if len(records) != 1:
        raise RuntimeError("Expected exactly one consumer result")
    record = records[0]
    expected = CASES | (NATIVE_CASES if native else WEB_CASES)
    cases = record.get("cases")
    if (
        not isinstance(cases, list)
        or any(not isinstance(case, str) for case in cases)
        or sorted(cases) != sorted(expected)
    ):
        raise RuntimeError("Missing, duplicate or unexpected consumer cases")
    generation = record.get("generation")
    if not isinstance(generation, dict) or set(generation) != {"tokens", "position"}:
        raise RuntimeError("Missing generation record")
    tokens = generation["tokens"]
    if (
        not isinstance(tokens, list)
        or len(tokens) != 3
        or any(type(t) is not int or t not in (0, 1) for t in tokens)
    ):
        raise RuntimeError("Invalid generated tokens")
    if type(generation["position"]) is not int or generation["position"] != 5:
        raise RuntimeError("Invalid session position")
    return record


def architecture_header(architecture):
    def string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    header = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
    header += (
        string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    )
    return header + bytes((-len(header)) % 32)


def run_consumers(commands, generated, web, native, model):
    output = commands.output
    for arch in (
        "bert",
        "modernbert",
        "whisper",
        "silero_vad",
        "kws",
        "unknown",
        "llama",
    ):
        (output / f"{arch}.gguf").write_bytes(
            architecture_header("future_probe" if arch == "unknown" else arch)
        )
    fixtures = {p.name: digest(p) for p in output.glob("*.gguf")}
    # Public dependency pin shared with the existing export probes. No Leap runtime.
    pins = REPO / "tests/leap_compat/artifacts.json"
    pins_digest = digest(pins)
    entry = json.loads(pins.read_text())["jna"]
    jna = Path(commands.environment["LOADING_JNA"])
    if digest(jna) != entry["sha256"]:
        raise RuntimeError("JNA SHA256 mismatch")
    coroutines_pin = json.loads(pins.read_text())["coroutines"]
    coroutines = jna.parent / coroutines_pin["file"]
    if digest(coroutines) != coroutines_pin["sha256"]:
        raise RuntimeError("Coroutines SHA256 mismatch")
    classpath = os.pathsep.join(map(str, (jna, coroutines)))
    include = output / "include"
    include.mkdir()
    for name in ("cera_ffi", "loading_native"):
        shutil.copyfile(generated / f"{name}FFI.h", include / f"{name}FFI.h")
    (include / "module.modulemap").write_text(
        "\n".join(
            (generated / f"{name}FFI.modulemap").read_text()
            for name in ("cera_ffi", "loading_native")
        )
    )
    swift = [
        "xcrun",
        "swiftc",
        "-swift-version",
        "5",
        "-target",
        "arm64-apple-macosx15.0",
        "-module-cache-path",
        output / "swift-cache",
        "-I",
        include,
        "-I",
        output,
    ]
    commands.run(
        "swift-bindings",
        swift
        + [
            "-emit-library",
            "-emit-module",
            "-module-name",
            "loading_native",
            "-L",
            native.parent,
            "-lloading_native",
            generated / "cera_ffi.swift",
            generated / "loading_native.swift",
            "-o",
            output / "libloading_swift.dylib",
            "-emit-module-path",
            output / "loading_native.swiftmodule",
        ],
    )
    commands.run(
        "swift-compile",
        swift
        + [
            "-warnings-as-errors",
            "-parse-as-library",
            "-I",
            output,
            "-L",
            output,
            "-lloading_swift",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            output,
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            native.parent,
            ROOT / "consumers/LoadingProbe.swift",
            ROOT / "consumers/RemoteProbe.swift",
            ROOT / "consumers/ProductionProbe.swift",
            "-o",
            output / "swift-probe",
        ],
    )
    swift_stdout, swift_remote = run_remote(
        commands, "swift", [output / "swift-probe"], model
    )
    swift_result = result(swift_stdout, True)
    jar = output / "loading-probe.jar"
    commands.run(
        "kotlin-compile",
        [
            "kotlinc",
            "-jvm-target",
            "21",
            "-classpath",
            classpath,
            generated / "uniffi/cera_ffi/cera_ffi.kt",
            generated / "uniffi/loading_native/loading_native.kt",
            ROOT / "consumers/LoadingProbe.kt",
            ROOT / "consumers/RemoteProbe.kt",
            ROOT / "consumers/ProductionProbe.kt",
            "-include-runtime",
            "-d",
            jar,
        ],
    )
    kotlin_stdout, kotlin_remote = run_remote(
        commands,
        "kotlin",
        [
            Path(commands.environment["JAVA_HOME"]) / "bin/java",
            f"-Duniffi.component.loading_native.libraryOverride={native}",
            f"-Duniffi.component.cera_ffi.libraryOverride={native}",
            "-classpath",
            os.pathsep.join(map(str, [jar, jna, coroutines])),
            "loadingprobe.LoadingProbeKt",
        ],
        model,
    )
    kotlin_result = result(kotlin_stdout, True)
    node_result = result(
        commands.run(
            "node-run",
            [
                "node",
                ROOT / "consumers/loading_probe.cjs",
                web / "loading_web.js",
                model.parent,
            ],
        ),
        False,
    )
    if (
        swift_result != kotlin_result
        or swift_result["generation"] != node_result["generation"]
    ):
        raise RuntimeError("Cross-language generation/results mismatch")
    if digest(pins) != pins_digest or digest(jna) != entry["sha256"]:
        raise RuntimeError("JNA pin/artifact changed during execution")
    if digest(coroutines) != coroutines_pin["sha256"]:
        raise RuntimeError("Coroutines artifact changed during execution")
    if fixtures != {p.name: digest(p) for p in output.glob("*.gguf")}:
        raise RuntimeError("Fixture changed during execution")
    return {
        "remote": {"swift": swift_remote, "kotlin": kotlin_remote},
        "swift": swift_result,
        "kotlin": kotlin_result,
        "node": node_result,
        "fixtures_sha256": fixtures,
        "jna": entry,
        "jna_path": str(jna),
        "coroutines": coroutines_pin,
        "coroutines_path": str(coroutines),
        "pins_sha256": pins_digest,
        "consumer_artifact_sha256": {
            p.name: digest(p)
            for p in (
                output / "swift-probe",
                output / "libloading_swift.dylib",
                output / "loading_native.swiftmodule",
                jar,
            )
        },
    }
