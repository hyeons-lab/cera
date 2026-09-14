"""Execute existing Cera bindings with the extended candidate's HiddenStates type."""

import json
import os
import shutil
from pathlib import Path

from native_fixture import tiny_model

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def native_probes(probes, artifacts, temporary, compiler, target, digest):
    output = temporary / "native"
    output.mkdir()
    fixture = output / "model.gguf"
    fixture.write_bytes(tiny_model())
    probes.run(
        "native-cera-build",
        [
            "cargo",
            "build",
            "-p",
            "cera-ffi",
            "--release",
            "--lib",
            "--locked",
            "--offline",
            "--target-dir",
            target,
        ],
        timeout=600,
        cwd=REPO,
    )
    library = target / "release/libcera_ffi.dylib"
    bindings = REPO / "cera-ffi/bindings"
    include = output / "include"
    include.mkdir()
    shutil.copyfile(bindings / "swift/CeraFFI.h", include / "CeraFFI.h")
    shutil.copyfile(bindings / "swift/CeraFFI.modulemap", include / "module.modulemap")
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
    ]
    probes.run(
        "native-swift-bindings",
        swift
        + [
            "-emit-library",
            "-emit-module",
            "-module-name",
            "Cera",
            "-L",
            library.parent,
            "-lcera_ffi",
            bindings / "swift/cera_ffi.swift",
            "-o",
            output / "libCera.dylib",
            "-emit-module-path",
            output / "Cera.swiftmodule",
        ],
    )
    candidate = ROOT / "candidates/kmp/build/extended"
    framework = candidate / "bin/macosArm64/debugFramework"
    probes.run(
        "native-swift-compile",
        swift
        + [
            "-warnings-as-errors",
            "-parse-as-library",
            "-I",
            output,
            "-L",
            output,
            "-lCera",
            "-F",
            framework,
            "-framework",
            "LeapSDK",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            output,
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            library.parent,
            ROOT / "native/NativeProbe.swift",
            "-o",
            output / "swift-probe",
        ],
    )
    swift_result = probes.run("native-swift-run", [output / "swift-probe", fixture])
    kotlin_binding = bindings / "kotlin/uniffi/cera_ffi/cera_ffi.kt"
    dependencies = [
        candidate / "libs/leap-export-probe-jvm.jar",
        artifacts["coroutines"],
        artifacts["jna"],
    ]
    jar = output / "native-probe.jar"
    probes.run(
        "native-kotlin-compile",
        [
            compiler,
            "-jvm-target",
            "21",
            "-classpath",
            os.pathsep.join(map(str, dependencies)),
            kotlin_binding,
            ROOT / "native/NativeProbe.kt",
            "-include-runtime",
            "-d",
            jar,
        ],
    )
    java = (
        Path(probes.environment["JAVA_HOME"]) / "bin/java"
        if probes.environment.get("JAVA_HOME")
        else "java"
    )
    kotlin_result = probes.run(
        "native-kotlin-run",
        [
            java,
            f"-Duniffi.component.cera_ffi.libraryOverride={library}",
            "-classpath",
            os.pathsep.join(map(str, [jar, *dependencies])),
            "nativeprobe.NativeProbeKt",
            fixture,
        ],
    )

    # Native logs may precede stdout. Require a single result record per process.
    def result(text):
        records = [
            json.loads(line) for line in text.splitlines() if line.startswith('{"')
        ]
        if len(records) != 1:
            raise RuntimeError("Expected one native result record")
        return records[0]

    swift_data, kotlin_data = result(swift_result), result(kotlin_result)
    if swift_data != kotlin_data:
        raise RuntimeError("Swift and Kotlin native results differ")
    linkage = probes.run(
        "native-swift-linkage",
        ["otool", "-L", output / "swift-probe", output / "libCera.dylib", library],
    )
    if "inference_engine" in linkage or "LeapSDK.framework/LeapSDK" in linkage:
        raise RuntimeError("Unexpected deprecated or dynamic Leap runtime linkage")
    return {
        "validated": True,
        "scope": "macOS arm64 CPU raw bindings and candidate HiddenStates; not a ModelRunner facade",
        "fixture_sha256": digest(fixture),
        "library_sha256": digest(library),
        "binding_sha256": {
            str(path.relative_to(REPO)): digest(path)
            for path in (
                bindings / "swift/cera_ffi.swift",
                bindings / "swift/CeraFFI.h",
                bindings / "swift/CeraFFI.modulemap",
                kotlin_binding,
            )
        },
        "result": swift_data,
    }
