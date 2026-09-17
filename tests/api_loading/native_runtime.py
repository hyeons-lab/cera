"""Compile and execute native Swift/Kotlin consumers against an exact staged library."""

import os
import shutil
from pathlib import Path


def stage_library(commands, source, output):
    native = output / "libcera_ffi.dylib"
    shutil.copyfile(source, native)
    # Cargo dylibs can carry an absolute install name pointing back into target/.
    # Bind the consumer to this staged, hashed copy instead.
    commands.run(
        "native-install-name",
        [
            "install_name_tool",
            "-id",
            "@rpath/libcera_ffi.dylib",
            native,
        ],
    )
    commands.run("native-sign", ["codesign", "--force", "--sign", "-", native])
    return native


def run_consumers(
    commands,
    bindings,
    output,
    native,
    swift_source,
    kotlin_source,
    kotlin_main,
    dependencies,
    arguments,
    *,
    jar_name="consumer-probe.jar",
):
    env = commands.environment
    results = {}
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
    commands.run(
        "swift-bindings",
        swift
        + [
            "-emit-library",
            "-emit-module",
            "-module-name",
            "Cera",
            "-L",
            output,
            "-lcera_ffi",
            bindings / "swift/cera_ffi.swift",
            "-o",
            output / "libCera.dylib",
            "-emit-module-path",
            output / "Cera.swiftmodule",
        ],
    )
    linkage = commands.run("swift-linkage", ["otool", "-L", output / "libCera.dylib"])
    ffi_links = [
        line.strip() for line in linkage.splitlines() if "libcera_ffi.dylib" in line
    ]
    if len(ffi_links) != 1 or not ffi_links[0].startswith("@rpath/libcera_ffi.dylib ("):
        raise RuntimeError("Swift wrapper does not link the staged native library")
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
            "-lCera",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            output,
            swift_source,
            "-o",
            output / "swift-probe",
        ],
    )
    env["DYLD_PRINT_LIBRARIES"] = "1"
    try:
        results["swift"] = commands.run(
            "swift-run", [output / "swift-probe", *arguments]
        )
    finally:
        del env["DYLD_PRINT_LIBRARIES"]
    loaded = [
        line
        for line in (output / "swift-run.stderr").read_text().splitlines()
        if line.rstrip().endswith("/libcera_ffi.dylib")
    ]
    if len(loaded) != 1 or not loaded[0].rstrip().endswith(str(native)):
        raise RuntimeError("Swift did not load exactly the staged native library")
    results["native"] = str(native)
    jar = output / jar_name
    commands.run(
        "kotlin-compile",
        [
            "kotlinc",
            "-jvm-target",
            "21",
            "-classpath",
            os.pathsep.join(map(str, dependencies)),
            bindings / "kotlin/uniffi/cera_ffi/cera_ffi.kt",
            kotlin_source,
            "-include-runtime",
            "-d",
            jar,
        ],
    )
    results["kotlin"] = commands.run(
        "kotlin-run",
        [
            Path(env["JAVA_HOME"]) / "bin/java",
            f"-Duniffi.component.cera_ffi.libraryOverride={native}",
            "-classpath",
            os.pathsep.join(map(str, [jar, *dependencies])),
            kotlin_main,
            *arguments,
        ],
    )
    return results
