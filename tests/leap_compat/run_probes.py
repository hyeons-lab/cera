"""Check Leap exports and optional isolated Cera native boundary probes."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def artifact(cache, entry, fetch):
    path = cache / entry["file"]
    if not path.exists():
        if not fetch:
            raise RuntimeError(
                f"Missing {path}; use --fetch to download public artifacts"
            )
        # A failed download must not become a reusable cache entry.
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            download = Path(temporary) / "download"
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "2",
                    "--max-time",
                    "180",
                    entry["url"],
                    "--output",
                    str(download),
                ],
                check=True,
            )
            if digest(download) != entry["sha256"]:
                raise RuntimeError(f"SHA256 mismatch for {entry['file']}")
            download.replace(path)
    if digest(path) != entry["sha256"]:
        raise RuntimeError(
            f"SHA256 mismatch for {path}; refusing to compile changed artifacts"
        )
    return path


class Probes:
    def __init__(self, output, environment):
        self.output = output
        self.environment = environment
        self.results = []

    def run(self, name, command, rejection=(), timeout=120, cwd=None):
        command = [str(arg) for arg in command]
        completed = subprocess.run(
            command,
            env=self.environment,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        diagnostic = completed.stdout + completed.stderr
        (self.output / f"{name}.log").write_text(diagnostic)
        expected = (
            completed.returncode == 1
            and all(re.search(pattern, diagnostic) for pattern in rejection)
            if rejection
            else completed.returncode == 0
        )
        outcome = "accepted" if not rejection else "rejected-as-expected"
        if not expected:
            outcome = "UNEXPECTED"
        self.results.append(
            {
                "name": name,
                "outcome": outcome,
                "command": command,
                "cwd": str(cwd) if cwd else None,
                "returncode": completed.returncode,
                "required_diagnostics": list(rejection),
            }
        )
        print(f"{name}: {outcome}", flush=True)
        if not expected:
            raise RuntimeError(
                f"Unexpected compiler result; see {self.output / (name + '.log')}"
            )
        return diagnostic.strip()


def swift_probes(probes, artifacts, temporary):
    command = [
        "xcrun",
        "swiftc",
        "-warnings-as-errors",
        "-swift-version",
        "5",
        "-target",
        "arm64-apple-macosx15.0",
        "-module-cache-path",
        temporary / "swift-cache",
    ]
    consumers = ROOT / "swift"
    core = [
        consumers / name
        for name in (
            "CoreConsumer.swift",
            "ConversationConsumer.swift",
            "PayloadConsumer.swift",
            "TypedStreamConsumer.swift",
            "ObjectiveCStreamConsumer.swift",
            "EnumConsumer.swift",
        )
    ]
    runners = [
        consumers / "RunnerConsumer.swift",
        consumers / "AsyncRunnerConsumer.swift",
    ]
    newer = consumers / "NewerConsumer.swift"
    for lane in ("swift-stable", "swift-snapshot"):
        # Always unpack the verified archive afresh; do not trust extracted caches.
        unpacked = temporary / lane
        with ZipFile(artifacts[lane]) as archive:
            archive.extractall(unpacked)
        framework = unpacked / "LeapSDK.xcframework/macos-arm64"
        check = command + ["-typecheck", "-F", framework]
        probes.run(f"{lane}-core", check + core)
        if lane == "swift-stable":
            probes.run(f"{lane}-runner", check + runners)
            probes.run(
                f"{lane}-newer-absent",
                check + [newer],
                (
                    r"cannot find 'LoraAdapterConfig' in scope",
                    r"cannot find type 'HiddenStates' in scope",
                ),
            )
        else:
            # Swift may stop after diagnosing the first invalid input file.
            # Check each conformer independently so neither failure is inferred.
            for runner in runners:
                probes.run(
                    f"{lane}-{runner.stem}-rejected",
                    check + [runner],
                    (
                        rf"type '{runner.stem}' does not conform to protocol 'ModelRunner'",
                        r"protocol requires function '__hiddenStates",
                        r"protocol requires function '__setLoraAdapters",
                    ),
                )
            probes.run(
                f"{lane}-defaults-rejected",
                check + [runners[0], consumers / "ProtocolDefaultsRejected.swift"],
                (
                    r"non-'@objc' method.*does not satisfy requirement of '@objc' protocol 'ModelRunner'",
                ),
            )
            probes.run(
                f"{lane}-newer",
                check
                + runners
                + [
                    newer,
                    consumers / "NewerRunnerMethods.swift",
                    consumers / "NewerAsyncRunnerMethods.swift",
                ],
            )

    for name in ("AsyncThrowingStream", "AsyncStream"):
        module = temporary / name
        module.mkdir()
        # These independent shape sketches import no reference SDK. The exact
        # same stream consumers are then compiled against each sketch.
        probes.run(
            f"candidate-{name}-module",
            command
            + [
                "-emit-module",
                "-module-name",
                "LeapSDK",
                ROOT / "candidates" / f"{name}Surface.swift",
                "-emit-module-path",
                module / "LeapSDK.swiftmodule",
            ],
        )
        check = command + ["-typecheck", "-I", module]
        if name == "AsyncThrowingStream":
            probes.run(
                f"candidate-{name}-failure-type",
                check
                + [
                    consumers / "TypedStreamConsumer.swift",
                ],
                (r"requires the types .*Never.*be equivalent",),
            )
        else:
            probes.run(
                f"candidate-{name}-nonthrowing",
                check
                + [
                    consumers / "TypedStreamConsumer.swift",
                ],
            )
        probes.run(
            f"candidate-{name}-objc-bridge",
            check
            + [
                consumers / "ObjectiveCStreamConsumer.swift",
            ],
            (r"has no member '_bridgeToObjectiveC'",),
        )


def kotlin_probes(probes, artifacts, temporary, compiler):
    command = [
        compiler,
        "-Werror",
        "-jvm-target",
        "21",
        "-classpath",
        os.pathsep.join(
            str(artifacts[name]) for name in ("kotlin-stable", "coroutines")
        ),
    ]
    probes.run(
        "kotlin-stable-core",
        command
        + [
            ROOT / "kotlin/CoreConsumer.kt",
            ROOT / "kotlin/ConversationConsumer.kt",
            ROOT / "kotlin/PayloadConsumer.kt",
            ROOT / "kotlin/TypedStreamConsumer.kt",
            "-d",
            temporary / "core.jar",
        ],
    )
    probes.run(
        "kotlin-stable-newer-absent",
        command
        + [
            ROOT / "kotlin/NewerConsumer.kt",
            "-d",
            temporary / "newer.jar",
        ],
        (
            r"unresolved reference ['\"]?HiddenStates",
            r"unresolved reference ['\"]?LoraAdapterConfig",
            r"unresolved reference ['\"]?setLoraAdapters",
            r"unresolved reference ['\"]?hiddenStates",
        ),
    )


def kmp_probes(probes, artifacts, temporary, compiler, fetch):
    project = ROOT / "candidates/kmp"
    command = [
        ROOT.parents[1] / "cera-ffi-kotlin/gradlew",
        "-p",
        project,
        "--no-daemon",
        "--console=plain",
    ]
    if not fetch:
        command += ["--offline"]
    outputs = {}
    for profile in ("stable", "extended"):
        name = f"candidate-kmp-{profile}"
        probes.run(
            name + "-build",
            command
            + [
                f"-PcompatibilityProfile={profile}",
                "linkDebugFrameworkMacosArm64",
                "jvmJar",
            ],
            timeout=600,
        )
        framework = project / f"build/{profile}/bin/macosArm64/debugFramework"
        jar = project / f"build/{profile}/libs/leap-export-probe-jvm.jar"
        swift = [
            "xcrun",
            "swiftc",
            "-typecheck",
            "-warnings-as-errors",
            "-swift-version",
            "5",
            "-target",
            "arm64-apple-macosx15.0",
            "-module-cache-path",
            temporary / (name + "-cache"),
            "-F",
            framework,
        ]
        fixtures = [
            ROOT / "swift" / filename
            for filename in (
                "CoreConsumer.swift",
                "ConversationConsumer.swift",
                "PayloadConsumer.swift",
                "RunnerConsumer.swift",
                "AsyncRunnerConsumer.swift",
                "TypedStreamConsumer.swift",
                "ObjectiveCStreamConsumer.swift",
                "EnumConsumer.swift",
            )
        ]
        if profile == "extended":
            fixtures += [
                ROOT / "swift" / filename
                for filename in (
                    "NewerConsumer.swift",
                    "NewerRunnerMethods.swift",
                    "NewerAsyncRunnerMethods.swift",
                )
            ]
            for runner in ("RunnerConsumer", "AsyncRunnerConsumer"):
                probes.run(
                    name + f"-{runner}-rejected",
                    swift + [ROOT / f"swift/{runner}.swift"],
                    (
                        rf"type '{runner}' does not conform to protocol 'ModelRunner'",
                        r"protocol requires function '__hiddenStates",
                        r"protocol requires function '__setLoraAdapters",
                    ),
                )
            probes.run(
                name + "-defaults-rejected",
                swift
                + [
                    ROOT / "swift/RunnerConsumer.swift",
                    ROOT / "swift/ProtocolDefaultsRejected.swift",
                ],
                (
                    r"non-'@objc' method.*does not satisfy requirement of '@objc' protocol 'ModelRunner'",
                ),
            )
        probes.run(name + "-swift", swift + fixtures)
        kotlin = [
            ROOT / "kotlin" / filename
            for filename in (
                "CoreConsumer.kt",
                "ConversationConsumer.kt",
                "PayloadConsumer.kt",
                "TypedStreamConsumer.kt",
            )
        ]
        if profile == "extended":
            kotlin += [ROOT / "kotlin/NewerConsumer.kt"]
        probes.run(
            name + "-kotlin",
            [
                compiler,
                "-Werror",
                "-jvm-target",
                "21",
                "-classpath",
                os.pathsep.join(str(path) for path in (jar, artifacts["coroutines"])),
                *kotlin,
                "-d",
                temporary / (name + ".jar"),
            ],
        )
        outputs[profile] = {
            "framework_sha256": digest(framework / "LeapSDK.framework/LeapSDK"),
            "header_sha256": digest(framework / "LeapSDK.framework/Headers/LeapSDK.h"),
            "jvm_sha256": digest(jar),
        }
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "build/artifacts")
    parser.add_argument(
        "--fetch", action="store_true", help="Download missing pinned public artifacts"
    )
    parser.add_argument("--platform", choices=("all", "swift", "kotlin"), default="all")
    parser.add_argument("--kotlinc", default="kotlinc")
    parser.add_argument(
        "--with-kmp",
        action="store_true",
        help="Build and check the limited KMP candidate",
    )
    parser.add_argument(
        "--with-native",
        action="store_true",
        help="Run CPU Cera boundary probes; requires --with-kmp",
    )
    parser.add_argument("--cera-target", type=Path, default=ROOT.parents[1] / "target")
    args = parser.parse_args()
    if args.with_native and not args.with_kmp:
        parser.error("--with-native requires --with-kmp")
    if args.with_kmp and args.platform != "all":
        parser.error("--with-kmp requires --platform all (macOS arm64 and JVM)")
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=ROOT / "build"))
    environment = os.environ.copy()
    # JVM startup echoes this variable verbatim; avoid leaking local settings
    # into compiler logs. These probes need no JVM network access.
    environment.pop("JAVA_TOOL_OPTIONS", None)
    probes = Probes(output, environment)
    manifest = json.loads((ROOT / "artifacts.json").read_text())
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "runtime_validated": False,
        "c0_complete": False,
        "artifacts": {},
        "tools": {},
        "fixture_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for pattern in (
                "swift/*.swift",
                "kotlin/*.kt",
                "candidates/*.swift",
                "candidates/kmp/*.kts",
                "candidates/kmp/src/**/*.kt",
                "candidates/kmp/src/**/*.swift",
                "native/*",
                "*.py",
                "artifacts.json",
            )
            for path in sorted(ROOT.glob(pattern))
        },
        "kmp_candidate": None,
        "native_boundary": None,
        "checks": probes.results,
        "success": False,
    }
    try:
        names = []
        if args.platform in ("all", "swift"):
            report["tools"]["swift"] = probes.run(
                "swift-version", ["xcrun", "swiftc", "--version"]
            )
            names += ["swift-stable", "swift-snapshot"]
        if args.platform in ("all", "kotlin"):
            if not shutil.which(args.kotlinc):
                raise RuntimeError(
                    "kotlinc not found; install Kotlin 2.4.0 or pass --kotlinc"
                )
            report["tools"]["kotlin"] = probes.run(
                "kotlin-version", [args.kotlinc, "-version"]
            )
            names += ["kotlin-stable", "coroutines"]
        if args.with_native:
            names += ["jna"]
        artifacts = {
            name: artifact(cache, manifest[name], args.fetch) for name in names
        }
        report["artifacts"] = {name: manifest[name] for name in names}
        with tempfile.TemporaryDirectory(prefix="compile-", dir=output) as directory:
            temporary = Path(directory)
            if args.platform in ("all", "swift"):
                swift_probes(probes, artifacts, temporary)
            if args.platform in ("all", "kotlin"):
                kotlin_probes(probes, artifacts, temporary, args.kotlinc)
            if args.with_kmp:
                report["kmp_candidate"] = kmp_probes(
                    probes, artifacts, temporary, args.kotlinc, args.fetch
                )
            if args.with_native:
                from run_native import native_probes

                report["native_boundary"] = native_probes(
                    probes,
                    artifacts,
                    temporary,
                    args.kotlinc,
                    args.cera_target.resolve(),
                    digest,
                )
        report["success"] = True
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(error, flush=True)
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence: {output / 'results.json'}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    (ROOT / "build").mkdir(exist_ok=True)
    raise SystemExit(main())
