"""Run recovery contracts and public examples against generated native bindings."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/api_loading"))
from commands import Commands  # noqa: E402
from native_runtime import run_consumers, stage_library  # noqa: E402
from prepare import digest  # noqa: E402
from run import probe_environment  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "metal", "wgpu"), default="cpu")
    parser.add_argument("--dependencies", type=Path, default=Path("/private/tmp/cera-leap-api-baseline"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"{args.backend}-", dir=args.output))
    env = probe_environment(os.environ)
    env["PATH"] = "/opt/homebrew/bin:" + env["PATH"]
    env["JAVA_HOME"] = str(Path.home() / ".sdkman/candidates/java/21.0.9-zulu")
    commands = Commands(out, env, REPO)
    bindings = REPO / "cera-ffi/bindings"
    pins = json.loads((REPO / "tests/leap_compat/artifacts.json").read_text())
    deps = [args.dependencies / pins[k]["file"] for k in ("jna", "coroutines")]
    for key, dependency in zip(("jna", "coroutines"), deps):
        if digest(dependency) != pins[key]["sha256"]:
            raise RuntimeError(f"Dependency digest mismatch: {key}")
    sources = [args.library, args.model, *bindings.rglob("*"), *Path(__file__).parent.glob("*"),
               REPO / "cera-ffi/examples/IngestionRecovery.swift", REPO / "cera-ffi/examples/IngestionRecovery.kt"]
    source_hashes = {str(p): digest(p) for p in sources if p.is_file()}
    report = {"status": "incomplete", "backend": args.backend, "inputs_sha256": source_hashes}
    try:
        native = stage_library(commands, args.library, out)
        runtime = run_consumers(commands, bindings, out, native,
                                REPO / "tests/api_recovery/RecoveryProbe.swift",
                                REPO / "tests/api_recovery/RecoveryProbe.kt", "RecoveryProbeKt", deps,
                                [args.model, args.backend])
        expected = f"passed 9 recovery cases: {args.backend}"
        if runtime["swift"] != expected or runtime["kotlin"] != expected:
            raise RuntimeError(f"Unexpected consumer result: {runtime}")
        report["runtime"] = runtime
        # Public examples select CPU explicitly. GPU contract probes run separately.
        if args.backend == "cpu":
            commands.run("swift-example-build", ["xcrun", "swiftc", "-swift-version", "5", "-warnings-as-errors",
                         "-parse-as-library", "-target", "arm64-apple-macosx15.0", "-module-cache-path", out / "swift-cache",
                         "-I", out / "include", "-I", out, "-L", out, "-lCera", "-Xlinker", "-rpath", "-Xlinker", out,
                         REPO / "cera-ffi/examples/IngestionRecovery.swift", "-o", out / "swift-example"])
            classpath = os.pathsep.join(map(str, [out / "consumer-probe.jar", *deps]))
            commands.run("kotlin-example-build", ["kotlinc", "-jvm-target", "21", "-classpath", classpath,
                         REPO / "cera-ffi/examples/IngestionRecovery.kt", "-d", out / "example.jar"])
            examples = {}
            for compressed in (False, True):
                arguments = [args.model, "ab", "baba", *(["--compressed"] if compressed else [])]
                mode = "compressed" if compressed else "standard"
                swift = commands.run(f"swift-example-{mode}", [out / "swift-example", *arguments])
                kotlin = commands.run(f"kotlin-example-{mode}", [Path(env["JAVA_HOME"]) / "bin/java",
                            f"-Duniffi.component.cera_ffi.libraryOverride={native}", "-classpath",
                            os.pathsep.join([str(out / "example.jar"), classpath]), "IngestionRecoveryKt", *arguments])
                expected = f"recovery: {'reset' if compressed else 'restored'}; position: {0 if compressed else 2}\nretry succeeded; position: 6"
                if swift.lower() != expected or kotlin.lower() != expected:
                    raise RuntimeError(f"Unexpected example output: {swift!r} / {kotlin!r}")
                examples[mode] = {"swift": swift, "kotlin": kotlin}
            report["examples"] = examples
        if source_hashes != {str(p): digest(p) for p in sources if p.is_file()}:
            raise RuntimeError("Consumer inputs changed during execution")
        report.update(status="passed", artifacts_sha256={str(p): digest(p) for p in out.iterdir() if p.is_file()})
    finally:
        report["commands"] = commands.results
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(out, flush=True)


if __name__ == "__main__":
    main()
