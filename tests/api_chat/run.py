"""Verify the pinned public tokenizer and execute every chat contract fixture."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

# Exact case counts per mode. Adding or removing a fixture must update these so
# a silently dropped test cannot pass as complete evidence.
ISOLATED_CASES = 15
CORE_CASES = 43

sys.path.insert(0, str(ROOT.parent / "api_loading"))
from commands import Commands, cargo_artifact  # noqa: E402
from prepare import inventory  # noqa: E402
from run import probe_environment  # noqa: E402


def source_hashes():
    files = (
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml",
        "cera/Cargo.toml", "cera/build.rs", "cera/tests/chat_contract.rs",
    )
    hashes = {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in files}
    directories = (
        ROOT, REPO / "cera/src", REPO / "cera/build_support",
        REPO / "cera/tests/api_chat", REPO / "cera/schema",
        ROOT.parent / "api_loading",
    )
    for directory in directories:
        for relative, digest in inventory(
            directory, excluded=("build", "target", "__pycache__")
        ).items():
            hashes[(directory / relative).relative_to(REPO).as_posix()] = digest
    return hashes


def require_executed_tests(output, *, core_transactions=False):
    """An exit status alone cannot prove a harness actually ran the fixtures."""
    prefix = "session::chat::contract_tests::" if core_transactions else "tests::"
    required = [f"test {prefix}public_lfm2_tokenizer_boundary_and_ten_turns ... ok"]
    if core_transactions:
        # The lib binary is filtered to session::chat::, which must include the
        # actual-Session public case, shared contract, and real model R1 proofs.
        required.extend([
            "test session::chat::tests::public_tokenizer_actual_session_ten_turns ... ok",
            "test session::chat::tests::real_model_r1_ten_warm_turns_and_kv_retention ... ok",
            "test session::chat::tests::real_model_r1_stochastic_rng_determinism_and_divergence ... ok",
            "test session::chat::tests::real_model_r1_interrupted_turn_and_replacement_recovery ... ok",
        ])
    summary = re.search(
        r"test result: ok\. (\d+) passed; 0 failed; 0 ignored; 0 measured; (\d+) filtered out;",
        output,
    )
    expected = CORE_CASES if core_transactions else ISOLATED_CASES
    complete = (
        summary is not None
        and all(line in output for line in required)
        and int(summary[1]) == expected
        # The isolated binary holds nothing else, so a filter would hide cases.
        and (core_transactions or int(summary[2]) == 0)
    )
    if not complete:
        raise RuntimeError("Missing positive evidence that all chat fixtures executed")
    return expected


def file_digest_sha256(path):
    with path.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        h = hashlib.sha256()
        while chunk := stream.read(65536):
            h.update(chunk)
        return h.hexdigest()


def select_profile(pin_data, model_hash):
    pins = (
        pin_data["profiles"]
        if isinstance(pin_data, dict) and "profiles" in pin_data
        else (pin_data if isinstance(pin_data, list) else [pin_data])
    )
    matched_pin = next(
        (p for p in pins if isinstance(p, dict) and p.get("sha256") == model_hash),
        None,
    )
    if matched_pin is None:
        expected = ", ".join(
            p.get("sha256", "<missing>") for p in pins if isinstance(p, dict)
        )
        raise ValueError(f"Model hash mismatch: expected one of [{expected}], got {model_hash}")
    return matched_pin


def execution_scope(pin, *, core_transactions):
    if core_transactions:
        return pin["runtime_validation"]
    return (
        "Production tokenizer and isolated chat contract; "
        "no runtime Session or warm KV performance claim"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--no-default-features", action="store_true")
    parser.add_argument(
        "--core-transactions",
        action="store_true",
        help="run the actual Session transaction adapter and shared contract",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output.resolve()))
    model = args.model.resolve(strict=True)
    model_hash = file_digest_sha256(model)
    pin_data = json.loads((ROOT / "profile.json").read_text(encoding="utf-8"))
    pin = select_profile(pin_data, model_hash)
    env = probe_environment(os.environ)
    env.update(
        CERA_CHAT_PROFILE_MODEL=str(model),
        CARGO_TARGET_DIR=str(args.target_dir.resolve()), CARGO_INCREMENTAL="0",
    )
    commands = Commands(output, env, REPO)
    hashes = source_hashes()
    result = {
        "status": "incomplete",
        "scope": execution_scope(pin, core_transactions=args.core_transactions),
        "model_sha256": model_hash, "source_sha256": hashes,
        "commands": commands.results,
        "core_transactions": args.core_transactions,
    }
    print(output, flush=True)
    try:
        command = [
            "cargo", "test", "--offline", "--locked", "-p", "cera",
            "--target", args.target,
        ]
        command += ["--lib"] if args.core_transactions else ["--test", "chat_contract"]
        if args.no_default_features:
            command.append("--no-default-features")
        build = commands.run(
            "build", command + ["--no-run", "--message-format=json"], timeout=1200
        )
        source = "cera/src/lib.rs" if args.core_transactions else "cera/tests/chat_contract.rs"
        artifact = cargo_artifact(build, REPO / source, executable=True)
        artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        result["artifact"] = {"path": str(artifact), "sha256": artifact_hash}
        # Invoke Cargo's exact emitted binary directly. An inherited Cargo runner
        # must not replace native test execution with an arbitrary successful tool.
        test_filter = ["session::chat::"] if args.core_transactions else []
        output_text = commands.run(
            "core-transactions" if args.core_transactions else "chat-contract",
            [
                artifact, *test_filter, "--include-ignored", "--nocapture",
                "--color", "never", "--test-threads=1",
            ],
            timeout=1200,
        )
        result["tests_passed"] = require_executed_tests(
            output_text, core_transactions=args.core_transactions
        )
        if source_hashes() != hashes:
            raise RuntimeError("Source changed during validation")
        if file_digest_sha256(model) != model_hash:
            raise RuntimeError("Model changed during validation")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != artifact_hash:
            raise RuntimeError("Test artifact changed during validation")
        result["status"] = "complete"
    finally:
        (output / "results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
