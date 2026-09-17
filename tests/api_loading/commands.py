"""Bounded commands and strict negative-control diagnostics for loading probes."""

import json
import os
import signal
import subprocess
from pathlib import Path


def cargo_artifact(stdout, source, *, suffix=None, executable=False):
    """Use the artifact emitted for this source, including Cargo target overrides."""
    paths = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(item, dict)
            or item.get("reason") != "compiler-artifact"
            or item.get("target", {}).get("src_path") != str(source)
        ):
            continue
        candidates = (
            [item.get("executable")] if executable else item.get("filenames", [])
        )
        paths.extend(
            Path(p) for p in candidates if p and (suffix is None or p.endswith(suffix))
        )
    if len(paths) != 1 or not paths[0].is_absolute() or not paths[0].is_file():
        raise RuntimeError(f"Expected one existing Cargo artifact for {source}")
    return paths[0]


def exhaustive_rejection(returncode, stdout):
    errors = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(item, dict)
            and item.get("reason") == "compiler-message"
            and item["message"]["level"] == "error"
        ):
            errors.append(item["message"])
    return (
        returncode == 101
        and len(errors) == 1
        and (errors[0].get("code") or {}).get("code") == "E0004"
        and "non-exhaustive" in errors[0]["message"]
        and any(
            s["is_primary"] and s["file_name"].endswith("exhaustive.rs")
            for s in errors[0]["spans"]
        )
    )


class Commands:
    def __init__(self, output, environment, workspace):
        self.output = output
        self.environment = environment
        self.workspace = workspace
        self.results = []

    def run(
        self, name, command, *, timeout=600, exhaustive=False, cwd=None, combined=False
    ):
        command = [str(v) for v in command]
        record = {
            "name": name,
            "command": command,
            "cwd": str(cwd or self.workspace),
            "status": "running",
        }
        self.results.append(record)
        print(f"{name}: running", flush=True)
        with (
            (self.output / f"{name}.stdout").open("w+") as stdout,
            (self.output / f"{name}.stderr").open("w+") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd or self.workspace,
                    env=self.environment,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError:
                record["status"] = "launch-failed"
                raise
            try:
                returncode = process.wait(timeout=timeout)
            except BaseException:
                record["status"] = "interrupted-or-timeout"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    # The leader may exit while a descendant ignores SIGTERM.
                    # Escalate for the entire group regardless of the leader's exit.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)
                raise
            stdout.seek(0)
            out = stdout.read()
            stderr.seek(0)
            diagnostic = out + stderr.read()
        expected = (
            exhaustive_rejection(returncode, out) if exhaustive else returncode == 0
        )
        record.update(
            returncode=returncode,
            status="expected-rejection"
            if expected and exhaustive
            else "passed"
            if expected
            else "FAILED",
        )
        print(f"{name}: {record['status']}", flush=True)
        if not expected:
            raise RuntimeError(f"Unexpected result: {name}; see {self.output}")
        return (diagnostic if combined else out).strip()
