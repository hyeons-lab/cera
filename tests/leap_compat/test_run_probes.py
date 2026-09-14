"""Check that artifact corruption and unrelated compiler failures cannot pass C0."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_probes import Probes, artifact


class ProbeIntegrityTests(unittest.TestCase):
    def test_changed_cached_artifact_is_rejected_even_with_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            entry = {
                "file": "sdk.jar",
                "sha256": hashlib.sha256(b"original").hexdigest(),
            }
            (cache / entry["file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                artifact(cache, entry, fetch=True)
            self.assertEqual((cache / entry["file"]).read_bytes(), b"changed")

    def test_failed_download_does_not_populate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            entry = {
                "file": "sdk.jar",
                "url": "https://example.invalid/sdk.jar",
                "sha256": "unused",
            }

            def interrupted_download(command, **kwargs):
                Path(command[-1]).write_bytes(b"partial")
                raise subprocess.CalledProcessError(7, command)

            with (
                patch("run_probes.subprocess.run", side_effect=interrupted_download),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                artifact(cache, entry, fetch=True)
            self.assertEqual(list(cache.iterdir()), [])

    def test_expected_rejection_requires_diagnostic_and_normal_failure(self):
        for code, text, accepted in (
            (1, "missing export", True),
            (1, "bad SDK", False),
            (2, "missing export", False),
        ):
            with (
                self.subTest(code=code, text=text),
                tempfile.TemporaryDirectory() as directory,
            ):
                probes = Probes(Path(directory), {})
                command = [
                    sys.executable,
                    "-c",
                    f"print({text!r}); raise SystemExit({code})",
                ]
                if accepted:
                    probes.run("negative", command, ("missing export",))
                    self.assertEqual(
                        probes.results[-1]["outcome"], "rejected-as-expected"
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "Unexpected compiler result"
                    ):
                        probes.run("negative", command, ("missing export",))
                    self.assertEqual(probes.results[-1]["outcome"], "UNEXPECTED")


if __name__ == "__main__":
    unittest.main()
