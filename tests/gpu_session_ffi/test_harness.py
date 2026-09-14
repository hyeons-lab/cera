"""Result validation must not turn missing or duplicate backend evidence into success."""

import json
import tempfile
import unittest
from pathlib import Path

from run import REPO, digest, expected_cases, result, source_inputs


class ResultTests(unittest.TestCase):
    def test_complete_matrix(self):
        cases = expected_cases()
        self.assertEqual(len(cases), 51)
        self.assertEqual(result(json.dumps({"cases": cases})), {"cases": cases})

    def test_reject_incomplete_or_forged_matrix(self):
        cases = expected_cases()
        records = [
            {"cases": cases[:-1]},
            {"cases": cases + [cases[0]]},
            {"cases": [case.replace("wgpu/", "cpu/") for case in cases]},
            {"cases": cases, "passed": False},
            {"cases": []},
            {"cases": {case: False for case in cases}},
            {"cases": None},
            {"cases": [1] + cases[1:]},
        ]
        for record in records:
            with self.subTest(record=record), self.assertRaises(RuntimeError):
                result(json.dumps(record))

    def test_reject_missing_or_multiple_reports(self):
        record = json.dumps({"cases": expected_cases()})
        for text in ("PASS", record + "\n" + record, "{}"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                result(text)


class SourceTests(unittest.TestCase):
    def test_native_build_inputs_are_fingerprinted(self):
        inputs = source_inputs()
        for name in (
            "cera/build.rs",
            "cera-ffi/build.rs",
            "cera/build_support/msl_postpass.rs",
        ):
            self.assertIn(REPO / name, inputs)
        for suffix in (".wgsl", ".metal", ".slang", ".spv", ".tmpl"):
            shaders = list((REPO / "cera/src/backend/shaders").rglob("*" + suffix))
            self.assertTrue(shaders, suffix)
            self.assertTrue(set(shaders).issubset(inputs), suffix)

    def test_shader_changes_and_additions_invalidate_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            shader = repo / "cera/src/backend/shaders/kernel.wgsl"
            shader.parent.mkdir(parents=True)
            shader.write_text("original shader")

            def snapshot():
                return {str(p): digest(p) for p in source_inputs(repo) if p.is_file()}

            original = snapshot()
            self.assertIn(str(shader), original)
            shader.write_text("modified shader")
            modified = snapshot()
            self.assertNotEqual(original, modified)
            added = shader.with_name("new-kernel.wgsl")
            added.write_text("new shader")
            self.assertNotEqual(modified, snapshot())
            shader.unlink()
            self.assertNotIn(str(shader), snapshot())


if __name__ == "__main__":
    unittest.main()
