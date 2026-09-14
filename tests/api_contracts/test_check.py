"""Mutation controls against reviewed source declarations, without editing the repo."""

import json
import tempfile
import unittest
from pathlib import Path

import check


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(check.BASELINE.read_text())

    def changed(self, source, before, after):
        original = (check.ROOT / source).read_text()
        self.assertEqual(original.count(before), 1)
        baseline = {
            "surfaces": [r for r in self.baseline["surfaces"] if r["source"] == source]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / source
            path.parent.mkdir(parents=True)
            path.write_text(original.replace(before, after, 1))
            return check.check(root, baseline)

    def test_current_declarations_match(self):
        self.assertEqual(check.check(check.ROOT, self.baseline), [])

    def test_removal_rename_and_new_method_are_visible(self):
        for after, expected in (
            ("    async fn from_parts_async(", "missing"),
            ("    pub async fn renamed_parts_async(", "added"),
            (
                "    pub fn new_loading_operation(&self) {}\n    pub async fn from_parts_async(",
                "added",
            ),
        ):
            with self.subTest(after=after):
                errors = self.changed(
                    "cera-ffi/src/lib.rs", "    pub async fn from_parts_async(", after
                )
                self.assertTrue(
                    any("CeraEngine" in e and expected in e for e in errors), errors
                )

    def test_async_and_payload_changes_are_visible(self):
        errors = self.changed(
            "cera-ffi/src/lib.rs",
            "    pub async fn from_parts_async(",
            "    pub fn from_parts_async(",
        )
        self.assertTrue(
            any("from_parts_async" in e and "changed" in e for e in errors), errors
        )
        errors = self.changed(
            "cera-ffi/src/lib.rs", "pub context_size: u64,", "pub context_size: u32,"
        )
        self.assertTrue(
            any("EngineConfig" in e and "record fields" in e for e in errors), errors
        )

    def test_foreign_callback_payload_is_covered(self):
        errors = self.changed(
            "cera-ffi/src/lib.rs",
            "fn on_progress(&self, url: String, bytes_downloaded: u64, total_bytes: Option<u64>);",
            "fn on_progress(&self, url: String, bytes_downloaded: u32, total_bytes: Option<u64>);",
        )
        self.assertTrue(
            any("DownloadProgressSink" in e and "record fields" in e for e in errors),
            errors,
        )

    def test_block_commented_native_method_is_missing(self):
        method = """    pub fn clear_prefix_cache(&self) {
        self.inner.clear_warm_cache();
    }"""
        errors = self.changed("cera-ffi/src/lib.rs", method, "/*\n" + method + "\n*/")
        self.assertTrue(
            any("clear_prefix_cache" in e and "missing" in e for e in errors), errors
        )

    def test_comments_and_literals_cannot_supply_declarations(self):
        source = r"""/* nested /* comment */
impl Engine {
    pub fn commented(&self) {}
}
pub struct Hidden {
    pub value: u64,
}
pub fn commented_function() {}
*/
// pub fn line_comment() {}
impl Engine {
    pub fn live<'a>(&'a self) -> &'a str {
        let escaped = "\" /* still a string */";
        let quote = '\'';
        let byte = b'"';
        let value = br##"
}
impl Engine {
    pub fn fake(&self) {}
}
pub fn fake_function() {}
pub struct Hidden {
    pub value: u64,
}
// a literal line, not a comment
"##;
        "https://example.com/*literal*/"
    }
}
pub fn helper() {}
pub struct Options {
    #[uniffi(default = "/*default*/ //value")]
    pub value: String,
}
"""
        self.assertEqual(
            check.methods(source, "Engine"),
            {"live": "pub fn live<'a>(&'a self) -> &'a str"},
        )
        self.assertEqual(check.functions(source), {"helper": "pub fn helper()"})
        with self.assertRaisesRegex(ValueError, "Expected one record"):
            check.record(source, "Hidden")
        declaration = check.record(source, "Options")
        self.assertIn('default = "/*default*/ //value"', declaration)
        self.assertNotEqual(
            declaration, check.record(source.replace("//value", "//changed"), "Options")
        )

    def test_unclosed_lexical_context_fails(self):
        for source in ("/* comment", 'r#"raw', '"string'):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "Unclosed"),
            ):
                check.functions(source)

    def test_literal_whitespace_changes_are_visible(self):
        errors = self.changed(
            "cera-ffi/src/lib.rs",
            '"modality not supported by this model"',
            '"modality\nnot supported by this model"',
        )
        self.assertTrue(any("FfiError" in e for e in errors), errors)
        for value in ('"a b"', 'r#"a b"#'):
            with self.subTest(value=value):
                source = (
                    "pub struct Options {\n"
                    f"    #[uniffi(default = {value})]\n"
                    "    pub value: String,\n}"
                )
                original = check.record(source, "Options")
                self.assertNotEqual(
                    original, check.record(source.replace("a b", "a  b"), "Options")
                )
                self.assertEqual(
                    original,
                    check.record(
                        source.replace("pub value:", "pub   value:"), "Options"
                    ),
                )

    def test_nested_webgpu_factory_is_covered(self):
        errors = self.changed(
            "cera-wasm/src/lib.rs",
            "        pub async fn create_with_parts(",
            "        async fn create_with_parts(",
        )
        self.assertTrue(
            any("WebGpuSession" in e and "create_with_parts" in e for e in errors),
            errors,
        )

    def test_missing_record_and_duplicate_rows_fail(self):
        errors = self.changed(
            "cera/src/engine.rs", "pub struct ModelFiles {", "struct ModelFiles {"
        )
        self.assertTrue(any("Expected one record" in e for e in errors), errors)
        with self.assertRaisesRegex(ValueError, "Duplicate inventory row"):
            check.check(check.ROOT, {"surfaces": [self.baseline["surfaces"][0]] * 2})


if __name__ == "__main__":
    unittest.main()
