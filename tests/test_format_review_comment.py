#!/usr/bin/env python3
"""Unit tests for scripts/format_review_comment.py."""

from __future__ import annotations

import unittest
import json
import sys
import os
import datetime
import subprocess
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts.format_review_comment as formatter


class TestFormatReviewComment(unittest.TestCase):
    def test_unescape_json_string_all_escape_sequences(self):
        escaped = r'Slash: \/ Backspace: \b Formfeed: \f Newline: \n Tab: \t Quote: \" Backslash: \\ Unicode: \u2602'
        unescaped = formatter.unescape_json_string(escaped)
        self.assertIn("Slash: /", unescaped)
        self.assertIn("Backspace: \b", unescaped)
        self.assertIn("Formfeed: \f", unescaped)
        self.assertIn("Newline: \n", unescaped)
        self.assertIn("Tab: \t", unescaped)
        self.assertIn('Quote: "', unescaped)
        self.assertIn("Backslash: \\", unescaped)
        self.assertIn("Unicode: ☂", unescaped)

    def test_unescape_json_string_escaped_backslash_not_cascading(self):
        # Escaped backslash before unicode literal (e.g. \\u2602 should unescape to \u2602 and not ☂)
        escaped = r'Literal: \\u2602'
        unescaped = formatter.unescape_json_string(escaped)
        self.assertEqual(unescaped, r'Literal: \u2602')

    def test_unescape_json_string_surrogate_pairs(self):
        # UTF-16 surrogate pair for Saturn emoji 🪐 (\uD83E\uDE90)
        escaped = r'Planet: \uD83E\uDE90'
        unescaped = formatter.unescape_json_string(escaped)
        self.assertIn("🪐", unescaped)

    def test_extract_valid_json_response(self):
        raw = json.dumps({"response": "### 1. Executive Summary\nLooks good!"})
        self.assertEqual(
            formatter.extract_review_text(raw),
            "### 1. Executive Summary\nLooks good!",
        )

    def test_extract_empty_or_whitespace_json_response(self):
        self.assertEqual(formatter.extract_review_text('{"response": ""}'), "")
        self.assertEqual(formatter.extract_review_text('{"response": "   "}'), "")
        self.assertEqual(formatter.extract_review_text('{"response": null}'), "")

    def test_extract_markdown_wrapped_json(self):
        raw_wrapped = '```json\n{"response": "### 1. Executive Summary\\nWrapped in code block."}\n```'
        self.assertEqual(
            formatter.extract_review_text(raw_wrapped),
            "### 1. Executive Summary\nWrapped in code block.",
        )

    def test_extract_markdown_code_fence_wrapping(self):
        raw_md = '```markdown\n### 1. Executive Summary\nWrapped in markdown fence.\n```'
        self.assertEqual(
            formatter.extract_review_text(raw_md),
            "### 1. Executive Summary\nWrapped in markdown fence.",
        )

    def test_extract_does_not_strip_internal_language_code_fences(self):
        # A markdown document containing language code blocks starting with comments
        raw_with_code = "```python\n# Header comment\ndef foo(): pass\n```\nSome discussion\n```rust\n// Rust code\n```"
        self.assertEqual(formatter.extract_review_text(raw_with_code), raw_with_code)

    def test_extract_candidates_structure(self):
        raw = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Thinking steps..."},
                            {"thought": False, "text": "### 1. Summary\nAll tests pass."},
                        ]
                    }
                }
            ]
        })
        self.assertEqual(
            formatter.extract_review_text(raw),
            "### 1. Summary\nAll tests pass.",
        )

    def test_extract_candidates_content_none(self):
        raw = json.dumps({
            "candidates": [
                {"content": None}
            ]
        })
        self.assertEqual(formatter.extract_review_text(raw), "")

    def test_extract_error_envelope(self):
        raw = json.dumps({
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED"
            }
        })
        self.assertEqual(formatter.extract_review_text(raw), "")

    def test_extract_candidates_part_text_none(self):
        raw = json.dumps({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": False, "text": None},
                            {"thought": False, "text": "### 1. Summary\nValid text."},
                        ]
                    }
                }
            ]
        })
        self.assertEqual(
            formatter.extract_review_text(raw),
            "### 1. Summary\nValid text.",
        )

    def test_extract_unescaped_control_characters(self):
        # Raw string with unescaped literal newline
        raw = '{\n  "response": "### 1. Summary\nLine 1\nLine 2"\n}'
        self.assertEqual(
            formatter.extract_review_text(raw),
            "### 1. Summary\nLine 1\nLine 2",
        )

    def test_extract_regex_fallback_malformed_json(self):
        # Malformed JSON (invalid syntax that fails json.loads, e.g. unquoted trailing keys or invalid tokens)
        malformed = '{"session_id": "123", "response": "### 1. Summary 🪐\\nLine with emoji 🚨\\nCode snippet: foo(\\"bar\\", 1)", invalid_unquoted_key: 123'
        extracted = formatter.extract_review_text(malformed)
        self.assertIn("🪐", extracted)
        self.assertIn("🚨", extracted)
        self.assertIn('foo("bar", 1)', extracted)

    def test_extract_raw_markdown_containing_response_keyword(self):
        raw = '### 1. Executive Summary\nHere is an API payload: {"response": "test"}\nEnd of review.'
        self.assertEqual(
            formatter.extract_review_text(raw),
            '### 1. Executive Summary\nHere is an API payload: {"response": "test"}\nEnd of review.',
        )

    def test_resolve_head_sha_stringified_nulls(self):
        try:
            git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_sha = ""
        self.assertEqual(formatter.resolve_head_sha("null"), git_sha)
        self.assertEqual(formatter.resolve_head_sha("undefined"), git_sha)
        self.assertEqual(formatter.resolve_head_sha("none"), git_sha)
        self.assertEqual(formatter.resolve_head_sha("  "), git_sha)
        self.assertEqual(formatter.resolve_head_sha("abc1234567"), "abc1234567")

    def test_build_comment_body_header_and_metadata(self):
        fixed_time = datetime.datetime(2026, 8, 18, 12, 0, 0, tzinfo=datetime.timezone.utc)
        body = formatter.build_comment_body(
            "### 1. Summary\nVerified.",
            "high",
            "hyeons-lab/cera",
            "1234567890abcdef",
            now_utc=fixed_time,
        )
        self.assertIn("<!-- antigravity-code-review -->", body)
        self.assertIn("## 🪐 Antigravity Code Review", body)
        self.assertIn("[`1234567`](https://github.com/hyeons-lab/cera/commit/1234567890abcdef)", body)
        self.assertIn("high effort", body)
        self.assertIn("12:00:00 UTC", body)
        self.assertIn("05:00:00 AM PDT", body)
        self.assertIn("### 1. Summary\nVerified.", body)

    def test_build_comment_body_strips_various_headers_completely(self):
        headers = [
            "## 🪐 Antigravity Code Review - PR #391\n\n### 1. Summary\nContent",
            "# Antigravity Code Review\n### 1. Summary\nContent",
            "🪐 antigravity code review - Deep Reasoning Pass\n### 1. Summary\nContent",
            "# ANTIGRAVITY CODE REVIEW\n### 1. Summary\nContent",
        ]
        for raw_review in headers:
            body = formatter.build_comment_body(
                raw_review,
                "max",
                "hyeons-lab/cera",
                "1234567",
            )
            self.assertEqual(body.count("## 🪐 Antigravity Code Review"), 1)
            self.assertNotIn("- PR #391", body)
            self.assertIn("### 1. Summary\nContent", body)

    def test_find_comment_id_in_items_defensive_checks(self):
        comments_payload = [
            {"id": None, "body": None},
            {"id": "None", "body": "<!-- antigravity-code-review -->\nOld bad comment"},
            {"id": 111, "body": None},
            {"id": 222, "body": "<!-- antigravity-code-review -->\n## 🪐 Antigravity Code Review\nVerified."},
            {"id": 333, "body": "Standard comment"},
            {"id": 444, "body": "<!-- antigravity-code-review -->\nLatest comment"},
        ]
        # Reversed iteration should pick the latest valid matching ID (444)
        found_id = formatter.find_comment_id_in_items(comments_payload)
        self.assertEqual(found_id, "444")

    def test_find_existing_comment_id_multi_page(self):
        page1 = json.dumps([{"id": 101, "body": "first comment"}])
        page2 = json.dumps([{"id": 202, "body": "<!-- antigravity-code-review -->\nReview"}])
        concatenated = f"{page1}\n{page2}"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=concatenated)
            result = formatter.find_existing_comment_id("hyeons-lab/cera", "391")
            self.assertEqual(result, "202")

    def test_truncation_limit_with_code_fence_closure(self):
        # Open code fence that would be cut in the middle
        huge_text = "```rust\nlet x = 1;\n" + ("A" * 70000)
        body = formatter.build_comment_body(
            huge_text,
            "high",
            "hyeons-lab/cera",
            "1234567",
        )
        self.assertLessEqual(len(body), 65000)
        self.assertIn("Review truncated due to GitHub character limit", body)
        # Even number of ``` fences
        self.assertEqual(body.count("```") % 2, 0)

    def test_dynamic_timezone_offsets_naive_and_aware(self):
        # Test summer date (PDT, UTC-7) with tzinfo
        summer_utc = datetime.datetime(2026, 7, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        ts_summer = formatter.get_formatted_timestamps(summer_utc)
        self.assertIn("PDT", ts_summer)
        self.assertIn("12:00:00 UTC", ts_summer)

        # Test winter date (PST, UTC-8) with naive datetime
        winter_naive = datetime.datetime(2026, 1, 15, 12, 0, 0)
        ts_winter = formatter.get_formatted_timestamps(winter_naive)
        self.assertIn("PST", ts_winter)
        self.assertIn("12:00:00 UTC", ts_winter)


if __name__ == "__main__":
    unittest.main()
