#!/usr/bin/env python3
"""Unit tests for scripts/gsm8k_eval.py answer extraction and scoring."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts.gsm8k_eval as gsm8k


class TestGsm8kEval(unittest.TestCase):
    def test_extract_answer(self) -> None:
        cases = [
            ("She earns 9 * 2 = 18 dollars.\n\\boxed{18}", "18"),
            ("The total is \\boxed{1,250} and 3 boxes remain.", "1250"),
            ("5 + 7 = 12 apples.\n#### 12", "12"),
            ("It drops 4 degrees and ends at -3 degrees.", "-3"),
            ("Each costs $2.50, so 4 cost $10.00.", "10.00"),
            ("No numbers here.", None),
            ("#### Step 1: Add\n5 + 7 = 12 apples.\n#### Step 2: Answer\nShe has 12 apples.", "12"),
            ("#### 1. Add the apples\n5 + 7 = 12 apples.", "12"),
            ("The profit is \\boxed{70{,}000}.", "70000"),
            ("The profit is \\boxed{70\\,000}.", "70000"),
            ("We put it in \\boxed{} later. 3 + 4 = 7, so she pays 7 dollars.", "7"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(gsm8k.extract_answer(text), expected)

    def test_gold_answer_and_scoring(self) -> None:
        self.assertEqual(gsm8k.gold_answer("48 / 2 = 24 clips\n#### 1,024"), "1024")
        self.assertTrue(gsm8k.is_correct("10.00", "10"))
        self.assertFalse(gsm8k.is_correct("11", "10"))
        self.assertFalse(gsm8k.is_correct(None, "10"))

    def test_prompt_turns_thinking_off(self) -> None:
        prompt = gsm8k.build_prompt("What is 2 + 3?")
        self.assertTrue(prompt.startswith("<|startoftext|><|im_start|>user\nWhat is 2 + 3?\n"))
        self.assertTrue(prompt.endswith("<|im_end|>\n<|im_start|>assistant\n<think></think>"))


if __name__ == "__main__":
    unittest.main()
