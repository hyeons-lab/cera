"""Negative controls for false-success and incomplete validation provenance."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("chat_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_requires_public_fixture_and_complete_nonempty_suite(self):
        public = "test tests::public_lfm2_tokenizer_boundary_and_ten_turns ... ok\n"
        count = runner.ISOLATED_CASES
        summary = f"test result: ok. {count} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"
        self.assertEqual(runner.require_executed_tests(public + summary), count)
        for output in (
            "",
            public,
            summary,
            public + summary.replace(f"{count} passed", f"{count - 1} passed"),
            public + summary.replace(f"{count} passed", f"{count + 1} passed"),
            public + summary.replace("0 ignored", "1 ignored"),
            public + summary.replace("0 filtered out", "1 filtered out"),
        ):
            with self.subTest(output=output), self.assertRaisesRegex(RuntimeError, "positive evidence"):
                runner.require_executed_tests(output)

    def test_core_mode_requires_both_public_cases_and_the_exact_suite(self):
        contract = "test session::chat::contract_tests::public_lfm2_tokenizer_boundary_and_ten_turns ... ok\n"
        actual = (
            "test session::chat::tests::public_tokenizer_actual_session_ten_turns ... ok\n"
            "test session::chat::tests::real_model_r1_ten_warm_turns_and_kv_retention ... ok\n"
            "test session::chat::tests::real_model_r1_stochastic_rng_determinism_and_divergence ... ok\n"
            "test session::chat::tests::real_model_r1_interrupted_turn_and_replacement_recovery ... ok\n"
        )
        count = runner.CORE_CASES
        summary = f"test result: ok. {count} passed; 0 failed; 0 ignored; 0 measured; 722 filtered out;"
        self.assertEqual(runner.require_executed_tests(contract + actual + summary, core_transactions=True), count)
        for output in (
            contract + summary,
            actual + summary,
            contract + actual,
            contract + actual + summary.replace(f"{count} passed", f"{count - 1} passed"),
            contract + actual + summary.replace(f"{count} passed", f"{count + 1} passed"),
            contract + actual + summary.replace("0 ignored", "1 ignored"),
        ):
            with self.subTest(output=output), self.assertRaisesRegex(RuntimeError, "positive evidence"):
                runner.require_executed_tests(output, core_transactions=True)

    def test_core_build_and_harness_inputs_are_fingerprinted(self):
        hashes = runner.source_hashes()
        for path in ("cera/src/session/chat.rs", "cera/src/session/chat/tests.rs", "cera/src/sampler.rs", "cera/src/model/mod.rs", "cera/src/kv_cache.rs", "cera/src/gguf.rs", "Cargo.lock", "cera/Cargo.toml", "cera/build.rs", "tests/api_loading/commands.py", "tests/api_loading/run.py", "cera/tests/api_chat/fixtures.rs", "tests/api_chat/test_runner.py"):
            self.assertIn(path, hashes)

    def test_source_addition_and_change_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for path in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/config.toml", "cera/Cargo.toml", "cera/build.rs", "cera/tests/chat_contract.rs", "cera/src/sampler.rs", "cera/build_support/generated.rs", "tests/api_loading/commands.py", "cera/tests/api_chat/contract.rs", "tests/api_chat/run.py", "cera/schema/kv_cache.fbs"):
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("initial")
            with patch.object(runner, "REPO", repo), patch.object(runner, "ROOT", repo / "tests/api_chat"):
                before = runner.source_hashes()
                (repo / "cera/src/sampler.rs").write_text("changed")
                after = runner.source_hashes()
                self.assertNotEqual(before, after)
                (repo / "cera/src/new.rs").write_text("new module")
                self.assertNotEqual(after, runner.source_hashes())

    def test_select_profile_matches_candidate_and_rejects_unknown(self):
        pins = {
            "profiles": [
                {"id": "p1", "sha256": "abc111"},
                {"id": "p2", "sha256": "def222"},
                "malformed-entry",
            ]
        }
        self.assertEqual(runner.select_profile(pins, "abc111")["id"], "p1")
        self.assertEqual(runner.select_profile(pins, "def222")["id"], "p2")
        with self.assertRaisesRegex(ValueError, "Model hash mismatch: expected one of \\[abc111, def222\\], got unknown"):
            runner.select_profile(pins, "unknown")

    def test_execution_scope_distinguishes_core_transactions_and_isolated_modes(self):
        pin = {"runtime_validation": "Full warm turns and numerical KV claim"}
        core_scope = runner.execution_scope(pin, core_transactions=True)
        self.assertEqual(core_scope, "Full warm turns and numerical KV claim")
        isolated_scope = runner.execution_scope(pin, core_transactions=False)
        self.assertIn("isolated chat contract", isolated_scope)
        self.assertNotIn("Full warm turns", isolated_scope)

    def test_budgets_schema_and_numerical_ranges(self):
        import json

        budgets_path = HERE / "budgets.json"
        self.assertTrue(budgets_path.is_file(), f"Missing budgets file: {budgets_path}")
        with open(budgets_path, encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("version", data)
        self.assertIn("scope", data)
        self.assertIn("reference_architecture", data)

        invariants = data.get("deterministic_work_invariants", {})
        self.assertTrue(invariants.get("delta_only_prompt_evaluation"))
        self.assertFalse(invariants.get("history_replay_on_normal_turn"))
        self.assertEqual(invariants.get("normal_turn_resets"), 0)
        self.assertEqual(invariants.get("full_cache_checkpoint_copies"), 0)
        self.assertTrue(invariants.get("kv_cache_bit_exact_retention"))

        profiles = data.get("profiles", {})
        self.assertIn("lfm2-350m-gguf-simple-text-v1", profiles)
        self.assertIn("lfm2.5-350m-gguf-text-v1", profiles)

        for name, profile in profiles.items():
            targets = profile.get("targets", {})
            self.assertIn("native-cpu-arm64", targets, f"Missing native-cpu-arm64 target in {name}")
            target = targets["native-cpu-arm64"]
            self.assertLessEqual(target["max_turn_framing_overhead_ms"], 1.5)
            self.assertLessEqual(target["max_warm_ttft_ratio_vs_raw"], 1.05)
            self.assertGreaterEqual(target["min_decode_throughput_ratio_vs_raw"], 0.98)
            self.assertLessEqual(target["max_wrapper_heap_overhead_bytes"], 16384)
            self.assertLessEqual(target["max_per_turn_allocation_bytes"], 8192)


if __name__ == "__main__":
    unittest.main()
