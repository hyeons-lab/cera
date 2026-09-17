#!/usr/bin/env python3
"""Unit tests for tools/wake_word/train_kws.py."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
    import torch
    from tools.wake_word.train_kws import (
        KwsDataset,
        get_training_texts,
        is_phrase_in_text,
        read_wav,
        seed_everything,
        synthesize_dataset,
    )
    from scripts.convert_kws import KwsModel
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "torch and numpy required for train_kws tests")
class TestTrainKws(unittest.TestCase):
    """Test suite verifying training pipeline utilities, phrase filtering, and audio checks."""

    def test_single_word_phrase_training_texts(self):
        """Verify single-word wake word phrase generates valid texts without onset crashes."""
        pos, neg = get_training_texts("Computer")
        self.assertGreater(len(pos), 0)
        self.assertGreater(len(neg), 0)
        self.assertIn("Computer", pos)
        self.assertIn("computer", pos)

        # Ensure single-word keyword is filtered out of negative pool
        pos_lower = {p.lower() for p in pos}
        for n in neg:
            self.assertNotIn(n.lower(), pos_lower)
            self.assertFalse(
                is_phrase_in_text("Computer", n),
                f"Negative text '{n}' contains target phrase 'Computer'",
            )

        # Single word without 'hey'/'ok' onset should not generate competing onset variants
        self.assertNotIn("they Computer", neg)
        self.assertNotIn("play Computer", neg)

    def test_multi_word_phrase_onsets_and_negatives(self):
        """Verify multi-word phrase generates onset confusers and conversational contexts."""
        pos, neg = get_training_texts("Hey Liquid")
        self.assertGreater(len(pos), 0)
        self.assertGreater(len(neg), 0)
        self.assertIn("Hey Liquid", pos)
        self.assertIn("hey liquid", pos)

        # Competing onset prefixes should be present in negatives
        self.assertIn("They Liquid", neg)
        self.assertIn("Play Liquid", neg)
        self.assertIn("Say Liquid", neg)

        # Conversational mentions without wake intent should be in negatives
        self.assertIn("liquid", neg)
        self.assertIn("my Liquid", neg)
        self.assertIn("the Liquid", neg)

        # Positive variations must not leak into negatives
        pos_lower = {p.lower() for p in pos}
        for n in neg:
            self.assertNotIn(n.lower(), pos_lower)
            self.assertFalse(
                is_phrase_in_text("Hey Liquid", n),
                f"Negative text '{n}' contains target phrase 'Hey Liquid'",
            )

    def test_is_phrase_in_text_subsequence_matching(self):
        """Verify word-level contiguous subsequence matching and confuser preservation."""
        # Exact and surrounding matches
        self.assertTrue(is_phrase_in_text("Hey Liquid", "Hey Liquid"))
        self.assertTrue(is_phrase_in_text("Hey Liquid", "and hey liquid please"))
        self.assertTrue(is_phrase_in_text("Hey Liquid", "Hey, Liquid!"))
        self.assertTrue(is_phrase_in_text("Hey Liquid", "talking about Hey Liquid today"))

        # Competing onsets and rhymes must NOT match (crucial for hard negatives)
        self.assertFalse(is_phrase_in_text("Hey Liquid", "They liquid"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", "Play liquid"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", "Say liquid"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", "hey squid"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", "hey livid"))

        # Incomplete fragments must NOT match
        self.assertFalse(is_phrase_in_text("Hey Liquid", "hey"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", "liquid"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", "liquid soap"))

        # Substring inside a single word must NOT match
        self.assertFalse(is_phrase_in_text("Hey", "They"))
        self.assertFalse(is_phrase_in_text("Hey", "Obey"))
        self.assertFalse(is_phrase_in_text("Liquid", "liquidify"))

        # Edge cases
        self.assertFalse(is_phrase_in_text("", "sample text"))
        self.assertFalse(is_phrase_in_text("Hey Liquid", ""))

    def test_extra_negatives_and_contamination_filtering(self):
        """Verify custom negatives are accepted and contaminated samples are purged."""
        custom_negs = [
            "custom user command",
            "ambient television noise",
            "and Hey Liquid please",  # Contaminated: contains wake word
            "say hey liquid",         # Contaminated: contains wake word
        ]
        _, neg = get_training_texts("Hey Liquid", extra_negatives=custom_negs)
        self.assertIn("custom user command", neg)
        self.assertIn("ambient television noise", neg)
        self.assertNotIn("and Hey Liquid please", neg)
        self.assertNotIn("say hey liquid", neg)

    def test_empty_negatives_file_robustness(self):
        """Verify pipeline handles empty negatives file and comment-only lines cleanly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name
            f.write("# Comment line\n\n   \n# Another comment\n")

        try:
            extra_negs = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        extra_negs.append(line)

            self.assertEqual(len(extra_negs), 0)
            _, neg = get_training_texts("Hey Liquid", extra_negatives=extra_negs)
            self.assertGreater(len(neg), 0)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_deterministic_seeding(self):
        """Verify seed_everything provides bit-for-bit deterministic weights and random states."""
        seed_everything(12345)
        np_arr1 = np.random.randn(10)
        t_arr1 = torch.randn(10)
        model1 = KwsModel(mel_bins=32, emb_dim=64, num_keywords=1)

        seed_everything(12345)
        np_arr2 = np.random.randn(10)
        t_arr2 = torch.randn(10)
        model2 = KwsModel(mel_bins=32, emb_dim=64, num_keywords=1)

        np.testing.assert_array_equal(np_arr1, np_arr2)
        self.assertTrue(torch.equal(t_arr1, t_arr2))

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_kws_dataset_tensor_preconversion(self):
        """Verify KwsDataset pre-converts numpy arrays and scalars to tensors during init."""
        sample_mel = np.ones((32, 20), dtype=np.float32)
        raw_samples = [
            (sample_mel, 1.0, 1.5),
            (sample_mel * 0.5, 0.0, 3.8),
        ]
        dataset = KwsDataset(raw_samples)
        self.assertEqual(len(dataset), 2)

        mel_t, label_t, weight_t = dataset[0]
        self.assertIsInstance(mel_t, torch.Tensor)
        self.assertIsInstance(label_t, torch.Tensor)
        self.assertIsInstance(weight_t, torch.Tensor)
        self.assertEqual(mel_t.shape, (32, 20))
        self.assertEqual(label_t.item(), 1.0)
        self.assertEqual(weight_t.item(), 1.5)

        # Ensure subsequent accesses return the pre-allocated tensor without reallocation
        mel_t2, _, _ = dataset[0]
        self.assertIs(mel_t, mel_t2)

    def test_read_wav_validation(self):
        """Verify read_wav validates header integrity, format, and channel layouts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Valid mono 16 kHz 16-bit PCM WAV
            valid_path = os.path.join(tmpdir, "valid_mono.wav")
            with wave.open(valid_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack("<4h", 0, 16384, -16384, 0))

            audio = read_wav(valid_path)
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(len(audio), 4)
            self.assertAlmostEqual(audio[1], 0.5, places=4)
            self.assertAlmostEqual(audio[2], -0.5, places=4)

            # 2. Stereo 16 kHz 16-bit WAV (should average to mono)
            stereo_path = os.path.join(tmpdir, "valid_stereo.wav")
            with wave.open(stereo_path, "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # Left = 16384 (0.5), Right = 0 (0.0) -> mean = 0.25
                wf.writeframes(struct.pack("<2h", 16384, 0))

            stereo_audio = read_wav(stereo_path)
            self.assertEqual(len(stereo_audio), 1)
            self.assertAlmostEqual(stereo_audio[0], 0.25, places=4)

            # 3. Invalid sample rate (44.1 kHz)
            wrong_sr_path = os.path.join(tmpdir, "wrong_sr.wav")
            with wave.open(wrong_sr_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(struct.pack("<2h", 0, 0))

            with self.assertRaises(ValueError) as ctx_sr:
                read_wav(wrong_sr_path)
            self.assertIn("16 kHz", str(ctx_sr.exception))

            # 4. Invalid sample width (8-bit)
            wrong_width_path = os.path.join(tmpdir, "wrong_width.wav")
            with wave.open(wrong_width_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(1)
                wf.setframerate(16000)
                wf.writeframes(b"\x80\x80")

            with self.assertRaises(ValueError) as ctx_w:
                read_wav(wrong_width_path)
            self.assertIn("16-bit", str(ctx_w.exception))

    def test_empty_positive_split_safeguard(self):
        """Verify synthesize_dataset raises RuntimeError when 0 positive clips are rendered."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock render_tts to always fail (returning False)
            with patch("tools.wake_word.train_kws.render_tts", return_value=False):
                with self.assertRaises(RuntimeError) as ctx:
                    synthesize_dataset(tmpdir, phrase="Hey Liquid")
                self.assertIn("No positive audio clips were successfully rendered", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
