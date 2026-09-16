#!/usr/bin/env python3
"""
Train native Keyword Spotting (KWS) model for Hey Liquid.

Synthesizes acoustic training samples across diverse macOS TTS voices in parallel,
extracts log-mel spectrograms matching Cera's front-end, trains the 1D
convolutional neural network with PyTorch, and exports folded GGUF weights.
"""

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import tempfile
import wave
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import model architecture, mel extractor, and export utilities from convert_kws
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.convert_kws import (
    KwsModel,
    export_kws_to_gguf,
    extract_log_mel_spectrogram,
    generate_verification_fixture,
)


# Natural English voices available on macOS (excluding eval held-out voices like Tara)
VOICES = [
    "Samantha",
    "Daniel",
    "Karen",
    "Kathy",
    "Moira",
    "Rishi",
    "Tessa",
    "Ralph",
    "Fred",
    "Albert",
    "Aman",
    "Junior",
    "Eddy",
    "Flo",
    "Reed",
    "Sandy",
    "Shelley",
    "Rocko",
]

# Rates in words per minute covering natural speech range
RATES = [150, 175, 200]

def get_training_texts(
    phrase: str = "Hey Liquid",
    extra_negatives: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """Generate positive variations and negative confusers for a target phrase."""
    pos_texts = [
        phrase,
        phrase.lower(),
        f"{phrase}.",
        f"{phrase.lower()}.",
        f"{phrase}!",
        f"{phrase}?",
        f"{phrase},",
        f"{phrase} please",
        f"{phrase} now",
    ]
    words = phrase.split()
    if len(words) == 2 and words[0].lower() in ("hey", "hi", "okay", "ok"):
        w0 = words[0]
        w1 = words[1]
        pos_texts.extend([
            f"{w0.capitalize()} {w1}",
            f"{w0.capitalize()}, {w1}",
            f"{w0.capitalize()} {w1}.",
            f"{w0.capitalize()}, {w1}.",
            f"{w0.capitalize()} {w1}!",
            f"{w0.capitalize()}, {w1}!",
            f"{w0.lower()} {w1.lower()}",
            f"{w0.lower()}, {w1.lower()}",
            f"{w0.lower()} {w1.lower()}.",
            f"{w0.lower()}, {w1.lower()}.",
            f"{w0.capitalize()} {w1} please",
            f"{w0.capitalize()} {w1}, please",
            f"{w0.capitalize()} {w1} now",
            f"{w0.capitalize()} {w1}, can you",
            f"{w0.capitalize()} {w1} can you",
            f"{w0.capitalize()} {w1}, help",
            f"{w0.capitalize()} {w1} help",
            f"{w0.capitalize()} {w1}, start",
            f"{w0.capitalize()} {w1}, turn on the lights",
            f"{w0.capitalize()} {w1}, what time is it",
            f"{w0.capitalize()} {w1}, how are you",
        ])

    phrase_lower = phrase.lower()
    neg_texts = list(NEGATIVE_TEXTS) + list(NEGATIVE_CONVERSATIONAL) + list(NEGATIVE_HEY)
    if "liquid" in phrase_lower:
        neg_texts.extend(NEGATIVE_LIQUID)

    if len(words) > 1 and words[0].lower() in ("hey", "hi", "okay", "ok"):
        target_name = " ".join(words[1:])
        # Competing onset prefixes
        synonyms = {"ok", "okay"}
        w0_lower = words[0].lower()
        for prefix in [
            "they", "play", "say", "may", "bay", "way", "ray", "pay", "day",
            "stay", "gray", "pray", "lay", "delay", "relay", "decay", "today",
            "away", "bye", "why", "no", "yes", "oh", "yo", "hello", "good", "bad",
            "okay", "hi", "hey",
        ]:
            is_synonym = prefix.lower() in synonyms and w0_lower in synonyms
            if prefix.lower() != w0_lower and not is_synonym:
                neg_texts.append(f"{prefix} {target_name}")
                neg_texts.append(f"{prefix.capitalize()} {target_name}")

        # In-cabin / conversational usage of target_name without wake intent
        neg_texts.extend([
            target_name,
            f"a {target_name}",
            f"the {target_name}",
            f"my {target_name}",
            f"this {target_name}",
            f"that {target_name}",
            f"our {target_name}",
            f"your {target_name}",
            f"his {target_name}",
            f"her {target_name}",
            f"in the {target_name}",
            f"inside the {target_name}",
            f"with the {target_name}",
            f"about the {target_name}",
            f"I like the {target_name}",
            f"I love my {target_name}",
            f"Where is the {target_name}",
            f"Look at that {target_name}",
            f"This is a {target_name}",
            f"Is that a {target_name}",
            f"The new {target_name} looks great",
            f"That old {target_name}",
            f"An old {target_name}",
            f"{target_name} is fast",
            f"{target_name} is in the shop",
            f"park the {target_name}",
            f"we took the {target_name}",
            f"driving a {target_name}",
            f"drives a {target_name}",
            f"she drives a {target_name}",
            f"he drives a {target_name}",
            f"my {target_name} needs service",
            f"my {target_name} needs gas",
            f"talking about {target_name}",
            f"rent a {target_name}",
            f"buy a {target_name}",
            f"sold the {target_name}",
            f"start the {target_name}",
            f"stop the {target_name}",
        ])

        # Target name as leading noun/modifier without wake onset
        for suffix in [
            "car", "vehicle", "app", "system", "model", "sedan", "brand",
            "company", "assistant", "software", "group", "engine", "driver",
            "service", "dealership", "parts",
        ]:
            neg_texts.append(f"{target_name} {suffix}")
            neg_texts.append(f"{target_name.capitalize()} {suffix}")

    if extra_negatives:
        neg_texts.extend(extra_negatives)

    # Prevent target phrase contamination: eliminate any negative texts
    # whose lowercased representation overlaps with the positive variations
    pos_set_lower = {p.lower() for p in pos_texts}
    cleaned_neg_texts = [
        t for t in neg_texts if t.lower() not in pos_set_lower
    ]

    return list(dict.fromkeys(pos_texts)), list(dict.fromkeys(cleaned_neg_texts))



NEGATIVE_LIQUID = [
    # Phonetic near-rhymes
    "they liquid",
    "play liquid",
    "say liquid",
    "may liquid",
    "bay liquid",
    "way liquid",
    "ray liquid",
    "pay liquid",
    "nay liquid",
    "day liquid",
    # Phonetic target confusers
    "hey squid",
    "hey livid",
    "hey lipid",
    "hey lizard",
    "hey limit",
    "hey rigid",
    "hey frigid",
    "hey wicked",
    "hey vivid",
    # Single words
    "liquid",
    "squid",
    "livid",
    "limit",
    "rigid",
    "frigid",
    "wicked",
    "vivid",
    # Hard negatives: 'liquid' without 'hey'
    "liquid agent",
    "liquid mobile app",
    "liquid soap",
    "liquid nitrogen",
    "liquid paper",
    "liquid detergent",
    "clear liquid",
    "drinking liquid",
    "cold liquid",
    "hot liquid",
    "pour some liquid",
    "spilled liquid on the phone",
    "is this a solid or a liquid",
    "water is a liquid",
    "viscous liquid",
    "a drop of liquid",
    "liquid crystal",
    "liquid state of matter",
    "colored liquid in a glass",
    "cleaning liquid for windows",
]

NEGATIVE_HEY = [
    # Hard negatives: 'hey' without target keyword
    "hey there",
    "hey how are you doing",
    "hey what is up",
    "hey you",
    "hey can you hear me",
    "hey look at this",
    "hey listen for a second",
    "hey check this out",
    "hey friend",
    "hey what are you doing today",
    "hey did you see the email",
    "hey we need to talk",
    "hey good morning",
    "hey good evening",
    "hey have you seen my keys",
    "hey let us go outside",
    "hey wait for me",
    "hey please help me with this",
]



NEGATIVE_CONVERSATIONAL = [
    "That is the type of thing that a model can do",
    "So it can respond to text and give helpful answers",
    "It is not like Claude or whatever that has access to the internet",
    "You have never been punctual, so why start now",
    "I am talking to someone in the other room",
    "Can you hear what I am saying right now",
    "We are developing a new application for mobile phones",
    "The neural network runs directly on the device hardware",
    "What did you think about the presentation earlier today",
    "Let me check if the meeting is still happening this afternoon",
    "I need to go to the grocery store after work today",
    "Where did I leave my keys and my wallet",
    "The weather forecast says it might rain later this evening",
    "Can you please pass the salt and pepper across the table",
    "I am going to make some coffee before we begin the meeting",
    "How was your weekend with the family in the mountains",
    "Let us take a quick break and come back in five minutes",
    "This is an interesting problem that we need to solve together",
    "I agree with what you just mentioned in the chat",
    "Could you send me an email with all the details",
    "Have you seen the latest news update on the screen",
    "I will be ready in about fifteen minutes or so",
    "That sounds like a great idea to everyone involved",
    "Let me know when the package arrives at the front door",
    "We should schedule a call for tomorrow morning at nine",
    "I do not think that will work well in this scenario",
    "Are you planning to go to the tech conference next month",
    "Let us review the code changes one more time carefully",
    "Everything seems to be running smoothly on the system now",
    "I will follow up with you later this evening after dinner",
    "Is there anything else you wanted to discuss today",
    "Let us get started on the next item on our checklist",
    "That was a very helpful explanation, thank you so much",
    "I am not sure if that is the best approach to take",
    "We have plenty of time to finish the project before Friday",
    "Can we talk about the architecture of the neural network",
    "The audio capture loop reads thirty millisecond frames",
    "Voice recognition works by converting audio into spectrograms",
    "The phone battery lasts longer when running on the NPU",
    "Please make sure the test passes before submitting the pull request",
]


NEGATIVE_TEXTS = [
    # Voice assistant wake phrases
    "Hey Assistant",
    "Okay Assistant",
    "Hey Device",
    "Hey System",
    "Computer",
    "Jarvis",
    # Phonetic and sibilant rhyming near-misses
    "Hey series",
    "Hey ferris",
    "Hey service",
    "Hey serious",
    "Hey surface",
    "Hey cereals",
    "Hey circus",
    "Hey secret",
    "Hey server",
    "Hey setup",
    "Hey safety",
    "Hey sensor",
    # Common voice commands
    "what time is it",
    "turn on the lights",
    "turn off the lights",
    "play some music",
    "stop the music",
    "call mom",
    "set an alarm",
    "how is the weather today",
    "hello world",
    "good morning",
    "cancel that",
    "navigate home",
    "turn on the air conditioner",
    "turn off the air conditioner",
    "set the temperature",
    "lower the temperature",
    "turn up the volume",
    "turn down the volume",
    "mute audio",
]


def get_available_voices() -> List[str]:
    """Return available voices from macOS say system."""
    if sys.platform != "darwin":
        return []
    try:
        res = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        if res.returncode != 0:
            return []
        installed = {line.split()[0] for line in res.stdout.splitlines() if line.split()}
        return [v for v in VOICES if v in installed]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def render_tts(text: str, voice: str, rate: int, output_wav: str) -> bool:
    """Render text to 16 kHz mono 16-bit PCM WAV using macOS say."""
    if sys.platform != "darwin":
        return False
    say_cmd = [
        "say",
        "-v", voice,
        "-r", str(rate),
        "--data-format=LEI16@16000",
        "-o", output_wav,
        text,
    ]
    if subprocess.run(say_cmd, capture_output=True).returncode == 0:
        return os.path.exists(output_wav) and os.path.getsize(output_wav) > 100

    # Fallback to AIFF + afconvert
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp_aiff:
        aiff_path = tmp_aiff.name
    try:
        say_cmd = ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text]
        if subprocess.run(say_cmd, capture_output=True).returncode != 0:
            return False

        convert_cmd = [
            "afconvert",
            "-f", "WAVE",
            "-d", "LEI16@16000",
            "-c", "1",
            aiff_path,
            output_wav,
        ]
        if subprocess.run(convert_cmd, capture_output=True).returncode != 0:
            return False

        return os.path.exists(output_wav) and os.path.getsize(output_wav) > 100
    finally:
        if os.path.exists(aiff_path):
            os.remove(aiff_path)


def read_wav(path: str) -> np.ndarray:
    """Read a 16 kHz mono 16-bit PCM WAV into float32 array in [-1.0, 1.0]."""
    with wave.open(path, "rb") as wf:
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def extract_framed_windows(audio: np.ndarray, is_positive: bool, target_len: int = 19200) -> List[np.ndarray]:
    """
    Extract multiple 19200-sample windows from audio to ensure temporal invariance.
    For positives: generate multiple alignments (left, center, right, intermediate jitter).
    For negatives: slide across long clips with 300ms overlap.
    """
    windows = []
    n = len(audio)
    if is_positive:
        if n <= target_len:
            max_offset = target_len - n
            offsets = {
                0,
                max_offset // 5,
                (2 * max_offset) // 5,
                (3 * max_offset) // 5,
                (4 * max_offset) // 5,
                max_offset,
            }
            for off in sorted(offsets):
                w = np.zeros(target_len, dtype=np.float32)
                w[off : off + n] = audio
                windows.append(w)
        else:
            # Positive wake word starts near the beginning of the clip.
            # Slice strictly near the front (0 to 300ms jitter) so the wake word
            # onset and core phonemes are fully present in the window.
            max_start = min(n - target_len, 4800)
            offsets = {0, max_start // 3, (2 * max_start) // 3, max_start}
            for off in sorted(offsets):
                if off + target_len <= n:
                    windows.append(audio[off : off + target_len])
    else:
        if n <= target_len:
            max_offset = target_len - n
            offsets = {
                0,
                max_offset // 5,
                (2 * max_offset) // 5,
                (3 * max_offset) // 5,
                (4 * max_offset) // 5,
                max_offset,
            }
            for off in sorted(offsets):
                w = np.zeros(target_len, dtype=np.float32)
                w[off : off + n] = audio
                windows.append(w)
        else:
            hop = 4800  # 300 ms hop
            for start in range(0, n - target_len + 1, hop):
                windows.append(audio[start : start + target_len])
            if (n - target_len) % hop != 0:
                windows.append(audio[n - target_len :])

    return windows


def apply_spec_augment(mel: np.ndarray) -> np.ndarray:
    """Apply SpecAugment (frequency and time masking) to log-mel spectrogram."""
    out = mel.copy()
    num_mels, num_frames = out.shape

    # Frequency masking (mask 1 to 3 mel channels)
    if random.random() < 0.6:
        f0 = random.randint(0, max(0, num_mels - 3))
        f_width = random.randint(1, 3)
        out[f0 : f0 + f_width, :] = 0.0

    # Time masking (mask 3 to 10 frames)
    if random.random() < 0.6 and num_frames > 15:
        t0 = random.randint(0, max(0, num_frames - 10))
        t_width = random.randint(3, 10)
        out[:, t0 : t0 + t_width] = 0.0

    return out




def add_acoustic_perturbations(audio: np.ndarray) -> np.ndarray:
    """Add wide-dynamic-range gain jitter, background noise, and AGC normalization."""
    # Sample gain from -26 dB (0.05) to 0 dB (1.0) on log scale
    gain = 10.0 ** random.uniform(-1.3, 0.0)
    out = audio * gain

    if random.random() < 0.85:
        noise = np.random.randn(len(out)).astype(np.float32)
        noise_level = random.uniform(0.0005, 0.008)
        out = out + noise * noise_level

    # Simulate runtime Automatic Gain Control (AGC) peak normalization
    peak = float(np.max(np.abs(out)))
    if peak > 0.002:
        scale = min(0.7 / peak, 30.0)
        out = out * scale

    return np.clip(out, -1.0, 1.0)


class KwsDataset(Dataset):
    """PyTorch Dataset yielding (mel_spectrogram, label, weight)."""

    def __init__(self, samples: List[Tuple[np.ndarray, float, float]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel, label, weight = self.samples[idx]
        return (
            torch.from_numpy(mel),
            torch.tensor([label], dtype=torch.float32),
            torch.tensor([weight], dtype=torch.float32),
        )


def synthesize_dataset(
    output_dir: str,
    phrase: str = "Hey Liquid",
    extra_negatives: List[str] = None,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, float, float]], List[Tuple[np.ndarray, float, float]]]:
    """Synthesize positive and negative speech clips in parallel and extract mel spectrograms."""
    os.makedirs(output_dir, exist_ok=True)

    available_voices = get_available_voices()
    print(f"Available voices ({len(available_voices)}): {', '.join(available_voices)}", flush=True)
    if not available_voices:
        raise RuntimeError("No suitable macOS TTS voices found")

    pos_texts, neg_texts = get_training_texts(phrase, extra_negatives=extra_negatives)
    tasks = []

    # 1. Plan positive tasks
    pos_idx = 0
    for text in pos_texts:
        for voice in available_voices:
            for rate in RATES:
                wav_path = os.path.join(output_dir, f"pos_{pos_idx:04d}.wav")
                tasks.append((text, voice, rate, wav_path, 1.0, False))
                pos_idx += 1

    # 2. Plan negative tasks
    neg_idx = 0
    words = [w.lower() for w in phrase.split()]
    if not words:
        raise ValueError(f"The target wake word '--phrase' cannot be empty or whitespace, got: {phrase!r}")
    target_name = " ".join(words[1:]) if len(words) > 1 else words[0]
    extra_set = {t.lower() for t in (extra_negatives or [])}
    onset = words[0] if words else ""

    for text in neg_texts:
        text_lower = text.lower()
        # Hard negatives include target name mentions, user extra negatives, and wake onset confusers
        is_hard_negative = (
            (target_name in text_lower)
            or (text_lower in extra_set)
            or (bool(onset) and (text_lower == onset or text_lower.startswith(f"{onset} ")))
        )
        if is_hard_negative:
            for v_idx, voice in enumerate(available_voices):
                rate = 175 if (v_idx % 3 == 0) else (150 if v_idx % 3 == 1 else 200)
                wav_path = os.path.join(output_dir, f"neg_{neg_idx:04d}.wav")
                tasks.append((text, voice, rate, wav_path, 0.0, True))
                neg_idx += 1
        else:
            for voice in available_voices[:6]:
                rate = random.choice(RATES)
                wav_path = os.path.join(output_dir, f"neg_{neg_idx:04d}.wav")
                tasks.append((text, voice, rate, wav_path, 0.0, False))
                neg_idx += 1

    print(f"Synthesizing {len(tasks)} audio clips using 8 parallel workers...", flush=True)

    # Parallel TTS rendering
    def run_task(task):
        text, voice, rate, wav_path, label, is_hard_neg = task
        ok = render_tts(text, voice, rate, wav_path)
        return (wav_path, label, is_hard_neg, ok)

    rendered_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(run_task, tasks):
            rendered_results.append(res)

    valid_results = [r for r in rendered_results if r[3] and os.path.exists(r[0])]
    print(
        f"Rendered {len(valid_results)} clips successfully. Partitioning dataset at source level...",
        flush=True,
    )

    # Stratified split at the source audio recording level to prevent data leakage across train/val
    rng = random.Random(seed)
    pos_records = [r for r in valid_results if r[1] > 0.5]
    hard_neg_records = [r for r in valid_results if r[1] <= 0.5 and r[2]]
    soft_neg_records = [r for r in valid_results if r[1] <= 0.5 and not r[2]]

    rng.shuffle(pos_records)
    rng.shuffle(hard_neg_records)
    rng.shuffle(soft_neg_records)

    def split_records(records: List, ratio: float):
        split_idx = int(len(records) * (1.0 - ratio))
        return records[:split_idx], records[split_idx:]

    train_pos, val_pos = split_records(pos_records, val_ratio)
    train_hard, val_hard = split_records(hard_neg_records, val_ratio)
    train_soft, val_soft = split_records(soft_neg_records, val_ratio)

    train_records = train_pos + train_hard + train_soft
    val_records = val_pos + val_hard + val_soft
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    def process_records(records: List, is_training: bool) -> List[Tuple[np.ndarray, float, float]]:
        samples: List[Tuple[np.ndarray, float, float]] = []
        for wav_path, label, is_hard_neg, _ in records:
            audio = read_wav(wav_path)
            is_pos = label > 0.5
            # Tiered penalty: hard negatives get 3.8x penalty; general negatives get 1.0x; positives get 1.5x
            sample_weight = 1.5 if is_pos else (3.8 if is_hard_neg else 1.0)
            framed_list = extract_framed_windows(audio, is_pos)
            for framed in framed_list:
                if is_training:
                    augmented = add_acoustic_perturbations(framed)
                    mel = extract_log_mel_spectrogram(augmented)
                    samples.append((mel, label, sample_weight))
                    # Add SpecAugment version to enhance acoustic invariance
                    mel_spec = apply_spec_augment(mel)
                    samples.append((mel_spec, label, sample_weight))
                else:
                    mel = extract_log_mel_spectrogram(framed)
                    samples.append((mel, label, sample_weight))
        return samples

    train_samples = process_records(train_records, is_training=True)
    val_samples = process_records(val_records, is_training=False)

    # 3. Add synthetic silence and noise negatives (split proportionally)
    silence_count = 150
    train_silence = int(silence_count * (1.0 - val_ratio))
    val_silence = silence_count - train_silence

    for _ in range(train_silence):
        silence = np.random.randn(19200).astype(np.float32) * random.uniform(0.0001, 0.02)
        mel = extract_log_mel_spectrogram(silence)
        train_samples.append((mel, 0.0, 2.0))

    for _ in range(val_silence):
        silence = np.random.randn(19200).astype(np.float32) * random.uniform(0.0001, 0.02)
        mel = extract_log_mel_spectrogram(silence)
        val_samples.append((mel, 0.0, 2.0))

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_pos_cnt = sum(1 for _, l, _ in train_samples if l > 0.5)
    train_neg_cnt = len(train_samples) - train_pos_cnt
    val_pos_cnt = sum(1 for _, l, _ in val_samples if l > 0.5)
    val_neg_cnt = len(val_samples) - val_pos_cnt

    print(
        f"Train dataset: {len(train_samples)} samples (Pos: {train_pos_cnt}, Neg: {train_neg_cnt}) | "
        f"Val dataset: {len(val_samples)} samples (Pos: {val_pos_cnt}, Neg: {val_neg_cnt})",
        flush=True,
    )
    return train_samples, val_samples


def train_model(
    train_samples: List[Tuple[np.ndarray, float, float]],
    val_samples: List[Tuple[np.ndarray, float, float]],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> KwsModel:
    """Train KwsModel on acoustic mel spectrograms."""
    train_loader = DataLoader(KwsDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KwsDataset(val_samples), batch_size=batch_size, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nTraining on device: {device}", flush=True)

    model = KwsModel(mel_bins=32, emb_dim=64, num_keywords=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion = nn.BCELoss(reduction="none")

    print("\n--- Starting Training ---", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for mels, labels, weights in train_loader:
            mels, labels, weights = mels.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()

            preds = model(mels)
            loss_raw = criterion(preds, labels)
            loss = (loss_raw * weights).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_samples)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        pos_scores: List[float] = []
        neg_scores: List[float] = []

        with torch.no_grad():
            for mels, labels, weights in val_loader:
                mels, labels, weights = mels.to(device), labels.to(device), weights.to(device)
                preds = model(mels)
                loss_raw = criterion(preds, labels)
                loss = (loss_raw * weights).mean()
                val_loss += loss.item() * len(labels)

                for p, l in zip(preds.squeeze(-1).cpu().numpy(), labels.squeeze(-1).cpu().numpy()):
                    if l > 0.5:
                        pos_scores.append(float(p))
                    else:
                        neg_scores.append(float(p))

        val_loss /= len(val_samples)
        pos_min = min(pos_scores) if pos_scores else 0.0
        pos_mean = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
        neg_max = max(neg_scores) if neg_scores else 0.0
        neg_mean = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{epochs} (lr: {current_lr:.6f}) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Pos [mean: {pos_mean:.2f}, min: {pos_min:.2f}] | "
            f"Neg [mean: {neg_mean:.2f}, max: {neg_max:.2f}]",
            flush=True,
        )

    model.eval()
    model.cpu()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KWS model for wake word detection")
    parser.add_argument(
        "--phrase",
        default="Hey Liquid",
        help="Target wake word phrase (default: 'Hey Liquid')",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Keyword labels to embed in GGUF metadata (default: [phrase])",
    )
    parser.add_argument(
        "--output",
        default="models/hey_liquid.gguf",
        help="Path to output GGUF model",
    )
    parser.add_argument(
        "--fixture",
        default="models/hey_liquid_fixture.json",
        help="Path to output oracle verification fixture",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs (default: 30)",
    )
    parser.add_argument(
        "--extra-negatives",
        nargs="*",
        default=None,
        help="Additional custom negative phrases to synthesize",
    )
    parser.add_argument(
        "--negatives-file",
        default=None,
        help="Optional path to a text file with negative phrases (one per line)",
    )
    args = parser.parse_args()
    if not args.phrase or not args.phrase.strip():
        raise ValueError("The target wake word '--phrase' cannot be empty or whitespace.")

    keywords = args.keywords if args.keywords is not None else [args.phrase]
    if len(keywords) != 1:
        raise ValueError(
            f"train_kws.py currently only supports training single-keyword models, got {len(keywords)} keywords: {keywords}"
        )

    extra_negs = list(args.extra_negatives or [])
    if args.negatives_file:
        if not os.path.exists(args.negatives_file):
            raise FileNotFoundError(f"Negatives file not found: {args.negatives_file}")
        with open(args.negatives_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    extra_negs.append(line)

    # 1. Synthesize audio dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Synthesizing dataset for '{args.phrase}' in {tmpdir}...", flush=True)
        train_samples, val_samples = synthesize_dataset(
            tmpdir,
            phrase=args.phrase,
            extra_negatives=extra_negs,
        )

        # 2. Train model
        model = train_model(train_samples, val_samples, epochs=args.epochs)

    # 3. Export to GGUF with folded BatchNorm
    print(f"\n--- Exporting Trained GGUF to {args.output} ---", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    export_kws_to_gguf(
        output_path=args.output,
        keywords=keywords,
        model=model,
        default_threshold=0.75,
    )

    # 4. Export verification fixture for Rust oracle parity tests
    if args.fixture:
        print(f"\n--- Exporting Verification Fixture to {args.fixture} ---", flush=True)
        generate_verification_fixture(
            fixture_path=args.fixture,
            model=model,
            keywords=keywords,
            seed=42,
        )

    print("\nTraining and GGUF export complete!", flush=True)


if __name__ == "__main__":
    main()
