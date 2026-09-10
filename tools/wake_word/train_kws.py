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


# Natural English voices available on macOS
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
]

# Rates in words per minute
RATES = [150, 175, 200]

def get_training_texts(phrase: str = "Hey Liquid") -> Tuple[List[str], List[str]]:
    """Generate positive variations and negative confusers for a target phrase."""
    pos_texts = [
        phrase,
        phrase.lower(),
        f"{phrase} please",
        f"{phrase} wake up",
        f"{phrase} what time is it",
        f"{phrase} turn on the lights",
        f"{phrase} can you help me",
        f"{phrase} what is the weather",
        f"{phrase} add milk to my list",
        f"{phrase} set a timer",
        f"{phrase} open messages",
    ]
    words = phrase.split()
    if len(words) == 2 and words[0].lower() in ("hey", "hi", "okay", "ok"):
        pos_texts.extend([
            f"Hey {words[1]}",
            f"Hi {words[1]}",
            f"Okay {words[1]}",
        ])

    neg_texts = list(NEGATIVE_TEXTS) + list(NEGATIVE_CONVERSATIONAL) + list(NEGATIVE_LIQUID) + list(NEGATIVE_HEY)
    if phrase.lower() != "hey liquid":
        if len(words) > 1 and words[0].lower() == "hey":
            for prefix in ["they", "play", "say", "may", "bay", "way", "ray", "pay", "day"]:
                neg_texts.append(f"{prefix} {' '.join(words[1:])}")
        for w in words:
            neg_texts.append(w)

    return list(dict.fromkeys(pos_texts)), list(dict.fromkeys(neg_texts))


NEGATIVE_LIQUID = [
    # Crucial hard negatives: 'liquid' without 'hey'
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
    # Crucial hard negatives: 'hey' without 'liquid'
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
    # Collocations & phrases with 'liquid'
    "liquid soap",
    "liquid nitrogen",
    "liquid paper",
    "liquid detergent",
    "clear liquid",
    "dishwashing liquid",
    "spilled some liquid on the desk",
    "drinking cold liquid after exercise",
    "is this a solid or a liquid state of matter",
    # Competitor wake words
    "Hey Siri",
    "Hey Google",
    "Okay Google",
    "Alexa",
    "Computer",
    "Jarvis",
    # General conversational phrases
    "what time is it",
    "turn on the lights",
    "play some music",
    "call mom",
    "set an alarm",
    "how is the weather today",
    "hello world",
    "good morning",
    "cancel that",
    "stop music",
]


def check_voice_available(voice: str) -> bool:
    """Check if a specific TTS voice is installed and functional."""
    if sys.platform != "darwin":
        return False
    try:
        cmd = ["say", "-v", voice, "test"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def render_tts(text: str, voice: str, rate: int, output_wav: str) -> bool:
    """Render text to 16 kHz mono 16-bit PCM WAV using macOS say and afconvert."""
    if sys.platform != "darwin":
        return False
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp_aiff:
        aiff_path = tmp_aiff.name
    try:
        # 1. Render speech to AIFF
        say_cmd = ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text]
        if subprocess.run(say_cmd, capture_output=True).returncode != 0:
            return False

        # 2. Convert to 16 kHz mono 16-bit PCM WAV via afconvert
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
    For negatives: slide across long clips with 400ms overlap.
    """
    windows = []
    n = len(audio)
    if is_positive:
        if n <= target_len:
            max_offset = target_len - n
            offsets = {0, max_offset // 3, (2 * max_offset) // 3, max_offset}
            for off in sorted(offsets):
                w = np.zeros(target_len, dtype=np.float32)
                w[off : off + n] = audio
                windows.append(w)
        else:
            for off in [0, 800, 1600]:
                if off + target_len <= n:
                    windows.append(audio[off : off + target_len])
            if n > target_len and (n - target_len) not in [0, 800, 1600]:
                windows.append(audio[n - target_len :])
    else:
        if n <= target_len:
            max_offset = target_len - n
            offsets = {0, max_offset // 3, (2 * max_offset) // 3, max_offset}
            for off in sorted(offsets):
                w = np.zeros(target_len, dtype=np.float32)
                w[off : off + n] = audio
                windows.append(w)
        else:
            hop = 6400  # 400 ms hop
            for start in range(0, n - target_len + 1, hop):
                windows.append(audio[start : start + target_len])
            if (n - target_len) % hop != 0:
                windows.append(audio[n - target_len :])

    return windows



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
    """PyTorch Dataset yielding (mel_spectrogram, label)."""

    def __init__(self, samples: List[Tuple[np.ndarray, float]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, label = self.samples[idx]
        return torch.from_numpy(mel), torch.tensor([label], dtype=torch.float32)


def synthesize_dataset(output_dir: str, phrase: str = "Hey Liquid") -> List[Tuple[np.ndarray, float]]:
    """Synthesize positive and negative speech clips in parallel and extract mel spectrograms."""
    os.makedirs(output_dir, exist_ok=True)

    available_voices = [v for v in VOICES if check_voice_available(v)]
    print(f"Available voices ({len(available_voices)}): {', '.join(available_voices)}", flush=True)
    if not available_voices:
        raise RuntimeError("No suitable macOS TTS voices found")

    pos_texts, neg_texts = get_training_texts(phrase)
    tasks = []

    # 1. Plan positive tasks
    pos_idx = 0
    for text in pos_texts:
        for voice in available_voices:
            for rate in RATES:
                wav_path = os.path.join(output_dir, f"pos_{pos_idx:04d}.wav")
                tasks.append((text, voice, rate, wav_path, 1.0))
                pos_idx += 1

    # 2. Plan negative tasks
    neg_idx = 0
    words = [w.lower() for w in phrase.split()]
    for text in neg_texts:
        text_lower = text.lower()
        is_hard_negative = any(w in text_lower for w in words if len(w) > 2)
        if is_hard_negative:
            for voice in available_voices:
                for rate in RATES:
                    wav_path = os.path.join(output_dir, f"neg_{neg_idx:04d}.wav")
                    tasks.append((text, voice, rate, wav_path, 0.0))
                    neg_idx += 1
        else:
            for voice in available_voices[:7]:
                rate = random.choice(RATES)
                wav_path = os.path.join(output_dir, f"neg_{neg_idx:04d}.wav")
                tasks.append((text, voice, rate, wav_path, 0.0))
                neg_idx += 1

    print(f"Synthesizing {len(tasks)} audio clips using 8 parallel workers...", flush=True)

    # Parallel TTS rendering
    def run_task(task):
        text, voice, rate, wav_path, label = task
        ok = render_tts(text, voice, rate, wav_path)
        return (wav_path, label, ok)

    rendered_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(run_task, tasks):
            rendered_results.append(res)

    print(f"Rendered {sum(1 for _, _, ok in rendered_results if ok)} clips successfully. Processing features...", flush=True)

    samples: List[Tuple[np.ndarray, float]] = []
    pos_count = 0
    neg_count = 0

    for wav_path, label, ok in rendered_results:
        if not ok or not os.path.exists(wav_path):
            continue
        audio = read_wav(wav_path)
        is_pos = label > 0.5
        framed_list = extract_framed_windows(audio, is_pos)
        for framed in framed_list:
            augmented = add_acoustic_perturbations(framed)
            mel = extract_log_mel_spectrogram(augmented)
            samples.append((mel, label))
            if is_pos:
                pos_count += 1
            else:
                neg_count += 1

    # 3. Add synthetic silence and noise negatives
    for _ in range(150):
        silence = np.random.randn(19200).astype(np.float32) * random.uniform(0.0001, 0.02)
        mel = extract_log_mel_spectrogram(silence)
        samples.append((mel, 0.0))
        neg_count += 1

    print(f"Total dataset size: {len(samples)} samples (Pos: {pos_count}, Neg: {neg_count})", flush=True)
    return samples


def train_model(
    samples: List[Tuple[np.ndarray, float]],
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> KwsModel:
    """Train KwsModel on acoustic mel spectrograms."""
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.85)
    train_data = samples[:split_idx]
    val_data = samples[split_idx:]

    train_loader = DataLoader(KwsDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KwsDataset(val_data), batch_size=batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nTraining on device: {device}", flush=True)

    model = KwsModel(mel_bins=32, emb_dim=64, num_keywords=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Weighted BCE: penalize false positives 2.5x higher than false negatives
    criterion = nn.BCELoss(reduction="none")

    print("\n--- Starting Training ---", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()

            preds = model(mels)
            loss_raw = criterion(preds, labels)
            weights = torch.where(labels < 0.5, 2.5, 1.0)
            loss = (loss_raw * weights).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_data)

        # Validation
        model.eval()
        val_loss = 0.0
        pos_scores: List[float] = []
        neg_scores: List[float] = []

        with torch.no_grad():
            for mels, labels in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                preds = model(mels)
                loss_raw = criterion(preds, labels)
                weights = torch.where(labels < 0.5, 2.5, 1.0)
                loss = (loss_raw * weights).mean()
                val_loss += loss.item() * len(labels)

                for p, l in zip(preds.squeeze(-1).cpu().numpy(), labels.squeeze(-1).cpu().numpy()):
                    if l > 0.5:
                        pos_scores.append(float(p))
                    else:
                        neg_scores.append(float(p))

        val_loss /= len(val_data)
        pos_min = min(pos_scores) if pos_scores else 0.0
        pos_mean = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
        neg_max = max(neg_scores) if neg_scores else 0.0
        neg_mean = sum(neg_scores) / len(neg_scores) if neg_scores else 0.0

        print(
            f"Epoch {epoch:02d}/{epochs} | "
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
        default=25,
        help="Training epochs (default: 25)",
    )
    args = parser.parse_args()
    keywords = args.keywords if args.keywords is not None else [args.phrase]

    # 1. Synthesize audio dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Synthesizing dataset for '{args.phrase}' in {tmpdir}...", flush=True)
        samples = synthesize_dataset(tmpdir, phrase=args.phrase)

        # 2. Train model
        model = train_model(samples, epochs=args.epochs)

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
