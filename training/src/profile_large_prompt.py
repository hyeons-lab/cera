"""
Profile Cera KV Prefix Caching with a rich, multi-shot schema (1,000+ tokens).
"""

import os
import time
import subprocess
import tempfile
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CERA_BIN = os.environ.get("CERA_BIN", str(REPO_ROOT / "target" / "release" / "cera"))
DRAFT_GGUF = os.environ.get(
    "DRAFT_GGUF",
    str(REPO_ROOT / "training" / "checkpoints" / "lfm2.5-vl-450m-dspark-Q4_0.gguf"),
)
PHOTO_DIR = Path(os.environ.get("PHOTO_DIR", str(REPO_ROOT / "training" / "data" / "test_photos")))
CACHE_DIR = Path(tempfile.gettempdir()) / f"cera_kv_bench_cache_{os.getuid() if hasattr(os, 'getuid') else 'user'}"

if CACHE_DIR.exists() and not CACHE_DIR.is_symlink():
    shutil.rmtree(CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Build a comprehensive cataloging prompt with 10 few-shot examples (~1,200 tokens)
examples = "\n".join([
    f"Example {i}:\n"
    f"Input: Image showing scene {i}\n"
    f"Output JSON: {{\"scene\": \"outdoor\", \"taxonomy_category\": \"landscape.nature\", \"confidence\": 0.95, \"tags\": [\"tree\", \"sky\", \"green\", \"sun\", \"nature\"], \"attributes\": {{\"lighting\": \"daylight\", \"quality\": \"high\", \"composition\": \"rule_of_thirds\"}}}}\n"
    for i in range(1, 12)
])

LONG_SYSTEM_PROMPT = (
    "You are an enterprise image cataloging and taxonomy tagging engine.\n"
    "Follow this strict taxonomy and reference examples when classifying all incoming media:\n\n"
    f"{examples}\n\n"
    "Output valid JSON only matching the schema shown above."
)

photos = sorted(list(PHOTO_DIR.glob("*.jpg")))[:3]

print(f"=== Testing Long Prompt (~1,200 tokens) with KV Cache Enabled ===")
for i, p in enumerate(photos):
    cmd = [
        CERA_BIN, "run",
        "--bundle-id", "LFM2.5-VL-450M",
        "--quant", "Q4_0",
        "--max-long-size", "256",
        "-d", DRAFT_GGUF,
        "--cache-dir", str(CACHE_DIR),
        "--system", LONG_SYSTEM_PROMPT,
        "--image", str(p),
        "--prompt", "Classify this image.",
        "--max-tokens", "32"
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True)
    dt = (time.perf_counter() - t0) * 1000.0
    status = "Cold (initial prefill + cache write)" if i == 0 else "Warm (KV Cache Hit)"
    print(f"  Photo {i+1} ({p.name}): {dt:.1f} ms [{status}]")

print(f"\n=== Testing Long Prompt (~1,200 tokens) WITHOUT Cache (--no-cache) ===")
for i, p in enumerate(photos):
    cmd = [
        CERA_BIN, "run",
        "--bundle-id", "LFM2.5-VL-450M",
        "--quant", "Q4_0",
        "--max-long-size", "256",
        "-d", DRAFT_GGUF,
        "--no-cache",
        "--system", LONG_SYSTEM_PROMPT,
        "--image", str(p),
        "--prompt", "Classify this image.",
        "--max-tokens", "32"
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, capture_output=True, text=True)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f"  Photo {i+1} ({p.name}): {dt:.1f} ms [Always Cold Prefill]")
