"""
Profile Cera KV Prefix Caching across a photo catalog loop.
Tests processing 4 distinct photos with and without KV Prefix Caching.
"""

import os
import time
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CERA_BIN = os.environ.get("CERA_BIN", str(REPO_ROOT / "target" / "release" / "cera"))
DRAFT_GGUF = os.environ.get(
    "DRAFT_GGUF",
    str(REPO_ROOT / "training" / "checkpoints" / "lfm2.5-vl-450m-dspark-Q4_0.gguf"),
)
PHOTO_DIR = Path(os.environ.get("PHOTO_DIR", str(REPO_ROOT / "training" / "data" / "test_photos")))
CACHE_DIR = Path(tempfile.gettempdir()) / f"cera_kv_bench_cache_{os.getuid() if hasattr(os, 'getuid') else 'user'}"

# Clean bench cache dir
if CACHE_DIR.exists() and not CACHE_DIR.is_symlink():
    import shutil
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

# Rich system prompt with JSON schema (~200 tokens)
SYSTEM_PROMPT = (
    "You are an automated photo library indexing engine. "
    "For every image provided, extract visual metadata and strictly output valid JSON matching this schema: "
    "{\n"
    '  "scene": "nature" | "urban" | "indoor" | "beach" | "food",\n'
    '  "palette": ["primary_color", "secondary_color"],\n'
    '  "tags": ["tag1", "tag2", "tag3"],\n'
    '  "mood": "calm" | "vibrant" | "dark" | "warm"\n'
    "}\n"
    "Output JSON only, with no markdown or additional explanation."
)

USER_PROMPT = "Catalog this image."

photos = sorted(list(PHOTO_DIR.glob("*.jpg")))
print(f"Profiling {len(photos)} photos across test configurations...\n")

def run_pass(enable_cache: bool):
    mode_name = "WITH KV Prefix Cache (Warm-hit)" if enable_cache else "WITHOUT KV Cache (--no-cache)"
    print(f"=== Running Pass: {mode_name} ===")
    results = []

    for i, photo in enumerate(photos, 1):
        cmd = [
            CERA_BIN, "run",
            "--bundle-id", "LFM2.5-VL-450M",
            "--quant", "Q4_0",
            "--max-long-size", "256",
            "-d", DRAFT_GGUF,
            "--system", SYSTEM_PROMPT,
            "--image", str(photo),
            "--prompt", USER_PROMPT,
            "--max-tokens", "64",
        ]
        
        if enable_cache:
            cmd.extend(["--cache-dir", str(CACHE_DIR)])
        else:
            cmd.append("--no-cache")

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stdout = proc.stdout
        stderr = proc.stderr

        # Parse metrics from cera output
        prefill_ms = 0.0
        decode_toks = 0.0
        for line in stdout.splitlines():
            if "Image prefill:" in line:
                # e.g. "Image prefill: 88 KV tokens, 941.2 ms"
                parts = line.split(",")
                if len(parts) >= 2 and "ms" in parts[1]:
                    try:
                        prefill_ms = float(parts[1].replace("ms", "").strip())
                    except ValueError:
                        pass
            if "Decode:" in line:
                # e.g. "Decode: 147.8 tok/s"
                try:
                    decode_toks = float(line.replace("Decode:", "").replace("tok/s", "").strip())
                except ValueError:
                    pass

        print(f"  Photo {i} ({photo.name}): Total={elapsed_ms:.1f}ms | Prefill={prefill_ms:.1f}ms | Decode={decode_toks:.1f} tok/s")
        results.append({
            "photo": photo.name,
            "total_ms": elapsed_ms,
            "prefill_ms": prefill_ms,
            "decode_toks": decode_toks,
            "output": stdout.strip()
        })
    print()
    return results

# 1. Run with KV Cache enabled
cached_results = run_pass(enable_cache=True)

# 2. Run without KV Cache (--no-cache)
uncached_results = run_pass(enable_cache=False)

# Summary Comparison
print("=========================================================================")
print(" SUMMARY COMPARISON: KV CACHING ON MULTIMODAL PHOTO CATALOG LOOP")
print("=========================================================================")
print(f"{'Photo':<25} | {'With KV Cache':<16} | {'Without Cache':<16} | {'Speedup'}")
print("-" * 73)

for c, u in zip(cached_results, uncached_results):
    speedup = (u['total_ms'] / c['total_ms']) if c['total_ms'] > 0 else 1.0
    print(f"{c['photo']:<25} | {c['total_ms']:>8.1f} ms      | {u['total_ms']:>8.1f} ms      | {speedup:.2f}x")

avg_cached_warm = sum(r['total_ms'] for r in cached_results[1:]) / len(cached_results[1:])
avg_uncached = sum(r['total_ms'] for r in uncached_results) / len(uncached_results)
overall_warm_speedup = avg_uncached / avg_cached_warm

print("-" * 73)
print(f"Avg Warm Photo Latency:     {avg_cached_warm:.1f} ms")
print(f"Avg Uncached Photo Latency: {avg_uncached:.1f} ms")
print(f"Warm KV Prefix Advantage:   {overall_warm_speedup:.2f}x throughput on repeated schema prompts")
print("=========================================================================")
