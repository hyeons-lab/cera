import os
import subprocess
import re
import csv
import sys
import argparse
from pathlib import Path

DEFAULT_FLAGSHIPS = [
    "LFM2.5-230M-Q4_0.gguf",
    "LFM2.5-VL-450M-Q4_0.gguf",
    "LFM2.5-VL-1.6B-Q8_0.gguf",
    "LFM2.5-2.6B-Q5_K_M.gguf",
    "LFM2.5-8B-A1B-Q4_0.gguf",
]

def find_models(models_dir, target_names=None):
    models = []
    models_dir = Path(models_dir).expanduser().resolve()
    if not models_dir.exists():
        return models
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if not file.endswith(".gguf") or "vocoder" in file or "Audio" in file or "mmproj" in file:
                continue
            if target_names and file not in target_names:
                continue
            models.append(Path(root) / file)
    return sorted(models, key=lambda p: p.name)

def parse_time_output(stderr):
    """Parse /usr/bin/time -l output for RSS and peak memory footprint."""
    rss = 0
    footprint = 0
    for line in stderr.splitlines():
        if "maximum resident set size" in line:
            rss = int(line.strip().split()[0])
        if "peak memory footprint" in line:
            footprint = int(line.strip().split()[0])
    return rss, footprint

def run_cera(cera_bin, model_path, prompt_len, gen_len, runs):
    print(f"  [cera] p={prompt_len} n={gen_len}...")
    cmd = [
        "/usr/bin/time", "-l",
        str(Path(cera_bin).expanduser().resolve()), "bench",
        "--model", str(Path(model_path).expanduser().resolve()),
        "--prompt-tokens", str(prompt_len),
        "--max-tokens", str(gen_len),
        "--runs", str(runs),
        "--device", "metal",
        "--context-size", "16384",
        "--no-cache"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        combined = proc.stdout + "\n" + proc.stderr
        decode_match = re.search(r"decode tok/s: p50=([\d.]+)", combined)
        prefill_match = re.search(r"prefill tok/s: p50=([\d.]+)", combined)
        
        decode_tps = float(decode_match.group(1)) if decode_match else 0.0
        prefill_tps = float(prefill_match.group(1)) if prefill_match else 0.0
        
        rss, footprint = parse_time_output(proc.stderr)
        return prefill_tps, decode_tps, rss, footprint
    except Exception as e:
        print(f"    Error running cera: {e}")
        return 0.0, 0.0, 0, 0

def run_llama(llama_bench, model_path, prompt_len, gen_len, runs):
    print(f"  [llama] p={prompt_len} n={gen_len}...")
    cmd = [
        "/usr/bin/time", "-l",
        str(Path(llama_bench).expanduser().resolve()),
        "-m", str(Path(model_path).expanduser().resolve()),
        "-p", str(prompt_len),
        "-n", str(gen_len),
        "-ngl", "99",
        "-r", str(runs),
        "--no-warmup"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        prefill_tps = 0.0
        decode_tps = 0.0
        
        for line in proc.stdout.splitlines():
            if f"pp{prompt_len}" in line:
                m = re.search(r"([\d.]+)\s±", line)
                if m: prefill_tps = float(m.group(1))
            if f"tg{gen_len}" in line:
                m = re.search(r"([\d.]+)\s±", line)
                if m: decode_tps = float(m.group(1))
        
        rss, footprint = parse_time_output(proc.stderr)
        return prefill_tps, decode_tps, rss, footprint
    except Exception as e:
        print(f"    Error running llama: {e}")
        return 0.0, 0.0, 0, 0

def resolve_llama_bench():
    import shutil
    env_value = os.environ.get("LLAMA_BENCH")
    if env_value:
        return Path(env_value).expanduser()

    resolved = shutil.which("llama-bench")
    if resolved:
        return Path(resolved)

    return Path("llama-bench")

def main():
    parser = argparse.ArgumentParser(description="Cera vs llama.cpp benchmark matrix")
    parser.add_argument("--models-dir", type=Path, default=Path.home() / ".leap" / "models", help="Directory containing GGUF models")
    parser.add_argument("--cera-bin", type=Path, default=Path.cwd() / "target" / "release" / "cera", help="Path to cera binary")
    parser.add_argument("--llama-bench", type=Path, default=resolve_llama_bench(), help="Path to llama-bench binary")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"), help="Output CSV file")
    parser.add_argument("--output-md", type=Path, default=Path("benchmarks/deltas_table.md"), help="Output Markdown deltas table")
    parser.add_argument("--model-name", type=str, default=None, help="Target specific model name (e.g. LFM2.5-2.6B-Q5_K_M.gguf)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--all-models", action="store_true", help="Benchmark all discovered LFM models instead of flagships")
    parser.add_argument("--append", action="store_true", help="Append to existing output files rather than overwriting")
    
    args = parser.parse_args()

    PROMPT_LENGTHS = [128, 1024, 4096]
    GEN_LENGTHS = [64, 256, 1024]

    target_names = [args.model_name] if args.model_name else (None if args.all_models else DEFAULT_FLAGSHIPS)
    models = find_models(args.models_dir, target_names)
    if not models:
        print(f"No compatible models found in {args.models_dir}")
        return

    fields = ["model", "prompt_len", "gen_len", "engine", "prefill_tps", "decode_tps", "rss_mb", "footprint_mb"]
    
    md_rows = []
    mode = "a" if args.append and args.output.exists() else "w"
    
    with open(args.output, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        
        for model in models:
            model_name = model.name
            print(f"\nBenchmarking model: {model_name}")
            for p_len in PROMPT_LENGTHS:
                for g_len in GEN_LENGTHS:
                    # Run Cera
                    c_ptps, c_dtps, c_rss, c_foot = run_cera(args.cera_bin, model, p_len, g_len, args.runs)
                    c_rss_mb = c_rss / 1024 / 1024
                    c_foot_mb = c_foot / 1024 / 1024
                    writer.writerow({
                        "model": model_name, "prompt_len": p_len, "gen_len": g_len,
                        "engine": "cera", "prefill_tps": c_ptps, "decode_tps": c_dtps,
                        "rss_mb": c_rss_mb, "footprint_mb": c_foot_mb
                    })
                    
                    # Run Llama.cpp
                    l_ptps, l_dtps, l_rss, l_foot = run_llama(args.llama_bench, model, p_len, g_len, args.runs)
                    l_rss_mb = l_rss / 1024 / 1024
                    l_foot_mb = l_foot / 1024 / 1024
                    writer.writerow({
                        "model": model_name, "prompt_len": p_len, "gen_len": g_len,
                        "engine": "llama.cpp", "prefill_tps": l_ptps, "decode_tps": l_dtps,
                        "rss_mb": l_rss_mb, "footprint_mb": l_foot_mb
                    })
                    f.flush()

                    # Format Markdown delta row
                    p_ratio = f"**{c_ptps / l_ptps:.2f}x**" if l_ptps > 0 and c_ptps >= l_ptps else f"{c_ptps / l_ptps:.2f}x" if l_ptps > 0 else "n/a"
                    d_ratio = f"**{c_dtps / l_dtps:.2f}x**" if l_dtps > 0 and c_dtps >= l_dtps else f"{c_dtps / l_dtps:.2f}x" if l_dtps > 0 else "n/a"
                    rss_ratio = f"{c_rss_mb / l_rss_mb:.2f}x" if l_rss_mb > 0 else "n/a"
                    foot_ratio = f"{c_foot_mb / l_foot_mb:.2f}x" if l_foot_mb > 0 else "n/a"
                    
                    md_rows.append(
                        f"| {model_name} | {p_len} | {g_len} | {c_ptps:.1f} / {l_ptps:.1f} ({p_ratio}) | {c_dtps:.1f} / {l_dtps:.1f} ({d_ratio}) | {c_rss_mb:.1f} / {l_rss_mb:.1f} ({rss_ratio}) | {c_foot_mb:.1f} / {l_foot_mb:.1f} ({foot_ratio}) |"
                    )

    md_header = [
        "| Model | Prompt | Gen | Prefill tok/s (Cera / Llama) | Decode tok/s (Cera / Llama) | RSS MB (Cera / Llama) | Footprint MB (Cera / Llama) |",
        "|-------|--------|-----|------------------------------|-----------------------------|-----------------------|-----------------------------|",
    ]
    with open(args.output_md, "w") as f:
        f.write("\n".join(md_header + md_rows) + "\n")

    print(f"\nDone! Results saved to {args.output} and {args.output_md}")

if __name__ == "__main__":
    main()
