#!/usr/bin/env python3
"""GSM8K accuracy of a cera LFM3 model on CPU or NPU, one `cera run` per question.

Protocol: the first N (default 150) questions of the official GSM8K test split,
a zero-shot chat prompt (the question plus a request for a `\\boxed{}` answer),
greedy decoding, thinking off. The prompt is the LFM3 chat template rendered
with `enable_thinking=false`, whose assistant turn opens with `<think></think>`.
cera passes no `enable_thinking` to the template, so `cera chat` would think,
and the CLI has no server or batch mode: each question is a `cera run` of the
rendered text, which `run` tokenizes with its special tokens. The per-question
JSONL lines line up by `index`, so a CPU and an NPU run diff line by line;
`truncated` marks an answer cut off by --max-new-tokens, and the summary gives
accuracy with and without those. The prompt and the token cap set the absolute
score, so a published number from another engine compares only when both match.

  python3 scripts/gsm8k_eval.py --model LFM3-1.7B-QAD-Q4_0.gguf --device npu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from decimal import Decimal
from pathlib import Path

TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
HASHES = re.compile(r"####\s*\$?(" + NUMBER.pattern + r")\.?[ \t]*$", re.MULTILINE)
GENERATED_TOKENS = re.compile(r"^Generated tokens: (\d+)$", re.MULTILINE)


def build_prompt(question: str) -> str:
    return (
        f"<|startoftext|><|im_start|>user\n{question}\n{INSTRUCTION}<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )


def extract_answer(text: str) -> str | None:
    """The number in the last `\\boxed{}` holding one, else of the last `#### N` line, else the last number."""
    text = text.replace("{,}", ",").replace("\\,", "")
    for content in reversed(BOXED.findall(text)):
        if found := NUMBER.search(content.replace(" ", "")):
            return found.group().replace(",", "")
    numbers = HASHES.findall(text) or NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def gold_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def is_correct(prediction: str | None, gold: str) -> bool:
    return prediction is not None and Decimal(prediction) == Decimal(gold)


def ratio(hits: int, total: int) -> str:
    return f"{hits}/{total} = {100 * hits / total:.1f}%" if total else f"{hits}/{total}"


def load_questions(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(TEST_URL, part)
        part.replace(path)
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) < limit:
        sys.exit(f"{path} holds {len(rows)} questions, fewer than --limit {limit}")
    return rows[:limit]


def generate(cera: Path, model: Path, device: str, max_new_tokens: int, prompt: str) -> tuple[str, bool]:
    """The generated text, and whether it stopped at the `max_new_tokens` cap."""
    cmd = [
        str(cera), "run",
        "--model", str(model),
        "--device", device,
        "--temperature", "0",
        "--max-tokens", str(max_new_tokens),
        # GSM8K prompts stay well under 1024 tokens.
        "--context-size", str(max_new_tokens + 1024),
        "--no-cache",
        "--no-spec",
        "--prompt", prompt,
    ]
    # cera writes its tracing logs to stdout, which is scored as the answer.
    env = {**os.environ, "RUST_LOG": "off"}
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", env=env)
    if proc.returncode != 0:
        sys.exit(f"cera run exited with {proc.returncode}:\n{proc.stderr[-4000:]}")
    count = GENERATED_TOKENS.search(proc.stderr)
    if not count:
        sys.exit(f"cera run printed no 'Generated tokens:' line:\n{proc.stderr[-4000:]}")
    return proc.stdout, int(count.group(1)) >= max_new_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="GSM8K accuracy of a cera LFM3 model")
    parser.add_argument("--model", type=Path, required=True, help="LFM3 GGUF")
    parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--cera-bin", type=Path, default=Path("target") / "release" / "cera")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / ".cache" / "cera" / "gsm8k" / "test.jsonl",
        help="GSM8K test.jsonl, downloaded here when missing",
    )
    parser.add_argument("--limit", type=int, default=150, help="first N test questions")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, help="per-question JSONL (default: gsm8k-<device>.jsonl)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    cera = args.cera_bin
    if os.name == "nt" and not cera.exists() and cera.with_suffix(".exe").exists():
        cera = cera.with_suffix(".exe")
    out = args.out or Path(f"gsm8k-{args.device}.jsonl")
    rows = load_questions(args.data, args.limit)

    scores = []  # (correct, truncated) per question
    with out.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            prompt = build_prompt(row["question"])
            text, truncated = generate(cera, args.model, args.device, args.max_new_tokens, prompt)
            prediction, gold = extract_answer(text), gold_answer(row["answer"])
            ok = is_correct(prediction, gold)
            scores.append((ok, truncated))
            record = {
                "index": index,
                "prediction": prediction,
                "answer": gold,
                "correct": ok,
                "truncated": truncated,
                "generated": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            cut = f" (cut off at {args.max_new_tokens} tokens)" if truncated else ""
            print(f"[{index + 1}/{len(rows)}] prediction={prediction} answer={gold} {'ok' if ok else 'wrong'}{cut}", flush=True)
    kept = [ok for ok, truncated in scores if not truncated]
    print(f"GSM8K accuracy ({args.device}, first {len(scores)}): {ratio(sum(ok for ok, _ in scores), len(scores))}")
    print(f"  without the {len(scores) - len(kept)} cut off at {args.max_new_tokens} tokens: {ratio(sum(kept), len(kept))}")


if __name__ == "__main__":
    main()
