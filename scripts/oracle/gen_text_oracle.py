#!/usr/bin/env python3
"""Generate golden reference fixtures for cera text-model validation.

Runs the vendored llama.cpp (see vendor_llama_cpp.sh) on a fixed prompt set and
captures, per prompt:
  - input token ids  (tokenizer parity)
  - ordered (node_name, op, sum) per-substep checksums from llama-eval-callback
    (localizes any divergence to the exact sub-step: rope, qkv-bias, attn, ffn)
  - greedy continuation text from llama-completion  (end-to-end correctness)

Output is small JSON committed under cera/tests/fixtures/oracle/<model>/ ; CI diffs
cera's forward pass against it (no llama.cpp needed in CI).

IMPORTANT: the oracle runs CPU-only (-ngl 0) so its float accumulation order
matches cera's CPU forward pass — backend (Metal) accumulation would inflate the
sum deltas and muddy the gate.
"""
import argparse
import json
import os
import re
import subprocess
import sys

# Header: "common_debug_cb_eval:   NAME = (TYPE) OP(src0{..}, src1{..}) = {shape}"
HDR_RE = re.compile(r"common_debug_cb_eval:\s*(.+?)\s*=\s*\((\w+)\)\s*(\w+)\(")
# Backend-specific KV-cache plumbing nodes — cera's CPU path doesn't replicate
# their decomposition, so they're noise for cross-impl comparison. Drop them.
SKIP_NODE_RE = re.compile(r"cache_[kv]|\(view\)|\(permuted\)|\(copy\)")
SUM_RE = re.compile(r"^\s*sum\s*=\s*(-?[\d.eE+]+)")
# Token lines carry a log prefix, e.g. "0.00.566.560 I   16"
TOK_RE = re.compile(r"\bI\s+(\d+)\s*$")


def run(cmd, env):
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # Fail fast: a non-zero exit (missing model, loader error, wrong binary)
    # would otherwise yield empty/partial fixtures and silently commit bad
    # goldens. Surface stdout+stderr so the cause is obvious.
    if out.returncode != 0:
        raise SystemExit(
            f"command failed ({out.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}"
        )
    return out


def capture_nodes_and_tokens(bin_dir, model, prompt, env):
    """Run llama-eval-callback; return (token_ids, [{name, op, sum}, ...])."""
    out = run(
        [f"{bin_dir}/llama-eval-callback", "-m", model, "-p", prompt, "-ngl", "0"],
        env,
    )
    text = out.stdout + "\n" + out.stderr
    lines = text.splitlines()

    tokens, in_tok_block = [], False
    nodes, pending = [], None
    for ln in lines:
        if "number of input tokens" in ln:
            in_tok_block = True
            continue
        if in_tok_block:
            m = TOK_RE.search(ln)
            if m:
                tokens.append(int(m.group(1)))
                continue
            in_tok_block = False
        h = HDR_RE.search(ln)
        if h:
            pending = {"name": h.group(1), "op": h.group(3)}
            continue
        s = SUM_RE.match(ln)
        if s and pending is not None:
            pending["sum"] = float(s.group(1))
            if not SKIP_NODE_RE.search(pending["name"]):
                nodes.append(pending)
            pending = None
    return tokens, nodes


def capture_greedy(bin_dir, model, prompt, n_predict, env):
    out = run(
        [
            f"{bin_dir}/llama-completion", "-m", model, "-p", prompt,
            "--temp", "0", "--top-k", "1", "-n", str(n_predict),
            "-ngl", "0", "--no-display-prompt",
            # RAW completion: -no-cnv disables conversation mode, which otherwise
            # auto-enables when the model ships a chat template and would wrap the
            # prompt in user/assistant turns — making the greedy text inconsistent
            # with eval-callback's raw tokenization (and with cera, which prefills
            # the raw prompt tokens).
            "-no-cnv",
        ],
        env,
    )
    # llama-completion may append an EOF/marker line; keep the raw model text.
    return out.stdout.replace("> EOF by user", "").rstrip()


def slug(p):
    return re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-")[:40] or "empty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--llama-sha", required=True)
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("prompts", nargs="+")
    args = ap.parse_args()

    # Locate the just-built llama.cpp shared libs: DYLD_* on macOS, LD_* on Linux.
    env = dict(os.environ, DYLD_LIBRARY_PATH=args.bin_dir, LD_LIBRARY_PATH=args.bin_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    index = {
        "model_file": os.path.basename(args.model),
        "llama_cpp_sha": args.llama_sha,
        "n_predict": args.n_predict,
        "prompts": [],
    }
    for prompt in args.prompts:
        tokens, nodes = capture_nodes_and_tokens(args.bin_dir, args.model, prompt, env)
        greedy = capture_greedy(args.bin_dir, args.model, prompt, args.n_predict, env)
        if not nodes:
            print(f"[oracle] WARNING: no nodes captured for {prompt!r}", file=sys.stderr)
        fx = {
            "prompt": prompt,
            "input_tokens": tokens,
            "greedy_text": greedy,
            "nodes": nodes,
        }
        name = slug(prompt) + ".json"
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(fx, f, indent=2)
        index["prompts"].append({"prompt": prompt, "fixture": name,
                                 "n_tokens": len(tokens), "n_nodes": len(nodes)})
        print(f"[oracle] {name}: {len(tokens)} tokens, {len(nodes)} nodes, "
              f"greedy={greedy[:40]!r}")

    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[oracle] wrote {len(index['prompts'])} fixtures + index.json to {args.out_dir}")


if __name__ == "__main__":
    main()
