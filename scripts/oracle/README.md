# Text-model reference oracle

Golden fixtures for validating cera's text-model forward pass against upstream
llama.cpp. Covers both RoPE families cera serves through `LlamaModel`:
NEOX-rope (Qwen2/Qwen3, Phase A) and NORM-rope (LLaMA/Mistral/Granite 3.x,
Phase B). Granite also exercises its embedding/residual/attention/logit scalar
multipliers (they fold into the gated `embd`/`l_out-{i}`/`result_output` sums); the
`llama-3_2-1b-long` set exercises Llama-3's RoPE frequency scaling
(`rope_freqs.weight`), whose effect on the sums only surfaces at long context.

## Design

The source of truth is **llama.cpp running on the exact same quantized GGUF** cera
loads (not HuggingFace on the unquantized model) — so numeric differences reflect
*implementation* bugs, not quantization noise. The oracle runs **CPU-only**
(`-ngl 0`) so its float accumulation order matches cera's CPU forward pass.

Fixtures are generated **once, locally**, and committed. CI diffs cera against the
committed fixtures and needs neither llama.cpp nor these scripts.

Two gate signals per prompt (see `cera/tests/fixtures/oracle/<model>/*.json`):

- **`input_tokens`** — tokenizer parity.
- **`nodes`** — ordered `(name, op, sum)` per-substep full-tensor checksums from
  `llama-eval-callback`. Localizes any divergence to the exact sub-step. Key
  layer-0 nodes: `embd → norm → attn_norm → Qcur(MUL_MAT→ADD(bias)→ROPE) →
  Kcur(…→ROPE) → Vcur → kqv_out → ffn_{norm,gate,up,swiglu,out} → l_out`, then
  final `result_norm`/`result_output`. The `ADD` after each Q/K/V `MUL_MAT` is the
  Qwen2 QKV bias; the `ROPE` nodes are the post-rotation checksums (catch the
  NEOX-vs-NORM rope bug). Backend KV-cache view/permute nodes are filtered out.
- **`greedy_text`** — end-to-end greedy (`--temp 0 --top-k 1`) continuation.

## Regenerating

```bash
scripts/oracle/vendor_llama_cpp.sh        # clone+build pinned llama.cpp (~target/, gitignored)

# Fixture models (gitignored). Q8_0 — uniform, fully supported by cera, tightest
# numeric match. NB: Qwen's "Q4_0" GGUFs store ffn_down as Q4_1, which cera
# doesn't support; use Q8_0.
hf download Qwen/Qwen2-0.5B-Instruct-GGUF     qwen2-0_5b-instruct-q8_0.gguf     --local-dir target/oracle/models
hf download Qwen/Qwen3-0.6B-GGUF              Qwen3-0.6B-Q8_0.gguf              --local-dir target/oracle/models
hf download bartowski/Llama-3.2-1B-Instruct-GGUF   Llama-3.2-1B-Instruct-Q8_0.gguf   --local-dir target/oracle/models
hf download bartowski/granite-3.1-2b-instruct-GGUF granite-3.1-2b-instruct-Q8_0.gguf --local-dir target/oracle/models

# gen_fixtures.sh <model-gguf-basename> <fixture-subdir>
scripts/oracle/gen_fixtures.sh qwen2-0_5b-instruct-q8_0.gguf     qwen2-0_5b
scripts/oracle/gen_fixtures.sh Qwen3-0.6B-Q8_0.gguf              qwen3-0_6b
scripts/oracle/gen_fixtures.sh Llama-3.2-1B-Instruct-Q8_0.gguf   llama-3_2-1b
scripts/oracle/gen_fixtures.sh granite-3.1-2b-instruct-Q8_0.gguf granite-3_1-2b

# Long-context fixture exercising Llama-3 RoPE frequency scaling (rope_freqs.weight).
# Its effect on the gated sums is sub-noise at the short prompts above but dominates
# near ~140 tokens (cera diverges ~3.5x more without the factors), so this one long
# prompt is what actually catches it. The full prompt is recorded in the fixture's
# index.json; regenerate with:
BIN=target/oracle/llama.cpp/build/bin
DYLD_LIBRARY_PATH=$BIN LD_LIBRARY_PATH=$BIN python3 scripts/oracle/gen_text_oracle.py \
  --bin-dir "$BIN" --model target/oracle/models/Llama-3.2-1B-Instruct-Q8_0.gguf \
  --out-dir cera/tests/fixtures/oracle/llama-3_2-1b-long \
  --llama-sha "$(git -C target/oracle/llama.cpp rev-parse HEAD)" --n-predict 8 "<long prompt>"
```

The pinned llama.cpp SHA lives in `vendor_llama_cpp.sh`; bump it deliberately and
regenerate. Text models use upstream `ggml-org/llama.cpp`. (LFM2.5-Audio is not in
upstream — see `devlog/plans/000150-01-llama-qwen2-support.md` for the audio-only
fork note; not needed here.)
