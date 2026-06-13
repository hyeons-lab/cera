# Text-model reference oracle

Golden fixtures for validating cera's text-model forward pass against upstream
llama.cpp, used by Phase A (Qwen2/Qwen3) model-correctness tests.

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
hf download Qwen/Qwen2-0.5B-Instruct-GGUF qwen2-0_5b-instruct-q8_0.gguf --local-dir target/oracle/models
hf download Qwen/Qwen3-0.6B-GGUF          Qwen3-0.6B-Q8_0.gguf          --local-dir target/oracle/models

# gen_fixtures.sh <model-gguf-basename> <fixture-subdir>
scripts/oracle/gen_fixtures.sh qwen2-0_5b-instruct-q8_0.gguf qwen2-0_5b
scripts/oracle/gen_fixtures.sh Qwen3-0.6B-Q8_0.gguf          qwen3-0_6b
```

The pinned llama.cpp SHA lives in `vendor_llama_cpp.sh`; bump it deliberately and
regenerate. Text models use upstream `ggml-org/llama.cpp`. (LFM2.5-Audio is not in
upstream — see `devlog/plans/000150-01-llama-qwen2-support.md` for the audio-only
fork note; not needed here.)
