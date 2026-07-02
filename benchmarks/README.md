# Cera Benchmarks

Detailed performance numbers and the backend optimization notes that back the
"Performance" summary in the [project README](../README.md).

Measured on Apple M-series (aarch64), single-socket, unless noted. All models
loaded from GGUF with memory-mapped weights. Raw per-run data and long-context
profiles live alongside this file:

- [`results_table.md`](results_table.md) — full result grid
- [`deltas_table.md`](deltas_table.md) — per-optimization deltas
- [`profile_longctx.md`](profile_longctx.md) — long-context prefill profiling
- [`cache_compare.md`](cache_compare.md) — KV-cache comparison

## CPU throughput

### Decode (single-token generation)

| Model | Quant | tok/s |
|-------|-------|------:|
| LFM2-450M | Q4_0 | 119 |
| LFM2-450M | Q8_0 | 107 |
| LFM2.5-1.6B | Q4_0 | 57 |
| LFM2.5-1.6B | Q8_0 | 51 |

### Prefill (prompt processing)

| Model | Quant | 32 tok | 117 tok |
|-------|-------|-------:|--------:|
| LFM2-450M | Q4_0 | 475 | 539 |
| LFM2-450M | Q8_0 | 407 | 451 |
| LFM2.5-1.6B | Q4_0 | 160 | 191 |
| LFM2.5-1.6B | Q8_0 | 131 | 158 |

> These short-prompt numbers predate the Accelerate BLAS wiring (below). At
> 32-117 tokens, overhead and per-call dequant cost dilute the BLAS win — the
> long-prompt table further down shows the larger speedup BLAS delivers at scale.

Q4_0 is faster than Q8_0 for both decode and prefill (less weight data to read
per row), matching llama.cpp behavior. Prefill scales well with prompt length
due to batched GEMM amortizing weight reads across all tokens.

### CPU prefill via Accelerate BLAS (Apple AMX) — opt-in, aarch64 only

The batched prefill GEMM path is currently `#[cfg(target_arch = "aarch64")]`, so
the BLAS rewrite only takes effect on Apple Silicon (and aarch64 Linux). On
x86_64 Linux `forward_prefill` still falls through to the per-token GEMV loop
regardless of the `blas` feature — enabling BLAS on x86_64 just pulls in OpenBLAS
for nothing. Extending the batched path to x86_64 is a separate follow-up.

On aarch64 with the feature on, SGEMM dispatches through Apple's Accelerate
framework (unlocking the AMX matrix unit) or through OpenBLAS on aarch64 Linux.
Weights are dequantized row-by-row into a reusable `InferenceState` scratch, then
multiplied by the f32 input columns — eight call sites per layer (conv in/out
proj, attn Q/K/V/output, FFN gate/up/down).

On LFM2.5-VL-1.6B-Q4_0 with a 2002-token prompt (CPU, M-series):

| Path | Prefill (tok/s, p50) |
|---|---:|
| NEON integer GEMM (default) | 146 |
| **Accelerate SGEMM (AMX, `--features blas`)** | **279** |

That's a **1.91× end-to-end prefill speedup**. A standalone GEMM microbench on
the ffn_up shape `(m=6912, n=2002, k=2048)` shows Accelerate SGEMM at **1885
GFLOPs/s** vs the NEON Q4_0 × Q8_0 kernel at **645 GFLOPs/s** — a ~3× kernel
speedup, diluted at the end-to-end level by non-GEMM attention compute.

Gated behind the **opt-in** `blas` feature so default builds stay zero-dependency:

```bash
# macOS (Accelerate is system-provided, no install needed)
cargo build --release -p cera-cli --features blas

# Linux (requires a system OpenBLAS install)
sudo apt-get install libopenblas-dev pkg-config
cargo build --release -p cera-cli --features blas
```

Default builds — `cargo build --release` with no features — use the pure-NEON
integer GEMM path on aarch64 and need no system libraries.

## GPU backends

Two GPU backends with runtime selection via `--device`:

- **Native Metal** (`--device metal`, macOS/iOS) — hand-written MSL shaders,
  single-encoder dispatch, GPU argmax. Decodes ~2× faster than llama.cpp on all
  tested Q4_0 models; prefill is competitive at short prompts and trails at long
  prompts (tracked in [`profile_longctx.md`](profile_longctx.md)).
- **wgpu** (`--device gpu`, cross-platform) — WGSL shaders targeting
  Metal/Vulkan/DX12/WebGPU. Portable but slower due to API translation overhead.

### Decode throughput vs llama.cpp (greedy, M1 Max, Q4_0)

| Model             | Test  | llama.cpp | cera Metal       |
|-------------------|-------|----------:|-----------------:|
| LFM2.5-VL-450M    | tg128 | 142       | **351** (+147%)  |
| LFM2.5-VL-450M    | tg512 | 139       | **321** (+131%)  |
| LFM2.5-VL-1.6B    | tg128 | 128       | **261** (+104%)  |
| LFM2.5-VL-1.6B    | tg512 | 122       | **223** (+83%)   |
| LFM2.5-Audio-1.5B | tg128 | 107*      | **226** (+111%)  |
| LFM2.5-Audio-1.5B | tg512 | 115       | **222** (+93%)   |

\* Audio tg128 is llama-bench noisy at r=10 (σ ≈ ±45). Steady-state Audio decode
sits near the tg512 number.

Both engines are primed with a 128-token prefill, then decode tok/s is timed over
the next 128 or 512 tokens (no timing includes the prefill). Reproduction:

```
# cera (per row, swap --max-tokens)
cera bench -m model.gguf --device metal --no-cache --prompt-tokens 128 --max-tokens 128 --runs 20 --warmup 3
cera bench -m model.gguf --device metal --no-cache --prompt-tokens 128 --max-tokens 512 --runs 20 --warmup 3

# llama.cpp (per row, swap -n)
llama-bench -m model.gguf -p 128 -n 128 -ngl 99 -r 10
llama-bench -m model.gguf -p 128 -n 512 -ngl 99 -r 10
```

### Prefill throughput vs llama.cpp (Q4_0, Metal, M1 Max)

| Model          | Prompt | llama.cpp | cera Metal      | Ratio |
|----------------|-------:|----------:|----------------:|------:|
| LFM2.5-VL-450M | 128    | 7619      | **8315** (+9%)  | 1.09× |
| LFM2.5-VL-450M | 1024   | 8213      | 6411            | 0.78× |
| LFM2.5-VL-450M | 4096   | 7008      | 2817            | 0.40× |
| LFM2.5-VL-1.6B | 128    | 2750      | 2567            | 0.93× |
| LFM2.5-VL-1.6B | 1024   | 2481      | 1864            | 0.75× |
| LFM2.5-VL-1.6B | 4096   | 2178      | 1135            | 0.52× |

Cera leads on 450M at p=128 and is competitive on 1.6B at p=128; llama.cpp's
BLAS-backed GEMM still wins at p=1024 and p=4096 on both models. PR
[#20](https://github.com/hyeons-lab/cera/pull/20) landed a **+26% improvement** to
Metal prefill at p=4096 (2227 → 2817 tok/s on LFM2.5-VL-450M) via a
one-simdgroup-per-query rewrite of `attention_prefill.metal`. Further work on the
long-prompt gap is tracked in [`profile_longctx.md`](profile_longctx.md).

```
# cera (per row, swap --prompt-tokens)
cera bench -m model.gguf --device metal --no-cache --context-size 8192 --prompt-tokens 128  --max-tokens 0 --runs 20 --warmup 3
cera bench -m model.gguf --device metal --no-cache --context-size 8192 --prompt-tokens 1024 --max-tokens 0 --runs 20 --warmup 3
cera bench -m model.gguf --device metal --no-cache --context-size 8192 --prompt-tokens 4096 --max-tokens 0 --runs 20 --warmup 3

# llama.cpp (per row, swap -p)
llama-bench -m model.gguf -p 128  -n 0 -ngl 99 -r 20
llama-bench -m model.gguf -p 1024 -n 0 -ngl 99 -r 20
llama-bench -m model.gguf -p 4096 -n 0 -ngl 99 -r 20
```

## Key optimizations

### Metal

- **GPU argmax** — greedy sampling on GPU, avoids 256KB logits readback (+57%)
- **Q6_K native embedding GEMV** — reads 52 MB Q6_K bytes directly, no f32 dequant
- **llama.cpp-derived fast Q4_0 GEMV** — pre-scaled y, uint16 nibble loads, sumy bias hoisting
- **Fused gate+up GEMV** — single dispatch for both FFN projections
- **Fused QK norm + RoPE** — 3 dispatches → 1 per attention layer
- **Vectorized attention V loads** — float2 loads in weighted-sum phase
- **Residual accumulate in GEMV** — `y += W×x` instead of separate add

### wgpu

- **Compute pass batching** — 300 passes → ~80 per token (+30%)
- **Fast Q4_0 GEMV** — ported Metal algorithm to WGSL with subgroupAdd
- **Multi-row f32 GEMV** — 8 rows per workgroup, 8× less input bandwidth

### CPU

- **Batched GEMM prefill** — reads each weight matrix once for all N tokens (vs N times with per-token GEMV)
- **8-column grouped Q4_0 GEMM** — decode weight blocks once, dot against 8 input columns
- **Integer Q4_0/Q8_0/Q6_K GEMV** — quantize activations to Q8_0, integer dot product via `vdotq_s32`
- **Pre-quantize shared inputs** — one Q8_0 quantization reused across Q/K/V and gate/up projections
- **NEON attention** — vectorized Q*K scores and softmax*V weighted sums with `vfmaq_f32`
- **3-phase batched prefill** — batch input projections (GEMM) → sequential core (conv/attention) → batch output projections (GEMM)
- **Software prefetch** in GEMV/GEMM inner loops
