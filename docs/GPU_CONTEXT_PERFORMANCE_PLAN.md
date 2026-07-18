# GPU Context Size + Performance Plan

This plan targets the wgpu backend first. Native Metal already has separate
long-context work in `benchmarks/profile_longctx.md`; the same direction applies
where layout or kernels are shared, but the immediate blocker observed on
Linux/wgpu is a buffer-binding limit:

```text
Buffer binding 4 range 268435456 exceeds max_*_buffer_binding_size limit 134217728
```

The failing run used `--context-size 8192` on `LFM2.5-350M-GGUF/Q4_0`. Cera tried
to bind a 256 MiB KV range, while the selected adapter allowed only a 128 MiB
storage binding. Dropping to `--context-size 4096` made the same model run, which
confirms this is a layout/binding problem rather than a model-load problem.

## Current Evidence

- `docs/IMPLEMENTATION_PLAN.md` still lists the original KV cache as simple
  contiguous storage and V2.1 as paged attention replacing contiguous KV.
- `benchmarks/BASELINE.md` says decode still issues 19 submits/token, but also
  records that batching decode into one command buffer was measured about 30%
  slower on Mac and Adreno. Do not reopen that as the first performance lever.
- `benchmarks/BASELINE.md` attributes decode time to quantized GEMV memory access:
  quantized decode GEMVs sustained roughly 25 GB/s while the f16 LM-head GEMV
  reached 106 GB/s on the same GPU.
- `benchmarks/GPU_FINDINGS_CORRECTION.md` reframes the main decode lever as
  quantized GEMV load pattern/vectorization, not submit count or logits readback.
- `benchmarks/profile_longctx.md` says long-context decode is dominated by
  attention: attention share rises to about 65% at ctx=4096, and prefill
  attention is about 80% at p=4096 after the later attribution fix.
- Recent history has already moved wgpu prefill forward:
  - `41a13b3` native wgpu flash-attention decode kernel.
  - `8bd9798` removed the old wgpu decode-attention path.
  - `5fbf275`, `34e3d6b`, `ac81ec1` register-tiled wgpu GEMM work for prefill.
  - `635942e` vectorized Q4_K/Q5_K GEMV weight loads.
  - `0f9abf8` added GPU I/O counters and benchmark harnesses.
  - `479e320` made KV allocation fallible, which helps large-context error
    reporting but does not solve per-binding limits.

## Goals

1. Make wgpu support larger context sizes without lowering `--context-size` just
   to fit adapter binding limits.
2. Preserve correctness for context shift, prefix-cache restore, hidden-state
   scratch paths, and batched prefill.
3. Improve GPU throughput where prior evidence says the time actually goes:
   long-context attention, quantized decode GEMV memory access, prefill readbacks,
   and non-greedy sampling readbacks.
4. Keep `--gpu-io` and benchmark output capable of proving whether a change
   helped.

## Non-Goals

- Do not pursue single-command-buffer decode as the primary fix. It was measured
  slower and the docs now mark that path WONTFIX.
- Do not treat the 128 MiB binding limit as a reason to reduce default context.
  The backend should adapt its layout to the adapter.
- Do not introduce a CUDA-only design. wgpu needs to remain portable across
  Vulkan, Metal, D3D12, and WebGPU.

## Phase 0: Baseline + Guardrails

Add a reproducible wgpu baseline matrix before changing layout:

```bash
./target/release/cera bench --bundle-id LFM2.5-350M-GGUF --quant Q4_0 \
  --device gpu --prompt-tokens 128 --max-tokens 128 --runs 5 --warmup 1 \
  --no-cache --context-size 4096 --gpu-io

./target/release/cera bench --bundle-id LFM2.5-350M-GGUF --quant Q4_0 \
  --device gpu --prompt-tokens 128 --max-tokens 128 --runs 1 --warmup 0 \
  --no-cache --context-size 8192 --gpu-io
```

Expected starting point:

- `context_size=4096` runs.
- `context_size=8192` may fail on adapters with 128 MiB max storage binding.
- For the local run on 2026-07-17, `LFM2.5-350M/Q4_0` at p128/tg128 measured:
  GPU prefill p50 112 tok/s, GPU decode p50 7.4 tok/s, CPU decode p50
  20.7 tok/s.

Add a small diagnostic to print adapter limits in the GPU load path:

- `max_storage_buffer_binding_size`
- `max_buffer_size`
- selected backend/adapter name
- computed KV bytes per layer and largest binding range

This makes future failures actionable instead of surfacing only a wgpu validation
panic.

## Phase 1: Make Large Context Legal

Start with a minimally invasive KV sharding layer for wgpu.

### 1.1 Introduce a GPU KV Layout Abstraction

Current shader and host code assume one contiguous `[seq_len x kv_dim]` K buffer
and one contiguous V buffer per attention layer. Replace direct buffer assumptions
in the wgpu model with a layout enum:

```rust
enum GpuKvLayout {
    Contiguous {
        k: wgpu::Buffer,
        v: wgpu::Buffer,
    },
    Paged {
        page_tokens: u32,
        k_pages: Vec<wgpu::Buffer>,
        v_pages: Vec<wgpu::Buffer>,
    },
}
```

Implementation surface:

- `cera/src/model/gpu_lfm2.rs`
- shared dense-transformer wgpu paths in `cera/src/model/transformer.rs` if they
  own separate KV buffers
- `cera/src/backend/shaders/flash_attention.wgsl`
- `cera/src/backend/shaders/attention_prefill.wgsl`
- `cera/src/backend/shaders/qk_norm_rope_batch.wgsl`
- `cera/src/backend/shaders/kv_shift.wgsl`

Keep `Contiguous` as the fast path when the computed binding range fits the
adapter.

### 1.2 Use Fixed Token Pages

Pick `page_tokens` from adapter limits:

```text
page_bytes = page_tokens * kv_dim * sizeof(f32)
page_bytes <= min(max_storage_buffer_binding_size / safety_factor, max_buffer_size)
```

For the adapter that failed at 128 MiB, use a much smaller operational page such
as 256 or 512 tokens. Smaller pages reduce binding pressure and line up with
existing prefill chunking.

Acceptance:

- `--context-size 8192` loads and runs p128/tg128 on the adapter that previously
  failed.
- `--context-size 16384` fails gracefully with a Cera error only if total
  allocation exceeds device memory or `max_buffer_size`, not because a single
  binding is too large.

### 1.3 First Shader Strategy: Page Loop on Host

For the first correctness-oriented version, avoid complex bindless-style
indirection. Dispatch attention over one page at a time and accumulate
online-softmax state across pages.

For decode:

- Reuse online-softmax state across pages: running max `m`, denominator `l`, and
  output accumulator.
- Dispatch one page-range kernel per page, then a small merge/finalize kernel if
  needed.
- This may increase submits at long context, but it removes the hard binding cap
  and creates a correctness baseline.

For prefill:

- Keep existing chunking.
- Process K/V pages in order for each query block.
- Validate against CPU on short contexts first, then contexts that require
  multiple pages.

Acceptance:

- Existing GPU parity tests pass at one-page context.
- Add a new test that forces tiny pages, for example 32 or 64 tokens, so paging is
  exercised without needing a huge model.
- `cera bench --device gpu --context-size 8192` no longer panics on binding size.

## Phase 2: Make Large Context Fast

Once correctness is stable, reduce the overhead introduced by page dispatch.

### 2.1 Tiled/Paged Flash Attention

Port the long-context guidance from `benchmarks/profile_longctx.md` to wgpu:

- one online-softmax flash-attention kernel that scans K/V tiles;
- page-aware addressing inside each tile;
- bounded threadgroup memory;
- decode path for `M=1`;
- prefill path for `M>1`, likely query blocks of 8-32 depending on head_dim and
  adapter limits.

This should replace the host-loop page prototype where supported.

Acceptance:

- ctx128/ctx2048/ctx4096 decode profile shows attention no longer scaling as
  steeply.
- p4096 prefill improves materially over the baseline and does not regress p128.
- `--gpu-io` does not show a large new readback count.

### 2.2 Avoid Full-KV Binding When Only Active KV Is Needed

Even before full paging is done, bind active ranges rather than capacity ranges
where wgpu allows it. The failing run bound the full context-sized allocation even
though p128/tg128 needed only 256 live slots. The backend should prefer:

- range binding for `[0, live_seq_len)` during attention;
- capacity binding only for kernels that need to write arbitrary future positions;
- diagnostic assertion that bound range <= adapter limit.

This is a lower-risk fix than full paging and may be enough for many bench shapes.

### 2.3 Context Shift on Paged KV

Current `kv_shift.wgsl` and host copies assume overlapping ranges inside a
contiguous buffer. Paged KV should implement shift as metadata movement first:

- rotate/drop page descriptors for full-page shifts;
- copy only partial page boundaries;
- apply RoPE correction to retained K tokens as today;
- avoid moving V data unless a boundary copy is unavoidable.

Acceptance:

- Existing `wgpu_kv_shift_oracle` still passes.
- Add a paged/tiny-page variant.
- Long chat generation across context rollover remains token-equivalent to CPU
  for a small fixture.

## Phase 3: Decode Performance Work

The docs point to quantized GEMV memory access as the decode bottleneck. Keep this
separate from context capability.

### 3.1 Finish Vectorized/Coalesced Quantized GEMV Loads

`635942e` already vectorized Q4_K/Q5_K GEMV loads. Extend the same discipline to
the hot Q4_0/Q8_0 paths observed in the local `LFM2.5-350M/Q4_0` run:

- avoid scalar per-weight extraction when rows can be read as aligned vectors;
- split kernels by quant layout where that improves coalescing;
- tune rows-per-workgroup per shape rather than using one global setting.

Acceptance:

- `CERA_GPU_PROFILE=1` shows FFN quantized GEMV GPU time falling.
- Achieved bandwidth closes part of the gap toward the f16 GEMV path.

### 3.2 Per-Shape Tuning

Implement the V2.7 direction from `docs/IMPLEMENTATION_PLAN.md`:

- introduce a `cera tune` or hidden benchmark mode for candidate GEMV workgroup
  parameters;
- cache results by adapter name, shader variant, quant type, and matrix shape;
- use conservative built-in defaults when no tuning cache exists.

Acceptance:

- no correctness changes;
- measurable decode improvement on at least two matrix shape classes: small
  projection rows and FFN rows.

## Phase 4: Reduce Readbacks

### 4.1 Prefill Readbacks

`benchmarks/BASELINE.md` reports about 12.9 MiB read back for a 512-token prompt,
independent of whether batched prefill or fallback prefill runs. Track where those
readbacks originate and keep intermediate state GPU-resident where possible.

Acceptance:

- `--gpu-io` prefill readback bytes drop for p512 without changing decode output.

### 4.2 Non-Greedy Sampling

Greedy decode reads back only a 4-byte token id. Non-greedy sampling still
downloads full vocab logits per token. Add GPU-side top-k/top-p or a two-stage
compact readback:

- GPU filter/reduction to a small candidate set;
- read back only candidate ids/logits;
- CPU RNG/sample over the compact set.

Acceptance:

- non-greedy `cera run` no longer reads full vocab logits every token;
- greedy path remains unchanged.

## Phase 5: Documentation + Bench Matrix

Update docs after each phase:

- `benchmarks/BASELINE.md`: measured before/after numbers and `--gpu-io` counters.
- `benchmarks/GPU_FINDINGS_CORRECTION.md`: only if a previous conclusion changes.
- `benchmarks/profile_longctx.md`: long-context attention results after
  paged/tiled flash attention.
- `docs/IMPLEMENTATION_PLAN.md`: mark contiguous KV replacement progress under
  V2.1.

Minimum benchmark table per milestone:

| Model | Quant | Context | Prompt | Decode | Device | Metrics |
|---|---|---:|---:|---:|---|---|
| LFM2.5-350M | Q4_0 | 4096, 8192, 16384 | 128 | 128 | wgpu | p50 tok/s + gpu-io |
| LFM2.5-350M | Q4_0 | 8192 | 4096 | 0 | wgpu | prefill p50 + gpu-io |
| LFM2.5-350M | Q4_K_M | 4096, 8192 | 512 | 128 | wgpu | prefill gate + decode |

## Recommended Order

1. Add adapter-limit diagnostics and convert the validation panic into a clear Cera
   error with computed KV sizes.
2. Bind active KV ranges where possible. This may quickly fix small live-sequence
   benches at large configured context.
3. Add the `GpuKvLayout` abstraction and paged buffers behind a tiny-page test mode.
4. Implement host-loop paged attention for correctness.
5. Replace host-loop paging with a page-aware tiled flash-attention kernel.
6. Continue decode GEMV vectorization and per-shape tuning.
7. Reduce prefill and non-greedy sampling readbacks.

This order produces useful checkpoints: first better errors, then the immediate
large-context capability fix, then the kernels that make the larger context worth
using.
