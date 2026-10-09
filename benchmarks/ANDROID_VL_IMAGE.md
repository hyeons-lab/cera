# LFM2.5-VL-450M with an image on Android: Cera vs llama.cpp

Cera (CPU, GPU, NPU) against llama.cpp (CPU, GPU, NPU) on one vision-language
model and one image, measured on a Snapdragon 8 Elite phone. The goal is to find
where Cera is slower and by how much, so the perf work can be aimed.

Raw data: [`android_vl_image_raw/`](android_vl_image_raw/). Harnesses:
[`../scripts/bench_android_vl.py`](../scripts/bench_android_vl.py) (the six-way matrix) and
[`../scripts/bench_android_ab.py`](../scripts/bench_android_ab.py) (paired, thermal-annotated A/B of one
change).

## Summary

Two measurements on the same phone, harness and image: the baseline (Cera at
`c5f16106`), and after the perf work in this branch. At an equal 210-token prompt
(the pug image at 512 px on its long side, 64 generated tokens, greedy), medians
of 5 passes, all 30 runs per measurement successful and the same caption:

| | TTFT | vision tower | prefill (TTFT minus tower) | decode | time to 64th token |
|---|---:|---:|---:|---:|---:|
| Cera CPU | **1,892 -> 438 ms** | 1,697 -> **252 ms** | 195 -> **186 ms** | 207 -> **242 tok/s** | **2,204 -> 699 ms** |
| llama.cpp CPU | 1,149 -> 1,091 ms | 498 -> 475 ms | 651 -> 616 ms | 218 -> 236 tok/s | 1,452 -> 1,355 ms |
| Cera GPU (wgpu/Vulkan) | **4,723 -> 246 ms** | 2,229 -> **138 ms** | 2,494 -> **108 ms** | 119 -> 163 tok/s | **5,269 -> 632 ms** |
| llama.cpp GPU (OpenCL) | 529 -> 469 ms | 206 -> 182 ms | 323 -> 287 ms | 156 -> 166 tok/s | 956 -> 863 ms |
| Cera NPU (Hexagon) | 138 -> 133 ms | 99 -> 97 ms | 39 -> **36 ms** | 126 -> **160 tok/s** | 639 -> **530 ms** |
| llama.cpp NPU (HTP) | 277 -> 267 ms | 111 -> 105 ms | 166 -> 162 ms | 153 -> 156 tok/s | 685 -> 667 ms |

TTFT is everything before the first decoded token (preprocess, vision tower, image-token prefill, text
prefill); "prefill" is the derived remainder after the tower. Measured through `llama-mtmd-cli`, Cera's
image-and-text prefill is 3.3x (CPU), 2.7x (GPU) and 4.5x (NPU) faster than llama.cpp's here. That is a
comparison with llama.cpp's multimodal path, not with its text prefill: on the same weights `llama-bench`
prefills at about 2,000 tok/s on a cool phone (1,150 warm), 3 to 7x what `llama-mtmd-cli` reaches on the
image tokens, and a gated text-only comparison puts Cera 17% (CPU) and 27% (GPU) behind llama.cpp (see
"Prefill and decode, text only").

(The llama.cpp rows are the same binary both times; their movement is run-to-run
noise, 3 to 12%. The "after" figures are one six-way run,
`matrix_512px_final.json`, after every change below.)

Where Cera now stands on this image:

- **CPU:** ahead on every metric: first token (438 vs 1,091 ms), vision tower (252 vs 475 ms), decode
  (242 vs 236 tok/s, ranges 234 to 250 and 214 to 241) and end to end (699 vs 1,355 ms). Text-only decode
  against llama.cpp is 1.01 in paired runs (see "CPU decode").
- **GPU:** ahead on first token (246 vs 469 ms), tower (138 vs 182 ms) and end to end (632 vs 863 ms),
  from 9x behind on first token. Decode is level (163 vs 166 tok/s, ranges overlapping).
- **NPU:** ahead on every metric: first token (133 vs 267 ms), decode (160 vs 156 tok/s) and end to end
  (530 vs 667 ms).

### The same comparison on a large image (1024x771)

LFM2-VL tiles a large image, and both engines now do the same work: 6 tiles in a 3x2 grid plus a 576x416
thumbnail, 1,795 prompt tokens, all six cells successful, and the same caption on five of six (the NPU
llama.cpp cell diverges after a few words). Medians of 5 passes
(`matrix_native_1024x771_final.json`):

| | TTFT | vision tower | prefill (TTFT minus tower) | decode | time to 64th token |
|---|---:|---:|---:|---:|---:|
| Cera CPU | **4,930 ms** | **3,303 ms** | **1,627 ms** | 135 tok/s | **5,387 ms** |
| llama.cpp CPU | 11,811 ms | 5,190 ms | 6,621 ms | 174 tok/s | 12,177 ms |
| Cera GPU | **2,622 ms** | **1,398 ms** | **1,224 ms** | 100 tok/s | **3,251 ms** |
| llama.cpp GPU | 4,023 ms | 1,613 ms | 2,410 ms | 122 tok/s | 4,558 ms |
| Cera NPU | **1,253 ms** | 997 ms | **256 ms** | **143 tok/s** | **1,691 ms** |
| llama.cpp NPU | 2,169 ms | 901 ms | 1,268 ms | 133 tok/s | 2,646 ms |

Cera is ahead on first token and end to end on all three backends, and its prefill through `llama-mtmd-cli`
is 4.1x (CPU), 2.0x (GPU) and 5.0x (NPU) faster than llama.cpp's (a comparison with llama.cpp's multimodal
path; see "Prefill and decode, text only" for the like-for-like text figures). This single matrix run exposed decode at a long context as the one
lag: 22% behind on CPU and 18% on GPU at 1,795 tokens, and the NPU tower 10% behind at 1,024 patches per
tile. The decode lags have since been closed (see "Long-context decode"): paired against llama.cpp at the
same 1,795 tokens, CPU decodes at 172.4 against 168.7 tok/s (f16 KV cache, the new default, x1.007) and GPU at
154.8 against 122 tok/s. The NPU tower gap remains.

## Prefill and decode, text only (re-run, 2026-10-09)

The prefill ratios above divide by `llama-mtmd-cli`'s "prompt eval time", and that path is much slower than
llama.cpp's own text prefill on the same weights, so they do not say how the two engines' kernels compare. A
re-run on the same phone (79% battery on USB) measured both: this branch's tip, the same two six-way
matrices, and a text-only benchmark of `cera bench` against `llama-bench` on the same
`LFM2.5-VL-450M-Q4_0.gguf` (pp512 for prefill; a 128-token decode, from a 128-token context in Cera and from
an empty one in `llama-bench`, which slightly disfavours Cera). `llama-bench` ran 8 threads with its default
flash attention and f16 KV cache, Cera with its defaults.

CPU prefill depends strongly on how hot the SoC is (it is compute bound and runs every core flat out), so the
text-only run is thermally gated: before every invocation the harness waits for the AP sensor to fall to 28 C
or less (it read 21 to 28 C at every start), the engine order alternates between rounds, and the result is the
median of three rounds (`android_vl_image_raw/rerun_20261009/cool_pp_tg.py`). An earlier, ungated pass of the
same binaries, run straight after the 20 minutes of matrices, measured CPU prefill at 983 tok/s for Cera and
1,158 for llama.cpp, about 40% below the gated figures for both; its GPU and NPU figures matched the gated ones.

| Text only, gated | Cera prefill | llama.cpp prefill | ratio | Cera decode | llama.cpp decode | ratio |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 1,715 tok/s | 2,056 | 0.83 | 252 | 245 | 1.03 |
| GPU | 2,419 | 3,301 | 0.73 | 173 | 182 | 0.96 |
| NPU | 8,644 | 8,037 | 1.08 | 159 | 161 | 0.99 |

Decode is level on all three backends. Prefill is behind llama.cpp on CPU (0.83x; rounds 1,774, 1,590 and 1,715
tok/s against 1,880, 2,056 and 2,082) and on GPU (0.73x; the rounds agree within 1%), and ahead on the NPU
(1.08x).

The same re-run through the image path, prefill as prompt tokens over (TTFT minus tower) for both engines,
medians of 5 passes, all 60 runs successful:

| | 210 tokens: Cera prefill | llama.cpp | decode, Cera | llama.cpp | 1,795 tokens: Cera prefill | llama.cpp | decode, Cera | llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 1,252 tok/s | 386 | 237 | 226 | 1,091 | 275 | 172 | 175 |
| GPU | 1,930 | 745 | 178 | 168 | 1,464 | 748 | 155 | 129 |
| NPU | 6,122 | 1,338 | 160 | 154 | 7,103 | 1,425 | 140 | 133 |

These matrices run the six cells back to back with no cooldown (as the method above says), so the CPU rows are
warm-phone figures. TTFT on the large image: 4.8 s against 11.6 s (CPU), 2.6 against 4.0 s (GPU), 1.2 against
2.2 s (NPU). Cera's image prefill is in line with its own text prefill (1,091 tok/s on 1,795 image tokens
against 1,715 cool at pp512). `llama-mtmd-cli` is not: its CPU image prefill (275 to 386 tok/s) is 5 to 7x below
`llama-bench`'s gated text prefill on the same weights (2,056 tok/s; 3 to 4x below even the warm 1,158), and its
GPU image prefill (745 tok/s) is 4x below `llama-bench`'s (3,301). Why was not investigated here. The consequence is that the CPU first-token lead on a
large image comes from the vision tower (3.1 against 5.1 s) and from that slow multimodal prefill, not from
faster prefill kernels; CPU decode is level with llama.cpp within the run-to-run spread (a 5 to 10% edge in
some cells is inside the ranges). Raw data: `android_vl_image_raw/rerun_20261009/`.

### GPU prefill: a register-tiled causal attention kernel

The text-only profile of a 512-token GPU prefill (`CERA_GPU_PROFILE=1`, wgpu timestamps) put 64% of the 205 ms
in the Q4_0 GEMMs and 22.6% (46 ms, six layers) in `attention_prefill`, which ran at about 0.07 TFLOPS. The
scalar kernel dates from before the head_dim-64 tiled kernel (`attention_flash_hd64.wgsl`, 0.2 TFLOPS on this
Adreno), which only handled f32 K/V and bidirectional windows, so the causal prefill over the packed-f16 cache
never used it. `attention_prefill_hd64.wgsl` is that tiled kernel with the cache's packed-f16 loads and the
prefill's windows (causal with `start_pos`, bidirectional with a media prefix, split into `q_base`/`n_sub`
dispatches); it is chosen for head_dim 64 on desktop adapters and Adreno, and `CERA_WGPU_ATTN_SCALAR=1` selects
the old kernel for an A/B. The attention phase falls from 45.9 to 14.8 ms and the whole 512-token prefill from
206 to 174 ms under the profiler.

Gated like the text-only runs above (AP sensor at 28 C or less before every invocation, rotating order, 3 rounds,
medians; `android_vl_image_raw/causal_attention_20261009/`), prefill tok/s on the GPU:

| Prompt | scalar attention | tiled attention | llama.cpp (OpenCL) | tiled / scalar | tiled / llama.cpp |
|---|---:|---:|---:|---:|---:|
| 512 tokens | 2,414 | 2,844 | 3,275 | 1.18 | 0.87 |
| 1,024 | 1,978 | 2,636 | 3,135 | 1.33 | 0.84 |
| 2,048 | 1,427 | 2,251 | 2,891 | 1.58 | 0.78 |

The gain grows with the context because attention is quadratic; the rest of the gap is the Q4_0 GEMMs
(about 2.2 TFLOPS) and, at 2,048 tokens, attention again. Accuracy on the device: over seven prompts of 60 to
3,600 characters the full next-token logits of the tiled kernel agree with the CPU's as closely as the scalar
kernel's do (cosine 0.9999 or better on six; the seventh, a noisy prompt, is 0.9989 against the scalar kernel's
0.9957), with the same argmax on every prompt.

#### What the Adreno 830 can do (`wgpu_peak_bench`)

The GPU kernels' TFLOPS mean little without the ceiling, so `cera/examples/wgpu_peak_bench.rs` measures it
through the same SPIR-V passthrough path as the production GEMMs: dependent-free FMA chains, a packed int8
dot product, and a coalesced streaming read. Each kernel runs about 25 ms per dispatch, 8 dispatches per
submit, median of 5 rounds; three runs agree within 1%.

| Ceiling | Measured | Production kernel | Share of ceiling |
|---|---:|---|---:|
| fp32 FMA | 3.60 TFLOPS | tiled causal attention, 0.23 TFLOPS | 6% |
| fp16 FMA | 7.10 TFLOPS | `gemm_stream_q4_0_k64`, 2.2 TFLOPS in the model (2.73 in `gemm-bench`) | 31% (38%) |
| int8 packed dot (`SPV_KHR_integer_dot_product`) | 6.83 TOPS | none | no gain over fp16 |
| read bandwidth | 62.5 GB/s | decode: 220 MB of weights per token at 173 tok/s is 38 GB/s | 61% |

fp16 runs at twice the fp32 rate, and the int8 dot product is no faster than fp16 FMA, so an int8
formulation of the prefill GEMM would buy nothing on this GPU. The GEMM does more than FMAs (it unpacks
and scales each 4-bit weight, about 5 instructions against 8 half4 FMAs per weight), so its practical
ceiling is below the 7.1; roughly 60% of it, 4 TFLOPS, is a fair target and would take the 512-token
chunk's GEMMs from 133 ms to about 70. Attention is the furthest from its ceiling in relative terms (6% of
fp32, 3% of fp16) but is the smaller block at 512 tokens. fp16 requires a SPIR-V passthrough kernel: wgpu
does not advertise `SHADER_F16` on this driver, so a WGSL kernel cannot use it.

#### What holds the GEMM at 31%

`scripts/gemm-ablate/gen.py` builds variants of `gemm_stream_q4_0_k64.slang` that each remove one component,
and `cera gemm-bench --spv` times them against the production kernel on the phone (same bindings and grid; the
ablations compute wrong values on purpose). One thread owns one weight row and 32 output columns: per weight
it dequantizes a nibble, reads the 8 half4 of the B tile from shared memory (the same address for the whole
workgroup) and does 8 half4 FMAs. `gemm-bench` reports `a0_base`, the same source as the built-in kernel, at
1.94 ms for 4608x512x1024 (2.5 TFLOPS); the first kernel in a run reads 20% slower from clock ramp, so
compare within the list.

| Variant (4608x512x1024) | ms | TFLOPS | Cost of what was removed |
|---|---:|---:|---|
| `a0_base` (production source) | 1.94 | 2.5 | |
| `a1_nodequant` | 1.69 | 2.9 | dequant, 0.25 |
| `a5_noglobal` (no weight loads) | 1.74 | 2.8 | weight loads, 0.20 |
| `a4_nostage` (no B staging or barrier) | 1.65 | 2.9 | staging, 0.29 |
| `a2_nolds` (B from registers) | 1.14 | 4.2 | shared-memory reads, 0.80 |
| `a7_ldsfma` (shared reads + FMA only) | 1.60 | 3.0 | |
| `a3_nolds_nodequant` | 0.78 | 6.2 | |
| `a6_fmaonly` (nothing else) | 0.77 | 6.3 | |

- **The FMA pipe is not the limit.** The structure with only FMAs runs at 6.3 TFLOPS, 88% of the 7.1
  ceiling. The 31% is everything around the FMAs.
- **The costs add up** rather than overlap (shared reads +0.83 and dequant +0.37 on top of 0.77 gives 1.97, the
  measured 1.94), which is what an issue-bound kernel does and a bandwidth-bound one does not. Shared-memory
  reads are the largest piece (about 43% of the time), then FMAs (40%) and dequant (19%); staging and weight
  loads overlap with those.
- **Register blocking helps less than the ablation suggests.** Two weight rows per thread (`r2`, 128 threads per
  workgroup, bit-exact) shares each shared read between two rows and takes the GEMMs 5% faster at 4608x512x1024,
  18% at 1024x512x4608 (ffn_down), 8% at 3072x512x1024, 20% at 1024x512x1024, 8% at 10752x128x2048 and 8% at
  4608x2048x1024; it is neutral at 9216x512x1024 and 32% slower at 2048x64x2048, where the smaller workgroups
  under-fill the GPU. Four rows per thread (`r4`) is 60% slower (64 accumulator registers). Weighted by the
  model's shapes `r2` saves about 9% of the GEMM time, 7% of a prefill.
- **Reading B straight from a storage buffer** (no staging, no barrier, no shared memory) is 2.8x slower than
  shared memory (0.87 against 2.5 TFLOPS), so only a texture path could beat it. The adapter reports subgroups of
  64, 32 KiB of workgroup memory and no cooperative matrices.
- **Tried and not worth it:** a magic-number nibble conversion (`e1`, +/-1%), and 64-bit staging loads (`e2`,
  24% slower). A 128-bit shared-memory read variant lost the GPU context and was dropped.
- **No hardware matrix unit to use:** `wgpu_coopmat_probe` lists zero cooperative-matrix configurations on this
  adapter, and the int8 dot product is no faster than fp16 FMA (above).

What it would take to do much better than `r2` is a different tile: dequantize the weights once into a half
tile in shared memory and run a register outer product (4x4 or 4x8 per thread) so each shared read feeds
many FMAs, which is what the older `mul_mat_reg_tile` kernels do and why they lost to this one here. The
streaming kernel's structure caps it near 40% of the fp16 ceiling; `r2` is the cheap step toward that.

### GPU prefill: a two-row GEMM and a fused gate/up

The ablation above led to two kernels, both generated by `scripts/gemm-ablate/gen.py --production`:

- `gemm_stream_q4_0_k64_r2` gives each thread two weight rows, so one read of the B tile from shared memory
  feeds two rows. It is bit-exact with the one-row kernel and the host uses it from 64 workgroups
  (`ceil(m/256) * n_pad/32`) up, where it was 5 to 20% faster; below that it loses, up to 32% at 2048x64x2048,
  so small chunks keep the one-row kernel (`CERA_WGPU_GEMM_R1=1` forces it everywhere).
- `gemm_stream_q4_0_k64_gateup` runs the FFN's gate and up projections as one dispatch: a thread owns the same
  row of both, both share each B tile read, and the epilogue stores `silu(gate) * up` (the clamp and formula of
  `silu_mul_inplace`) instead of two outputs and a separate pass. It also needs one transpose of the
  activations instead of two. It is skipped while a LoRA adapter is active (the delta applies to the raw
  projections), and `CERA_WGPU_FUSE_GATE_UP=0` turns it off. Its workgroup size matters: 256 threads took 4.3 ms
  per layer, 128 took 3.6 (less than two two-row GEMMs, 3.7), 64 took 6.0.

Per 512-token chunk under the profiler (`CERA_GPU_PROFILE=1`), GPU time falls from 173 to 153 ms: the Q4_0 GEMMs
from 132 to 119 ms (gate/up 57 ms in 16 fused calls, 48 two-row GEMMs 58 ms, 12 one-row GEMMs 4 ms), the
transposes from 8.7 to 7.8 ms (76 dispatches instead of 92) and `silu_mul` (6.5 ms) is gone. Gated like the runs above (AP sensor at 28 C or
less, rotating order, 3 rounds, medians; `android_vl_image_raw/gemm_two_row_20261009/`), prefill tok/s on the GPU:

| Prompt | previous | two-row only | this change | llama.cpp (OpenCL) | vs previous | vs llama.cpp |
|---|---:|---:|---:|---:|---:|---:|
| 512 tokens | 2,841 | 3,048 | 3,230 | 3,293 | 1.14 | 0.98 |
| 1,024 | 2,626 | 2,809 | 2,960 | 3,144 | 1.13 | 0.94 |
| 2,048 | 2,255 | 2,360 | 2,470 | 2,889 | 1.10 | 0.85 |

The full-vocabulary logits are identical to the previous kernels' (maximum difference 0.0000 on five prompts of 60
to 3,600 characters) and equally close to the CPU's. What remains at 2,048 tokens is attention, which reads more
keys per chunk.

## Baseline results (before the perf work)

| | TTFT | vision tower | decode | time to 64th token |
|---|---:|---:|---:|---:|
| Cera CPU | 1,892 ms | 1,697 ms | 207 tok/s | 2,204 ms |
| llama.cpp CPU | 1,149 ms | 498 ms | 218 tok/s | 1,452 ms |
| Cera GPU (wgpu/Vulkan) | 4,723 ms | 2,229 ms | 119 tok/s | 5,269 ms |
| llama.cpp GPU (OpenCL) | 529 ms | 206 ms | 155 tok/s | 956 ms |
| **Cera NPU (Hexagon)** | **138 ms** | 99 ms | 126 tok/s | **639 ms** |
| llama.cpp NPU (HTP) | 277 ms | 111 ms | **153 tok/s** | 685 ms |

- **CPU:** the whole gap was the vision tower. Cera's LLM prefill was 3.3x faster
  than `llama-mtmd-cli`'s (194 ms vs 641 ms; see "Prefill and decode, text only" for why that is not a text
  prefill comparison), but its tower was 3.4x slower.
- **GPU:** Cera was slower in every phase: tower 10.8x, LLM prefill 7.6x, decode
  1.3x.
- **NPU:** Cera reached the first token in half the time because its prefill was
  5x faster than `llama-mtmd-cli`'s, but decoded 18% slower.

## Setup

| | |
|---|---|
| Device | Samsung Galaxy S25 Ultra (SM-S938U1), Snapdragon 8 Elite (SM8750): 2 prime cores at 4.47 GHz plus 6 cores at 3.53 GHz, Adreno 830, Hexagon v79. Android 17. |
| Date | 2026-10-06 |
| Power | On USB, battery 77 to 79% throughout. Baseline: skin temperature 36 C at the start of the 512 px run, 40 C at the end; the native-size run started at 40 C and ended at 42 C. After: 37.8 C throughout. |
| Model | LFM2.5-VL-450M, `Q4_0` weights (219,311,264 bytes), `Q8_0` mmproj (102,815,168 bytes) |
| Image | `cera/tests/fixtures/pug.jpg` (1024x771). The equal-token scenario uses the same file resized to 512x385 (`sips -Z 512`). |
| Prompt | `Describe this image in detail.` Greedy (`--temperature 0`), at most 64 new tokens, all runs generate 64. |
| Cera | `cera-cli` with `--features gpu,hexagon`, NDK r30.0.16248370, Cera's own DSP skels (`cera/src/backend/hexagon/skels/`). Baseline: commit `c5f16106`. After: this branch's tip. |
| llama.cpp | `llama-mtmd-cli` from the `llama-cpp-android-arm64-snapdragon` CI artifact (id 10896183786) of upstream commit `4e7481175`, built with OpenCL and Hexagon. That is one commit after the `171e8846` that the Leap SDK pins; the difference touches only a Hexagon CMake file and a script. It is upstream, so it does not carry the Leap patch stack, and its CPU backend is one `armv8.7a` variant, not the runtime-selected variants the Leap SDK ships. llama.cpp used its own skels from its own library directory. |

## Method

Follows `scripts/bench_android.sh`, adapted to an image prompt:

- **Placement:** a sweep over core masks (all 8, the 6 mid cores, the 2 prime
  cores) picked the best CPU placement for each engine. All 8 cores won for both
  (table below). GPU and NPU cells are not pinned.
- **Passes:** 2 discarded warm passes, then 5 measured passes. Each pass runs all
  six cells in a fixed interleaved order with no idle in between, so thermal
  drift lands on every engine alike.
- **Equal input:** both engines saw exactly 210 prompt tokens (192 image tokens
  plus 18 text tokens). Both read the same GGUF files. The vision tower runs on
  the cell's own device in both engines (`-mmdev` follows `-dev` for llama.cpp).
- **Safety:** the shell is marked OOM-killable before each run (a GPU run rebooted
  this phone in the past), and the harness refuses to run below 30% battery.
- **Output check:** every cell produced the same opening caption (`A pug dog with
  a wrinkled face and ...`). CPU runs say "dark brown eyes" and accelerator runs
  say "dark eyes", consistently in both engines.

Metric definitions (each engine's own report, no wall-clock stitching):

- **TTFT**, everything before the first decoded token. Cera: its `Image prefill`
  line (preprocess, vision tower, image-token prefill, text prefill).
  llama.cpp: the sum of `mtmd batch encoding done` (vision tower) and
  `prompt eval time` (decode of the image and text tokens). llama.cpp's number
  leaves out JPEG decode and resize, which Cera includes (2 to 3 ms here).
- **Decode**, tokens per second as each engine reports it.
- **Time to 64th token**, TTFT plus 63 / decode.

## Baseline detail at an equal token count (512 px pug, 210 tokens)

Median of 5 measured passes, with the min to max range:

| Cell | TTFT ms (range) | Decode tok/s (range) | Time to 64th token |
|---|---:|---:|---:|
| Cera CPU | 1,892 (1,838 to 1,916) | 207.1 (202 to 209) | 2,204 |
| llama.cpp CPU | 1,149 (1,131 to 1,369) | 217.7 (185 to 228) | 1,452 |
| Cera GPU | 4,723 (3,807 to 4,738) | 118.7 (113 to 152) | 5,269 |
| llama.cpp GPU | 529 (466 to 561) | 155.5 (146 to 158) | 956 |
| Cera NPU | 138 (135 to 143) | 125.5 (123 to 128) | 639 |
| llama.cpp NPU | 277 (258 to 280) | 153.4 (150 to 157) | 685 |

All 30 measured runs (15 per engine) succeeded.

### Where the time goes

Cera (its `VL phases` line), median ms:

| Cell | preprocess | vision tower | image prefill (192 tok) | text prefill (18 tok) | total |
|---|---:|---:|---:|---:|---:|
| Cera CPU | 2.3 | 1,696.8 | 137.8 | 56.2 | 1,892.4 |
| Cera GPU | 2.8 | 2,229.3 | 1,556.9 | 900.6 | 4,723.1 |
| Cera NPU | 3.0 | 98.6 | 32.9 | 0.0 | 138.1 |

llama.cpp, median ms. Its prompt eval covers all 210 tokens (image and text):

| Cell | vision tower | prompt eval (210 tok) | TTFT |
|---|---:|---:|---:|
| llama.cpp CPU | 498 | 641 | 1,149 |
| llama.cpp GPU | 206 | 323 | 529 |
| llama.cpp NPU | 111 | 166 | 277 |

Cera's NPU text prefill reads 0.0 ms for 18 tokens, so that phase is either
folded into the image-prefill batch or not timed separately; the total is
unaffected, but do not read the 0.0 as a measurement.

### CPU placement sweep

4 runs per cell after 1 warm run, same prompt and image:

| Cell | TTFT ms | vision tower ms | decode tok/s | time to 64th token ms |
|---|---:|---:|---:|---:|
| Cera, 8 cores | 1,702 | 1,531 | 212.4 | 2,007 |
| Cera, 6 mid cores | 1,767 | 1,583 | 205.9 | 2,095 |
| Cera, 2 prime cores | 1,884 | 1,692 | 199.7 | 2,206 |
| llama.cpp, 8 threads | 1,105 | 478 | 219.1 | 1,393 |
| llama.cpp, 6 threads | 1,429 | 620 | 203.1 | 1,768 |
| llama.cpp, 2 threads | 2,294 | 1,036 | 202.8 | 2,605 |

Cera's tower barely moves with the core count (1,531 to 1,692 ms from 8 cores
down to 2), which suggests the tower is not parallelizing across rows the way the
LLM prefill does. That is a hypothesis from this one observation, not a finding.

## Baseline at native size (1024x771, not an equal-token comparison; superseded)

This is the baseline, before tiling was implemented; the equal-work result is in
the Summary. The engines did different amounts of work here, and the difference was a Cera
gap, not a design choice: LFM2-VL tiles large images, llama.cpp follows that
reference (6 tiles plus a thumbnail, 1,795 prompt tokens), and Cera encodes only
the thumbnail (252 prompt tokens). See "Vision tiling" below. These numbers
therefore flatter Cera by about 7x less vision and prefill work, and the run
started 4 C warmer than the equal-token run.

| Cell | tokens | TTFT ms | vision tower ms | decode tok/s |
|---|---:|---:|---:|---:|
| Cera CPU | 252 | 2,427 | 2,177 | 198.1 |
| llama.cpp CPU | 1,795 | 11,595 | 5,084 | 167.6 |
| Cera GPU | 252 | 7,811 | 4,218 | 77.9 |
| llama.cpp GPU | 1,795 | 5,886 | 2,366 | 74.5 |
| Cera NPU | 252 | 181 | 132 | 123.3 |
| llama.cpp NPU | 1,795 | 2,466 | 1,043 | 131.5 |

## Vision tiling (a correctness gap found while benchmarking; fixed)

LFM2-VL does not simply downscale a large image. Its reference processor
(Hugging Face `Lfm2VlImageProcessorFast`, mirrored by llama.cpp's
`mtmd_image_preprocessor_lfm2` in `tools/mtmd/mtmd-image.cpp`, constants from the
model's `processor_config.json`) works like this:

1. Align each side to 32 px (patch 16 times the 2x2 merge). The single-image
   budget is 256 tokens, which is 262,144 pixels.
2. If the aligned area exceeds that budget times a 2.0 tolerance (524,288 px,
   roughly a 724x724 image), **tile it**: pick a grid of 2 to 10 tiles whose
   aspect ratio is closest to the image's, resize to 512 px per tile, encode each
   tile (256 tokens), and add a thumbnail (the aspect-preserving single-image
   resize) after the tiles.
3. Otherwise encode the single resized image, with no thumbnail.

Worked through for the two images used here (checked against what both engines
actually produced):

| Image | Aligned area | Tiles? | Image tokens | llama.cpp prompt | Cera prompt |
|---|---:|---|---:|---:|---:|
| 512x385 | 196,608 | no, 512x384 single image | 192 | 210 | 210 |
| 1024x771 | 786,432 | yes, 3x2 grid = 6 tiles of 256 = 1,536, plus a 576x416 thumbnail of 234 | 1,770 | 1,795 | 252 |

Cera has no tiling code (nothing in `vision_preprocessor.rs`, `vision_encoder.rs`
or the session mentions tiles or a thumbnail). For a large image it encodes only
what the reference would use as the thumbnail, 234 tokens, and drops the six
tiles. The model sees a downscaled picture it was not trained to receive for
that size, so detail is lost (small text, fine features) even though nothing
errors. `--max-long-size` can only shrink the image further, and the CLI help
calls the unset behavior "full model resolution", which is not what happens.

This is not a perf gap in the usual sense (Cera is faster at native size because
it does about 7x less work), but fixing it will add vision-tower and prefill work
for large images, so it belongs next to the perf items. Verifying a fix needs the
tile layout and special tokens (`<|img_start|>`, per-tile row and column markers,
`<|img_thumbnail|>`, `<|img_end|>`) to match the reference token for token.

### Resolution

Implemented as the reference does it (`vision_preprocessor.rs`, `session.rs`):

- **Layout.** A large image becomes `<|image_start|>`, each tile preceded by its `<|img_row_R_col_C|>`
  marker in row-major order, `<|img_thumbnail|>` and the thumbnail, `<|image_end|>`. The session returns
  that interior from `encode_image_rows` with the markers embedded as ordinary token rows, so both
  prompt routes and the raw-pixel path pick it up. A model that cannot embed the markers falls back to
  the thumbnail with a warning that says why. A `--max-long-size` cap still means one image.
- **A second discrepancy, the resize.** The reference calls a Pillow-compatible bilinear (separable,
  widened by the scale factor when shrinking so it averages the pixels it covers, 22-bit fixed point);
  Cera used a two-tap bilinear that reads two source pixels per output sample and drops the rest on a
  downscale. llama.cpp's `resize_pillow` is ported exactly and used for every VL resize.
- **Verification** against llama.cpp built from source on the Mac with the same model and projector. The
  1024x771 pug gives 6 tiles in a 3x2 grid and a 576x416 thumbnail, **1,795 prompt tokens (llama.cpp:
  1,795)**, and the same scene in the caption on CPU, GPU and NPU. The resize digests match an independent
  implementation of the algorithm. Feeding both engines the same decoded PNG, llama.cpp's own thumbnail
  and Cera's thumbnail pixels give **bit-identical embeddings** (relative RMS 0.00000%), so the resize is
  pixel-exact; Cera's f32 tower then differs from llama.cpp's by the known 2.8%.
- **What is still different.** On a JPEG input the thumbnail embeddings differ by 3.6% from llama.cpp's,
  all of it the JPEG decoder (stb_image against the `image` crate); it predates tiling. The per-tile
  statistics could not be compared one by one because the reference's debug dump keeps only the last chunk.

## What the perf work found and changed

### CPU vision tower: 1,375 ms to 540 ms, then to about 265 ms (281 ms in the matrix)

Profiling the tower on the phone split it into linears 980 ms, attention 255 ms
and the rest about 120 ms. Three changes, in `vision_encoder.rs`:

- The 74 Q8_0 linears dequantized every weight row to f32 and ran one `dot_f32`
  per (row, token) pair, about 130 GFLOP/s. They now quantize the activations to
  Q8_0 once per distinct input (Q/K/V share one) and run the int8 GEMM the LLM
  prefill uses (980 -> 300 ms). llama.cpp does the same for a Q8_0 weight.
- Every parallel section ran on rayon, which pays a park/unpark each time it
  follows work on the GEMM's spinning `RowPool` threads. Moving them all to
  `par_rows_n` removed about 35 ms that moving only the transpose had made worse.
- Attention re-read a head's whole K and V (about 400 KB) from L2 for every
  query. It now uses the tiled flash-attention kernel, one task per (head, 32
  queries) (245 -> 125 ms).

That left 550 ms, and a per-phase profile (`CERA_VIT_PROFILE=1`, 768 tokens)
showed where it went:

| Phase | Before | After |
|---|---:|---:|
| Q/K/V, output, up and down GEMMs | 300 ms | 87 ms |
| Attention | 128 ms | 123 ms |
| GELU | 55 ms | 9 ms |
| Projector (mm.1, mm.2) | 29 ms | 3.6 ms |
| Patch embed | 14 to 21 ms | 8 to 14 ms |
| LayerNorm, quantize, bias, residual | 25 ms | 24 ms |
| Total | 552 ms | 262 ms |

- **GEMMs.** The Q8_0 i8mm kernel was a 2x2 tile with a scalar epilogue per
  block, about 19% of the phone's int8 peak. A Q8_0 block's quants are exactly the
  int8 values a recentered Q4_0 nibble decodes to, so each weight is repacked once
  at load (43 ms for the whole mmproj, one extra copy of each weight) into the
  8-row smmla layout the LLM's Q4_0 prefill already uses, and that tested 8x4
  row-major kernel runs it unchanged, writing straight into the `[token][row]`
  buffer (no transpose pass). Hosts without i8mm, or weights whose rows are not a
  multiple of 8, keep the old kernel.
- **GELU.** The scalar `tanh` is a libm call per element, 55 ms for 29 M elements.
  A NEON path computes `tanh(u)` as `1 - 2 / (exp(2u) + 1)` with the lane-exact exp
  twin; it saturates without branches and matches the scalar to 1e-6 over
  [-60, 60].
- **Projector.** It ran the f32 batched matmul; it now takes the same int8 GEMM.

The embedding stays 2.46% relative RMS from the f32 path (cosine 0.9997), the
same as before these three, so none of them added error. `CERA_VIT_INT8=0`
restores f32 end to end. Attention was then 47% of the tower: 22 GFLOP in 123 ms is
about 176 GFLOP/s, well under the f32 peak.

**Attention, 123 to 79 ms.** The tower ran its attention through the LLM's causal flash kernel, which
re-transposes every key tile once per group of queries (8 times per tile), spends about 60 instructions per
32 FMAs on scores and carries online-softmax rescaling a ViT does not need. A ViT-shaped NEON kernel
(`vit_attention_chunk_neon`) transposes the keys once per head, computes scores as an 8-query x 8-key
register tile (64 lane-FMAs per 16 loads), runs an exact softmax over the full score rows (they fit in
L1), and forms the weighted sum of values as another 8 x 8 tile. 176 to 275 GFLOP/s, the tower 262 to 213
ms, embedding error unchanged (cosine 0.9997). Tested against a naive softmax attention for ragged token
counts (37, 101), a single token, head_dim 8 to 64, and large-magnitude scores, on the phone.

### CPU decode: a hidden slow state, and why Cera is hit harder by it

The text-only decode gap to llama.cpp (LFM2.5-VL-450M-Q4_0, depth 210, 8 threads) looked like 4 to 8% in
the matrix and measured anywhere from 2% to 25% depending on when it was run. Two things explain that,
and the second sets what is left to win.

- **The phone has a slow state.** Decode flips between a fast regime (about 215 to 230 tok/s for Cera,
  225 to 245 for llama.cpp) and a slow one (about 150 to 160 for Cera, about 205 for llama.cpp), in blocks
  of seconds to minutes, with no change in the workload, the flags (`--no-cache`, KV dtype, pinning, spin
  window and thread count all leave it alone) or the other processes (the pool is the only thing running).
  The clock is the difference: `simpleperf stat` over a run gives an effective 2.9 GHz in a fast run and
  1.9 GHz in a slow one, while `scaling_cur_freq` reads about 3.3 and 4.0 GHz either way, so the sysfs
  number is not the real clock. Only paired runs mean anything: alternate the two engines a few seconds
  apart, in rotating order, and compare adjacent results.
- **Cera needs more instructions per token, so a lower clock costs it more.** Counting user-space
  instructions per decoded token (difference two run lengths so load and warmup cancel): Cera 254 M,
  llama.cpp 208 M at 8 threads; on one thread (no spin-waiting in either) 249 M against 147 M. With 8 cores
  at 2.4 IPC, 254 M instructions is 4.5 ms at 2.9 GHz (220 tok/s) and 6.9 ms at 1.9 GHz (145 tok/s), which
  is both regimes. llama.cpp's runs are closer to memory bound, so the same clock drop costs it 9%, not 25%.
  The matrix's CPU decode row is therefore a mixture of the two regimes.

A per-function profile (`simpleperf record -e instructions:u`, symbols kept by building with
`CARGO_PROFILE_RELEASE_STRIP=none`) put the budget at: Q6_K lm_head argmax 31%, fused gate/up SwiGLU
GEMV 33%, other Q4_0 GEMVs 24%, with load-time repack and the rest the remainder. The Q4_0 kernels cost
12 to 19 instructions per 32-weight block-row; the lm_head cost about 300 per 256-weight block-row, and
that was the cheapest to fix:

- The Q6_K decode GEMVs (logits and fused argmax) now take four rows per iteration, one row per lane.
  Each lane runs exactly the serial chain `sumf += ((d * sc) * xs) * hsum`, so every output is bit
  identical to before (the prefill GEMMs are tested bit-exact against it, and a reordered sum drifted
  3.4e-4 at k=4608). The four rows' sub-block dots share one `vpaddq` tree, the scale products and
  converts run four rows wide, the 6-bit unpack merges with `vbsl` (11 instructions per four sub-blocks
  against 18), and the activation loads are shared. The fused argmax is now exactly argmax of the logits.
- 65536 x 1024 Q6_K: 80 M to 57 M instructions per call, 8 threads 1.08 to 0.93 ms (59 GB/s, the
  bandwidth ceiling). In the model the lm_head phase drops from about 1.27 to 1.08 ms per token.
- Paired against llama.cpp, 14 rotating rounds: Cera 196.6 to 204.3 tok/s, paired ratio 0.859 to 0.895.

The Q4_0 decode GEMVs were the rest (about 60% of instructions), and llama.cpp spends about 7 instructions
per block-row there to Cera's 12 to 19. They now read a 4-row interleaved repack built at load
(`repack_q4_0_dec4`, 72 bytes per four rows and block, the standard layout's size, nibbles still packed):

- One `vdotq_laneq_s32` against four activation bytes gives a partial dot for each of four rows in the
  four lanes, so a block's whole integer dot accumulates in one vector (lane = row). It starts at
  `-8 * sum(x)`, an exact integer correction computed once per call, so no weight is unpacked to signed.
- The float step is `acc = fma(float(blockdot), d * xs, acc)`, four rows wide, with no horizontal sum.
  Gate+up SwiGLU and the Q/K/V concat share the kernel; activation loads are shared across two row groups.
- **That formula is exactly what the repacked prefill GEMMs compute per output** (vdot 8x8 and smmla), so
  decode and prefill now agree bit for bit on a weight that has the repack. The standard-layout decode
  kernels did not on the i8mm tier (they accumulate per element group in even and odd float
  accumulators), a gap the test comments already called real. Prefill itself needed no change: it was
  already on this arithmetic, and the new tests pin the equality (vdot kernel on the Mac, smmla on the
  phone, 1 to 5 columns, k up to 4608).
- Weights outside the repack's condition (rows not in whole 16-row groups, MoE experts, hosts without
  dotprod and fp16) keep the standard path on both sides, unchanged.

Results on the phone: user instructions per decoded token 249 M to **134 M** single-threaded (llama.cpp
147 M); the kernel microbenchmark runs 2x faster at 8 threads and 2.3 to 3x on one thread (`q4_0_gemv_bench`);
paired against llama.cpp text decode at depth 210, 12 rotating rounds, **208.6 to 236.8 tok/s, paired
ratio 0.895 to 1.012**. Cera is now bandwidth bound rather than instruction bound, so the slow clock state
costs it about what it costs llama.cpp. Llama-3.2-1B Q4_0 gives the identical text and +5 to 8% (it was
already bandwidth bound). The generated text is unchanged and prefill speed is unchanged.

The cost is memory: one extra copy of each repacked weight, 380 to 535 MB anonymous resident on the 450M
model (about +155 MB; the file-backed pages are unchanged). A later text-only run, polling
`RssAnon` in `/proc/<pid>/status` with the repack on and off, gave 645 against 487 MB (+158 MB). The
figure was measured on a Galaxy S25 Ultra only; an Apple-silicon Mac showed no resident-size change. `CERA_Q4_DEC4=0` turns the repack off. What is
left on CPU decode is the Q6_K lm_head (about 38% of the remaining instructions), which is pinned to
bit-exactness with its prefill GEMMs by the existing tests.

### GPU: 4,723 ms to 390 ms first token, 119 to 169 tok/s decode

Five separate causes, found in this order:

1. **Image prefill ran one frame at a time.** The 192 image embeddings went
   through the decode path (7.5 ms each, 1,472 ms). `forward_prefill_from_embeddings`
   now feeds the batched prefill (77 ms).
2. **The batched GEMM was slow for small batches.** For Q4_0, `n < 32` used the
   generic register-tile kernel, about 167 ms per forward at `n = 1` to 12
   (about 1 GB/s of weight traffic) against about 30 ms for the streaming GEMM,
   which pads to 32 columns anyway. Every batch size now takes the streaming
   kernel (`CERA_WGPU_STREAM_MIN_N`): a 15-token text prompt prefills at 170
   tok/s instead of 62. The session also fuses text, image and text into one
   prefill once the GPU model implements `embed_token_rows`, so the whole
   210-token prompt is one forward of about 105 ms.
3. **The vision tower's kernels.** Its Q8_0 linears ran at about 40 GFLOP/s on
   the generic kernel and its attention at about 19 GFLOP/s (the tiled kernel is
   disabled on Android because its runtime-indexed arrays spill and trip the
   driver watchdog). New fp16 Slang kernels: a Q8_0 streaming GEMM, and attention
   as three GEMM-shaped passes (scores, column softmax, P.V). One trap worth
   recording: a 64-way unrolled inner loop made the driver's code so large that a
   single dispatch took over a second and lost the device; the loops are 8 wide.
4. **The tower was host-bound, not GPU-bound.** Per-kernel GPU timestamps showed
   181 ms of kernel time inside a 307 ms first-to-last span, and the CPU encode
   took the entire wall time. Timing the host put 280 ms in buffer allocation
   (about 600 fresh Vulkan buffers per image at about 0.45 ms each), 80 ms in
   submit, and almost nothing in bind groups or recording. Pooling and reusing
   the buffers took the tower from 400 to 272 ms. (Batching submits alone made
   it slightly slower.)
5. **Decode paid 2.2 ms of GPU idle per token.** 0.3 ms recording, then 1.6 ms in
   `CommandEncoder::finish` and 0.3 ms in `queue.submit`, before the GPU saw
   work. Nothing recorded depends on the position, so the next token's command
   buffer is now finished while the current one executes (`CERA_GPU_PREBUILD`):
   140 -> 170 tok/s with identical output.

With these, the tower's GPU timeline was saturated (192 ms first-to-last against
181 ms of kernel time). Two further changes, found with per-kernel device timestamps and a new
`vit_gpu_probe`:

- **`attn_scores` took 63.6 ms, `attn_pv` 9.2 ms, for the same arithmetic.** The score kernel gave each thread
  one key and wrote 32 contiguous floats, so neighbouring threads stored 3 KB apart and every 16-byte store
  hit its own cache line. A thread now owns one query against a tile of 32 keys; the output layout
  `S^T[head][key][query]` is unchanged, so softmax and P.V are untouched, but neighbouring threads are
  neighbouring queries and each key's store is one contiguous run across the wave. 63.6 to 7.2 ms of GPU
  time per image, bit-identical output (cosine 0.999393 against the CPU before and after).
- **The first encode was 65 to 80 ms slower than every later one.** About 30 ms is the driver's first use of
  each pipeline and about 50 ms is the first allocation of the pooled buffers (the pool was keyed by exact
  size, so every new image size allocated again). Pool sizes now round up to a power of two, and one blank
  512x384 image is encoded when the encoder is built (`CERA_VIT_WARMUP=0` skips it; about 190 ms once, at
  load). llama.cpp's mtmd runs a warm-up encode by default, which is why its tower figure never included
  this.

Warm, the tower is now 135.7 ms per encode (was 184.6); through the CLI the first token is about 260 ms
(was 380) and the matrix tower 138 ms against llama.cpp's 182. The kernel times are now the Q8_0 GEMMs 64
ms, softmax 13 ms, P.V 9 ms, `attn_scores` 7 ms, bias 6 ms; fusing the bias, GELU and residual into the GEMM
epilogue would take roughly 10 ms more off.

The fp16 kernels cost accuracy: the GPU embedding is 3.7% RMS from the old f32
GPU path, 4.3% from the CPU path and 5.3% from llama.cpp's. `CERA_VIT_STREAM=0`
restores the previous kernels exactly.

### Long-context decode: the gap that tiling exposes

At 210 tokens of context decode is level with llama.cpp. At the 1,795 tokens of a tiled image it is not:
CPU 135 against 174 tok/s and GPU 100 against 122 tok/s before the work below (the NPU leads, 143 against 133).

- **CPU.** The attention phase takes 2.7 ms per token at 1,795 tokens against 0.37 ms at 210. Each pair of
  query heads shares one KV head, but each head was its own task, so two workers fetched the same keys and
  values from memory; the unit is now a whole GQA group, so the second head reads them from L2 (attention
  2.95 to 2.55 ms, decode 140 to 155 tok/s in a back-to-back pair, output bit-identical). What is left is
  the f32 KV cache's bytes: with `--kv-cache-keys f16` the same run decodes about 160 tok/s, as llama.cpp's
  f16 cache does. The CLI now defaults to `--kv-cache-keys auto`, which is f16 wherever the model honors it
  (CPU LFM2 and the dense transformers) and the backend's own cache otherwise; `f32` restores full precision.
  The library default is the same: `SessionConfig::default()` is `KvCompression::F16` (except on wasm32, whose
  f16 kernels have no SIMD path), and a model that does not honor f16 (the GPU and NPU backends keep their own
  cache) is configured with its uncompressed KV when the session is built. `KvCompression::None` asks for
  the backend's own full-precision cache; in the FFI an omitted `kv_compression` is the default, so it is
  `Some(KvCompression::None)` that opts out.
  llama.cpp also splits its single-token attention across the key range (one slice per thread, merged
  through partial softmax states, from 512 keys). Cera's CPU path already had one unit per GQA group, which
  balances 8 KV heads on 8 threads, so a key-range split could only help where cores differ in speed. It was
  built (units of GQA group x key range, about two per worker) and measured, and it lost, so it was removed.
  Ten paired, rotating-order rounds at 1,795 tokens of context, 128 tokens decoded, all 8 cores, skin
  37.3 C throughout, headroom drifting 0.49 to 0.78 over the run (7 of 10 rounds thermally matched; the
  matched-round medians agree with all rounds):

  | Variant | Decode median, tok/s | Paired vs llama.cpp |
  |---|---|---|
  | Cera f32 KV (before) | 141.8 | x0.847, never ahead (0 of 10) |
  | Cera f16 KV (new default) | 172.4 | x1.007, ahead in 6 of 10 (range 0.85 to 1.08) |
  | Cera f16 KV + key split | 163.9 | x0.973, and x0.961 against f16 alone |
  | llama.cpp CPU (`-t 8`) | 168.7 | 1.0 |

  The f16 default is worth x1.15 and brings the CPU to parity with llama.cpp, not clearly past it: the
  spread between rounds (slow and fast clock regimes) is as wide as the lead. Decoded text is identical across
  the Cera variants; llama.cpp's differs at "textured" against "woven". The key split lost 4% against the
  per-group split on this phone, so it was dropped. First token on this image is 4.9 s for Cera against
  11.8 s for llama.cpp in the matrix run above; the paired harness reports the same like-for-like TTFT.
- **GPU.** The decode attention launched one workgroup per head (16), each walking the whole KV cache
  serially and doing the value sum on 64 of its 256 threads, so `attn_core` took 4.67 ms per token at 1,795
  tokens against 0.89 ms at 210. It is now split-K flash decoding: a (head x 16 key splits) grid with the
  online softmax per split, all 256 threads on the value sum, and a merge kernel that combines the 16 partial
  states per head (`flash_attention_split.wgsl`, `flash_attention_merge.wgsl`; head_dim 64, the single
  kernel remains for other shapes and with `CERA_GPU_ATTN_SPLIT=0`). A second change reads each key row
  with four lanes and 16-byte loads instead of one lane per row. Paired, alternating runs with the thermal
  state recorded around each (skin 37.3 C throughout, headroom 0.41 to 0.50, matched within each round),
  decode at 1,795 tokens: 100.4 tok/s (single kernel) -> 110.2 (split) -> 154.8 (vectorized key reads,
  x1.45 over the split with scalar reads), against llama.cpp's 122. At 210 tokens: 172.9 -> 182.6 -> 181.1
  tok/s, unchanged within noise by the second step. Output is identical in every variant. `cera thermal`
  prints the headroom now and at +10 s / +30 s for annotating runs.

### NPU decode: the irreproducibility was a Cera scratch bug

Until this fix NPU decode was capped at 32 tensors per DSP batch
(`CERA_HEXAGON_BATCH_TENSORS`), about 26 flushes per token, because longer
batches gave irreproducible output. Measured before the fix, on the same phone,
8 runs of 160 tokens (6 for the caps marked *):

| Setup | Decode | Distinct outputs |
|---|---:|---:|
| Cera, cap 32 (old default) | 129 tok/s | 1 of 8 |
| Cera, cap 64 * | 142 tok/s | 2 of 6 |
| Cera, cap 128 * | 146 tok/s | 4 of 6 |
| Cera, cap 256 * | 152 tok/s | 4 of 6 |
| Cera, cap 1024 * | 158 tok/s | 3 of 6 |
| Cera, no cap (single flush) | 158 to 160 tok/s | 3 of 8 |
| llama.cpp, batches of up to 1,280 ops | 152 tok/s | 1 of 8 |

llama.cpp sends up to 1,280 ops per batch reproducibly, and the DSP code was
ruled out (Cera's `htp-tensor.c` is byte-identical to upstream's, and the
failure persists with one DSP thread), which pointed at what Cera's host emits.

**Root cause.** Every conv layer wrote its `b*x` row to one shared scratch
address, and the strided scalar `Cpy` that updates the layer's rolling `[C, 2]`
state read that row back. In a long batch the second conv layer's `Cpy` read the
previous conv layer's row (stale cached lines the DSP's range flush does not
remove), so whole 128-byte lines of its state held the earlier layer's values.
Per-block digests of the state, diffed across identical runs, showed exactly
that: the differing words were the other layer's correct values at the same
channels, only in conv layers that followed another conv layer in the batch. Any
flush between conv layers, which any cap provides, hid it. The exact reason the
range flush does not invalidate those lines on the DSP was not pinned down.

**Fix.** Each conv layer gets a private scratch slot (`conv_stage`, 64 slots of
8 KB). Decode builds `b*x` there; prefill copies the last two rows into it before
the strided writeback. The default cap is gone for every model; `CERA_HEXAGON_BATCH_TENSORS=N` still sets one.

| Check | Result |
|---|---|
| Teacher-forced replays, no cap (450M, 350M, 1.6B), 14 each | 0 differ from the first (max logit diff 0), every replay differed before |
| State after the fix vs the old capped state | identical at all 7 prefill and decode snapshots |
| 8 full VL runs of 160 tokens through the CLI | 1 distinct output, equal to the capped text, 157 to 160 tok/s (capped: 128 to 130) |
| Six-way matrix, 64 tokens, medians of 5 | Cera NPU 155.7 tok/s (151.7 to 157.6) vs llama.cpp 150.4 |

Other models, no cap, same phone:

| Model | Replays (48 steps) | CLI runs, capped vs uncapped | Decode capped -> uncapped | Prefill capped -> uncapped |
|---|---|---|---|---|
| LFM2.5-8B-A1B (routed experts) | before the fix 7 of 10 differed, now 0 of 10 | 1 text in 4 + 4 | 15.5 -> 17.2 tok/s | 60 -> 72 tok/s |
| Llama-3.2-1B (plain, 16 layers) | 0 of 10 differ (capped too) | 1 text in 6 + 6 | 50 -> 53 tok/s | unchanged |
| Qwen3.5-0.8B (DeltaNet) | 0 of 12 differ (capped, uncapped and pre-fix) | 1 text in 6 + 6 | 57 -> 60 tok/s | +3% |

## Remaining gaps

| # | Gap | Size | Notes |
|---|---|---|---|
| 1 | Long-context decode, CPU | level: 172.4 vs 168.7 tok/s at 1,795 tokens (paired x1.007) | Closed by the f16 KV default; a key-split attention was tried and lost 4%, so it was removed. GPU is closed too: 154.8 vs 122 tok/s. See "Long-context decode". |
| 2 | NPU tower at 1,024 patches | 997 vs 901 ms on 7 chunks | The NPU wins everywhere else; its tower is 10% behind per large tile. Not investigated. |
| 3 | Q6_K lm_head | none now | Text decode is level with llama.cpp (paired 1.01). The lm_head is the largest remaining share of CPU decode instructions, but decode is bandwidth bound and it already streams near the ceiling; a 4-row-lane layout like Q4_0's would also require changing the Q6_K prefill GEMMs. |
| 4 | GPU tower epilogues | about 10 ms | Fuse bias, GELU and residual into the GEMM. The tower already wins (138 vs 182 ms). |
| 5 | JPEG decoder | 3.6% on the thumbnail | stb_image against the `image` crate; matching the reference's decode bit for bit would mean porting stb's decoder. |
| 6 | Decode repack memory | +155 MB on the 450M | One extra copy of each repacked Q4_0 weight (`CERA_Q4_DEC4=0` to disable). Scales with the model. |

## Caveats

- One device, one model, one image, one 64-token continuation. Decode is measured
  over 63 steps, at a context of 210 tokens (512 px image) or 1,795 (1024x771).
- The in-app benchmark recorded much higher Cera NPU vision TTFT (about 3.1 s)
  than the 138 ms here. That test used a different image (a 290-token generated
  gradient) and measures through the whole app stack, and the cause has not been
  investigated; these CLI numbers are engine-only.
- llama.cpp numbers carry the build above. A different llama.cpp commit, a
  different CPU backend variant, or the Leap SDK's runtime variant selection can
  change the CPU rows.
- The phone was on USB power; in the baseline run skin temperature rose 4 C across the equal-token
  run. Medians and ranges are shown so a thermal trend would be visible. The later paired measurements
  record skin temperature (steady at 37.3 C) and thermal headroom (which drifts upward over a ten-minute
  run, 0.49 to 0.78 in the CPU decode comparison), and report the thermally matched rounds separately.
- CPU decode against llama.cpp at 1,795 tokens is level, not clearly ahead: the spread between rounds
  (0.85 to 1.08 of llama.cpp per round) is as wide as the 0.7% median lead.

## Reproduce

```bash
# Cera CLI
cargo ndk -t arm64-v8a build --release -p cera-cli --features gpu,hexagon
# push: cera -> $CERA_DIR/cera, cera/src/backend/hexagon/skels/ -> $CERA_DIR/skels/
# push: a llama.cpp Snapdragon build -> $LLAMA_DIR (bin/ and lib/)
SERIAL=<adb-serial> python3 scripts/bench_android_vl.py out sweep
SERIAL=<adb-serial> python3 scripts/bench_android_vl.py out matrix ff ff 8
```

For a before/after or engine-vs-engine comparison of one change, use the paired harness instead of two
separate runs (CPU decode flips between a fast and a slow clock regime, so a single pair can show a 2% to
25% gap). It interleaves the variants in rotating order, records skin temperature and the Android thermal
headroom (`cera thermal`) around every run, and reports all-round and thermally matched medians:

```bash
SERIAL=<adb-serial> python3 scripts/bench_android_ab.py \
  '[["f32 KV","cera","cera","","--kv-cache-keys f32"],["f16 KV","cera","cera","",""],["llama.cpp","llama","","",""]]' \
  /data/local/tmp/pug.jpg 10 128
```

The CLI defaults to an f16 KV cache on CPU (`--kv-cache-keys auto`), as llama.cpp does, so a default Cera run
is compared with a default llama.cpp run; pass `--kv-cache-keys f32` for the full-precision cache.
