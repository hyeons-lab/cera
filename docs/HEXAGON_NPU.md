# Qualcomm Hexagon NPU Architecture & Acceleration Guide

Cera provides a native, hardware-accelerated backend for Qualcomm Hexagon Neural Processing Units (NPUs) and Compute DSPs (CDSPs). This backend targets Snapdragon mobile and compute platforms, executing LLM text generation, vision transformer encoding, Whisper speech-to-text transcription, and vocoder audio synthesis directly on the Hexagon Tensor Processor (HTP) and Hexagon Vector eXtensions (HVX) without host CPU roundtrips.

---

## 1. System Architecture & Execution Model

```
+-----------------------------------------------------------------------------------+
| Host Application (Android App, Kotlin / Flutter / Rust CLI)                       |
+-----------------------------------------------------------------------------------+
       |                                      |
       | HexagonNpu.setup(context)            | cera::Session / CeraEngine
       v                                      v
+-----------------------------+        +--------------------------------------------+
| FastRPC Client Loader       |        | cera::backend::hexagon                     |
| (libcdsprpc.so)             |        | - HexagonDevice & dynamic probe            |
| - Extracts bundled skels    |        | - DspQueue & FastRPC batch builder         |
| - Sets ADSP_LIBRARY_PATH    |        | - Static queue template cache              |
+-----------------------------+        | - CDSP Latency QoS (100 µs vote)           |
       |                               +--------------------------------------------+
       | ioctl / Binder                       |
       v                                      v
+-----------------------------------------------------------------------------------+
| Qualcomm FastRPC Kernel Driver (/dev/fastrpc-cdsp)                                |
| Zero-Copy Shared DMA Memory Subsystem (rpcmem / DMA-BUF / ION)                    |
+-----------------------------------------------------------------------------------+
       |                                      |
       | FastRPC RPC messages                 | Zero-copy physical memory access
       v                                      v
+-----------------------------------------------------------------------------------+
| Qualcomm CDSP (Unsigned Process Domain)                                            |
| Worker Library: libggml-htp-v{73,75,79,81,85}.so                                  |
| - HTP Matrix Multiplication Engine (HMX)                                          |
| - HVX 128-byte SIMD Vector Units                                                  |
| - 8 MB Tightly-Coupled Vector Memory (VTCM)                                       |
+-----------------------------------------------------------------------------------+
```

### FastRPC Unsigned Process Domain (PD)
Inference runs inside Qualcomm's FastRPC Unsigned User Process Domain on the CDSP (`/dev/fastrpc-cdsp`). Unlike traditional DSP development that required signed Qualcomm engineering certificates or device rooting, the unsigned process domain allows any standard Android application (installed from Google Play or sideloaded) to execute compute workloads with full access to HTP matrix coprocessors and HVX vector units.

### Zero-Copy Shared DMA (`rpcmem`)
All model weights, dynamic activations, scratch buffers, and key-value (KV) caches reside in memory allocated via `rpcmem` (backed by Linux DMA-BUF or `/dev/ion`). Because this memory is mapped simultaneously into the host CPU address space and the DSP MMU:
- Tensor dispatches do not incur CPU-to-DSP copying over PCIe or system buses.
- Model weights staged once by the host CPU are read directly by the DSP DMA engine.
- Explicit host write-back cache flushes (`flush_cpu_cache()`) guarantee physical coherency before DSP dispatches execute.

### Dynamic Architecture Probing & Skel Selection
Snapdragon chipsets span multiple Hexagon DSP architecture revisions. Cera probes device capabilities at runtime using `HexagonDevice::probe()`:
- Interrogates FastRPC hardware info and identifies the DSP core architecture (`v73`, `v75`, `v79`, `v81`).
- Dynamically locates and loads the corresponding versioned skeleton dynamic library from application-staged storage (`context.noBackupFilesDir/cera_skels`).
- Automatically falls back to CPU execution if the device lacks a compatible CDSP or if FastRPC access is denied by system policy.

---

## 2. Hardware Support Matrix

| Qualcomm SoC Family | Hexagon Architecture | Bundled Skeleton | Hardware Capabilities | Validation Status |
|---------------------|----------------------|------------------|------------------------|-------------------|
| **Snapdragon 8 Elite** (SM8750) | `v79` / `v81` | `libggml-htp-v79.so` / `v81` | HTP GEMV, 8 MB VTCM, dual HVX units | Verified on Samsung Galaxy S25 Ultra |
| **Snapdragon 8 Gen 3** / 8s Gen 3 (SM8650) | `v75` | `libggml-htp-v75.so` | HTP GEMV, 4 MB VTCM, dual HVX units | Supported (same codebase) |
| **Snapdragon 8 Gen 2** / 8+ Gen 1 (SM8550) | `v73` | `libggml-htp-v73.so` | HTP GEMV, 2 MB VTCM, dual HVX units | Supported (same codebase) |
| **Snapdragon X Elite** (Compute) | `v73` | `libggml-htp-v73.so` | HTP compute engine, Windows/Linux on ARM | Supported |

---

## 3. Skeleton Libraries & DSP Additions

The DSP-side worker libraries (`libggml-htp-v{73,75,79,81}.so`) are precompiled Hexagon ELF shared objects that implement the low-level compute kernels. Cera embeds all four libraries directly into the host binary (`libcera_ffi.so` and the `cera` crate) and extracts them at startup.

### Additions to the Skeleton Libraries
1. **Extended Operator Set**:
   - `Conv1D`: 1D convolution kernel with configurable dilation, stride, and padding for waveform audio encoding and speech feature extraction.
   - `ConvTranspose1D`: Transposed 1D convolution kernel for vocoder audio synthesis and spectrogram upsampling.
   - `Snake` / `Snake1D`: Specialized sinusoidal activation function ($\sin^2(\alpha x)$) required for neural vocoders (BigVGAN, Vocos).
   - `UnaryStep` and `Sum`: Element-wise step activations and reductions executing directly on HVX vector registers.
2. **Opcode Alignment (`HtpOpCode`)**:
   - Realigned all discriminant values in Cera's `HtpOpCode` enum with the upstream DSP firmware ABI (`UnaryStep = 21`, `Sum = 39`, `Scale = 32`, `Cpy = 33`, `Conv1D = 63`, `UnarySnake = 64`). This prevents command displacement where non-scale operators were previously misinterpreted as `Scale` operations with invalid parameters.
3. **Compilation**:
   - Built with Qualcomm Hexagon SDK 6.6.0.0 and Hexagon Tools 19.0.07 (`hexagon-clang` with whole-program Link-Time Optimization), compiling dedicated targets with `-DDSP_VERSION`.
4. **16 KB Page Alignment**:
   - All ELF headers and program segments comply with Android 16+ 16 KB page-alignment requirements (`0x4000`), ensuring compatibility with Google Play store distribution.

---

## 4. Accelerated Kernels & Subsystems

### 4.1 LLM Text Generation (`HexagonLfmModel`)
- **32x32 Tiled Weight Repacking**: Weights quantized in Q4_0 and Q8_0 formats are repacked into 32x32 tiles. This layout maps directly to HTP matrix multiplier hardware blocks, maximizing SIMD register utilization and eliminating runtime unpacking overhead.
- **Single-Flush Forward Decode (opt-in)**: With `CERA_HEXAGON_BATCH_TENSORS=0` (the default is a 32-tensor batch cap, see the 'Decode was not reproducible' note in section 7), the full forward decode pass (attention projections, multi-head attention, RMSNorm, SwiGLU feed-forward, and residual adds) across all layers is staged into a single contiguous FastRPC batch submission. This removes approximately 22 host-to-DSP synchronization roundtrips per token.
- **Ping-Pong Scratch Buffering**: Alternating layer buffers (`activation` and `activation_b`, `normed` and `normed_b`) isolate memory access between consecutive layers, preventing data races in the asynchronous DSP execution pipeline.
- **Static Command Queue Template Caching (only with `CERA_HEXAGON_BATCH_TENSORS=0`; the default cap bypasses it)**: Because decode graph topology is static across tokens, Cera constructs the FastRPC batch template once on the initial step. Subsequent steps patch only sequence position (`pos`) and attention KV pointers in place, achieving zero-allocation host dispatch.
- **FastRPC Latency QoS**: Sets `FASTRPC_CONTROL_LATENCY = 100 µs` via `remote_session_control()` during initialization. This prevents Qualcomm's CDSP frequency governor from scaling clocks down during brief inter-token pauses, sustaining peak throughput throughout the generation turn.
- **Q8_0 Quantized KV Cache**: Stores key and value cache rows in 32-element Q8_0 quantized blocks (34 bytes per block) instead of uncompressed F16 (64 bytes). This yields a 47% reduction in memory bandwidth and physical rpcmem footprint, expanding the usable context length on device.

### 4.2 Multimodal Vision Transformer (`HexagonVisionEncoder`)
- **Batched 24-Block ViT Pipeline**: All 24 blocks of the Vision Transformer (ViT) are dispatched in a single batch submission to the DSP, executing patch embeddings, LayerNorm, multi-head attention, and MLP projections without CPU intervention.
- **On-NPU Flash Attention**: Computes self-attention over image patch tokens directly on the DSP using causal flash attention kernels with F16 KV scratch conversion.
- **Quantized Projector Dispatch**: MLP multimodal projector weights are repacked for HTP matrix kernels, projecting visual representations into the language model embedding dimension on device.
- **Two defects fixed on the way to a correct ViT (S25 Ultra).** Before them the NPU ViT did not run at all above about 60 patches (`VtcmTooSmall`, the session fell back to the CPU encoder silently) and its embeddings were off by 25 to 45%. (1) *Row chunking.* The quantized HVX matmul keeps its activations in VTCM at 36 bytes per element, so the 3072-wide feed-forward down projection passes the 8 MB VTCM past about 57 rows; the DSP walks the rows in chunks when `kparams.m_chunk` says so, and the host builder always left it 0 (a decode-era assumption). `build_mul_mat_kernel_params` now solves the chunk like the DSP's `htp_mm_hvx_solve_vtcm_params`. The fused Q/K/V `MulMatNx` cannot chunk, so past `mm_hvx_fused_nx_max_rows` (a couple of hundred tokens at 768 wide) the three projections run separately. (2) *GELU.* The DSP's `UNARY_GELU` is the quick approximation `x * sigmoid(1.702 x)`, up to about 2% per element away from the tanh GELU the ViT is trained with; the wide down projection turned that into a 20% error in the first block alone. `dispatch::gelu_tanh` builds the tanh form from seven DSP ops (`x * sigmoid(2 sqrt(2/pi) x (1 + 0.044715 x^2))`) and the quick-GELU helper is gone, so nothing calls `UNARY_GELU` any more. Block by block against the CPU encoder (`examples/hexagon_vit_probe.rs`): the first block's output is at relative RMS 0.7%, the final embeddings at cosine 0.9983 to 0.9989 at every grid from 8x8 to 32x24 patches (0.92 to 0.97 before), and with the quantizer-fixed skel of PR 467 the mid-network error is lower again (2.8% against 8.2% after 8 blocks). Image prefill at 768 px is 1.06 s instead of 2.77 s on the CPU fallback. The ViT runs its whole forward as one DSP batch (thousands of tensors with the GELU's seven ops per tile), which is the batch size the decode defect depends on; `hexagon_vit_probe` prints a bitwise fingerprint of each size's embeddings and repeats each size in-process; on the S25 Ultra all eight sizes (8x8 to 32x24 patches) gave identical fingerprints across six fresh processes and identical in-process repeats, so none of the eight sizes showed the decode defect in these runs (one image pattern, six processes: evidence of reproducibility, not proof that long ViT batches are immune). Whisper (conv stem and both MLPs) and the audio-encoder adapter use the same tanh GELU (within about 5e-4 of the exact erf form the CPU adapter uses, measured 4.7e-4): the Whisper transcript of the weather clip is unchanged and identical across 10 NPU runs, and the audio encoder's end-to-end embeddings are at cosine 0.99983 to 0.99984 against the CPU encoder (10 s and 30 s).

### 4.3 Speech Recognition (`HexagonWhisperModel`)
- **64-Token VTCM Chunking**: Large audio convolutions and multi-head attention dispatches can exceed the physical 8 MB Vector Tightly-Coupled Memory (VTCM) limit on Snapdragon 8 Elite. Cera partitions the 1,500-token audio sequence into 64-token tiles across all Conv1D, LayerNorm, linear GEMM, and GELU dispatches.
- **Autoregressive FlashAttnExt**: Whisper decoder autoregression utilizes on-DSP FlashAttnExt with cross-attention bound to the encoder representations resident in rpcmem.
- **Host-Side Control Token Suppression**: Applies timestamp and control token suppression masks (setting non-timestamp logits to negative infinity) directly on the host CPU during sampling, preventing premature segment termination.

### 4.4 Vocoder & Audio Synthesis (`HexagonAudioDecoder` / `HexagonDepthformer`)
- **Native Hexagon Depthformer**: The vocoder depthformer model runs 8 autoregressive codebook passes (totaling 48 transformer layers per synthesized audio frame). `HexagonDepthformer` executes all 8 passes entirely on the Hexagon NPU using HTP GEMV, per-head RMSNorm, interleaved RoPE, F16 KV cache, and FlashAttnExt without roundtrips to the host CPU.
- **On-DSP Audio Detokenizer**: Runs LayerNorm, linear GEMM, SwiGLU, and Conv1D operations directly from rpcmem buffers to generate spectrograms.
- **Throughput & Thermals**: Audio decode throughput reaches 36.67 tok/s on Snapdragon 8 Elite (1.59x faster than Leap CPU at 23.02 tok/s, and 2.82x faster than previous NPU decode of 13.00 tok/s). Offloading audio synthesis to the NPU protects real-time speech generation from Android CPU frequency throttling.

### 4.5 On-DSP Argmax (`HtpOpCode::Argmax`)
- **Direct DSP Reduction**: Offloads greedy token selection directly to the Hexagon HTP execution engine using `HtpOpCode::Argmax` (opcode 62).
- **Zero-Copy Host Retrieval**: The host CPU reads only 4 bytes containing the winning token ID from rpcmem rather than transferring or cache-invalidating 128 to 256 KB of float32 logits. This eliminates memory bandwidth bottlenecks and host vector allocations during greedy token generation.

### 4.6 Training-Free Speculative Decoding (`forward_prefill_logits_all`)
- **Batched LM-Head Verification**: Evaluates all candidate draft tokens in a single forward pass by batching LM-head projections across M rows via `dispatch_mul_mat_m`.
- **KV State Coherency**: Implements `check_kv_rewind` and `try_truncate_kv` with boundary validation, allowing non-fatal verification rejects or prompt tail rollbacks without context wipeouts.
- **Hardware Acceleration Uplift**: Pairing Hexagon NPU verification with Cera's prompt-lookup drafter (`ngram=2`, `k=4`) elevates 350M decode throughput from 164.0 tok/s to 259.5 tok/s (peak 260.2 tok/s) and 2.6B decode throughput from 26.7 tok/s to 47.7 tok/s (peak 48.5 tok/s) on Snapdragon 8 Elite without requiring fine-tuning or neural drafter sidecars.

### 4.7 FastRPC Power Management & Wakelock
- **Device Node Preservation**: Acquires a FastRPC driver wakelock (`FASTRPC_CONTROL_WAKELOCK`) during `HexagonDevice::new()` to prevent Android power management from suspending the FastRPC device node during active sessions. The votes are process-wide, so they are refcounted: the first live device takes them, the last one to drop releases them, and a failed device open (for example an unsupported arch during probing) never leaves them on.
- **Latency QoS Scaling**: Combines the wakelock with `FASTRPC_CONTROL_LATENCY = 100 µs` session votes, ensuring the CDSP frequency governor remains locked in peak performance corners during active inference.

### 4.8 Centralized Audio Accelerator Factory Integration
- **Unified Backend Resolution**: Vocoder detokenizer and depthformer acceleration routes through `cera::model::audio_decoder::build_audio_accelerator`, providing immediate parity with `--device hexagon` and honoring `CERA_AUDIO_GPU` environment overrides.
- **Automatic Depthformer Activation**: When targeting Hexagon NPU, hardware-accelerated depthformer codebook sampling executes automatically without requiring auxiliary experimental flags. The Metal and wgpu depthformers stay opt-in (`CERA_GPU_DF=1`); an accelerator opts in to the default through `AudioAccelerator::depthformer_default_on`.

### 4.9 Failure Handling: Torn State, Lock Poisoning, and Timeouts
- **Torn recurrent state**: a forward that fails after any batch reached the DSP may have advanced the short-conv or DeltaNet state in `kv_state_buf` while `seq_len` did not move. `HexagonLfmModel` then refuses every forward and partial rewind with a typed "recurrent state torn" error until `truncate_kv(0)` or `try_reset_kv` zeroes the state. A failure before any dispatch (emit-time validation, batch registration) leaves the state intact and does not tear it. Attention-only models rewrite their KV slots idempotently and never tear.
- **Lock poisoning**: the LFM2 device lock is taken through one helper. If a panic unwound through a forward, the helper marks the state torn, clears the poison and returns the guard, so recovery is the same full reset. The check that refuses torn state takes the device guard as an argument, so checking before locking does not compile. The audio-decoder, vision and Whisper paths recover a poisoned device lock through `LockOrRecover` (logged once) and drop any half-built pending batch before they stage a new one, since each call restages its batch in full. Whisper's decode keeps its scratch and state buffers behind locks that fail closed with a backend error, because that state resumes across calls and a torn one cannot be told from a good one. Cached decode templates that are patched in place are discarded on poison (`lock_or_discard`) and rebuilt.
- **DSP read timeout**: `dspqueue_read` gives up after 30 s and the queue counts the batch as outstanding, because the DSP may still complete it later and write the buffers it targets (KV, recurrent state). The LFM2 reset paths (`truncate_kv(0)`, `try_reset_kv`) first wait for every outstanding batch to answer and only then zero the state buffer; if the DSP still does not answer, `try_reset_kv` returns an error and `truncate_kv(0)` leaves the buffer untouched and the model torn, so the only way out is to retry the reset later or drop the model. Not validated on device: the S25 Ultra has not been driven into a real 30 s hang, so the timing of a late completion relative to the reset is covered only by the fake-driver tests. Forwards after a timeout on an attention-only model do not wait, and rely on their KV slots being rewritten before they are read.

### 4.10 Speech Input: the FastConformer Encoder (`HexagonAudioEncoder`)
LFM2-Audio's input encoder (the `mmproj` front end and Conformer: log-mel, a 3x3 conv stem, 17 blocks of feed-forward, relative-position attention, convolution module, feed-forward, LayerNorm, then an MLP adapter into the LLM width) runs on the NPU from the PCM samples to the embeddings for `--device hexagon` and `auto`. It exists for background transcription on Android, where the point is CPU time, not latency. Measured on the S25 Ultra with `examples/audio_encoder_bench.rs` and `examples/hexagon_conformer_probe.rs`:

| Audio | CPU encoder (CPU time, wall) | NPU encoder (CPU time, wall) | CPU time per audio second |
|---|---|---|---|
| 10 s | 2568 ms, 497 ms | 14 ms, 239 ms | 0.257 against 0.0014 |
| 30 s | 8272 ms, 2188 ms | 60 ms, 839 ms | 0.276 against 0.0020 |

What got it there, in order of impact:
- **Blocking waits.** The driver's default is to spin on `dspqueue_read` (`CERA_HEXAGON_OPPOLL`), which holds a core for the whole batch: with the blocks on the DSP but the host spinning, the encoder still cost 0.043 CPU-s per audio second (143 ms of CPU for 143 ms of blocks, 10 s of audio). The encoder's queue sleeps for its responses instead (`HexagonQueueSession::set_blocking_wait`, set per run and restored, so decode keeps polling), which took the blocks to 4 ms of CPU at the same wall time.
- **The stem on the DSP.** It was 283 of the remaining 322 ms of CPU.
- **Log-mel on the DSP.** 17 ms of CPU for 10 s, now 3 ms.
- **What still runs on the CPU.** Writing the input (the padded, pre-emphasised samples; the mel into a zero-bordered buffer), the log and per-feature normalization of the mel energies (a millisecond for 10 s; the statistics need every frame, and the normalization is the CPU path's own `finish_log_mel`), and the relative-position table. About 11 ms of CPU for 10 s in all.
- **How the ops map (blocks).** Linears are the repacked Q4_0 `MulMat`, tiled by 64 frames (a whole 126-frame sequence overflows the VTCM). The attention's `QK^T` and position scores are batched F32 `MulMat`s over per-head strided views of Q, K and the projected position embedding (exact to 1e-7 against the CPU); the rel-shift is not a copy but a view of the `[2t-1, t, heads]` position-score matrix with a row stride one element short, handed to `Softmax` as its additive mask (the softmax kernel addresses its own source and destination rows contiguously, so those are not padded). `attn @ V` contracts over a 32-padded key axis against a per-head transposed V made with a strided `Cpy`. The depthwise conv (kernel 9, padded 4 each side) is `SsmConv` over a channel-major sequence built by two `Concat`s; the GLU is two half-width matmuls and a sigmoid. The adapter's exact (erf) GELU runs as the tanh form built from seven DSP ops (`dispatch::gelu_tanh`, within about 5e-4 of erf, measured 4.7e-4); the DSP's own `UNARY_GELU` is only the quick approximation `x * sigmoid(1.702 x)`, see the vision section for what that did to a wider network.
- **How the ops map (stem).** Activations are channel-last (`[position, channel]`) in buffers with a one-element zero border, so padding costs no ops. The first convolution is an im2col (nine strided `Cpy` gathers into rows of 32: the nine taps, a constant one that carries the bias, zeros) and one F32 `MulMat` written straight into the interior of the bordered output, then one ReLU over the whole buffer (the border stays zero). Each depthwise convolution is nine taps of gather (a strided view of the bordered input), per-channel scale (`Mul` against a tap-major weight row) and add, then the bias. The pointwise convolutions are F32 `MulMat`s over all positions, a bias add and a ReLU. A transposing `Cpy` turns the activations' `(freq, channel)` order into the `(channel, freq)` columns the Q4_0 output projection expects, which runs in tiles of 16 rows (its 4096-wide input overflows the VTCM at 64). Every stage matches the CPU's convolutions to a relative RMS of 1e-7 to 5e-7.
- **How the ops map (log-mel).** The padded, pre-emphasised samples are framed with one strided `Cpy` (the frames overlap, and the matmul kernel faults the DSP when it reads overlapping activation rows directly), two F32 `MulMat`s against Hann-weighted cos and sin matrices (the CPU's window and 512-point DFT), the power as `re*re + im*im`, and one matmul against the Slaney filterbank. The matrices are built once, with the 257 bins padded to 288. The linear mel energies match the CPU's to a relative RMS of 2e-6.
- **Memory.** The stem's activations and the log-mel's buffer are allocated per call and freed after it, so they are not held between turns; the DSP's reference is released before each unmap. The stem runs in chunks of 64 output frames (about 5 s of audio), so its buffer is about 32 MB however long the clip is (a whole 32 s clip took 197 MB). Each output frame reads mel rows `8t-7 ..= 8t+7`, so a slice that starts and ends on a multiple of 8 mel rows (or at the clip's end) is exact except for its first output frame, whose receptive field reaches the row before the slice; every chunk after the first therefore starts one frame early and drops that frame, about 1.5% recomputed. On a DSP that stops answering, each encode waits up to 30 s (the read guard) before the session falls back to the CPU encoder, and there is no failure memo, so every later clip pays the same wait; the per-call buffers a timed-out call used are leaked rather than freed under the stale batch. Chunked output matches the CPU stem to the same 1e-7 to 5e-7 as the unchunked run (checked with 3-, 8- and 64-frame chunks on the device), and exactly against a CPU reference in a host test. The encoder's own weights and scratch, and every other Hexagon model's buffers, are released on drop (a model's DSP references are released before its host unmaps; before that, each teardown logged an unmap failure per buffer).
- **Accuracy.** The final embeddings match the CPU encoder at cosine 0.99983 (10 s) and 0.99984 (30 s) with the tanh-form adapter GELU (0.9993 and 0.9991 with the DSP's quick GELU before), and the ASR transcripts of the three JP clips are identical to the CPU encoder's. Encode plus prefill drops from 0.57 to 0.82 s to 0.15 to 0.19 s (measured before the stem was chunked; the chunked stem does 6 DSP round trips per 64 output frames instead of 6 per clip, and the probe's whole NPU encode is 0.024 to 0.029 s per audio second). On a long clip where the model falls into a repetition loop on both devices, the two transcripts differ in one character ("2" against "二"), consistent with the 2e-2 relative-RMS embedding difference.
- **Limits.** Sequences of up to 400 frames (about 32 s) are staged; a longer clip is declined before any work is done and the session encodes on the CPU (`audio encoder: GPU path declined, using CPU`). `CERA_DISABLE_HEXAGON` and a missing driver leave the CPU encoder as before.

---

## 5. Mobile & Android Integration

### Android Manifest Requirements
In your application's `AndroidManifest.xml`, include the following entry within `<application>`:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application ...>
        <!-- Grants runtime linker namespace access to vendor FastRPC -->
        <uses-native-library
            android:name="libcdsprpc.so"
            android:required="false" />
    </application>
</manifest>
```

The `cera-ffi-android` AAR includes this declaration automatically; it merges into consumer manifests at build time.

### Kotlin Setup (Android)

```kotlin
import com.hyeonslab.cera.android.HexagonNpu
import uniffi.cera_ffi.*

class MyApplication : Application() {
    override fun onCreate() {
        super.onCreate()

        // 1. Extract bundled skel libraries and configure ADSP_LIBRARY_PATH
        HexagonNpu.setup(this)
    }
}

// 2. Probe hardware readiness and configure the engine
val backend = try {
    val probe = hexagonProbe()
    Log.i("NPU", "Detected Hexagon ${probe.arch} with ${probe.hmxUnits} HMX units")
    BackendPreference.HEXAGON
} catch (e: Exception) {
    Log.w("NPU", "Hexagon NPU unavailable, falling back to CPU", e)
    BackendPreference.CPU
}

val config = EngineConfig(
    backend = backend,
    contextSize = 2048u
)
```

### Rust CLI & Native Usage

Enable the `hexagon` feature in `Cargo.toml`:

```toml
[dependencies]
cera = { version = "0.7", features = ["hexagon"] }
```

Run inference targeting the NPU:

```bash
# Explicitly select Hexagon NPU
cera run -m model-Q4_0.gguf -p "Hello from Qualcomm Hexagon!" --device hexagon

# Automatic hardware selection (chooses Hexagon on supported Snapdragon devices)
cera run -m model-Q4_0.gguf -p "Auto device selection" --device auto
```

### 5.1 Environment Switches

Every environment knob the Hexagon path reads. All are read at model load or first use unless noted, and are meant for debugging and A/B work rather than production configuration.

Boolean rule (the intended convention for the LFM2 loader knobs, consolidated in `HexagonKnobs`): an **opt-in** knob (default off) is enabled only by `1` or `true` (case-insensitive); a **default-on** knob is disabled only by `0` or `false`. Any other value keeps the default. Knobs marked "set" instead test only that the variable exists (any value, including `0`, turns them on). The kill switches are the one deliberate exception: they fail safe, so any value except empty, `0` or `false` disables the backend.

Diagnostics convention: warnings and failures on the model path are paired, emitted through both `tracing` and stderr by the `hexagon_warn!` / `hexagon_error!` macros (`backend/hexagon/mod.rs`, one formatted message, `cera-hexagon: ` prefix), since `cera-ffi` installs no tracing subscriber and stderr is what Android logcat shows. Opt-in debug knobs (`DEBUG`, `STEP`, `PROFILE`, `PROFILE_OPS`) use plain `eprintln!` because the output is the point of setting them.

| Variable | Default | Parse rule | Effect |
|----------|---------|------------|--------|
| `CERA_DISABLE_HEXAGON` | unset | Any value except empty, `0`, `false` disables | Kill switch, checked centrally in `HexagonContext::new`, `probe_device` and `probe` (the FFI `hexagon_probe`, which reports no NPU while it is set), so the LFM2, vision, audio and whisper loaders all honor it. Loaders fall back to the CPU/GPU path. |
| `CERA_NO_HEXAGON` | unset | Same as `CERA_DISABLE_HEXAGON` | Alias of the kill switch above; either one suffices. |
| `CERA_HEXAGON_ARCH` | auto-probe | Integer `73`, `75`, `79`, `81` or `85`; anything else ignored | Force the DSP architecture (skel) instead of probing `V79, V75, V73, V81, V85` in order. Read by every device probe (alphabetical, exhaustive: audio, LFM2, nemotron3, sortformer, VAD, vision, whisper) through the shared `arch_override` helper; LFM2 keeps its own injectable lookup seam around the same parse. |
| `CERA_HEXAGON_OPPOLL` | on | `0` disables (any other value keeps it) | DSP queue completion polling. On: non-blocking `dspqueue_read` with a spin (lower wakeup latency, one busy host core during batches). `0`: blocking reads. The audio encoder ignores it and always blocks (it runs long, latency-tolerant batches and is meant to cost no CPU), see section 4.10. |
| `CERA_HEXAGON_ADPF` | on | `0` disables | Android ADPF performance-hint session for the calling thread. |
| `CERA_HEXAGON_ADPF_TARGET_MS` | `10` | Unsigned integer milliseconds; invalid values fall back to the default | ADPF work-duration target. |
| `CERA_HEXAGON_SPIN` | off | `1` enables | Park a detached spinner thread for the process lifetime to hold CPU clocks across DSP waits (governor experiments; burns a core). |
| `CERA_HEXAGON_DEBUG` | off | set | Verbose flush dump (buffers, tensors, ops) and NX-fallback notices, printed to stderr (`eprintln!`, so it shows on Android logcat without a tracing subscriber). |
| `CERA_HEXAGON_STEP` | off | set | Flush at every op-group boundary (one `dispatch::*` helper or model-local op emitter, so ops that share tensor indices stay in one batch) instead of at the tensor cap (`CERA_HEXAGON_BATCH_TENSORS`); bisects a faulty group, very slow. Progress lines go to stderr. Read once per process. |
| `CERA_HEXAGON_PROFILE` | off | set | Enable the DSP profiler and aggregate per-op timings; a per-flush line and a table on session drop are printed to stderr. Read once per process. |
| `CERA_HEXAGON_PROFILE_OPS` | off | set | With `CERA_HEXAGON_PROFILE`, also print one stderr line per op per batch. |
| `CERA_HEXAGON_UNARY_T1` | off | `1` enables | Legacy single-threaded, single-row unary kernel params (pre-port behavior). Read once per process; ignored under `cfg(test)` so goldens stay hermetic. |
| `CERA_HEXAGON_BARRIERS` | off | `1` enables | LFM2 debug: flush between blocks (`debug_barriers`). |
| `CERA_HEXAGON_CPU_ROPE` | off | `1` enables | LFM2 decode applies RoPE on the host CPU instead of the NPU. |
| `CERA_HEXAGON_KV_Q8` | off (F16 KV) | `1` or `true` (case-insensitive) enables | Store the on-device KV cache as Q8_0 instead of F16. |
| `CERA_HEXAGON_SSM_CONV` | on | `0` disables | Use the `SsmConv` DSP op for the short-conv state update instead of the manual op chain. |
| `CERA_HEXAGON_HMX` | on | `0` disables | Prefer HMX matmul kernels (HVX fallback when off). |
| `CERA_HEXAGON_BATCH_TENSORS` | `32` | Unsigned integer; `0` disables the cap; unparsable keeps the default | Tensors per DSP batch for every decode step and prefill chunk of every model `HexagonLfmModel` runs (LFM2, the routed MoE and the dense transformers; the evidence behind the default covers the first two), read once when the model loads. An unparsable value logs a warning and keeps the default. The batch ends at the first op-group boundary at or past N tensors. A whole token as one batch computes nondeterministically on the device, see the 'Decode was not reproducible' note in section 7; `0` also brings back the single-batch resident decode template, which has that defect. |
| `CERA_HEXAGON_DECODE_OPS` | unset (no ops cap; the tensor cap still applies) | Positive integer caps ops per decode flush; `0`, unset or invalid means no cap | Decode ops-per-flush cap (bring-up and bisection aid, not a safety threshold). Captured once per model at load (via `HexagonKnobs`), not per decode step. |
| `CERA_HEXAGON_MAP_BUDGET_MIB` | `3584` | Unsigned integer MiB; invalid values are warned about and ignored; clamped to the driver's 3993 MiB ceiling | Cap on bytes mapped into the CDSP, below the hard ceiling because several mid-sized buffers fragment the address space before the total is reached. Decides when routed experts are paged. |
| `CERA_HEXAGON_MAP_BUDGET_KIB` | unset | Unsigned integer KiB; wins over `_MIB` when both are set | Same budget in KiB, for tiny test models where a MiB is more than the whole window. |
| `CERA_HEXAGON_PAGE_EXPERTS` | off | `1` or `true` (case-insensitive) enables | Page routed experts through the DSP mapping even when the model would fit (tests, A/B runs), and let `--device auto` page a model that does not fit instead of falling back to the CPU. |
| `CERA_DUMP_ACT` | off | set | LFM2 debug: flush and log RMS/max-abs of activations at layer boundaries for cross-backend diffing. |

The three paging variables are ignored by the host tests, so goldens stay hermetic.

Testing: host tests pin the emitted op sequence with a thread-local recorder (`backend/hexagon/op_capture.rs`, test-only). `CERA_UPDATE_GOLDEN=1` (or `true`; test-only, never read in production) dumps the recorded op text of each golden to `target/golden/<label>.txt` so a changed golden can be diffed. The hexagon test session is hermetic against `CERA_HEXAGON_STEP` and `CERA_HEXAGON_UNARY_T1`.

---

## 6. Physical Hardware Benchmarks

Measurements taken on a retail Samsung Galaxy S25 Ultra (Snapdragon 8 Elite, SM-S938U1, Android 16) running in foreground execution:

| Workload | Model Architecture | Metric | Cera Hexagon NPU | Baseline / Comparison |
|----------|-------------------|--------|-------------------|-----------------------|
| **Text Generation** | LFM2.5-1.2B (Q4_0) | Decode Throughput | **154.91 tok/s** | llama.cpp: 157.8 tok/s |
| **Speculative Text (350M)** | LFM2.5-350M (Q4_0) | Decode (Prompt Lookup) | **259.50 tok/s** | Sequential: 164.0 tok/s (1.58x uplift) |
| **Speculative Text (2.6B)** | LFM2.5-2.6B (Q4_0) | Decode (Prompt Lookup) | **47.70 tok/s** | Sequential: 26.7 tok/s, llama CPU: 34.2 tok/s |
| **Vision Ingestion** | LFM2.5-VL-450M (Q4_0) | Backbone Prefill | **3,805.38 tok/s** | CPU: 1,295.0 tok/s |
| **Vision Generation** | LFM2.5-VL-450M (Q4_0) | Decode Throughput | **146.35 tok/s** | CPU: 140.97 tok/s |
| **Audio Time-To-First-Token** | LFM2-Audio-1.5B (Q4_0) | Audio TTFT | **238.5 ms** | Leap CPU: 600.0 ms (2.52x faster) |
| **Audio Synthesis** | Audio Vocoder / Detok | Decode Throughput | **36.67 tok/s** | Leap CPU: 23.02 tok/s (1.59x faster) |

---

## 7. Operator Coverage & Model Tier Support

The following table inventories Hexagon DSP kernel implementation and Cera host driver dispatch coverage across model architecture tiers:

| Tier | Architectures | Operations Required | In DSP Skeleton | In Host Driver | Status |
|------|---------------|---------------------|-----------------|----------------|--------|
| **T0** | LFM2, LFM2-VL, Whisper, Audio Decoder | `MulMat`, `MulMatNx`, `RmsNormMul`, `GluSwiglu`, `Rope`, `FlashAttnExt`, `Add`, `Mul`, `SsmConv`, `Argmax`, `SetRows`, `Cpy`, `Concat` | Yes | Yes | LFM2, LFM2-VL and the audio decoder measured on device (section 6); Whisper is verified on a device for transcription of one clip (encoder accuracy against the CPU not measured) |
| **T1 (Dense)** | LLaMA, Qwen2, Qwen3, Granite, Mistral3, Ministral3, MiniCPM, MiniCPM5, Nanbeige, Olmo2/3, Phi/Phi3 | `RmsNormMul`, `MulMat`, `MulMatNx`, `Rope` (Norm and Neox), `FlashAttnExt`, `Add` (residual and QKV/output/FFN bias, row-broadcast), `Mul` (residual scalar), `GluSwiglu`, Granite attention scale (kparams) and host-side logit scale, `Argmax`, Per-head QK-norm | Yes | Yes | Verified on S25 Ultra for Qwen3-0.6B, Qwen2-0.5B, SmolLM-135M and Llama-3.2-1B (Q8_0 and Q4_K_M): top-3 logits match the CPU at 700 tokens |
| **T2 (Dense+)** | Gemma 2, Gemma 4 | `FlashAttnExt` (sliding window), Attn logit soft-capping (`kparams[5]`), Final logit soft-capping, Sandwich norms, `GluGeglu`, Q4_1 wire repacking | Yes | Yes | Implemented in Hexagon engine and routed in model loader |
| **T3 (MoE)** | LFM2-MoE (BailingMoE is CPU-only; the Hexagon loader does not accept it) | Router `MulMat`, `UnarySigmoid`, `Add` (bias), `Argsort`, `GetRows`, `MulMatId`, `MulScalar` | Yes | Yes | Implemented in Hexagon engine (dispatch_moe_token) and routed in model loader; top-k weights are renormalized on the DSP (`Add` chain, `max(sum, 2^-14)` via `Sub` + `UnaryRelu`, then `Div`), matching the CPU `select_experts`. Router and SSM gate weights ride the Q8_0 wire (no F32 matmul layout), so near-tied experts can flip |
| **T4 (Hybrid SSM)** | Qwen3.5, GraniteHybrid, Falcon-H1, Mamba2 | `GatedDeltaNet`, `SsmConv`, `Cumsum`, `SolveTri`, `L2Norm`, Mamba-2 1D state-space scan | Partial (Mamba-2 SSD scan missing in DSP firmware) | Partial (needs state layout) | Qwen3.5 verified on S25 Ultra (0.8B Q4_K_M, prompts up to 1100 tokens, matches CPU). DSP kernel needed for Mamba-2; others supported or CPU-split |

Notes on the tiers above:

- **Host CPU steps.** YaRN, Llama-3 `rope_freqs` scaling, Mistral 3 attention temperature and Qwen 3.5 partial rotary are not expressible in the DSP rope kernel. For those models the queue is flushed, the step runs on the host and the model continues (one warning at load). Decode then flushes once per attention layer, which costs throughput.
- **CPU-only architectures.** `stablelm`, `starcoder2`, `cohere` and `command-r` need LayerNorm and/or parallel residual, which only the CPU `LlamaModel` path implements. The wgpu, Metal and Hexagon loaders return a clear error for them.
- **Q4_1 wire.** Q4_1 weights are repacked onto the `Q4K` wire type with `block_bytes` 20 (Q4_K uses 144). The host layout is pinned by tests; confirm the DSP stride on a Q4_1 model on the S25 Ultra.
- **The DSP address space caps what can be mapped; routed experts are paged through it.** The CDSP maps at most about 3 to 4 GiB in total. Measured on an S25 Ultra with `cera/examples/hexagon_map_probe.rs` (maps buffers after opening a DSP session): a single buffer maps up to 3.9 GiB when nothing else is mapped, but three 1 GiB buffers fit and a fourth (or an extra 512 MiB) fails with `error 1`; with 256 MiB buffers the total ceiling is 3840 MiB. Unmapping returns the address space (eight map and unmap cycles of 1 GiB all succeed), a sliding window of four mapped 512 MiB buffers over nine buffers that all stay allocated works with no copy per page-in, and mapping costs about 0.16 ms for 16 MiB, 2.7 ms for 256 MiB and 11.7 ms for 1 GiB (unmapping 0.2 to 0.5 ms). `FastRpcDriver::ensure_map_fits` tracks the bytes mapped against a 3.9 GiB ceiling and refuses a mapping that can never fit before allocating anything. A model whose resident weights exceed it (any non-MoE model, or a MoE one whose non-expert weights plus KV do not fit) is refused with an error that says so, and `--device auto` then falls back to the CPU. A routed-expert model such as the 4.8 GiB LFM2.5-8B-A1B is paged instead (next bullet), so `--device auto` skips the NPU for such a model unless `CERA_HEXAGON_PAGE_EXPERTS=1` is set (paged decode is about 20 tok/s against the CPU's 48), while an explicit `BackendPreference::Hexagon` pages it, for background work that should leave the CPU free. On Android `--device auto` also skips wgpu when twice the weights exceed the available RAM, because that upload OOM-killed the process on this phone (the CPU path runs the same 4.8 GiB file at 38 to 48 tok/s).
- **Paged routed experts** (`model/hexagon_lfm2/pager.rs`). Each MoE layer's expert stacks live in their own `rpcmem` buffer, all of them stay allocated (no copy per page-in), and an `ExpertPager` keeps the leading layers mapped and rotates the rest through a two-layer window, flushing the pending batch once per rotation because the DSP must be done with a buffer before it is unmapped (the host then tells the DSP to drop its hold with `htp_iface_munmap`, IDL method 5, as llama.cpp does, or `fastrpc_munmap` fails with error 1). The router runs on the DSP, so a layer's whole stack has to be mapped for its batch. Paging engages when the weights, scratch, and the KV cache for the configured context would not fit under the map budget, or when forced; see the three variables in the table above. A DSP refusal while pinning pins fewer layers instead of failing the load. On the S25 Ultra LFM2.5-8B-A1B loads in about 5 s with 13 of 22 expert layers pinned and 3183 MiB mapped. Measured with paging: decode 20 tok/s against the CPU's 48 and prefill 70 tok/s against 83 on a 32-token chat prompt, so the NPU path is for offload, not speed.
- **Routed-expert host fixes.** The routed-expert op chain had never run on a device, and host-side gaps (the skel is unchanged) kept it from running, found by reading the llama.cpp fork's host code: `MulMatId` was emitted with all-zero kernel params and the DSP answered `InvalParams` (it now gets the `MulMat` HVX params); the scalar `Mul`/`Div` (expert-weight combine, top-k renormalization) ran under the same-shape kernel and faulted the DSP (`dspqueue_read` error 0x2e), where the host must select `HTP_BINARY_KERNEL_CHUNKED` with `is_scalar` (`build_binary_scalar_kernel_params`); `GetRows` (which gathers the chosen experts' unbiased weights) was also emitted with all-zero kernel params, so the DSP left its output untouched, the renormalization divisor sat at its 2^-14 floor and the routed output was exactly zero (`build_get_rows_f32_kernel_params` now mirrors the host precompute); and a routed prefill chunk emits a chain per row, so a long prompt overflowed the 4 MiB staging buffer (about 10 KB of descriptors per row), which is now avoided by flushing between rows once the pending batch is half full. With all of that, a small synthetic LFM2-MoE gives the CPU's top-5 tokens in the same order on the NPU, paged output with a real window rotation is bit-identical to unpaged, and LFM2.5-8B-A1B generates coherent text on the S25 Ultra that matches the CPU's reasoning (one rephrasing in 60 greedy tokens).
- **Why the 8B's logits differ from the CPU's.** They agree at cosine 0.90 to 0.95, not 0.999, and this is routing sensitivity, not an NPU defect; it was checked rather than assumed. The per-layer residual stream stays within 1 to 3.5% of the CPU's through layer 8 (the dense LFM2.5-2.6B stays within 1 to 4% across all 30 layers), then a near-tied expert flips and one swapped expert changes a layer by about 20%. The router is not the cause (an F32 router recomputed on the NPU's own inputs picks the same experts), and the NPU's routed FFN matches an exact host recomputation from its own input, experts and weights to about 1% at every layer. The control: the CPU's own NEON kernel tier against its default tier is bit-identical at layer 0, then flips an expert at layer 4 and ends 12 to 19% off, and two NPU runs (HMX and HVX-only) agree with each other at only cosine 0.87.
- **Qwen 3.5 prefill chunks.** A prefill chunk of 64 or more rows used to fail with `VtcmTooSmall` on Qwen 3.5, because the attention gate ran as one flat vector; it now runs one row per token.
- **Decode was not reproducible (S25 Ultra, v79 skel).** Greedy decode on the NPU gave different text on identical input, run to run (the CPU never does), and the cause is not precision: with teacher-forced tokens (`examples/hexagon_decode_probe.rs`, bitwise logit comparison) every one of 39 identical replays produced a different logit sequence, differing by up to 3 from decode step 1, while the NPU's own prefill of the same prompt was bit-identical on every run. It depends on how many tensors one DSP batch holds, not on any op: ending the batch at every 16 op groups or fewer gave one identical sequence on every run, at 24 or more every run differed; counted in tensors, caps of 40 or fewer were reproducible on the 450M, the 2.6B and the 8B MoE, 48 was not on the 2.6B, and batches past about 60 tensors were not on the 450M. `CERA_HEXAGON_STEP`, `_BARRIERS` and any flush-after-op setting hid it for the same reason, while `CERA_HEXAGON_DECODE_OPS` did not (it cuts a helper's ops apart, and was phase sensitive). A bigger DSP dirty-range table (64 to 256 entries, rebuilt skel) did not change it, and the rebuilt skel with the HVX quantizer fix shows it too, so the firmware's handling of long batches is the suspect; the mechanism is not identified. The host-side bound is `HexagonQueueSession::set_max_tensors_per_flush`, applied by default (32) to decode and to every prefill chunk, which also bypasses the single-batch resident template (a large prefill chunk run as one batch left the recurrent state different from process to process in about a third of fresh processes, visible as decode step 1 disagreeing with a warm replay while the chunk's own logits matched; capped, 25 of 25 fresh processes were bit-identical on the 450M, the 2.6B and the 8B MoE, and 60 of 60 CLI runs gave one text, where 2 of 60 had diverged before); it costs decode speed, measured on the S25 Ultra with the CLI: 170 to 137 tok/s on the 450M (-20%, it is dispatch bound), 29.3 to 27.0 on the 2.6B (-8%) and 18.8 to 16.9 on the 8B MoE (-10%), about a third of that from no longer replaying the single-batch resident template and the rest from the extra batches; capping the large prefill chunks costs 3 to 7% of prefill on the 450M, about 1% on the 2.6B and 17% on the 8B MoE (79 to 66 tok/s: each routed row chain is itself about 30 tensors, so the cap ends a batch per row). Whether the MoE's large prefill chunks need the cap is not established; they are capped because the recurrent-state defect was seen on the dense-conv models and the MoE shares that block. `CERA_HEXAGON_BATCH_TENSORS=0` turns it off for anyone who prefers the speed. The same defect is what the small-M prefill ops cap (`MAX_OPS_PER_FLUSH`) was first working around.
- **NPU text-to-speech needs a voice phrase.** LFM2.5-Audio-1.5B Q4_0 given only `Perform TTS.` is out of distribution: greedy runs ranged from 23 to 4070 audio frames on the NPU, and a long runaway is also why the CPU run on the phone did not finish in five minutes. With `Perform TTS. Use the US female voice.` the NPU gave 25, 25 and 23 frames (CPU model with NPU audio: 23, 23), against 24 on a Mac CPU. Use the voice prompts from the audio profile (PR 463).
- **LFM2-Audio on the NPU (S25 Ultra, greedy audio sampling).** Text-to-speech through `--device hexagon` produces speech the model's own ASR transcribes back correctly, for the merged Q4_0 vocoder (English) and for the split-layout JP Q8_0 bundle (`vocoder-*.gguf` plus the `tokenizer-*.gguf` sidecar, merged on load): the depthformer takes 11.5 ms per frame against 176 ms on the CPU (Q4_0, 380 against 52 tok/s overall), and 16 ms against 76 ms on the JP bundle. The NPU run is not frame-for-frame identical to the CPU's (108 frames against 65 on the English sentence, 33 against 34 on the JP one); both transcribe to the same text. Speech input with `--device hexagon` runs the LLM backbone on the NPU (JP ASR prefill 0.18 s against 0.60 s, decode 41 against 23 tok/s) and, as described next, the FastConformer encoder now runs there too. Whisper on the NPU is device-verified for transcription (see the ViT bullet for the GELU change) but its encoder accuracy against the CPU is not measured here.
- **Still needs S25 Ultra validation.** Routed experts (LFM2-MoE) on a model that fits, Q4_1 weights (the wire stride, block bytes 20 against 144), Gemma 2 and sliding-window models past the window, a forced 30 s DSP hang (quiesce and reset), and Whisper and ViT parity beyond the short samples tried (Whisper-base.en and LFM2.5-VL-450M matched the CPU on one clip each).
