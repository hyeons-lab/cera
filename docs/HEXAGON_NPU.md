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
- Interrogates FastRPC hardware info and identifies the DSP core architecture (`v73`, `v75`, `v79`, `v81`, `v85`).
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
| **Next-Gen Snapdragon** | `v85` | `libggml-htp-v85.so` | Extended HTP ops, unified Conv/Snake units | Prebuilt and ABI aligned |

---

## 3. Skeleton Libraries & DSP Additions

The DSP-side worker libraries (`libggml-htp-v{73,75,79,81,85}.so`) are precompiled Hexagon ELF shared objects that implement the low-level compute kernels. Cera embeds all five libraries directly into the host binary (`libcera_ffi.so` and the `cera` crate) and extracts them at startup.

### Additions to the Skeleton Libraries
1. **Extended Operator Set**:
   - `Conv1D`: 1D convolution kernel with configurable dilation, stride, and padding for waveform audio encoding and speech feature extraction.
   - `ConvTranspose1D`: Transposed 1D convolution kernel for vocoder audio synthesis and spectrogram upsampling.
   - `Snake` / `Snake1D`: Specialized sinusoidal activation function ($\sin^2(\alpha x)$) required for neural vocoders (BigVGAN, Vocos).
   - `UnaryStep` and `Sum`: Element-wise step activations and reductions executing directly on HVX vector registers.
2. **Opcode Alignment (`HtpOpCode`)**:
   - Realigned all discriminant values in Cera's `HtpOpCode` enum with the upstream v85 DSP firmware ABI (`UnaryStep = 21`, `Sum = 39`, `Cpy = 32`, `Scale = 33`, `Conv1d = 40`, `Snake = 41`). This prevents command displacement where non-scale operators were previously misinterpreted as `Scale` operations with invalid parameters.
3. **Architecture V85 Compilation**:
   - Built with Qualcomm Hexagon SDK 6.6.0.0 and Hexagon Tools 19.0.07 (`hexagon-clang` with whole-program Link-Time Optimization), compiling dedicated targets with `-DDSP_VERSION`.
4. **16 KB Page Alignment**:
   - All ELF headers and program segments comply with Android 16+ 16 KB page-alignment requirements (`0x4000`), ensuring compatibility with Google Play store distribution.

---

## 4. Accelerated Kernels & Subsystems

### 4.1 LLM Text Generation (`HexagonLfmModel`)
- **32x32 Tiled Weight Repacking**: Weights quantized in Q4_0 and Q8_0 formats are repacked into 32x32 tiles. This layout maps directly to HTP matrix multiplier hardware blocks, maximizing SIMD register utilization and eliminating runtime unpacking overhead.
- **Single-Flush Forward Decode**: The full forward decode pass (attention projections, multi-head attention, RMSNorm, SwiGLU feed-forward, and residual adds) across all layers is staged into a single contiguous FastRPC batch submission. This removes approximately 22 host-to-DSP synchronization roundtrips per token.
- **Ping-Pong Scratch Buffering**: Alternating layer buffers (`activation` and `activation_b`, `normed` and `normed_b`) isolate memory access between consecutive layers, preventing data races in the asynchronous DSP execution pipeline.
- **Static Command Queue Template Caching**: Because decode graph topology is static across tokens, Cera constructs the FastRPC batch template once on the initial step. Subsequent steps patch only sequence position (`pos`) and attention KV pointers in place, achieving zero-allocation host dispatch.
- **FastRPC Latency QoS**: Sets `FASTRPC_CONTROL_LATENCY = 100 µs` via `remote_session_control()` during initialization. This prevents Qualcomm's CDSP frequency governor from scaling clocks down during brief inter-token pauses, sustaining peak throughput throughout the generation turn.
- **Q8_0 Quantized KV Cache**: Stores key and value cache rows in 32-element Q8_0 quantized blocks (34 bytes per block) instead of uncompressed F16 (64 bytes). This yields a 47% reduction in memory bandwidth and physical rpcmem footprint, expanding the usable context length on device.

### 4.2 Multimodal Vision Transformer (`HexagonVisionEncoder`)
- **Batched 24-Block ViT Pipeline**: All 24 blocks of the Vision Transformer (ViT) are dispatched in a single batch submission to the DSP, executing patch embeddings, LayerNorm, multi-head attention, and MLP projections without CPU intervention.
- **On-NPU Flash Attention**: Computes self-attention over image patch tokens directly on the DSP using causal flash attention kernels with F16 KV scratch conversion.
- **Quantized Projector Dispatch**: MLP multimodal projector weights are repacked for HTP matrix kernels, projecting visual representations into the language model embedding dimension on device.

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
