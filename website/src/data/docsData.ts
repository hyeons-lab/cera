import type { DocCategory } from '../types/docs';

export const docsCategories: DocCategory[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    description: 'Introduction to Cera and rapid installation across environments',
    iconName: 'Sparkles',
    articles: [
      {
        id: 'overview',
        title: 'Why Cera?',
        description: 'Design principles and architectural foundations of Cera',
        badge: 'Core',
        sections: [
          {
            id: 'philosophy',
            title: 'Zero Runtime, Pure Systems Inference',
            content: `Cera is an inference engine written from the ground up in pure Rust. Unlike conventional machine learning toolchains that depend on multi-gigabyte Python environments, C++ shared library mismatches, or heavyweight runtime frameworks, Cera compiles down to a single compact binary or static library.

Key architectural pillars:
1. Zero Python or runtime dependencies: A single dependency-free static binary or library package.
2. Runs everywhere from one core: The identical Rust codebase drives desktop CLIs, iOS/macOS frameworks (Metal MSL), Android libraries (UniFFI), Flutter/Dart packages, and WebAssembly in browsers (WebGPU).
3. Standard GGUF and LeapBundles compatibility: Directly load GGUF models or auto-download from Hugging Face.
4. Universal multimodal envelopes: Vision (image to text) and audio (ASR and TTS) models load through the same clean session API.
5. First-class thinking and reasoning support: Native streaming thought delimiters and reasoning state machines.`,
          },
          {
            id: 'supported-architectures',
            title: 'Supported Model Architectures',
            content: `Cera dispatches dynamically on the GGUF general.architecture string:
- lfm2: Liquid LFM2 and LFM2.5 (the canonical LeapBundles family supporting text, vision, and audio).
- lfm2moe: Liquid LFM2.5-8B-A1B (routed mixture-of-experts activating 4 of 32 experts per token).
- llama: LLaMA 2 / 3 / 3.2, and classic Mistral 7B.
- qwen2 / qwen3: Qwen2, Qwen2.5, and Qwen3 instruction and reasoning models.
- granite: IBM Granite 3.x and dense Granite 4.1 lines.`,
          },
        ],
      },
      {
        id: 'installation',
        title: 'Quickstart Installation',
        description: 'Install Cera for CLI, Rust, Web, Flutter, Android, and Apple platforms',
        badge: 'Setup',
        sections: [
          {
            id: 'cli-install',
            title: 'CLI Installation',
            content: 'The CLI can be installed directly from crates.io or built from source with your preferred hardware acceleration:',
            codeSnippets: {
              cli: {
                language: 'bash',
                filename: 'terminal',
                code: `# Install the CPU-optimized CLI binary
cargo install cera-cli --locked

# Or install with native Apple Metal acceleration
cargo install cera-cli --locked --features metal

# Or install with cross-platform Vulkan / DX12 / WebGPU acceleration
cargo install cera-cli --locked --features gpu`,
              },
              rust: {
                language: 'toml',
                filename: 'Cargo.toml',
                code: `[dependencies]
cera = "0.5.1"`,
              },
              web: {
                language: 'bash',
                filename: 'terminal',
                code: `npm install @hyeons-lab/cera-wasm`,
              },
              flutter: {
                language: 'bash',
                filename: 'terminal',
                code: `flutter pub add cera_ffi_flutter`,
              },
            },
          },
          {
            id: 'one-shot-run',
            title: 'Running Your First Model',
            content: 'Download and run a quantized model from Hugging Face with a single command:',
            codeSnippets: {
              cli: {
                language: 'bash',
                filename: 'terminal',
                code: `# Auto-download and run LFM2.5-1.2B with interactive prompt
cera run --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0 --prompt "Explain quantum computing in two sentences."

# Run an interactive multi-turn chat session with warm prefix caching
cera chat --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0`,
              },
            },
          },
        ],
      },
    ],
  },
  {
    id: 'sdk-platforms',
    title: 'Multiplatform SDKs',
    description: 'In-depth guides and API examples for each supported language and platform',
    iconName: 'Code',
    articles: [
      {
        id: 'rust-sdk',
        title: 'Rust Core SDK (cera)',
        description: 'Native Rust inference engine, Session management, and ModalitySink streaming',
        badge: 'Rust',
        sections: [
          {
            id: 'rust-session-api',
            title: 'Engine Initialization and Streaming Generation',
            content: 'The cera crate exposes CeraEngine and Session. Token generation streams incrementally into any implementation of ModalitySink:',
            codeSnippets: {
              rust: {
                language: 'rust',
                filename: 'main.rs',
                code: `use cera::{
    CeraEngine, EngineConfig, GenerateOpts, ModalitySink,
    SessionConfig, UserMessage, FinishReason,
};
use std::sync::Arc;

struct MySink;

impl ModalitySink for MySink {
    fn on_text_chunk(&mut self, text: &str) {
        print!("{}", text);
    }

    fn on_thought_chunk(&mut self, thought: &str) {
        eprint!("[thinking: {}]", thought);
    }

    fn on_done(&mut self, reason: FinishReason) {
        println!("\\nFinished: {:?}", reason);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load engine from a local GGUF file or LeapBundle
    let engine = CeraEngine::from_path("model.gguf", EngineConfig::default())?;
    let mut session = engine.new_session(SessionConfig::default());

    // Dispatch canonical multimodal user message
    let msg = UserMessage {
        text: "Explain how TurboQuant compresses KV caches.".to_string(),
        images: vec![],
        audio: None,
    };
    session.append_user_message(msg)?;

    // Stream generated text and thoughts
    let mut sink = MySink;
    let opts = GenerateOpts {
        max_tokens: 256,
        temperature: 0.7,
        ..Default::default()
    };
    let summary = session.generate(&opts, &mut sink)?;

    println!("Performance: {:.1} tokens/sec", summary.decode_tok_per_sec);
    Ok(())
}`,
              },
            },
          },
        ],
      },
      {
        id: 'web-sdk',
        title: 'WebAssembly & WebGPU (cera-wasm)',
        description: 'In-browser zero-server LLM inference running on GPU and Web Workers',
        badge: 'Web / JS',
        sections: [
          {
            id: 'webgpu-setup',
            title: 'In-Browser WebGPU Session',
            content: 'cera-wasm provides hardware-accelerated LLM execution directly in modern browsers through WebGPU and WGSL compute shaders:',
            codeSnippets: {
              web: {
                language: 'typescript',
                filename: 'inference.ts',
                code: `import init, { WebGpuSession, TurboQuantConfig } from '@hyeons-lab/cera-wasm';

async function runWebGpuChat(ggufBuffer: Uint8Array) {
  // Initialize WASM runtime
  await init();

  // Configure TurboQuant 3-bit KV compression (optional)
  const turboQuant = new TurboQuantConfig(42n);

  // Allocate GPU buffers and upload model weights (e.g. 4096 context length)
  const session = await WebGpuSession.create(ggufBuffer, 4096, turboQuant);
  console.log(\`Loaded on \${session.adapter} with \${session.kvCompression} KV cache\`);

  // Stream tokens with dedicated thought callback
  await session.generate(
    "What are the benefits of edge computing?",
    256, // maxTokens
    0.7, // temperature
    null, null, null, // topP, topK, seed
    (tokenChunk: string) => {
      process.stdout.write(tokenChunk);
    },
    null, // onAudio
    (thoughtChunk: string) => {
      console.log("[thought]", thoughtChunk);
    }
  );
}`,
              },
            },
          },
        ],
      },
      {
        id: 'flutter-sdk',
        title: 'Flutter & Dart (cera_ffi_flutter)',
        description: 'Cross-platform mobile and desktop AI apps with native FFI and web workers',
        badge: 'Flutter',
        sections: [
          {
            id: 'flutter-usage',
            title: 'Unified Flutter API across iOS, Android, macOS, and Web',
            content: 'cera_ffi_flutter automatically embeds the native precompiled dylibs on mobile/desktop and dispatches to Web Workers on Flutter Web:',
            codeSnippets: {
              flutter: {
                language: 'dart',
                filename: 'chat_screen.dart',
                code: `import 'package:flutter/material.dart';
import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';

Future<void> runInference() async {
  // Download or open model bundle
  final cera = await Cera.openBundle(
    bundleId: 'LFM2.5-1.2B-Instruct',
    quant: 'Q4_0',
    options: const CeraOptions(
      backend: CeraBackend.auto,
      turboQuant: true,
    ),
  );

  // Create multimodal message envelope
  final userMessage = CeraUserMessage(
    text: 'What are the main advantages of running local models?',
  );

  // Send message and stream generated response with reasoning
  final stream = cera.sendMessage(
    userMessage,
    maxTokens: 512,
    onThought: (thoughtChunk) {
      debugPrint('[reasoning] $thoughtChunk');
    },
    onAudio: (pcm, sampleRate) {
      debugPrint('Synthesized \${pcm.length} audio samples at \${sampleRate} Hz');
    },
  );

  await for (final textPiece in stream) {
    debugPrint(textPiece);
  }

  await cera.close();
}`,
              },
            },
          },
        ],
      },
      {
        id: 'android-sdk',
        title: 'Android & Kotlin (cera-ffi-kotlin)',
        description: 'Android AAR bindings with Jetpack Compose, CeraDownloadService, and Coroutine flows',
        badge: 'Android',
        sections: [
          {
            id: 'android-usage',
            title: 'Foreground Model Downloads and Inference',
            content: 'cera-ffi-android ships with background download resilience, HTTP range resumption, and coroutine flows:',
            codeSnippets: {
              android: {
                language: 'kotlin',
                filename: 'CeraViewModel.kt',
                code: `import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.liquid.cera.CeraEngine
import ai.liquid.cera.EngineConfig
import ai.liquid.cera.SessionConfig
import ai.liquid.cera.UserMessage
import ai.liquid.cera.ModalitySink
import ai.liquid.cera.FinishReason
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

class CeraViewModel : ViewModel() {
    private var engine: CeraEngine? = null

    fun loadModel(modelPath: String) {
        viewModelScope.launch {
            val config = EngineConfig()
            engine = CeraEngine(modelPath, config)
        }
    }

    fun sendMessage(prompt: String) {
        val session = engine?.newSession(SessionConfig()) ?: return
        val message = UserMessage(
            text = prompt,
            images = emptyList(),
            audio = null,
        )

        session.sendMessageStreaming(
            message = message,
            opts = ai.liquid.cera.GenerateOpts(),
            sink = object : ModalitySink {
                override fun onTextChunk(text: String) {
                    // Update UI state
                }
                override fun onThoughtChunk(text: String) {
                    // Display thinking stream
                }
                override fun onAudioFrames(pcm: List<Double>, sampleRate: UInt) {}
                override fun onDone(reason: FinishReason) {}
            }
        )
    }
}`,
              },
            },
          },
        ],
      },
      {
        id: 'apple-sdk',
        title: 'Apple & Swift (Swift Package)',
        description: 'SwiftPM package with native Metal MSL shader acceleration for iOS and macOS',
        badge: 'Swift',
        sections: [
          {
            id: 'swift-usage',
            title: 'Swift Package Manager Integration',
            content: 'Add Cera to your Xcode project using the Swift Package Manager and consume native Metal inference:',
            codeSnippets: {
              apple: {
                language: 'swift',
                filename: 'InferenceService.swift',
                code: `import Foundation
import Cera

class InferenceService {
    private var engine: CeraEngine?

    func initialize(modelPath: String) throws {
        let config = EngineConfig()
        // Auto chooses native Metal GPU kernels on Apple Silicon
        self.engine = try CeraEngine(modelPath: modelPath, config: config)
    }

    func chat(prompt: String, onChunk: @escaping (String) -> Void) throws {
        guard let engine = engine else { return }
        let session = engine.newSession(config: SessionConfig())
        
        let msg = UserMessage(text: prompt, images: [], audio: nil)
        let opts = GenerateOpts()

        class Sink: ModalitySink {
            let onChunk: (String) -> Void
            init(onChunk: @escaping (String) -> Void) { self.onChunk = onChunk }
            func onTextChunk(text: String) { onChunk(text) }
            func onThoughtChunk(text: String) { print("[thought] \\(text)") }
            func onAudioFrames(pcm: [Double], sampleRate: UInt32) {}
            func onDone(reason: FinishReason) {}
        }

        try session.sendMessageStreaming(message: msg, opts: opts, sink: Sink(onChunk: onChunk))
    }
}`,
              },
            },
          },
        ],
      },
    ],
  },
  {
    id: 'innovations',
    title: 'Core Innovations',
    description: 'Technical deep-dives into TurboQuant, FreeToken, DSpark, and Silero VAD',
    iconName: 'Cpu',
    articles: [
      {
        id: 'turboquant',
        title: 'TurboQuant KV Cache Compression',
        description: 'Extreme KV cache compression (~12x vs f32) with near-lossless perplexity and zero calibration',
        badge: 'Algorithm',
        sections: [
          {
            id: 'turboquant-theory',
            title: 'How TurboQuant Works',
            content: `Cera provides the first production implementation of Google Research's TurboQuant (arXiv:2504.19874). In long-context inference, KV cache memory quickly dominates weight memory: at 32k tokens, standard FP32 or FP16 caches consume multiple gigabytes per session.

TurboQuant compresses keys to ~3 bits and values to ~2 bits:
1. PolarQuant transformation: Applies per-layer randomized orthogonal rotations (Hadamard transforms) to eliminate outlier activations across attention heads.
2. Quantized Johnson-Lindenstrauss (QJL) embeddings: Preserves inner products with minimal distortion, maintaining generation perplexity within +/-5% of uncompressed FP32.
3. Zero calibration requirement: Requires no calibration dataset or offline tuning. Works out-of-the-box on any supported architecture.
4. Tri-backend acceleration: Native SIMD kernels on CPU (AVX2/NEON), hand-written MSL compute shaders on Apple Metal, and WGSL pipelines on WebGPU/wgpu.`,
          },
        ],
      },
      {
        id: 'freetoken',
        title: 'FreeToken: Semantic Anchor Caching',
        description: 'Hierarchical warm and cold tiering for long-context multi-turn conversations',
        badge: 'Caching',
        sections: [
          {
            id: 'freetoken-explanation',
            title: 'Semantic-Aware KV Persistence',
            content: `Based on FreeToken (arXiv:2406.14588), Cera implements a hierarchical caching architecture:
- Semantic Anchor Extraction: Automatically identifies sentence boundaries and high-entropy prompt positions as persistent anchor points.
- Hierarchical Tiering: Maintains warm in-memory caches for recent tokens and compresses historical tokens into cold TurboQuant representations.
- FlatBuffers v2 Serialization: Saves and restores persistent KV caches to disk in a fast zero-copy format across session restarts.`,
          },
        ],
      },
      {
        id: 'dspark',
        title: 'DSpark: Neural Speculative Decoding',
        description: 'Parallel multi-token verification with lightweight sidecar drafters',
        badge: 'Speed',
        sections: [
          {
            id: 'dspark-explanation',
            title: 'Speculative Drafting & Verification',
            content: `DSpark (arXiv:2407.08608) accelerates inference throughput by pairing the target model with a lightweight drafter:
1. Sidecar Drafter: Loads a compact draft model sharing embedding weights with the base model to eliminate duplicate memory mapping.
2. Batched Parallel LM-Head Verification: Verifies candidate token sequences in a single batched GEMM step rather than sequential single-token evaluations.
3. Zero Loss of Correctness: The target model retains strict argmax equivalence; draft rejections only cost speed, never accuracy.`,
          },
        ],
      },
      {
        id: 'silero-vad',
        title: 'Silero VAD v5 Native Rust Engine',
        description: 'Zero-dependency streaming Voice Activity Detection running entirely on-device',
        badge: 'Audio',
        sections: [
          {
            id: 'silero-vad-explanation',
            title: 'Streaming Speech Boundary Detection',
            content: `Cera includes a native pure-Rust implementation of Silero VAD v5:
- Zero ONNX Runtime or Python required: Model weights run directly from GGUF packaging using Cera compute kernels.
- Real-time 512-sample streaming: State-machine VadIterator evaluates continuous 512-sample audio frames (32ms at 16kHz, or 256 samples at 8kHz).
- Automatic boundary timestamping: Emits SpeechStart and SpeechEnd events with configurable confidence thresholds and silence padding.
- Multiplatform FFI: Directly callable from Rust, Flutter/Dart, Swift, and Kotlin for hands-free interactive voice agents.`,
            codeSnippets: {
              rust: {
                language: 'rust',
                filename: 'vad_streaming.rs',
                code: `use cera::vad::{SileroVad, VadConfig, VadIterator, VadSampleRate, VadEvent};

// Load Silero VAD v5 natively from GGUF format (zero ONNX runtime):
let mut vad = SileroVad::from_file("models/silero_vad.gguf")?;

let config = VadConfig {
    threshold: 0.5,
    min_speech_duration_ms: 250,
    min_silence_duration_ms: 100,
    speech_pad_ms: 30,
    ..Default::default()
};

let mut iterator = VadIterator::new(VadSampleRate::Rate16kHz, Some(config));

// Stream continuous 512-sample PCM chunks (32ms frames at 16kHz):
for chunk in pcm_chunks {
    if let Some(event) = iterator.process_chunk(&mut vad, &chunk)? {
        match event {
            VadEvent::SpeechStart { sample, ms } => {
                println!("User started speaking at {ms}ms (sample {sample})");
            }
            VadEvent::SpeechEnd { start_ms, end_ms, .. } => {
                println!("Speech segment complete: {start_ms}ms to {end_ms}ms");
                // Trigger downstream ASR transcription or speech generation
            }
        }
    }
}`,
              },
              flutter: {
                language: 'dart',
                filename: 'voice_vad.dart',
                code: `import 'package:cera_ffi/cera_ffi.dart';

final vad = await FfiSileroVad.fromFile('silero_vad.gguf');
final iterator = FfiVadIterator(
  rate: FfiVadSampleRate.rate16kHz,
  config: FfiVadConfig(
    threshold: 0.5,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
    speechPadMs: 30,
  ),
);

// Feed streaming 512-sample audio frame from device microphone:
final event = iterator.processChunk(vad: vad, chunk: micFrame512);
if (event is FfiVadEventSpeechStart) {
  print('Speech started at \${event.ms}ms');
} else if (event is FfiVadEventSpeechEnd) {
  print('Speech ended: \${event.startMs}ms - \${event.endMs}ms');
}`,
              },
              apple: {
                language: 'swift',
                filename: 'VoiceAgent.swift',
                code: `import Cera

let vad = try FfiSileroVad.fromFile(path: "silero_vad.gguf")
let iterator = FfiVadIterator(rate: .rate16kHz, config: nil)

// In AVAudioEngine audio tap (512-sample PCM buffer):
if let event = try iterator.processChunk(vad: vad, chunk: buffer) {
    if let speechStart = event as? FfiVadEventSpeechStart {
        print("User began speaking at \\(speechStart.ms)ms")
    } else if let speechEnd = event as? FfiVadEventSpeechEnd {
        print("User finished speaking: \\(speechEnd.startMs) - \\(speechEnd.endMs)ms")
    }
}`,
              },
            },
          },
        ],
      },
    ],
  },
  {
    id: 'features',
    title: 'Advanced Features',
    description: 'Structured output with GBNF grammars and format-aware tool calling',
    iconName: 'Settings',
    articles: [
      {
        id: 'grammars',
        title: 'GBNF Grammars & Structured Output',
        description: 'Mask samplers at the byte level to guarantee valid JSON and custom schemas',
        badge: 'Grammar',
        sections: [
          {
            id: 'gbnf-overview',
            title: 'Byte-Level Constrained Sampling',
            content: `Cera includes a native byte-level GBNF grammar parser and sampler constraint engine. By masking invalid token logits before sampling, Cera guarantees that generated outputs strictly adhere to formal schemas:
- JSON mode: Set one flag (--json or GenerateOpts.jsonMode) to enforce syntactically valid JSON.
- Custom schemas: Compile arbitrary GBNF grammars supporting alternation, grouping, and repetition.
- Cross-platform availability: Exposed across CLI, Rust, Swift, Kotlin, Dart, and WebAssembly.`,
            codeSnippets: {
              cli: {
                language: 'bash',
                filename: 'terminal',
                code: `# Guarantee valid JSON response
cera run -m model.gguf -p "List 3 European capitals as JSON" --json

# Enforce custom grammar schema
cera run -m model.gguf -p "Output a valid SQL query" --grammar @sql.gbnf`,
              },
            },
          },
        ],
      },
      {
        id: 'tool-calling',
        title: 'Format-Aware Tool Calling',
        description: 'Automatic Pythonic and JSON tool calling with lazy grammar triggers',
        badge: 'Tools',
        sections: [
          {
            id: 'tool-calling-overview',
            title: 'Format Detection and Lazy Grammar Triggers',
            content: `Different model families utilize distinct tool calling formats:
- Liquid LFM2 / LFM2.5: Emits Pythonic function calls like [get_weather(city="Paris")].
- Qwen / Hermes: Emits JSON tool schemas like <tool_call>{"name": "...", "arguments": {...}}</tool_call>.

Cera inspects the model architecture, formats tools into the chat template, and parses tool invocations into structured records. With --constrain-tools, a lazy grammar trigger keeps token generation free until a tool invocation starts, then constrains arguments to the exact JSON-Schema types.`,
          },
        ],
      },
      {
        id: 'streaming-quantization',
        title: 'Hugging Face Streaming Quantization',
        description: 'Automatic SafeTensors streaming, zero-disk on-the-fly quantization, and local caching',
        badge: 'Zero-Disk',
        sections: [
          {
            id: 'streaming-quant-overview',
            title: 'Zero-Disk Remote Quantization & Caching',
            content: `Cera can point directly at any remote Hugging Face repository containing SafeTensors weights. When no primary GGUF exists, Cera streams the tensors over HTTP range requests and converts them on-the-fly into quantized GGUF format:
- Zero unquantized disk footprint: The remote 10-20 GB FP16/BF16 SafeTensors files are never saved to disk. Weights are quantized tensor-by-tensor directly in memory.
- Resumable checkpoints: If a download or conversion is interrupted, Cera preserves a checkpoint (model.gguf.checkpoint.json) and resumes exactly at the interrupted tensor index via HTTP Range requests.
- Automatic local caching: The final quantized GGUF is stored in ~/.cache/cera/huggingface.co/<owner>/<repo>/quantized/<quant>/. Subsequent runs load the model instantly from disk with zero network overhead.
- Supported target formats: Q4_K_M (default), Q5_K_M, Q6_K, Q8_0, and Q4_0.`,
            codeSnippets: {
              cli: {
                language: 'bash',
                filename: 'terminal',
                code: `# Stream remote SafeTensors from Hugging Face, quantize to Q4_K_M on-the-fly, and run:
cera run --model meta-llama/Llama-3.2-1B --quant Q4_K_M

# Pull and quantize ahead of time into local cache:
cera pull Qwen/Qwen2.5-0.5B --quant Q8_0

# Run directly from local cache (zero network calls):
cera run --model meta-llama/Llama-3.2-1B`,
              },
              rust: {
                language: 'rust',
                filename: 'main.rs',
                code: `use cera::{CeraConfig, CeraEngine, Session};

// Point directly at a Hugging Face repository spec:
// Cera checks the cache, downloads SafeTensors headers, streams & quantizes on-the-fly,
// and saves the cached GGUF model automatically.
let engine = CeraEngine::from_hf("meta-llama/Llama-3.2-1B", &CeraConfig::default())?;
let mut session = Session::new(&engine)?;

session.append_prompt("Explain quantum computing in one sentence.")?;
session.generate(&Default::default(), |token_chunk| {
    print!("{token_chunk}");
    Ok(true)
})?;`,
              },
            },
          },
        ],
      },
    ],
  },
];
