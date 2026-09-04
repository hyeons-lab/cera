import React from 'react';
import { Layers, Smartphone, Globe, Terminal, Cpu, HardDrive, Zap, CheckCircle2 } from 'lucide-react';

export const ArchitectureDiagram: React.FC = () => {
  return (
    <div className="py-20 bg-[#0c0e15] border-b border-[#222638]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-semibold mb-3">
            <Layers className="w-3.5 h-3.5" />
            <span>Layered Systems Design</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight mb-4">
            Unified Multiplatform Architecture
          </h2>
          <p className="text-slate-400 text-sm sm:text-base">
            From mobile devices to high-end workstations and browser tabs, Cera shares the exact same core inference engine without runtime fragmentation.
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-4">
          {/* Layer 1: Consumers */}
          <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a]">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-blue-400" />
                1. Applications & Frameworks
              </span>
              <span className="text-[11px] text-blue-400 font-mono">Consuming Applications</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <Terminal className="w-5 h-5 mx-auto mb-1 text-slate-300" />
                <span className="text-xs font-semibold text-white block">CLI / Shell</span>
                <span className="text-[10px] text-slate-400 font-mono">cera-cli</span>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <Smartphone className="w-5 h-5 mx-auto mb-1 text-slate-300" />
                <span className="text-xs font-semibold text-white block">Apple iOS/macOS</span>
                <span className="text-[10px] text-slate-400 font-mono">SwiftPM</span>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <Smartphone className="w-5 h-5 mx-auto mb-1 text-slate-300" />
                <span className="text-xs font-semibold text-white block">Android</span>
                <span className="text-[10px] text-slate-400 font-mono">Kotlin AAR</span>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <Layers className="w-5 h-5 mx-auto mb-1 text-slate-300" />
                <span className="text-xs font-semibold text-white block">Flutter</span>
                <span className="text-[10px] text-slate-400 font-mono">cera_ffi_flutter</span>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638] col-span-2 sm:col-span-1">
                <Globe className="w-5 h-5 mx-auto mb-1 text-slate-300" />
                <span className="text-xs font-semibold text-white block">Browser / Node</span>
                <span className="text-[10px] text-slate-400 font-mono">cera-wasm</span>
              </div>
            </div>
          </div>

          {/* Connection arrows */}
          <div className="flex justify-center">
            <div className="w-0.5 h-4 bg-gradient-to-b from-blue-500 to-indigo-500" />
          </div>

          {/* Layer 2: Universal Dispatch & Core Engine */}
          <div className="p-5 rounded-xl border border-blue-500/30 bg-[#121626] shadow-xl shadow-blue-500/5">
            <div className="text-xs font-bold uppercase tracking-wider text-blue-400 mb-3 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                2. Cera Core Engine (cera crate)
              </span>
              <span className="text-[11px] text-emerald-400 font-mono">Pure Rust · Zero Runtime</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-[#0b0d17] border border-blue-500/20">
                <div className="font-semibold text-slate-200 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                  Multimodal Session API
                </div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Canonical UserMessage envelope, automatic chat template formatting, vision patch embedding, and audio PCM dispatch.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-[#0b0d17] border border-blue-500/20">
                <div className="font-semibold text-slate-200 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                  StreamingThinkingParser
                </div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Zero-allocation thought delimiter matching and real-time reasoning chunk separation for DeepSeek-R1 and Qwen.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-[#0b0d17] border border-blue-500/20">
                <div className="font-semibold text-slate-200 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                  Structured Output & Tools
                </div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Byte-level GBNF grammar constraint engine, JSON-Schema enforcement, and format-aware tool calling.
                </p>
              </div>
            </div>
          </div>

          {/* Connection arrows */}
          <div className="flex justify-center">
            <div className="w-0.5 h-4 bg-gradient-to-b from-indigo-500 to-sky-500" />
          </div>

          {/* Layer 3: Hardware Compute Backends */}
          <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a]">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                3. Compute Acceleration Backends
              </span>
              <span className="text-[11px] text-amber-400 font-mono">Dynamic Dispatch</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <div className="font-semibold text-white text-xs mb-1">Native Metal MSL</div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Hand-written MSL shaders for Apple Silicon (M-series and A-series). Single-encoder dispatch and GPU argmax sampling.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <div className="font-semibold text-white text-xs mb-1">wgpu / WebGPU (WGSL)</div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Vulkan on Linux/Android, DX12 on Windows, and WebGPU in Chrome, Edge, and Safari for browser inference.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <div className="font-semibold text-white text-xs mb-1">CPU SIMD Kernels</div>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  ARM NEON (with DotProd), x86 AVX2 / AVX-512, with model-aware decode thread sizing and cache tiering.
                </p>
              </div>
            </div>
          </div>

          {/* Connection arrows */}
          <div className="flex justify-center">
            <div className="w-0.5 h-4 bg-gradient-to-b from-sky-500 to-blue-500" />
          </div>

          {/* Layer 4: Weights and KV Storage */}
          <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a]">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-emerald-400" />
                4. Model Formats & Memory Architecture
              </span>
              <span className="text-[11px] text-emerald-400 font-mono">Zero Calibration</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <span className="font-semibold text-slate-200 block mb-1">LeapBundles & GGUF</span>
                <span className="text-slate-400 text-[11px]">
                  Q4_0, Q4_K, Q8_0, and F32 with streaming HTTP range download resumption.
                </span>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <span className="font-semibold text-slate-200 block mb-1">TurboQuant TQ3</span>
                <span className="text-slate-400 text-[11px]">
                  3-bit keys and 2-bit values (~12x compression vs FP32) with near-lossless perplexity.
                </span>
              </div>
              <div className="p-3 rounded-lg bg-[#141724] border border-[#222638]">
                <span className="font-semibold text-slate-200 block mb-1">FreeToken Anchors</span>
                <span className="text-slate-400 text-[11px]">
                  Hierarchical warm and cold semantic prefix caching with FlatBuffers v2 serialization.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
