import React from 'react';
import { Cpu, Zap, Eye, Mic, Layers, Code, Sparkles, HardDrive } from 'lucide-react';

export const FeatureGrid: React.FC = () => {
  const features = [
    {
      icon: Cpu,
      title: 'Zero Runtime, Pure Rust Core',
      description: 'Single static binary or library with zero Python or runtime dependencies. Compiles to clean native code on desktop, mobile, and WebAssembly.',
      tag: 'Architecture',
    },
    {
      icon: Zap,
      title: 'TurboQuant KV Compression',
      description: 'First production implementation of Google Research TurboQuant. Compresses keys to ~3 bits and values to ~2 bits (~12x vs f32) with zero calibration.',
      tag: 'Memory',
    },
    {
      icon: Eye,
      title: 'Universal Multimodal Envelopes',
      description: 'Unified UserMessage dispatch for text, vision patches (ViT encoder on GPU), and streaming audio. Canonical ordering ensures robust inference across modalities.',
      tag: 'Multimodal',
    },
    {
      icon: Layers,
      title: 'Native GPU & WebGPU Kernels',
      description: 'Hand-written Metal MSL compute shaders on Apple Silicon and high-performance WGSL shaders on Vulkan, DX12, and WebGPU in browsers.',
      tag: 'Performance',
    },
    {
      icon: Sparkles,
      title: 'DSpark Speculative Decoding',
      description: 'Accelerate throughput via parallel multi-token draft verification. Batched LM-head verification achieves up to 2x speedup without accuracy loss.',
      tag: 'Throughput',
    },
    {
      icon: Code,
      title: 'Structured Output & Tool Calling',
      description: 'Byte-level GBNF grammar constraints guarantee valid JSON. Format-aware tool calling automatically handles Pythonic and Hermes JSON schemas.',
      tag: 'Correctness',
    },
    {
      icon: Mic,
      title: 'Native Silero VAD v5 Engine',
      description: 'Pure-Rust voice activity detection with zero ONNX Runtime dependencies. Real-time streaming speech chunk iteration for interactive voice apps.',
      tag: 'Speech',
    },
    {
      icon: HardDrive,
      title: 'Streaming SafeTensors Quantization',
      description: 'Stream remote Hugging Face SafeTensors repositories and quantize on-the-fly directly to GGUF in memory. Zero unquantized disk footprint, resumable HTTP range checkpoints, and automatic local caching.',
      tag: 'Zero-Disk',
    },
  ];

  return (
    <div className="py-20 bg-[#090a0f] border-b border-[#222638]">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight mb-4">
            Engineered for Extreme Efficiency
          </h2>
          <p className="text-slate-400 text-sm sm:text-base">
            Every layer of Cera is designed for low latency, minimal memory footprint, and reliable portability across edge hardware.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feat) => {
            const Icon = feat.icon;
            return (
              <div
                key={feat.title}
                className="group p-6 rounded-xl border border-[#222638] bg-[#0f111a] hover:border-blue-500/40 hover:bg-[#131622] transition-all duration-200"
              >
                <div className="flex items-center justify-between mb-4">
                  <div className="p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 group-hover:scale-110 transition-transform">
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 px-2 py-0.5 rounded bg-[#1a1d2c] border border-[#222638]">
                    {feat.tag}
                  </span>
                </div>
                <h3 className="text-base font-semibold text-white mb-2 group-hover:text-blue-300 transition-colors">
                  {feat.title}
                </h3>
                <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
                  {feat.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
