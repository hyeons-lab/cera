import { useState } from 'react';
import { Copy, Check, Sparkles, ArrowRight, Play, Cpu, ShieldCheck, ExternalLink } from 'lucide-react';

interface HeroProps {
  onOpenDemo?: () => void;
  onOpenDocs: () => void;
}

export const Hero = ({ onOpenDocs }: HeroProps) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const installCommands = [
    { label: 'CLI', command: 'cargo install cera-cli --locked' },
    { label: 'Rust', command: 'cargo add cera' },
    { label: 'Flutter', command: 'flutter pub add cera_ffi_flutter' },
    { label: 'Web / JS', command: 'npm install @hyeons-lab/cera-wasm' },
  ];

  const handleCopy = (command: string, index: number) => {
    navigator.clipboard.writeText(command);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  return (
    <div className="relative overflow-hidden pt-12 pb-20 border-b border-[#222638]">
      {/* Background glow effects */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-blue-600/10 blur-[120px] pointer-events-none rounded-full" />
      <div className="absolute top-1/3 left-1/4 w-[400px] h-[300px] bg-indigo-600/10 blur-[100px] pointer-events-none rounded-full" />

      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-blue-500/30 bg-blue-500/10 text-blue-400 text-xs font-semibold mb-6">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Pure Rust On-Device AI Engine</span>
            <span className="w-1 h-1 rounded-full bg-blue-400" />
            <span className="text-slate-300">Streaming SafeTensors & WebGPU Ready</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-white tracking-tight leading-[1.1] mb-6">
            On-Device AI Inference. <br />
            <span className="bg-gradient-to-r from-blue-400 via-indigo-300 to-sky-400 bg-clip-text text-transparent">
              Zero Runtime. Everywhere.
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed mb-8">
            A high-performance, pure-Rust inference engine. Stream remote Hugging Face SafeTensors directly and quantize on-the-fly to GGUF with zero unquantized disk footprint. Run multimodal models across Apple Metal MSL, Vulkan/wgpu, Android, iOS, or WebGPU in the browser.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <a
              href="https://cera-demo.pages.dev/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-5 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <Play className="w-4 h-4 fill-white" />
              <span>Open Live WebGPU Demo</span>
              <ExternalLink className="w-4 h-4 opacity-75" />
            </a>
            <button
              onClick={onOpenDocs}
              className="flex items-center gap-2 px-5 py-3 rounded-lg border border-[#222638] bg-[#151824] hover:bg-[#1c2030] text-slate-200 font-semibold text-sm transition-all hover:border-slate-600"
            >
              <span>Explore SDK Documentation</span>
              <ArrowRight className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Quick install tabs & Terminal Showcase */}
        <div className="max-w-4xl mx-auto mt-10">
          <div className="rounded-xl border border-[#222638] bg-[#0f111a] shadow-2xl overflow-hidden">
            {/* Terminal Window Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#222638] bg-[#0c0d14]">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500/80" />
                <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                <div className="w-3 h-3 rounded-full bg-green-500/80" />
                <span className="ml-2 text-xs font-mono text-slate-400">cera terminal</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-mono">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>zero-python</span>
              </div>
            </div>

            {/* Install commands tabs */}
            <div className="p-4 bg-[#12141f] border-b border-[#222638]">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Install Package</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                {installCommands.map((item, idx) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between p-2 rounded-lg bg-[#0a0b10] border border-[#222638] hover:border-blue-500/40 transition-colors"
                  >
                    <div className="truncate pr-2">
                      <span className="text-[11px] font-bold text-blue-400 block">{item.label}</span>
                      <code className="text-xs font-mono text-slate-300 truncate block">{item.command}</code>
                    </div>
                    <button
                      onClick={() => handleCopy(item.command, idx)}
                      title="Copy to clipboard"
                      className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors shrink-0"
                    >
                      {copiedIndex === idx ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* Terminal Live Output Simulation */}
            <div className="p-5 font-mono text-xs sm:text-sm text-slate-300 space-y-3 leading-relaxed bg-[#090a0f]">
              <div className="flex items-center gap-2 text-blue-400">
                <span className="text-slate-500">$</span>
                <span className="font-semibold text-slate-100">
                  cera run --bundle-id LFM2.5-1.2B-Instruct --quant Q4_0 --prompt "Explain TurboQuant KV compression"
                </span>
              </div>
              <div className="text-slate-500 text-xs">
                [cera:repo] Bundle cached at ~/.cache/cera/LFM2.5-1.2B-Instruct-Q4_0.gguf (724.8 MB)
                <br />
                [cera:backend] Hardware: Apple M4 Max · Backend: Native Metal MSL (40 cores) · TurboQuant: TQ3 active
              </div>
              <div className="p-3 rounded-lg bg-blue-950/30 border border-blue-500/20 text-slate-300 text-xs">
                <div className="font-semibold text-blue-400 mb-1 flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5" />
                  <span>Thinking Stream:</span>
                </div>
                <div className="italic text-slate-400">
                  Analyzing user query about KV cache memory compression... TurboQuant applies per-layer randomized orthogonal rotations (Hadamard transforms) and Quantized Johnson-Lindenstrauss embeddings to compress keys to ~3 bits and values to ~2 bits without calibration...
                </div>
              </div>
              <div className="text-slate-100 pl-1 border-l-2 border-blue-500">
                TurboQuant achieves extreme KV cache compression (~12x reduction vs FP32) by rotating key-value representations into outlier-free spaces before multi-bit quantization. This allows 32k context windows to run within hundreds of megabytes on consumer hardware with near-lossless perplexity.
              </div>
              <div className="pt-2 text-xs text-slate-400 flex flex-wrap gap-4 border-t border-[#1c2030]">
                <span>Prefill: <strong className="text-emerald-400">642 tok/s</strong></span>
                <span>Decode: <strong className="text-emerald-400">138.4 tok/s</strong></span>
                <span>KV Memory: <strong className="text-blue-400">16 MB</strong> (vs 192 MB f32)</span>
                <span>TTFT: <strong className="text-slate-200">14.2ms</strong></span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
