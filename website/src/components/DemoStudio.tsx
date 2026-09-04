import { useState, useEffect, useRef } from 'react';
import { Cpu, Play, Square, Sparkles, Zap, Activity, Clock, CheckCircle, AlertTriangle, ChevronDown, ChevronUp, Image as ImageIcon, Volume2 } from 'lucide-react';

interface PresetPrompt {
  id: string;
  title: string;
  category: string;
  prompt: string;
  thought: string;
  response: string;
}

const PRESET_PROMPTS: PresetPrompt[] = [
  {
    id: 'reasoning',
    title: 'Deep Logical Reasoning',
    category: 'Thinking Model',
    prompt: 'A farmer has 17 sheep and all but 9 die. How many sheep does the farmer have left? Explain your step-by-step reasoning.',
    thought: 'The problem statement asks: "A farmer has 17 sheep and all but 9 die. How many sheep does the farmer have left?"\n\nLet us carefully parse the phrasing "all but 9 die".\n1. Initial count: 17 sheep.\n2. The phrase "all but 9" means that 9 sheep did not die.\n3. Therefore, exactly 9 sheep survived.\n4. A common cognitive pitfall is calculating 17 minus 9 equals 8, which answers how many sheep died, not how many remain.\n5. Confirming conclusion: the farmer is left with exactly 9 living sheep.',
    response: 'The farmer has 9 sheep left.\n\nThe key to this riddle is the phrase "all but 9 die", which directly specifies that 9 sheep survived. 8 sheep died, leaving exactly 9 alive.',
  },
  {
    id: 'turboquant',
    title: 'TurboQuant Explanation',
    category: 'Architecture',
    prompt: 'Explain how TurboQuant compresses the KV cache to 3 bits without needing calibration data.',
    thought: 'Query: Explaining TurboQuant KV cache compression mechanisms.\nKey topics to cover:\n1. Attention memory bottleneck in long contexts.\n2. PolarQuant: randomized orthogonal rotations via Walsh-Hadamard transforms to disperse outlier channels.\n3. Quantized Johnson-Lindenstrauss (QJL) embeddings to preserve attention inner products.\n4. Zero-calibration property: rotations are data-independent, so no training or fine-tuning set is required.\n5. Compression ratio: ~12x reduction compared to standard FP32 KV caches.',
    response: 'TurboQuant compresses large language model KV caches down to ~3 bits per key and ~2 bits per value without requiring calibration data or fine-tuning.\n\nIt accomplishes this through two primary mathematical techniques:\n\n1. PolarQuant Transformations: In standard transformers, attention activations contain severe outlier dimensions that ruin uniform quantization. PolarQuant applies randomized orthogonal rotations (via fast Walsh-Hadamard transforms) across key vectors, spreading activation energy evenly without modifying attention dot products.\n\n2. Quantized Johnson-Lindenstrauss (QJL) Embeddings: By leveraging random projections, QJL guarantees that inner products between queries and keys are preserved with minimal distortion, keeping perplexity within +/-5% of full-precision FP32.\n\nBecause the orthogonal rotations are data-independent, TurboQuant requires zero calibration datasets and operates on any supported architecture immediately upon loading.',
  },
  {
    id: 'coding',
    title: 'Rust Concurrency',
    category: 'Systems Code',
    prompt: 'Write a thread-safe, lock-free ring buffer in Rust using atomic operations.',
    thought: 'Analyzing user request: Thread-safe ring buffer using atomic operations.\n- Need head and tail atomic usize indices.\n- Buffer capacity should ideally be a power of two for efficient bitwise masking.\n- Memory ordering: Acquire on reads, Release on writes.\n- Return Option<T> on push/pop for queue full/empty conditions.',
    response: `use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;

pub struct LockFreeRingBuffer<T, const N: usize> {
    buffer: [UnsafeCell<Option<T>>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

unsafe impl<T: Send, const N: usize> Sync for LockFreeRingBuffer<T, N> {}

impl<T, const N: usize> LockFreeRingBuffer<T, N> {
    pub fn new() -> Self {
        assert!(N.is_power_of_two(), "Capacity must be a power of two");
        Self {
            buffer: std::array::from_fn(|_| UnsafeCell::new(None)),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail - head >= N {
            return Err(item); // Buffer is full
        }
        unsafe {
            *self.buffer[tail & (N - 1)].get() = Some(item);
        }
        self.tail.store(tail + 1, Ordering::Release);
        Ok(())
    }

    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None; // Buffer is empty
        }
        let item = unsafe { (*self.buffer[head & (N - 1)].get()).take() };
        self.head.store(head + 1, Ordering::Release);
        item
    }
}`,
  },
  {
    id: 'tool_calling',
    title: 'Format-Aware Tool Call',
    category: 'Tool Calling',
    prompt: 'What is the weather in Tokyo right now? Call the get_weather tool.',
    thought: 'The user is requesting the weather in Tokyo and instructed to call the get_weather tool.\nAvailable tools: get_weather(city: String).\nModel architecture: Liquid LFM2.5 (uses Pythonic tool call syntax).\nEmitting Pythonic call: [get_weather(city="Tokyo")]',
    response: '[get_weather(city="Tokyo")]\n\nI have submitted a tool call to query real-time weather conditions for Tokyo.',
  },
];

export const DemoStudio = () => {
  const [hasWebGpu, setHasWebGpu] = useState<boolean | null>(null);
  const [gpuAdapterName, setGpuAdapterName] = useState<string>('');
  const [webGpuReason, setWebGpuReason] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState('LFM2.5-1.2B-Instruct');
  const [kvCompression, setKvCompression] = useState<'tq3' | 'none'>('tq3');
  const [prompt, setPrompt] = useState(PRESET_PROMPTS[0].prompt);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(256);
  const [isGenerating, setIsGenerating] = useState(false);
  const [thoughtOutput, setThoughtOutput] = useState('');
  const [textOutput, setTextOutput] = useState('');
  const [isThoughtExpanded, setIsThoughtExpanded] = useState(true);

  // Telemetry metrics
  const [stats, setStats] = useState({
    ttftMs: 0,
    tokensGenerated: 0,
    tokPerSec: 0,
    totalDurationSec: 0,
    kvMemoryMb: 16.4,
  });

  const abortControllerRef = useRef<boolean>(false);

  // Check WebGPU availability and diagnostic reason on mount
  useEffect(() => {
    async function checkWebGpu() {
      if (typeof window !== 'undefined' && !window.isSecureContext) {
        setHasWebGpu(false);
        setWebGpuReason(
          'Insecure HTTP context detected. WebGPU is disabled by browsers on non-HTTPS origins. When connecting over local Wi-Fi (http://...), please switch to the HTTPS Cloudflare Tunnel URL or http://localhost.'
        );
        return;
      }
      if (typeof navigator === 'undefined') {
        setHasWebGpu(false);
        setWebGpuReason('Navigator is undefined in this environment.');
        return;
      }
      if (!('gpu' in navigator)) {
        setHasWebGpu(false);
        const ua = typeof window !== 'undefined' && window.navigator ? window.navigator.userAgent : '';
        const isIOS = /iPad|iPhone|iPod/.test(ua);
        const isFirefox = /Firefox/.test(ua);
        if (isIOS) {
          setWebGpuReason(
            'iOS Safari requires enabling the WebGPU feature flag in Settings > Apps > Safari > Advanced > Feature Flags > WebGPU.'
          );
        } else if (isFirefox) {
          setWebGpuReason(
            'Firefox requires setting dom.webgpu.enabled to true in about:config.'
          );
        } else {
          setWebGpuReason(
            'WebGPU is not exposed by this browser. Use desktop Google Chrome 113+, Microsoft Edge 113+, or Safari 18+.'
          );
        }
        return;
      }

      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter) {
          setHasWebGpu(true);
          let name = 'WebGPU Device';
          if (adapter.info) {
            name = adapter.info.description || adapter.info.device || adapter.info.architecture || 'WebGPU Device';
          } else if (typeof adapter.requestAdapterInfo === 'function') {
            const info = await adapter.requestAdapterInfo();
            name = info.description || info.device || 'WebGPU Device';
          }
          setGpuAdapterName(name);
          return;
        } else {
          setHasWebGpu(false);
          setWebGpuReason(
            'WebGPU adapter request returned null. Hardware acceleration may be disabled in your browser settings.'
          );
        }
      } catch (err: any) {
        setHasWebGpu(false);
        setWebGpuReason(err?.message || 'Failed to initialize WebGPU adapter.');
      }
    }
    checkWebGpu();
  }, []);

  const handleSelectPreset = (preset: PresetPrompt) => {
    if (isGenerating) return;
    setPrompt(preset.prompt);
    setThoughtOutput('');
    setTextOutput('');
  };

  const handleStop = () => {
    abortControllerRef.current = true;
    setIsGenerating(false);
  };

  const handleGenerate = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    abortControllerRef.current = false;
    setThoughtOutput('');
    setTextOutput('');

    // Match with a preset or synthesize response
    const matched = PRESET_PROMPTS.find((p) => p.prompt === prompt) || {
      thought: `Analyzing prompt: "${prompt}"\nEvaluating reasoning parameters and structuring output...\nExtracting thinking process via StreamingThinkingParser delimiters...`,
      response: `[Cera Interactive Preview]\n\nPrompt: "${prompt}"\n\nThis preview demonstrates client-side thought extraction and streaming telemetry with ${kvCompression === 'tq3' ? 'TurboQuant 3-bit' : 'uncompressed FP32'} KV cache metrics.\n\nTo run actual GGUF weight files through WebGPU WGSL compute shaders, load cera-wasm/examples/webgpu/index.html with any supported GGUF model.`,
    };

    const startTime = performance.now();
    let firstTokenTime = 0;
    let tokensEmitted = 0;

    // Stream thoughts first
    const thoughtWords = matched.thought.split(' ');
    for (let i = 0; i < thoughtWords.length; i++) {
      if (abortControllerRef.current) break;
      await new Promise((r) => setTimeout(r, 18));
      setThoughtOutput((prev) => (prev ? prev + ' ' + thoughtWords[i] : thoughtWords[i]));
    }

    // Stream text response
    const textChars = matched.response.split('');
    let currentText = '';
    const chunkSize = 3;

    for (let i = 0; i < textChars.length; i += chunkSize) {
      if (abortControllerRef.current) break;
      const chunk = textChars.slice(i, i + chunkSize).join('');
      currentText += chunk;
      tokensEmitted += 1;

      if (!firstTokenTime) {
        firstTokenTime = performance.now();
      }

      await new Promise((r) => setTimeout(r, 22));
      setTextOutput(currentText);

      const elapsedSec = (performance.now() - startTime) / 1000;
      setStats({
        ttftMs: Math.round(firstTokenTime - startTime),
        tokensGenerated: tokensEmitted,
        tokPerSec: elapsedSec > 0 ? parseFloat((tokensEmitted / elapsedSec).toFixed(1)) : 0,
        totalDurationSec: parseFloat(elapsedSec.toFixed(2)),
        kvMemoryMb: kvCompression === 'tq3' ? 16.4 : 192.0,
      });
    }

    setIsGenerating(false);
  };

  return (
    <div className="py-12 bg-[#090a0f] min-h-screen">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Studio Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-8 border-b border-[#222638] mb-8">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                Cera WebGPU Studio
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 font-semibold border border-blue-500/30">
                Interactive Preview
              </span>
            </div>
            <p className="text-slate-400 text-xs sm:text-sm">
              Explore streaming thought extraction, TurboQuant KV cache metrics, and multi-turn prompt templates.
            </p>
          </div>

          {/* WebGPU Status Pill */}
          <div className="flex items-center gap-2 self-start md:self-auto px-3 py-1.5 rounded-lg border border-[#222638] bg-[#0f111a] text-xs font-mono">
            {hasWebGpu ? (
              <>
                <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                <span className="text-slate-300">
                  WebGPU Ready {gpuAdapterName ? `: ${gpuAdapterName}` : ''}
                </span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                <span className="text-slate-300">WebGPU Inactive</span>
              </>
            )}
          </div>
        </div>

        {/* Diagnostic Banner if WebGPU is unavailable */}
        {!hasWebGpu && webGpuReason && (
          <div className="mb-6 p-4 rounded-xl border border-amber-500/30 bg-amber-950/20 text-amber-200 text-xs flex items-start gap-3">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-amber-300 mb-1">WebGPU Hardware & Browser Diagnostic</div>
              <p className="text-slate-300 leading-relaxed">{webGpuReason}</p>
            </div>
          </div>
        )}


        {/* Main Grid: Settings & Execution */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Model Knobs & Presets (4 cols) */}
          <div className="lg:col-span-4 space-y-5">
            {/* Model & KV Selection */}
            <div className="p-4 rounded-xl border border-[#222638] bg-[#0f111a] space-y-4">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-blue-400" />
                Model & Hardware Configuration
              </div>

              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1.5">Model Family</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={isGenerating}
                  className="w-full bg-[#151824] border border-[#222638] rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 outline-none"
                >
                  <option value="LFM2.5-1.2B-Instruct">Liquid LFM2.5 1.2B Instruct (Q4_0)</option>
                  <option value="LFM2.5-VL-450M">Liquid LFM2.5 VL 450M (Multimodal Vision)</option>
                  <option value="LFM2-Audio-1.5B">Liquid LFM2 Audio 1.5B (Speech Synthesis)</option>
                  <option value="DeepSeek-R1-Distill-Qwen">DeepSeek R1 Distill Qwen 1.5B (Reasoning)</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-300 block mb-1.5">KV Cache Compression</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setKvCompression('tq3')}
                    className={`p-2 rounded-lg text-xs font-medium border text-left transition-colors ${
                      kvCompression === 'tq3'
                        ? 'bg-blue-600/20 border-blue-500/50 text-blue-300'
                        : 'bg-[#151824] border-[#222638] text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className="font-bold block text-white">TurboQuant TQ3</span>
                    <span className="text-[10px] text-emerald-400 font-mono">16 MB (~12x saved)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setKvCompression('none')}
                    className={`p-2 rounded-lg text-xs font-medium border text-left transition-colors ${
                      kvCompression === 'none'
                        ? 'bg-blue-600/20 border-blue-500/50 text-blue-300'
                        : 'bg-[#151824] border-[#222638] text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className="font-bold block text-white">Uncompressed FP32</span>
                    <span className="text-[10px] text-slate-400 font-mono">192 MB standard</span>
                  </button>
                </div>
              </div>

              {/* Sliders */}
              <div className="space-y-3 pt-2 border-t border-[#1c2030]">
                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Max Tokens</span>
                    <span className="font-mono text-blue-400">{maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    min="32"
                    max="1024"
                    step="32"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                    disabled={isGenerating}
                    className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs text-slate-300 mb-1">
                    <span>Temperature</span>
                    <span className="font-mono text-blue-400">{temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    disabled={isGenerating}
                    className="w-full accent-blue-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* Presets */}
            <div className="p-4 rounded-xl border border-[#222638] bg-[#0f111a] space-y-3">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                Prompt Presets
              </div>
              <div className="space-y-1.5">
                {PRESET_PROMPTS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handleSelectPreset(preset)}
                    disabled={isGenerating}
                    className="w-full text-left p-2.5 rounded-lg bg-[#141724] border border-[#222638] hover:border-blue-500/40 hover:bg-[#181c2c] transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-200">{preset.title}</span>
                      <span className="text-[10px] text-blue-400 font-mono">{preset.category}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Prompt Input, Streaming Telemetry, and Output (8 cols) */}
          <div className="lg:col-span-8 space-y-5">
            {/* Prompt Input Box */}
            <div className="p-4 rounded-xl border border-[#222638] bg-[#0f111a] space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold uppercase tracking-wider text-slate-400">User Prompt</label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    title="Vision Multimodal Input"
                    className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white transition-colors text-xs flex items-center gap-1"
                  >
                    <ImageIcon className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Image</span>
                  </button>
                  <button
                    type="button"
                    title="Audio Speech Input"
                    className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 hover:text-white transition-colors text-xs flex items-center gap-1"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">Voice</span>
                  </button>
                </div>
              </div>

              <textarea
                rows={4}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isGenerating}
                placeholder="Enter prompt or select a preset..."
                className="w-full bg-[#141724] border border-[#222638] rounded-lg p-3 text-xs sm:text-sm text-white placeholder-slate-500 focus:border-blue-500 outline-none resize-none font-sans leading-relaxed"
              />

              <div className="flex items-center justify-between pt-1">
                <div className="text-[11px] text-slate-500 font-mono">
                  Interactive UI & Telemetry Preview
                </div>
                <div className="flex gap-2">
                  {isGenerating ? (
                    <button
                      onClick={handleStop}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white font-semibold text-xs transition-colors shadow-lg shadow-red-600/20"
                    >
                      <Square className="w-3.5 h-3.5 fill-white" />
                      Stop
                    </button>
                  ) : (
                    <button
                      onClick={handleGenerate}
                      className="flex items-center gap-1.5 px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition-colors shadow-lg shadow-blue-600/20"
                    >
                      <Play className="w-3.5 h-3.5 fill-white" />
                      Run Preview
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Live Telemetry Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 rounded-xl border border-[#222638] bg-[#0c0e15] font-mono text-xs">
              <div className="p-2 rounded-lg bg-[#12141f] border border-[#1f2334]">
                <div className="text-slate-400 text-[10px] uppercase flex items-center gap-1 mb-0.5">
                  <Activity className="w-3 h-3 text-emerald-400" />
                  Decode Speed
                </div>
                <div className="text-emerald-400 font-bold text-sm">
                  {stats.tokPerSec > 0 ? `${stats.tokPerSec} tok/s` : '-'}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-[#12141f] border border-[#1f2334]">
                <div className="text-slate-400 text-[10px] uppercase flex items-center gap-1 mb-0.5">
                  <Clock className="w-3 h-3 text-blue-400" />
                  First Token (TTFT)
                </div>
                <div className="text-slate-200 font-bold text-sm">
                  {stats.ttftMs > 0 ? `${stats.ttftMs} ms` : '-'}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-[#12141f] border border-[#1f2334]">
                <div className="text-slate-400 text-[10px] uppercase flex items-center gap-1 mb-0.5">
                  <Zap className="w-3 h-3 text-amber-400" />
                  Tokens Generated
                </div>
                <div className="text-slate-200 font-bold text-sm">
                  {stats.tokensGenerated}
                </div>
              </div>
              <div className="p-2 rounded-lg bg-[#12141f] border border-[#1f2334]">
                <div className="text-slate-400 text-[10px] uppercase flex items-center gap-1 mb-0.5">
                  <Cpu className="w-3 h-3 text-indigo-400" />
                  KV Footprint
                </div>
                <div className="text-blue-400 font-bold text-sm">
                  {stats.kvMemoryMb} MB
                </div>
              </div>
            </div>

            {/* Reasoning / Thought Stream Box */}
            {thoughtOutput && (
              <div className="rounded-xl border border-blue-500/30 bg-[#0d101a] overflow-hidden">
                <button
                  type="button"
                  onClick={() => setIsThoughtExpanded(!isThoughtExpanded)}
                  className="w-full flex items-center justify-between px-4 py-2.5 bg-blue-950/25 border-b border-blue-500/20 text-xs font-semibold text-blue-400 hover:bg-blue-950/40 transition-colors"
                >
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-blue-400" />
                    <span>Thinking & Reasoning Stream</span>
                    {isGenerating && !textOutput && (
                      <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                    )}
                  </span>
                  {isThoughtExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {isThoughtExpanded && (
                  <div className="p-4 text-xs font-mono text-slate-300 italic whitespace-pre-wrap leading-relaxed max-h-56 overflow-y-auto bg-[#0a0c14]">
                    {thoughtOutput}
                  </div>
                )}
              </div>
            )}

            {/* Generated Response Box */}
            <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a] min-h-[14rem]">
              <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
                <span>Model Output (Preview)</span>
                {isGenerating && (
                  <span className="text-[11px] text-emerald-400 font-mono animate-pulse">
                    Streaming tokens...
                  </span>
                )}
              </div>
              <div className="font-mono text-xs sm:text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
                {textOutput ? (
                  <>
                    {textOutput}
                    {isGenerating && <span className="inline-block w-1.5 h-4 ml-0.5 bg-blue-400 animate-pulse align-middle" />}
                  </>
                ) : (
                  <span className="text-slate-500 italic">
                    Output will stream here in real time to demonstrate token generation and thinking parser extraction.
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
