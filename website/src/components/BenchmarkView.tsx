import { useState } from 'react';
import { benchmarksData } from '../data/benchmarksData';
import { Activity, Zap, HardDrive, Filter } from 'lucide-react';

export const BenchmarkView = () => {
  const [filterBackend, setFilterBackend] = useState<string>('all');

  const backends = ['all', 'Metal', 'WebGPU', 'CPU', 'wgpu'];

  const filteredData = benchmarksData.filter((b) => {
    if (filterBackend === 'all') return true;
    return b.backend.toLowerCase().includes(filterBackend.toLowerCase());
  });

  return (
    <div className="py-12 bg-[#090a0f] min-h-screen">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mb-10">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              Hardware Matrix
            </span>
            <span className="text-xs text-slate-400 font-mono">cera bench</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight mb-3">
            Inference Performance Benchmarks
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            Real throughput (tokens/sec), latency to first token (TTFT), and memory footprint measured across desktop workstations, mobile phones, and browser WebGPU runtimes.
          </p>
        </div>

        {/* Filter Bar */}
        <div className="flex items-center justify-between gap-4 p-2 bg-[#0f111a] border border-[#222638] rounded-xl mb-6 overflow-x-auto">
          <div className="flex items-center gap-1">
            <Filter className="w-3.5 h-3.5 text-slate-400 ml-2 mr-1" />
            <span className="text-xs font-semibold text-slate-400 mr-2">Compute Backend:</span>
            {backends.map((b) => (
              <button
                key={b}
                onClick={() => setFilterBackend(b)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                  filterBackend === b
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                {b === 'all' ? 'All Backends' : b}
              </button>
            ))}
          </div>
          <div className="text-xs text-slate-400 font-mono pr-2 hidden sm:block">
            Target Model: Liquid LFM2.5 1.2B (Q4_0)
          </div>
        </div>

        {/* Table View */}
        <div className="rounded-xl border border-[#222638] bg-[#0f111a] overflow-hidden shadow-xl mb-12">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#222638] bg-[#12141f] text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-3 px-4">Device & Hardware</th>
                  <th className="py-3 px-4">Chip</th>
                  <th className="py-3 px-4">Backend</th>
                  <th className="py-3 px-4 text-right">Prefill (tok/s)</th>
                  <th className="py-3 px-4 text-right">Decode (tok/s)</th>
                  <th className="py-3 px-4 text-right">TTFT (ms)</th>
                  <th className="py-3 px-4 text-right">RAM (MB)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1c2030] font-mono text-slate-300">
                {filteredData.map((item, idx) => (
                  <tr key={idx} className="hover:bg-[#141724] transition-colors">
                    <td className="py-3 px-4 font-sans font-medium text-white">
                      {item.device}
                    </td>
                    <td className="py-3 px-4 text-slate-300">{item.chip}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded text-[11px] bg-[#1a1d2c] border border-[#222638] text-blue-400">
                        {item.backend}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-slate-200">
                      {item.prefillTokPerSec.toFixed(1)}
                    </td>
                    <td className="py-3 px-4 text-right font-bold text-emerald-400">
                      {item.decodeTokPerSec.toFixed(1)}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-300">
                      {item.ttftMs.toFixed(1)}
                    </td>
                    <td className="py-3 px-4 text-right text-blue-400">
                      {item.memoryMb}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Visual Highlights Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a]">
            <div className="flex items-center gap-2 mb-2 text-blue-400 font-semibold text-sm">
              <Zap className="w-4 h-4" />
              Dynamic Decode Thread Sizing
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Cera dynamically sizes worker thread pools based on model bandwidth thresholds (bytes per dispatch) rather than flat processor caps, yielding an average +19% speedup across models.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a]">
            <div className="flex items-center gap-2 mb-2 text-emerald-400 font-semibold text-sm">
              <HardDrive className="w-4 h-4" />
              TurboQuant KV Compression
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Reduces 4K context KV cache memory from 192 MB down to 16.4 MB with less than 3% impact on generation latency, unlocking long-context windows in browser WebGPU runtimes.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-[#222638] bg-[#0f111a]">
            <div className="flex items-center gap-2 mb-2 text-amber-400 font-semibold text-sm">
              <Activity className="w-4 h-4" />
              Argmax on GPU
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Both Metal MSL and WebGPU WGSL pipelines execute greedy sampling directly on the GPU without reading back 32k+ logits to the host CPU, eliminating PCIe bus stalls.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
