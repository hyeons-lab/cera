import { useState } from 'react';
import { modelCatalog } from '../data/modelCatalog';
import { ExternalLink, Copy, Check, Eye, Mic, MessageSquare } from 'lucide-react';

export const ModelCatalogView = () => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [filterModality, setFilterModality] = useState<string>('all');

  const modalities = ['all', 'text', 'vision', 'audio'];

  const filteredModels = modelCatalog.filter((m) => {
    if (filterModality === 'all') return true;
    return m.modalities.includes(filterModality);
  });

  const handleCopyCli = (bundleId: string) => {
    const cmd = `cera run --bundle-id ${bundleId} --quant Q4_0`;
    navigator.clipboard.writeText(cmd);
    setCopiedId(bundleId);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="py-12 bg-[#090a0f] min-h-screen">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mb-10">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
              LeapBundles Catalog
            </span>
            <span className="text-xs text-slate-400 font-mono">cera list-bundles</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight mb-3">
            Supported Models & LeapBundles
          </h1>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            All models run natively in Cera with automatic Hugging Face downloading, manifest verification, and persistent local caching. Remote SafeTensors repositories can also be streamed and quantized on-the-fly directly to GGUF in memory with zero unquantized disk footprint.
          </p>
        </div>

        {/* Modality Filter */}
        <div className="flex items-center gap-2 p-1.5 bg-[#0f111a] border border-[#222638] rounded-xl mb-8 overflow-x-auto w-fit">
          <span className="text-xs font-semibold text-slate-400 px-3 uppercase tracking-wider">
            Modality:
          </span>
          {modalities.map((mod) => (
            <button
              key={mod}
              onClick={() => setFilterModality(mod)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold capitalize transition-colors ${
                filterModality === mod
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {mod}
            </button>
          ))}
        </div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels.map((model) => (
            <div
              key={model.id}
              className="flex flex-col justify-between p-6 rounded-xl border border-[#222638] bg-[#0f111a] hover:border-blue-500/40 transition-all duration-200"
            >
              <div>
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div>
                    <h3 className="text-base font-bold text-white mb-1">
                      {model.name}
                    </h3>
                    <div className="text-[11px] font-mono text-blue-400">
                      ID: {model.id}
                    </div>
                  </div>
                  {model.featured && (
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30 shrink-0">
                      Featured
                    </span>
                  )}
                </div>

                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  {model.description}
                </p>

                {/* Modality Badges */}
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {model.modalities.map((mod) => (
                    <span
                      key={mod}
                      className="inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded bg-[#151824] border border-[#222638] text-slate-300 capitalize"
                    >
                      {mod === 'vision' && <Eye className="w-3 h-3 text-indigo-400" />}
                      {mod === 'audio' && <Mic className="w-3 h-3 text-amber-400" />}
                      {mod === 'text' && <MessageSquare className="w-3 h-3 text-blue-400" />}
                      {mod}
                    </span>
                  ))}
                  <span className="text-[11px] px-2 py-0.5 rounded bg-[#151824] border border-[#222638] text-slate-300 font-mono">
                    {model.parameters}
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-[#151824] border border-[#222638] text-slate-300 font-mono">
                    {model.contextLength} ctx
                  </span>
                </div>

                {/* Quants */}
                <div className="mb-4">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Quantization Formats:
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {model.quants.map((q) => (
                      <span
                        key={q}
                        className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#12141f] border border-[#1f2334] text-slate-400"
                      >
                        {q}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-4 border-t border-[#1c2030] space-y-2">
                <button
                  onClick={() => handleCopyCli(model.id)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-[#141724] border border-[#222638] hover:border-blue-500/40 text-slate-300 hover:text-white transition-colors text-xs font-mono"
                >
                  <span className="truncate pr-2">cera run --bundle-id {model.id}</span>
                  {copiedId === model.id ? (
                    <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  ) : (
                    <Copy className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  )}
                </button>
                <a
                  href={model.huggingFaceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-[#0c0d14] border border-[#222638] hover:border-slate-600 text-slate-400 hover:text-slate-200 transition-colors text-xs"
                >
                  <span>View on Hugging Face</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
