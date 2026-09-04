import { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { FeatureGrid } from './components/FeatureGrid';
import { ArchitectureDiagram } from './components/ArchitectureDiagram';
import { DocsView } from './components/DocsView';
import { ModelCatalogView } from './components/ModelCatalogView';
import { BenchmarkView } from './components/BenchmarkView';
import { Footer } from './components/Footer';
import { Cpu, ExternalLink } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('overview');

  return (
    <div className="min-h-screen flex flex-col bg-[#090a0f] text-slate-100 font-sans selection:bg-blue-500/30 selection:text-blue-200">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1">
        {activeTab === 'overview' && (
          <>
            <Hero onOpenDocs={() => setActiveTab('docs')} />
            <FeatureGrid />
            <ArchitectureDiagram />

            {/* Bottom Call to Action */}
            <div className="py-20 bg-gradient-to-b from-[#090a0f] to-[#0d101a] border-b border-[#222638]">
              <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 text-center">
                <div className="inline-flex items-center justify-center p-3 rounded-2xl bg-blue-500/10 border border-blue-500/20 text-blue-400 mb-6">
                  <Cpu className="w-8 h-8" />
                </div>
                <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight mb-4">
                  Run Neural Models Locally on Any Hardware
                </h2>
                <p className="text-slate-400 text-sm sm:text-base max-w-2xl mx-auto mb-8 leading-relaxed">
                  Start building on-device AI applications with zero runtime dependencies. Stream and quantize SafeTensors on-the-fly, and deploy across desktop, mobile, and WebAssembly with a single portable engine.
                </p>
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <a
                    href="https://cera-demo.pages.dev/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/25 transition-all hover:scale-[1.02]"
                  >
                    <span>Open Live WebGPU Demo</span>
                    <ExternalLink className="w-4 h-4 opacity-75" />
                  </a>
                  <button
                    onClick={() => setActiveTab('docs')}
                    className="px-6 py-3 rounded-lg border border-[#222638] bg-[#151824] hover:bg-[#1c2030] text-slate-200 font-semibold text-sm transition-all"
                  >
                    Read Multiplatform SDK Guides
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {activeTab === 'docs' && <DocsView />}
        {activeTab === 'catalog' && <ModelCatalogView />}
        {activeTab === 'benchmarks' && <BenchmarkView />}
      </main>

      <Footer setActiveTab={setActiveTab} />
    </div>
  );
}

export default App;
