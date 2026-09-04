import { Cpu, ExternalLink } from 'lucide-react';

const GithubIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 24 24">
    <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
  </svg>
);

interface FooterProps {
  setActiveTab: (tab: string) => void;
}

export const Footer = ({ setActiveTab }: FooterProps) => {
  return (
    <footer className="border-t border-[#222638] bg-[#07080c] py-12 text-xs text-slate-400">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2">
              <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold">
                <Cpu className="w-4 h-4" />
              </div>
              <span className="font-bold text-base text-white tracking-tight">Cera</span>
              <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                v0.5.1
              </span>
            </div>
            <p className="text-slate-400 text-xs leading-relaxed max-w-sm">
              High-performance, pure-Rust on-device inference engine for GGUF and LeapBundles models. Run locally on CPU, Apple Metal, and WebGPU with zero Python or runtime dependencies.
            </p>
            <div className="text-[11px] text-slate-400">
              Open source under Apache 2.0 and MIT licenses.
            </div>
          </div>

          <div>
            <h4 className="font-bold text-slate-200 uppercase tracking-wider text-[11px] mb-3">
              Navigation
            </h4>
            <ul className="space-y-2">
              <li>
                <button
                  onClick={() => setActiveTab('overview')}
                  className="hover:text-white transition-colors"
                >
                  Overview & Architecture
                </button>
              </li>
              <li>
                <button
                  onClick={() => setActiveTab('demo')}
                  className="hover:text-white transition-colors"
                >
                  WebGPU Studio
                </button>
              </li>
              <li>
                <button
                  onClick={() => setActiveTab('docs')}
                  className="hover:text-white transition-colors"
                >
                  SDK Documentation
                </button>
              </li>
              <li>
                <button
                  onClick={() => setActiveTab('catalog')}
                  className="hover:text-white transition-colors"
                >
                  LeapBundles Catalog
                </button>
              </li>
              <li>
                <button
                  onClick={() => setActiveTab('benchmarks')}
                  className="hover:text-white transition-colors"
                >
                  Hardware Benchmarks
                </button>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-bold text-slate-200 uppercase tracking-wider text-[11px] mb-3">
              Community & Code
            </h4>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/hyeons-lab/cera"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors flex items-center gap-1.5"
                >
                  <GithubIcon className="w-3.5 h-3.5" />
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/LiquidAI/LeapBundles"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors flex items-center gap-1.5"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  Hugging Face Hub
                </a>
              </li>
              <li>
                <a
                  href="https://crates.io/crates/cera"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors flex items-center gap-1.5"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  crates.io (cera)
                </a>
              </li>
              <li>
                <a
                  href="https://pub.dev/packages/cera_ffi_flutter"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-white transition-colors flex items-center gap-1.5"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  pub.dev (cera_ffi_flutter)
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-[#1c2030] flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-400 text-[11px]">
          <div>
            &copy; {new Date().getFullYear()} Cera Project Contributors.
          </div>
          <div className="flex items-center gap-1">
            <span>Built with pure Rust and modern WebGPU.</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
