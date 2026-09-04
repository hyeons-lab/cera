import { Cpu, Terminal, BookOpen, Layers, Activity, ExternalLink } from 'lucide-react';

const GithubIcon = ({ className }: { className?: string }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 24 24">
    <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
  </svg>
);

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar = ({ activeTab, setActiveTab }: NavbarProps) => {
  return (
    <header className="sticky top-0 z-50 border-b border-[#222638] bg-[#090a0f]/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8 h-16">
        <div className="flex items-center gap-6">
          <button
            onClick={() => setActiveTab('overview')}
            className="flex items-center gap-2.5 text-left group focus:outline-none"
          >
            <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg text-white tracking-tight">Cera</span>
                <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  v0.5.1
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono hidden sm:block">Rust On-Device Inference</p>
            </div>
          </button>

          <nav className="hidden md:flex items-center gap-1 pl-4 border-l border-[#222638]">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'overview'
                  ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/40'
              }`}
            >
              <Terminal className="w-4 h-4" />
              Overview
            </button>
            <a
              href="https://cera-demo.pages.dev/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 rounded-md text-sm font-medium text-blue-400 hover:text-blue-300 hover:bg-blue-600/10 transition-colors flex items-center gap-1.5"
            >
              <Cpu className="w-4 h-4" />
              <span>Live Demo</span>
              <ExternalLink className="w-3 h-3 opacity-70" />
            </a>
            <button
              onClick={() => setActiveTab('docs')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'docs'
                  ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/40'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              Docs
            </button>
            <button
              onClick={() => setActiveTab('catalog')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'catalog'
                  ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/40'
              }`}
            >
              <Layers className="w-4 h-4" />
              Models
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-1.5 ${
                activeTab === 'benchmarks'
                  ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/40'
              }`}
            >
              <Activity className="w-4 h-4" />
              Benchmarks
            </button>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <a
            href="https://huggingface.co/LiquidAI/LeapBundles"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-1.5 text-xs text-slate-300 hover:text-white px-2.5 py-1.5 rounded-md border border-[#222638] bg-[#0f111a] hover:border-slate-600 transition-colors"
          >
            <span>LeapBundles</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </a>
          <a
            href="https://github.com/hyeons-lab/cera"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs font-medium text-white px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 shadow-md shadow-blue-600/20 transition-colors"
          >
            <GithubIcon className="w-4 h-4" />
            <span>GitHub</span>
          </a>
        </div>
      </div>

      {/* Mobile nav bar */}
      <div className="flex md:hidden border-t border-[#222638] px-2 py-1.5 bg-[#0f111a] overflow-x-auto gap-1">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-2.5 py-1 rounded text-xs font-medium shrink-0 ${
            activeTab === 'overview' ? 'bg-blue-600 text-white' : 'text-slate-400'
          }`}
        >
          Overview
        </button>
        <a
          href="https://cera-demo.pages.dev/"
          target="_blank"
          rel="noopener noreferrer"
          className="px-2.5 py-1 rounded text-xs font-medium shrink-0 text-blue-400 bg-blue-600/10 flex items-center gap-1"
        >
          <span>Live Demo</span>
          <ExternalLink className="w-3 h-3" />
        </a>
        <button
          onClick={() => setActiveTab('docs')}
          className={`px-2.5 py-1 rounded text-xs font-medium shrink-0 ${
            activeTab === 'docs' ? 'bg-blue-600 text-white' : 'text-slate-400'
          }`}
        >
          Docs
        </button>
        <button
          onClick={() => setActiveTab('catalog')}
          className={`px-2.5 py-1 rounded text-xs font-medium shrink-0 ${
            activeTab === 'catalog' ? 'bg-blue-600 text-white' : 'text-slate-400'
          }`}
        >
          Models
        </button>
        <button
          onClick={() => setActiveTab('benchmarks')}
          className={`px-2.5 py-1 rounded text-xs font-medium shrink-0 ${
            activeTab === 'benchmarks' ? 'bg-blue-600 text-white' : 'text-slate-400'
          }`}
        >
          Benchmarks
        </button>
      </div>
    </header>
  );
};
