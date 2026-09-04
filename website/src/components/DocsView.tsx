import { useState, useMemo } from 'react';
import { docsCategories } from '../data/docsData';
import type { PlatformId } from '../types/docs';
import { Search, BookOpen, Code, Copy, Check, Sparkles, Cpu, Settings } from 'lucide-react';

export const DocsView: React.FC = () => {
  const [selectedArticleId, setSelectedArticleId] = useState<string>('overview');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedPlatform, setSelectedPlatform] = useState<PlatformId>('rust');
  const [copiedSnippet, setCopiedSnippet] = useState<string | null>(null);

  // Flattened articles list for quick lookups
  const allArticles = useMemo(() => {
    return docsCategories.flatMap((c) => c.articles);
  }, []);

  const activeArticle = useMemo(() => {
    return allArticles.find((a) => a.id === selectedArticleId) || allArticles[0];
  }, [allArticles, selectedArticleId]);

  const filteredCategories = useMemo(() => {
    if (!searchQuery.trim()) return docsCategories;
    const q = searchQuery.toLowerCase();
    return docsCategories
      .map((cat) => ({
        ...cat,
        articles: cat.articles.filter(
          (art) =>
            art.title.toLowerCase().includes(q) ||
            art.description.toLowerCase().includes(q) ||
            art.sections.some((s) => s.title.toLowerCase().includes(q) || s.content.toLowerCase().includes(q))
        ),
      }))
      .filter((cat) => cat.articles.length > 0);
  }, [searchQuery]);

  const handleCopy = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedSnippet(id);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  const platforms: { id: PlatformId; label: string }[] = [
    { id: 'rust', label: 'Rust' },
    { id: 'web', label: 'Web / JS' },
    { id: 'flutter', label: 'Flutter' },
    { id: 'android', label: 'Android' },
    { id: 'apple', label: 'Apple / Swift' },
    { id: 'cli', label: 'CLI' },
  ];

  const getCategoryIcon = (iconName: string) => {
    switch (iconName) {
      case 'Sparkles':
        return <Sparkles className="w-4 h-4 text-blue-400" />;
      case 'Code':
        return <Code className="w-4 h-4 text-indigo-400" />;
      case 'Cpu':
        return <Cpu className="w-4 h-4 text-emerald-400" />;
      case 'Settings':
        return <Settings className="w-4 h-4 text-amber-400" />;
      default:
        return <BookOpen className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="py-10 bg-[#090a0f] min-h-screen">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Docs Sidebar Navigation (3 cols) */}
          <aside className="lg:col-span-3 space-y-6">
            {/* Search Input */}
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search documentation..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-[#0f111a] border border-[#222638] rounded-lg pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:border-blue-500 outline-none"
              />
            </div>

            {/* Categories & Articles List */}
            <nav className="space-y-6">
              {filteredCategories.map((category) => (
                <div key={category.id} className="space-y-2">
                  <div className="flex items-center gap-2 px-2 text-xs font-bold uppercase tracking-wider text-slate-400">
                    {getCategoryIcon(category.iconName)}
                    <span>{category.title}</span>
                  </div>
                  <div className="space-y-0.5">
                    {category.articles.map((article) => {
                      const isActive = article.id === selectedArticleId;
                      return (
                        <button
                          key={article.id}
                          onClick={() => setSelectedArticleId(article.id)}
                          className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center justify-between ${
                            isActive
                              ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30'
                              : 'text-slate-400 hover:text-slate-200 hover:bg-[#141724]'
                          }`}
                        >
                          <span className="truncate pr-2">{article.title}</span>
                          {article.badge && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-[#1c2030] text-slate-400 border border-[#222638]">
                              {article.badge}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </nav>
          </aside>

          {/* Docs Content Main View (9 cols) */}
          <main className="lg:col-span-9 space-y-8">
            {/* Article Header */}
            <div className="pb-6 border-b border-[#222638]">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-semibold px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  {activeArticle.badge || 'SDK Guide'}
                </span>
                <span className="text-xs text-slate-500 font-mono">cera docs · {activeArticle.id}</span>
              </div>
              <h1 className="text-3xl font-extrabold text-white tracking-tight mb-2">
                {activeArticle.title}
              </h1>
              <p className="text-sm text-slate-400">
                {activeArticle.description}
              </p>
            </div>

            {/* Platform Selector */}
            <div className="flex items-center gap-2 p-1.5 bg-[#0f111a] border border-[#222638] rounded-xl overflow-x-auto">
              <span className="text-xs font-semibold text-slate-400 px-3 uppercase tracking-wider shrink-0">
                Language / Target:
              </span>
              <div className="flex gap-1">
                {platforms.map((plat) => (
                  <button
                    key={plat.id}
                    onClick={() => setSelectedPlatform(plat.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors shrink-0 ${
                      selectedPlatform === plat.id
                        ? 'bg-blue-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                    }`}
                  >
                    {plat.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sections */}
            <div className="space-y-10">
              {activeArticle.sections.map((section) => {
                const snippet = section.codeSnippets?.[selectedPlatform] ||
                  section.codeSnippets?.rust ||
                  section.codeSnippets?.cli ||
                  Object.values(section.codeSnippets || {})[0];

                return (
                  <section key={section.id} id={section.id} className="space-y-4">
                    <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
                      <span className="w-1.5 h-5 rounded-full bg-blue-500" />
                      {section.title}
                    </h2>
                    <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">
                      {section.content}
                    </div>

                    {/* Code Snippet Box */}
                    {snippet && (
                      <div className="rounded-xl border border-[#222638] bg-[#0c0d14] overflow-hidden shadow-lg">
                        <div className="flex items-center justify-between px-4 py-2 border-b border-[#222638] bg-[#10121c]">
                          <span className="text-xs font-mono text-slate-400">
                            {snippet.filename || `${snippet.language} snippet`}
                          </span>
                          <button
                            onClick={() => handleCopy(snippet.code, section.id)}
                            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white py-1 px-2 rounded hover:bg-slate-800 transition-colors"
                          >
                            {copiedSnippet === section.id ? (
                              <>
                                <Check className="w-3.5 h-3.5 text-emerald-400" />
                                <span className="text-emerald-400">Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3.5 h-3.5" />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                        </div>
                        <pre className="p-4 text-xs sm:text-sm font-mono text-slate-200 overflow-x-auto leading-relaxed">
                          <code>{snippet.code}</code>
                        </pre>
                      </div>
                    )}
                  </section>
                );
              })}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};
