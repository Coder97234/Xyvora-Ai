import React, { useEffect, useState } from "react";
import {
  BookOpen,
  Download,
  ExternalLink,
  Globe,
  History,
  Printer,
  Search,
  ShieldCheck,
  Sparkles,
  Trash2,
} from "lucide-react";
import { api } from "../api";
import { ResearchReport } from "../types";
import { MarkdownView } from "./MarkdownView";

export const ResearchStudio: React.FC<{ activeProjectId?: string | null }> = ({ activeProjectId }) => {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"quick" | "deep">("quick");
  const [busy, setBusy] = useState(false);
  const [activeReport, setActiveReport] = useState<ResearchReport | null>(null);
  const [history, setHistory] = useState<ResearchReport[]>([]);
  const [providerStatus, setProviderStatus] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadHistoryAndStatus = async () => {
    try {
      const [h, st] = await Promise.all([api.listResearchHistory(), api.getResearchStatus()]);
      setHistory(h);
      setProviderStatus(st);
      if (!activeReport && h[0]) setActiveReport(h[0]);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    loadHistoryAndStatus();
  }, []);

  const runSearch = async (qOverride?: string) => {
    const q = (qOverride ?? query).trim();
    if (!q) return;
    setQuery(q);
    setBusy(true);
    setError(null);
    try {
      const rep = await api.runResearch({
        query: q,
        mode,
        max_results: mode === "deep" ? 7 : 5,
        follow_links: mode === "deep",
        project_id: activeProjectId,
      });
      setActiveReport(rep);
      const h = await api.listResearchHistory();
      setHistory(h);
    } catch (err: any) {
      setError(err.message || "Research request failed");
    } finally {
      setBusy(false);
    }
  };

  const exportMarkdown = () => {
    if (!activeReport) return;
    const body = activeReport.report_markdown || activeReport.summary || "";
    const cites = (activeReport.sources || [])
      .map((s, i) => `${i + 1}. [${s.title}](${s.url}) — ${s.credibility?.label || s.provider} (${s.published_date || "n.d."})`)
      .join("\n");
    const full = `# Xyvora Research Report: ${activeReport.query}\n\n${body}\n\n## Verified Sources\n${cites}\n`;
    const blob = new Blob([full], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `xyvora-research-${activeReport.query.slice(0, 28).replace(/[^a-z0-9]+/gi, "-")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDelete = async (id?: string) => {
    if (!id) return;
    await api.deleteResearchReport(id);
    const h = await api.listResearchHistory();
    setHistory(h);
    if ((activeReport?.id || activeReport?.report_id) === id) {
      setActiveReport(h[0] || null);
    }
  };

  return (
    <div className="h-full grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
      {/* Left History Rail */}
      <div className="lg:col-span-3 border-r border-slate-800/80 bg-[#060a15] p-3.5 overflow-y-auto space-y-3 no-print">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-bold text-white uppercase tracking-wider">Research History</h3>
        </div>
        {providerStatus && (
          <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-[10.5px] text-slate-400">
            <span className="text-cyan-300 font-semibold">Grounded Search Stack:</span> Wikipedia, GitHub Docs, DuckDuckGo
            {(providerStatus as any).tavily_configured ? ", Tavily" : ""}
            {(providerStatus as any).brave_configured ? ", Brave" : ""}
          </div>
        )}
        {history.length === 0 ? (
          <p className="text-xs text-slate-500">Run your first research topic to build an exportable report archive.</p>
        ) : (
          <div className="space-y-1.5">
            {history.map((item) => {
              const id = item.id || item.report_id || item.query;
              const isSel = (activeReport?.id || activeReport?.report_id) === (item.id || item.report_id);
              return (
                <div
                  key={id}
                  onClick={() => setActiveReport(item)}
                  className={`group p-2.5 rounded-xl border text-xs cursor-pointer flex items-center justify-between gap-2 ${
                    isSel
                      ? "bg-cyan-500/15 border-cyan-500/40 text-cyan-100"
                      : "bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-900"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="font-medium truncate">{item.query}</div>
                    <div className="text-[10px] font-mono text-slate-400">
                      {item.mode} • {item.sources?.length || 0} sources
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(item.id || item.report_id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Main Research Workspace */}
      <div className="lg:col-span-9 overflow-y-auto p-4 md:p-6 space-y-5">
        <div className="glass-panel rounded-2xl p-4 space-y-3 no-print">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Globe className="w-5 h-5 text-cyan-400" />
              <div>
                <h2 className="text-sm font-bold text-white">AI Web Research & Source Verification Studio</h2>
                <p className="text-[11px] text-slate-400">
                  Multi-source synthesis, source credibility scoring, publication dates & zero fabricated citations
                </p>
              </div>
            </div>

            <div className="flex gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
              <button
                type="button"
                onClick={() => setMode("quick")}
                className={`px-3 py-1 rounded-lg text-xs font-medium ${
                  mode === "quick" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
                }`}
              >
                Quick Research
              </button>
              <button
                type="button"
                onClick={() => setMode("deep")}
                className={`px-3 py-1 rounded-lg text-xs font-medium ${
                  mode === "deep" ? "bg-indigo-500/25 text-indigo-200 border border-indigo-400/45" : "text-slate-400"
                }`}
              >
                Deep Link-Follow
              </button>
            </div>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              runSearch();
            }}
            className="flex gap-2"
          >
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Research any technical or scientific topic (e.g., Rust vs C++ memory safety in operating systems)"
              className="flex-1 px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white"
            />
            <button
              type="submit"
              disabled={busy}
              className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950 flex items-center gap-1.5 disabled:opacity-50"
            >
              <Search className="w-4 h-4" /> {busy ? "Researching..." : "Research"}
            </button>
          </form>

          <div className="flex flex-wrap gap-1.5">
            {[
              "Transformer architecture self-attention mechanisms",
              "Python asyncio event loop and cooperative multitasking",
              "PostgreSQL MVCC concurrency control vs SQLite WAL",
            ].map((sample) => (
              <button
                key={sample}
                type="button"
                onClick={() => runSearch(sample)}
                className="text-[11px] px-2.5 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/35 text-xs text-rose-200">{error}</div>
        )}

        {activeReport && (
          <div className="space-y-4">
            {/* Report Header + Export Controls */}
            <div className="glass-panel rounded-2xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <div>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                    {activeReport.mode} mode • {activeReport.sources?.length || 0} cited sources
                  </span>
                  <h3 className="text-base font-bold text-white mt-1.5">{activeReport.query}</h3>
                </div>
                <div className="flex items-center gap-2 no-print">
                  <button
                    type="button"
                    onClick={exportMarkdown}
                    className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-cyan-200 flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" /> Export .MD
                  </button>
                  <button
                    type="button"
                    onClick={() => window.print()}
                    className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-indigo-200 flex items-center gap-1.5"
                  >
                    <Printer className="w-3.5 h-3.5" /> Print / PDF
                  </button>
                </div>
              </div>

              <div className="text-xs leading-relaxed">
                <MarkdownView content={activeReport.report_markdown || activeReport.summary || ""} />
              </div>
            </div>

            {/* Cited Sources & Credibility Assessment */}
            {activeReport.sources && activeReport.sources.length > 0 && (
              <div className="glass-panel rounded-2xl p-4 space-y-3">
                <h4 className="text-xs font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4" /> Grounded Source Citations & Credibility
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {activeReport.sources.map((src, idx) => (
                    <a
                      key={idx}
                      href={src.url}
                      target="_blank"
                      rel="noreferrer"
                      className="p-3 rounded-xl bg-slate-900/80 hover:bg-slate-900 border border-slate-800 transition block space-y-1.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold text-cyan-200 truncate">
                          [{idx + 1}] {src.title}
                        </span>
                        <ExternalLink className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-mono">
                        <span
                          className={`px-1.5 py-0.5 rounded flex items-center gap-1 ${
                            src.credibility?.tier === "high"
                              ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                              : "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                          }`}
                        >
                          <ShieldCheck className="w-3 h-3" /> {src.credibility?.label || "Web Source"}
                        </span>
                        <span className="text-slate-400">via {src.provider}</span>
                        {src.published_date && <span className="text-indigo-300">• {src.published_date}</span>}
                      </div>
                      <p className="text-[11px] text-slate-300 line-clamp-3">{src.snippet}</p>
                    </a>
                  ))}
                </div>
              </div>
            )}

            {/* Follow-up Research Questions */}
            {activeReport.follow_up_questions && activeReport.follow_up_questions.length > 0 && (
              <div className="glass-panel rounded-2xl p-4 space-y-2 no-print">
                <div className="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4" /> Suggested Follow-Up Investigations
                </div>
                <div className="flex flex-wrap gap-2">
                  {activeReport.follow_up_questions.map((fq, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => runSearch(fq)}
                      className="px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs text-slate-200 text-left"
                    >
                      {fq}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
