import React, { useEffect, useRef, useState } from "react";
import {
  Database,
  History,
  Pin,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
} from "lucide-react";
import { api } from "../api";
import { GraphEdge, GraphNode, MemoryGraphData, SemanticMemory } from "../types";

interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

interface MemoryPalaceProps {
  onRefreshVitals: () => void;
}

export const MemoryPalace: React.FC<MemoryPalaceProps> = ({ onRefreshVitals }) => {
  const [graphData, setGraphData] = useState<MemoryGraphData | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [probeQuery, setProbeQuery] = useState("");
  const [probeResults, setProbeResults] = useState<{
    semantic: SemanticMemory[];
    episodic: { id: string; user_text: string; ai_summary: string; similarity?: number }[];
  } | null>(null);

  // Manual fact builder state
  const [subject, setSubject] = useState("User");
  const [relation, setRelation] = useState("loves");
  const [objectVal, setObjectVal] = useState("");
  const [category, setCategory] = useState("preference");

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const simNodesRef = useRef<SimNode[]>([]);
  const dragNodeRef = useRef<SimNode | null>(null);

  const loadGraph = async () => {
    const data = await api.getMemoryGraph();
    setGraphData(data);
    onRefreshVitals();
  };

  useEffect(() => {
    loadGraph().catch(console.error);
  }, []);

  // Initialize & run force-directed physics canvas
  useEffect(() => {
    if (!graphData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.clientWidth || 680;
    const height = 340;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Preserve positions of existing nodes if already simulated
    const prevMap = new Map(simNodesRef.current.map((n) => [n.id, n]));
    const simNodes: SimNode[] = graphData.nodes.map((n, idx) => {
      const existing = prevMap.get(n.id);
      if (existing) {
        return { ...n, x: existing.x, y: existing.y, vx: 0, vy: 0 };
      }
      const angle = (idx / Math.max(1, graphData.nodes.length)) * Math.PI * 2;
      const rad = n.type === "user" || n.type === "ai" ? 55 : 115 + (idx % 3) * 24;
      return {
        ...n,
        x: width / 2 + Math.cos(angle) * rad,
        y: height / 2 + Math.sin(angle) * rad,
        vx: 0,
        vy: 0,
      };
    });
    simNodesRef.current = simNodes;

    const edges: GraphEdge[] = graphData.edges;
    let animId = 0;

    const colorForNode = (n: SimNode) => {
      if (n.type === "user") return "#22d3ee";
      if (n.type === "ai") return "#818cf8";
      if (n.type === "document") return "#c084fc";
      if (n.category === "preference") return "#34d399";
      if (n.category === "identity") return "#38bdf8";
      if (n.category === "knowledge" || n.category === "project") return "#fbbf24";
      return "#94a3b8";
    };

    const tick = () => {
      // Repulsion
      for (let i = 0; i < simNodes.length; i++) {
        for (let j = i + 1; j < simNodes.length; j++) {
          const a = simNodes[i];
          const b = simNodes[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          const distSq = Math.max(60, dx * dx + dy * dy);
          const dist = Math.sqrt(distSq);
          const force = 2600 / distSq;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          if (dragNodeRef.current !== a) {
            a.vx -= fx;
            a.vy -= fy;
          }
          if (dragNodeRef.current !== b) {
            b.vx += fx;
            b.vy += fy;
          }
        }
      }

      // Spring attraction along edges
      const nodeById = new Map(simNodes.map((n) => [n.id, n]));
      for (const e of edges) {
        const s = nodeById.get(e.source);
        const t = nodeById.get(e.target);
        if (!s || !t) continue;
        const dx = t.x - s.x;
        const dy = t.y - s.y;
        const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
        const targetLen = 108;
        const k = 0.028;
        const force = (dist - targetLen) * k;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        if (dragNodeRef.current !== s) {
          s.vx += fx;
          s.vy += fy;
        }
        if (dragNodeRef.current !== t) {
          t.vx -= fx;
          t.vy -= fy;
        }
      }

      // Gentle gravity toward center + damping + bounds
      for (const n of simNodes) {
        if (dragNodeRef.current === n) continue;
        n.vx += (width / 2 - n.x) * 0.004;
        n.vy += (height / 2 - n.y) * 0.004;
        n.vx *= 0.82;
        n.vy *= 0.82;
        n.x = Math.max(38, Math.min(width - 38, n.x + n.vx));
        n.y = Math.max(28, Math.min(height - 28, n.y + n.vy));
      }

      // Draw
      ctx.clearRect(0, 0, width, height);

      // Edges
      for (const e of edges) {
        const s = nodeById.get(e.source);
        const t = nodeById.get(e.target);
        if (!s || !t) continue;
        const highlighted =
          selectedNodeId && (e.source === selectedNodeId || e.target === selectedNodeId);
        ctx.strokeStyle = highlighted
          ? "rgba(34, 211, 238, 0.85)"
          : "rgba(148, 163, 184, 0.24)";
        ctx.lineWidth = highlighted ? 2.0 : 1.1;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.stroke();

        // Relation label midpoint
        const mx = (s.x + t.x) / 2;
        const my = (s.y + t.y) / 2;
        ctx.fillStyle = highlighted ? "#67e8f9" : "rgba(148, 163, 184, 0.68)";
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(e.relation.replace(/_/g, " "), mx, my - 3);
      }

      // Nodes
      for (const n of simNodes) {
        const radius = n.type === "user" || n.type === "ai" ? 13 : 8 + Math.min(5, n.weight * 1.5);
        const col = colorForNode(n);
        const isSel = selectedNodeId === n.id;

        if (isSel || n.type === "user" || n.type === "ai") {
          ctx.beginPath();
          ctx.arc(n.x, n.y, radius + 6, 0, Math.PI * 2);
          ctx.fillStyle = isSel ? "rgba(34, 211, 238, 0.25)" : "rgba(99, 102, 241, 0.14)";
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = col;
        ctx.fill();
        ctx.strokeStyle = isSel ? "#ffffff" : "#050811";
        ctx.lineWidth = isSel ? 2.2 : 1.5;
        ctx.stroke();

        ctx.fillStyle = "#f1f5f9";
        ctx.font = n.type === "user" || n.type === "ai" ? "bold 11.5px sans-serif" : "11px sans-serif";
        ctx.textAlign = "center";
        const shortLbl = n.label.length > 22 ? n.label.slice(0, 20) + "…" : n.label;
        ctx.fillText(shortLbl, n.x, n.y + radius + 13);
      }

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [graphData, selectedNodeId]);

  const findNodeAt = (e: React.MouseEvent<HTMLCanvasElement>): SimNode | null => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    for (const n of simNodesRef.current) {
      const dx = n.x - mx;
      const dy = n.y - my;
      if (dx * dx + dy * dy <= 20 * 20) return n;
    }
    return null;
  };

  const handleAddFact = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!subject.trim() || !relation.trim() || !objectVal.trim()) return;
    await api.addMemoryFact({
      subject: subject.trim(),
      relation: relation.trim().replace(/\s+/g, "_"),
      object: objectVal.trim(),
      category,
      pinned: 1,
    });
    setObjectVal("");
    await loadGraph();
  };

  const handleTogglePin = async (mem: SemanticMemory) => {
    await api.patchMemoryFact(mem.id, { pinned: mem.pinned ? 0 : 1 });
    await loadGraph();
  };

  const handleDeleteMem = async (id: string) => {
    await api.deleteMemoryFact(id);
    await loadGraph();
  };

  const handleProbe = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!probeQuery.trim()) return;
    const res = await api.probeMemoryRecall(probeQuery, 0.12);
    setProbeResults({ semantic: res.semantic, episodic: res.episodic });
  };

  return (
    <div className="h-full overflow-y-auto px-4 md:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            Tri-Layer Memory Palace & Semantic Knowledge Graph
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Drag synaptic nodes, inspect associative recall weights, or add & pin permanent knowledge triples.
          </p>
        </div>
        <button
          type="button"
          onClick={loadGraph}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 transition"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Sync Graph
        </button>
      </div>

      {/* Interactive Force-Directed Graph Canvas */}
      <div className="glass-panel rounded-2xl p-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2 text-xs">
          <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-300">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" /> User / Identity
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400" /> Xyvora Core
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" /> Preferences
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" /> Learned Knowledge
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-purple-400" /> RAG Docs
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-400">
            {graphData?.nodes.length || 0} Nodes • {graphData?.edges.length || 0} Synaptic Edges
          </span>
        </div>

        <canvas
          ref={canvasRef}
          onMouseDown={(e) => {
            const hit = findNodeAt(e);
            if (hit) {
              dragNodeRef.current = hit;
              setSelectedNodeId(hit.id);
            } else {
              setSelectedNodeId(null);
            }
          }}
          onMouseMove={(e) => {
            if (!dragNodeRef.current) return;
            const rect = e.currentTarget.getBoundingClientRect();
            dragNodeRef.current.x = e.clientX - rect.left;
            dragNodeRef.current.y = e.clientY - rect.top;
          }}
          onMouseUp={() => {
            dragNodeRef.current = null;
          }}
          onMouseLeave={() => {
            dragNodeRef.current = null;
          }}
          className="w-full h-[340px] rounded-xl bg-[#040812] border border-slate-800 cursor-grab active:cursor-grabbing"
        />
      </div>

      {/* Associative Vector Recall Simulator + Manual Fact Builder */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Vector Recall Probe */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-cyan-200">
            <Search className="w-4 h-4 text-cyan-400" />
            Test Associative Memory Recall
          </div>
          <form onSubmit={handleProbe} className="flex gap-2">
            <input
              type="text"
              value={probeQuery}
              onChange={(e) => setProbeQuery(e.target.value)}
              placeholder="e.g., Where do I live? / My hobbies / Architecture..."
              className="flex-1 rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <button
              type="submit"
              className="px-3.5 py-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/40 text-xs font-semibold text-cyan-200 transition"
            >
              Probe
            </button>
          </form>

          {probeResults && (
            <div className="space-y-2 pt-1 max-h-52 overflow-y-auto">
              {probeResults.semantic.length === 0 ? (
                <div className="text-xs text-slate-400">No semantic nodes matched above 0.12 similarity.</div>
              ) : (
                probeResults.semantic.map((m) => (
                  <div
                    key={m.id}
                    className="flex items-center justify-between gap-2 px-3 py-2 rounded-lg bg-slate-900/90 border border-slate-800 text-xs"
                  >
                    <span className="text-slate-200 truncate">
                      <strong>{m.subject}</strong>{" "}
                      <span className="text-cyan-400 font-mono">{m.relation}</span> →{" "}
                      <strong>{m.object}</strong>
                    </span>
                    <span className="font-mono text-[11px] text-emerald-300 shrink-0">
                      sim {(m.similarity ?? 0).toFixed(2)}
                    </span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Manual Fact Builder */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-emerald-200">
            <Plus className="w-4 h-4 text-emerald-400" />
            Inject New Knowledge Triple
          </div>
          <form onSubmit={handleAddFact} className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="Subject (e.g. User)"
              className="rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <input
              type="text"
              value={relation}
              onChange={(e) => setRelation(e.target.value)}
              placeholder="Relation (e.g. building)"
              className="rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <input
              type="text"
              value={objectVal}
              onChange={(e) => setObjectVal(e.target.value)}
              placeholder="Object (e.g. Mars Rover)"
              className="rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-slate-200 focus:outline-none"
            >
              <option value="identity">Identity</option>
              <option value="preference">Preference</option>
              <option value="knowledge">Knowledge / Fact</option>
              <option value="project">Project</option>
            </select>
            <button
              type="submit"
              className="sm:col-span-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-slate-950 font-semibold text-xs py-2 transition"
            >
              Consolidate Into Memory Graph
            </button>
          </form>
        </div>
      </div>

      {/* Stored Semantic Memories List */}
      <div className="glass-panel rounded-2xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            Consolidated Semantic Memories ({graphData?.memories.length || 0})
          </div>
          {graphData && graphData.memories.some((m) => m.category !== "system") && (
            <button
              type="button"
              onClick={async () => {
                await api.resetUserMemories();
                await loadGraph();
              }}
              className="text-[11px] text-rose-300 hover:text-rose-200 px-2.5 py-1 rounded-lg bg-rose-500/10 border border-rose-500/25 transition"
            >
              Clear Learned Memories
            </button>
          )}
        </div>

        <div className="divide-y divide-slate-800/70 max-h-72 overflow-y-auto">
          {graphData?.memories.map((m) => (
            <div key={m.id} className="py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2 min-w-0">
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono uppercase bg-slate-900 border border-slate-700 text-slate-300">
                  {m.category}
                </span>
                <span className="text-slate-100 font-medium">{m.subject}</span>
                <span className="font-mono text-cyan-400">→ {m.relation} →</span>
                <span className="text-emerald-200 font-semibold">{m.object}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <span className="text-[10.5px] font-mono">recalled {m.access_count}x</span>
                <button
                  type="button"
                  onClick={() => handleTogglePin(m)}
                  className={`p-1 rounded hover:bg-slate-800 ${
                    m.pinned ? "text-amber-300" : "text-slate-500"
                  }`}
                  title={m.pinned ? "Pinned to Working Memory" : "Pin to Working Memory"}
                >
                  <Pin className="w-3.5 h-3.5" />
                </button>
                {m.category !== "system" && (
                  <button
                    type="button"
                    onClick={() => handleDeleteMem(m.id)}
                    className="p-1 rounded hover:bg-rose-500/20 text-slate-500 hover:text-rose-300"
                    title="Delete memory node"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Consolidation History Audit */}
      {graphData && graphData.history.length > 0 && (
        <div className="glass-panel rounded-2xl p-4 space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
            <History className="w-4 h-4 text-indigo-400" />
            Recent Synaptic Consolidation & Revision Log
          </div>
          <div className="space-y-1 max-h-40 overflow-y-auto text-[11.5px] font-mono text-slate-400">
            {graphData.history.map((h) => (
              <div key={h.id} className="flex justify-between gap-2 py-0.5">
                <span>
                  [{h.action.toUpperCase()}]{" "}
                  {h.previous_value ? `${h.previous_value} → ${h.new_value}` : h.new_value}
                </span>
                <span className="text-slate-600">
                  {new Date(h.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
