import React, { useEffect, useState } from "react";
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock,
  Layers,
  Play,
  Plus,
  Puzzle,
  Shield,
  StopCircle,
  XCircle,
} from "lucide-react";
import { api } from "../api";
import { AgentRun, ExtensionItem } from "../types";
import { MarkdownView } from "./MarkdownView";

export const AgentsWorkspace: React.FC<{ activeProjectId?: string | null }> = ({ activeProjectId }) => {
  const [subTab, setSubTab] = useState<"agents" | "extensions">("agents");

  // Autonomous Agents state
  const [goal, setGoal] = useState("");
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [activeRun, setActiveRun] = useState<AgentRun | null>(null);
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentErr, setAgentErr] = useState<string | null>(null);

  // Extensions Marketplace state
  const [extensions, setExtensions] = useState<ExtensionItem[]>([]);
  const [catFilter, setCatFilter] = useState<string>("");
  const [selectedExt, setSelectedExt] = useState<ExtensionItem | null>(null);
  const [extInputs, setExtInputs] = useState<Record<string, string>>({});
  const [extOutput, setExtOutput] = useState<Record<string, unknown> | null>(null);
  const [creatingExt, setCreatingExt] = useState(false);
  const [newExtName, setNewExtName] = useState("");
  const [newExtDesc, setNewExtDesc] = useState("");
  const [newExtCat, setNewExtCat] = useState("productivity");
  const [newExtTmpl, setNewExtTmpl] = useState("Analyze and structure the following engineering note:\n{input}");

  const loadAgents = async () => {
    const list = await api.listAgentRuns();
    setRuns(list);
    if (!activeRun && list[0]) {
      const full = await api.getAgentRun(list[0].id);
      setActiveRun(full);
    }
  };

  const loadExts = async (cat?: string) => {
    const list = await api.listExtensions(cat || undefined);
    setExtensions(list);
    if (!selectedExt && list[0]) setSelectedExt(list[0]);
  };

  useEffect(() => {
    loadAgents();
    loadExts();
  }, []);

  const handleStartAgent = async (goalOverride?: string) => {
    const g = (goalOverride ?? goal).trim();
    if (!g) return;
    setGoal(g);
    setAgentBusy(true);
    setAgentErr(null);
    try {
      const created = await api.startAgentRun({ goal: g, project_id: activeProjectId });
      setActiveRun(created);
      const list = await api.listAgentRuns();
      setRuns(list);
    } catch (err: any) {
      setAgentErr(err.message || "Agent execution failed");
    } finally {
      setAgentBusy(false);
    }
  };

  const handleApproveStep = async (approved: boolean) => {
    if (!activeRun) return;
    setAgentBusy(true);
    try {
      const updated = await api.approveAgentRun(activeRun.id, approved);
      setActiveRun(updated);
      setRuns(await api.listAgentRuns());
    } finally {
      setAgentBusy(false);
    }
  };

  const handleCancelRun = async () => {
    if (!activeRun) return;
    const updated = await api.cancelAgentRun(activeRun.id);
    setActiveRun(updated);
    setRuns(await api.listAgentRuns());
  };

  const toggleInstall = async (ext: ExtensionItem) => {
    if (ext.installed) {
      await api.uninstallExtension(ext.id);
    } else {
      await api.installExtension(ext.id);
    }
    const list = await api.listExtensions(catFilter || undefined);
    setExtensions(list);
    const refreshed = list.find((e) => e.id === ext.id) || null;
    if (selectedExt?.id === ext.id) setSelectedExt(refreshed);
  };

  const handleRunExtension = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedExt) return;
    const coerced: Record<string, unknown> = {};
    const props = selectedExt.input_schema?.properties || {};
    for (const [k, rawVal] of Object.entries(extInputs)) {
      if (props[k]?.type === "number") coerced[k] = Number(rawVal);
      else if (props[k]?.type === "array") coerced[k] = rawVal.split(",").map((s) => s.trim()).filter(Boolean);
      else coerced[k] = rawVal;
    }
    const res = await api.runExtension(selectedExt.id, coerced);
    setExtOutput(res);
  };

  const handleCreateCustomExt = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newExtName.trim()) return;
    const created = await api.createCustomExtension({
      name: newExtName,
      description: newExtDesc,
      category: newExtCat,
      prompt_template: newExtTmpl,
    });
    setCreatingExt(false);
    setNewExtName("");
    setNewExtDesc("");
    await loadExts(catFilter);
    setSelectedExt(created);
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Sub-navigation */}
      <div className="px-4 py-2.5 border-b border-slate-800 bg-[#070b18]/90 flex items-center justify-between">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setSubTab("agents")}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 ${
              subTab === "agents"
                ? "bg-cyan-500/20 border border-cyan-500/40 text-cyan-200"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Bot className="w-4 h-4" /> Autonomous Multi-Step Agents
          </button>
          <button
            type="button"
            onClick={() => setSubTab("extensions")}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 ${
              subTab === "extensions"
                ? "bg-indigo-500/20 border border-indigo-400/40 text-indigo-200"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Puzzle className="w-4 h-4" /> Extension Marketplace ({extensions.filter((e) => e.installed).length} active)
          </button>
        </div>
      </div>

      {subTab === "agents" ? (
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
          {/* Left Run History */}
          <div className="lg:col-span-3 border-r border-slate-800 bg-[#060a15] p-3.5 overflow-y-auto space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400">Agent Execution History</div>
            {runs.length === 0 ? (
              <p className="text-xs text-slate-500">Launch a multi-step agent goal to track planning, retries, and tool checkpoints.</p>
            ) : (
              runs.map((r) => (
                <div
                  key={r.id}
                  onClick={async () => setActiveRun(await api.getAgentRun(r.id))}
                  className={`p-2.5 rounded-xl border text-xs cursor-pointer ${
                    activeRun?.id === r.id
                      ? "bg-cyan-500/15 border-cyan-500/40 text-white"
                      : "bg-slate-900/70 border-slate-800 text-slate-300"
                  }`}
                >
                  <div className="font-medium truncate">{r.goal}</div>
                  <div className="text-[10px] font-mono mt-1 flex items-center justify-between">
                    <span
                      className={
                        r.status === "completed"
                          ? "text-emerald-300"
                          : r.status === "awaiting_approval"
                          ? "text-amber-300"
                          : "text-cyan-300"
                      }
                    >
                      {r.status}
                    </span>
                    <span className="text-slate-500">{new Date(r.created_at * 1000).toLocaleTimeString()}</span>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Right Planner & Step Tracker */}
          <div className="lg:col-span-9 overflow-y-auto p-4 md:p-6 space-y-4">
            <div className="glass-panel rounded-2xl p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-cyan-400" />
                <div>
                  <h2 className="text-sm font-bold text-white">Autonomous Goal Planner & Guarded Executor</h2>
                  <p className="text-[11px] text-slate-400">
                    Bounded steps (max 8), automatic retries (max 2), live cancellation & mandatory approval checkpoints before file/destructive actions
                  </p>
                </div>
              </div>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleStartAgent();
                }}
                className="flex gap-2"
              >
                <input
                  type="text"
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="Enter a multi-step goal (e.g., Calculate derivative of x^3 + 4x, search vault for security specs, and create a task)"
                  className="flex-1 px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white"
                />
                <button
                  type="submit"
                  disabled={agentBusy}
                  className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950 flex items-center gap-1.5 disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5" /> {agentBusy ? "Executing..." : "Launch Agent"}
                </button>
              </form>

              <div className="flex flex-wrap gap-1.5">
                {[
                  "Differentiate sin(x)*x^2, search Knowledge Vault for Xyvora architecture, and create a review task",
                  "Run a Python Monte Carlo simulation of 500 trials and create a project file report.md",
                ].map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => handleStartAgent(sample)}
                    className="text-[11px] px-2.5 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300"
                  >
                    {sample}
                  </button>
                ))}
              </div>
            </div>

            {agentErr && (
              <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/35 text-xs text-rose-200">{agentErr}</div>
            )}

            {activeRun && (
              <div className="space-y-4">
                {/* Approval Checkpoint Banner */}
                {activeRun.status === "awaiting_approval" && activeRun.pending_approval && (
                  <div className="p-4 rounded-2xl bg-amber-500/15 border border-amber-400/50 space-y-2.5">
                    <div className="flex items-center gap-2 text-amber-200 text-xs font-bold">
                      <AlertTriangle className="w-4 h-4 shrink-0" /> Human-in-the-Loop Approval Checkpoint
                    </div>
                    <p className="text-xs text-slate-200">
                      Agent paused before consequential step{" "}
                      <span className="font-mono text-amber-200">
                        #{activeRun.pending_approval.step_index + 1}: {activeRun.pending_approval.title} (
                        {activeRun.pending_approval.tool_name})
                      </span>
                      . Approve execution?
                    </p>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => handleApproveStep(true)}
                        disabled={agentBusy}
                        className="px-3.5 py-1.5 rounded-xl bg-emerald-500 text-slate-950 text-xs font-bold"
                      >
                        Approve & Resume
                      </button>
                      <button
                        type="button"
                        onClick={() => handleApproveStep(false)}
                        disabled={agentBusy}
                        className="px-3.5 py-1.5 rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-200 text-xs"
                      >
                        Skip Consequential Step
                      </button>
                      <button
                        type="button"
                        onClick={handleCancelRun}
                        className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 text-xs flex items-center gap-1"
                      >
                        <StopCircle className="w-3.5 h-3.5" /> Cancel Run
                      </button>
                    </div>
                  </div>
                )}

                <div className="glass-panel rounded-2xl p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-xs font-bold text-white">{activeRun.goal}</div>
                      <div className="text-[11px] font-mono text-cyan-300">Status: {activeRun.status}</div>
                    </div>
                    {(activeRun.status === "running" || activeRun.status === "planning") && (
                      <button
                        type="button"
                        onClick={handleCancelRun}
                        className="px-3 py-1 rounded-lg bg-rose-500/20 border border-rose-500/40 text-xs text-rose-200 flex items-center gap-1"
                      >
                        <StopCircle className="w-3.5 h-3.5" /> Cancel
                      </button>
                    )}
                  </div>

                  {/* Step List */}
                  <div className="space-y-2">
                    {(activeRun.steps || []).map((st) => (
                      <div key={st.id} className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-semibold text-slate-100 flex items-center gap-2">
                            {st.status === "completed" ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                            ) : st.status === "failed" ? (
                              <XCircle className="w-4 h-4 text-rose-400" />
                            ) : (
                              <Clock className="w-4 h-4 text-amber-300" />
                            )}
                            Step {st.step_index + 1}: {st.title}
                          </span>
                          <span className="text-[10.5px] font-mono text-cyan-300">
                            {st.tool_name} • {st.status} {st.retries > 0 ? `(retries: ${st.retries})` : ""}
                          </span>
                        </div>
                        {st.output_summary && (
                          <div className="text-[11px] text-slate-300 font-mono pl-6 whitespace-pre-wrap">
                            {st.output_summary}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  {activeRun.final_output && (
                    <div className="pt-3 border-t border-slate-800">
                      <div className="text-[11px] font-mono uppercase text-emerald-300 mb-1.5">Synthesized Agent Outcome</div>
                      <div className="p-3.5 rounded-xl bg-slate-950/90 border border-slate-800 text-xs">
                        <MarkdownView content={activeRun.final_output} />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Extension Marketplace View */
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
          <div className="lg:col-span-5 border-r border-slate-800 bg-[#060a15] p-4 overflow-y-auto space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex flex-wrap gap-1">
                {["", "coding", "mathematics", "developer", "data", "research", "productivity"].map((c) => (
                  <button
                    key={c}
                    type="button"
                    onClick={() => {
                      setCatFilter(c);
                      loadExts(c);
                    }}
                    className={`px-2 py-1 rounded-lg text-[11px] capitalize ${
                      catFilter === c ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400 bg-slate-900"
                    }`}
                  >
                    {c || "All"}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setCreatingExt((s) => !s)}
                className="px-2.5 py-1 rounded-lg bg-indigo-500/20 border border-indigo-400/40 text-[11px] text-indigo-200 flex items-center gap-1"
              >
                <Plus className="w-3 h-3" /> Custom
              </button>
            </div>

            {creatingExt && (
              <form onSubmit={handleCreateCustomExt} className="p-3 rounded-xl bg-slate-900 border border-indigo-500/35 space-y-2">
                <div className="text-xs font-bold text-indigo-200">Create Safe Declarative Extension</div>
                <input
                  type="text"
                  required
                  value={newExtName}
                  onChange={(e) => setNewExtName(e.target.value)}
                  placeholder="Extension Name"
                  className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-white"
                />
                <input
                  type="text"
                  required
                  value={newExtDesc}
                  onChange={(e) => setNewExtDesc(e.target.value)}
                  placeholder="Short description"
                  className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-white"
                />
                <textarea
                  rows={2}
                  value={newExtTmpl}
                  onChange={(e) => setNewExtTmpl(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-white font-mono"
                />
                <button type="submit" className="w-full py-1.5 rounded bg-indigo-500 text-white text-xs font-bold">
                  Register & Install
                </button>
              </form>
            )}

            <div className="space-y-2">
              {extensions.map((ext) => (
                <div
                  key={ext.id}
                  onClick={() => {
                    setSelectedExt(ext);
                    setExtInputs({});
                    setExtOutput(null);
                  }}
                  className={`p-3 rounded-xl border cursor-pointer space-y-1.5 ${
                    selectedExt?.id === ext.id
                      ? "bg-cyan-500/15 border-cyan-500/40"
                      : "bg-slate-900/70 border-slate-800 hover:bg-slate-900"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-bold text-white">{ext.name}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleInstall(ext);
                      }}
                      className={`px-2.5 py-0.5 rounded-full text-[10.5px] font-mono ${
                        ext.installed
                          ? "bg-emerald-500/20 border border-emerald-500/40 text-emerald-200"
                          : "bg-slate-800 border border-slate-700 text-slate-300"
                      }`}
                    >
                      {ext.installed ? "Installed" : "Install"}
                    </button>
                  </div>
                  <p className="text-[11px] text-slate-400">{ext.description}</p>
                  <div className="flex flex-wrap gap-1.5 text-[10px] font-mono text-slate-400">
                    <span>v{ext.version}</span> • <span>{ext.category}</span> •{" "}
                    <span className="text-cyan-300 flex items-center gap-0.5">
                      <Shield className="w-3 h-3" /> {ext.permissions.join(", ")}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Extension Runner & Schema Inspector */}
          <div className="lg:col-span-7 p-5 overflow-y-auto space-y-4">
            {selectedExt && (
              <div className="glass-panel rounded-2xl p-5 space-y-4">
                <div className="flex items-start justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-white">{selectedExt.name}</h3>
                    <p className="text-xs text-slate-400 mt-0.5">{selectedExt.description}</p>
                    <div className="text-[10.5px] font-mono text-cyan-300 mt-1">
                      ID: `{selectedExt.id}` • Compatibility: `{selectedExt.compatibility}` • Permissions:{" "}
                      {selectedExt.permissions.join(", ")}
                    </div>
                  </div>
                </div>

                {!selectedExt.installed ? (
                  <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/35 text-xs text-amber-200">
                    Install this extension first to execute it inside your workspace.
                  </div>
                ) : (
                  <form onSubmit={handleRunExtension} className="space-y-3">
                    <div className="text-xs font-semibold text-slate-200">Interactive Extension Runner</div>
                    {Object.entries(selectedExt.input_schema?.properties || {}).map(([field, spec]) => (
                      <div key={field}>
                        <label className="block text-[11px] font-mono text-slate-400 mb-1">
                          {field} ({spec.type}) {spec.description ? `— ${spec.description}` : ""}
                        </label>
                        {field.includes("text") || field === "sql" || field === "input" ? (
                          <textarea
                            rows={3}
                            value={extInputs[field] || ""}
                            onChange={(e) => setExtInputs({ ...extInputs, [field]: e.target.value })}
                            className="w-full p-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white font-mono"
                          />
                        ) : (
                          <input
                            type="text"
                            value={extInputs[field] || ""}
                            onChange={(e) => setExtInputs({ ...extInputs, [field]: e.target.value })}
                            className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white font-mono"
                          />
                        )}
                      </div>
                    ))}
                    <button
                      type="submit"
                      className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950"
                    >
                      Execute Extension
                    </button>
                  </form>
                )}

                {extOutput && (
                  <div className="space-y-1.5 pt-2 border-t border-slate-800">
                    <div className="text-[11px] font-mono text-emerald-300">Structured Output:</div>
                    <pre className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-slate-200 overflow-x-auto">
                      {JSON.stringify(extOutput, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
