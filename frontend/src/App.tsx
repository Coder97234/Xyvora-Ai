import React, { useCallback, useEffect, useState } from "react";
import {
  Cpu,
  Database,
  FlaskConical,
  Menu,
  MessageSquare,
  Plus,
  Sliders,
  Sparkles,
  Trash2,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { api } from "./api";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { CognitiveControls } from "./components/CognitiveControls";
import { MemoryPalace } from "./components/MemoryPalace";
import { NeuralOrb } from "./components/NeuralOrb";
import { SandboxTools } from "./components/SandboxTools";
import { TrainingStudio } from "./components/TrainingStudio";
import {
  BrainState,
  ChatSession,
  CognitiveSettings,
  HealthVitals,
  SemanticMemory,
  WorkspaceTab,
} from "./types";

const DEFAULT_SETTINGS: CognitiveSettings = {
  persona: "nexus",
  creativity: 0.65,
  verbosity: "balanced",
  memory_threshold: 0.24,
  auto_learn_memory: true,
  show_thought_trace: true,
  voice_enabled: false,
  external_llm_enabled: false,
  external_llm_base_url: "https://api.openai.com/v1",
  external_llm_api_key: "",
  external_llm_model: "gpt-4o-mini",
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("chat");
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [vitals, setVitals] = useState<HealthVitals | null>(null);
  const [settings, setSettings] = useState<CognitiveSettings>(DEFAULT_SETTINGS);
  const [brainState, setBrainState] = useState<BrainState>("idle");
  const [controlsOpen, setControlsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sandboxInitCode, setSandboxInitCode] = useState<string | undefined>(undefined);
  const [memoryToast, setMemoryToast] = useState<SemanticMemory[] | null>(null);

  const refreshVitals = useCallback(async () => {
    try {
      const [h, sessList] = await Promise.all([api.getHealth(), api.listSessions()]);
      setVitals(h);
      setSettings(h.settings);
      setSessions(sessList);
      setActiveSessionId((curr) => {
        if (curr && sessList.some((s) => s.id === curr)) return curr;
        return sessList[0]?.id || "";
      });
    } catch (err) {
      console.error("Failed loading Xyvora vitals:", err);
    }
  }, []);

  useEffect(() => {
    refreshVitals();
  }, [refreshVitals]);

  const handleUpdateSettings = async (patch: Partial<CognitiveSettings>) => {
    const updated = await api.patchSettings(patch);
    setSettings(updated);
  };

  const handleNewSession = async () => {
    const created = await api.createSession();
    await refreshVitals();
    setActiveSessionId(created.id);
    setActiveTab("chat");
    setSidebarOpen(false);
  };

  const handleDeleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (sessions.length <= 1) return;
    await api.deleteSession(id);
    await refreshVitals();
  };

  const handleMemoryToast = (memories: SemanticMemory[]) => {
    if (!memories.length) return;
    setMemoryToast(memories);
    setTimeout(() => setMemoryToast(null), 4500);
  };

  const navigateTab = (tab: WorkspaceTab, codePayload?: string) => {
    if (codePayload) setSandboxInitCode(codePayload);
    setActiveTab(tab);
  };

  const NAV_ITEMS: {
    id: WorkspaceTab;
    label: string;
    icon: React.FC<{ className?: string }>;
    badge?: string | number;
  }[] = [
    { id: "chat", label: "Neural Chat", icon: MessageSquare },
    {
      id: "memory",
      label: "Memory Palace",
      icon: Database,
      badge: vitals?.semantic_memories ?? undefined,
    },
    {
      id: "training",
      label: "Training Studio",
      icon: Cpu,
      badge: vitals ? `${Math.round(vitals.neural_accuracy * 100)}%` : undefined,
    },
    {
      id: "lab",
      label: "AI Lab & Tools",
      icon: FlaskConical,
      badge: vitals?.open_tasks ? vitals.open_tasks : undefined,
    },
  ];

  return (
    <div className="h-screen w-screen overflow-hidden flex bg-[#050811] text-slate-100 neural-grid-bg">
      {/* Mobile Sidebar Backdrop */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/60 md:hidden"
        />
      )}

      {/* Left Cognitive Navigation Rail */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-68 bg-[#070b18]/95 border-r border-slate-800/80 flex flex-col transition-transform duration-200 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
        style={{ width: "268px" }}
      >
        {/* Brand + Interactive Neural Orb */}
        <div className="px-4 py-3.5 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <NeuralOrb
              state={brainState}
              size={42}
              onClick={() => setControlsOpen(true)}
            />
            <div>
              <div className="text-sm font-extrabold tracking-wide text-white flex items-center gap-1.5">
                XYVORA <span className="text-cyan-400">AI</span>
              </div>
              <div className="text-[10.5px] font-mono text-cyan-300/80">
                Neuro-Symbolic Cortex
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setSidebarOpen(false)}
            className="md:hidden p-1 text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Workspace Switcher */}
        <div className="p-3 space-y-1 border-b border-slate-800/80">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActiveTab(item.id);
                  setSidebarOpen(false);
                }}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition ${
                  active
                    ? "bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-cyan-200 border border-cyan-500/35 shadow-sm"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-900/70"
                }`}
              >
                <span className="flex items-center gap-2.5">
                  <Icon className="w-4 h-4 text-cyan-400" />
                  {item.label}
                </span>
                {item.badge !== undefined && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-slate-900 border border-slate-700 text-cyan-300">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Persistent Sessions List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <div className="flex items-center justify-between px-1">
            <span className="text-[10.5px] font-mono uppercase tracking-wider text-slate-500">
              Memory-Linked Threads
            </span>
            <button
              type="button"
              onClick={handleNewSession}
              className="flex items-center gap-1 text-[11px] text-cyan-300 hover:text-cyan-200 px-2 py-0.5 rounded-md bg-cyan-500/10 border border-cyan-500/25 transition"
            >
              <Plus className="w-3 h-3" /> New
            </button>
          </div>

          <div className="space-y-1">
            {sessions.map((s) => {
              const active = s.id === activeSessionId && activeTab === "chat";
              return (
                <div
                  key={s.id}
                  onClick={() => {
                    setActiveSessionId(s.id);
                    setActiveTab("chat");
                    setSidebarOpen(false);
                  }}
                  className={`group flex items-center justify-between gap-2 px-3 py-2 rounded-xl text-xs cursor-pointer transition ${
                    active
                      ? "bg-slate-800/90 text-white border border-slate-700"
                      : "text-slate-400 hover:bg-slate-900/70 hover:text-slate-200"
                  }`}
                >
                  <span className="truncate">{s.title}</span>
                  {sessions.length > 1 && (
                    <button
                      type="button"
                      onClick={(e) => handleDeleteSession(s.id, e)}
                      className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400 transition"
                      title="Delete thread"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Live Brain Telemetry Footer */}
        {vitals && (
          <div className="p-3 border-t border-slate-800/80 bg-[#050914]/80 text-[11px] space-y-1.5">
            <div className="flex items-center justify-between text-slate-400">
              <span className="flex items-center gap-1.5 text-emerald-300 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Cortex Online
              </span>
              <span className="font-mono text-slate-500">
                {(vitals.neural_params / 1000).toFixed(0)}k weights
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1 text-center font-mono text-[10px]">
              <div className="rounded bg-slate-900/90 py-1 border border-slate-800">
                <div className="text-cyan-300 font-bold">{vitals.semantic_memories}</div>
                <div className="text-slate-500">Facts</div>
              </div>
              <div className="rounded bg-slate-900/90 py-1 border border-slate-800">
                <div className="text-indigo-300 font-bold">{vitals.documents}</div>
                <div className="text-slate-500">Docs</div>
              </div>
              <div className="rounded bg-slate-900/90 py-1 border border-slate-800">
                <div className="text-amber-300 font-bold">{vitals.custom_skills}</div>
                <div className="text-slate-500">Skills</div>
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        {/* Top Interactive Status Header */}
        <header className="h-14 px-4 md:px-6 border-b border-slate-800/80 bg-[#060a16]/85 backdrop-blur-md flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2 text-xs font-mono">
              <span
                className={`px-2.5 py-0.5 rounded-full border ${
                  brainState === "idle"
                    ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                    : brainState === "listening"
                    ? "border-emerald-400/50 bg-emerald-500/20 text-emerald-200 animate-pulse"
                    : "border-amber-400/50 bg-amber-500/20 text-amber-200 animate-pulse"
                }`}
              >
                {brainState === "idle"
                  ? "READY"
                  : brainState === "listening"
                  ? "VOICE LISTENING"
                  : brainState === "thinking"
                  ? "NEURAL INFERENCE"
                  : "SYNTHESIZING"}
              </span>
              <span className="hidden sm:inline text-slate-400">
                Persona: <strong className="text-indigo-300 uppercase">{settings.persona}</strong>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => handleUpdateSettings({ voice_enabled: !settings.voice_enabled })}
              className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-xl border transition ${
                settings.voice_enabled
                  ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-200"
                  : "bg-slate-900/90 border-slate-800 text-slate-400 hover:text-slate-200"
              }`}
              title="Toggle auto-read aloud (Neural Voice TTS)"
            >
              {settings.voice_enabled ? (
                <Volume2 className="w-3.5 h-3.5" />
              ) : (
                <VolumeX className="w-3.5 h-3.5" />
              )}
              <span className="hidden sm:inline">Voice</span>
            </button>

            <button
              type="button"
              onClick={() => setControlsOpen(true)}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-xl bg-gradient-to-r from-cyan-500/15 to-indigo-500/15 hover:from-cyan-500/25 hover:to-indigo-500/25 border border-cyan-400/35 text-cyan-200 font-medium transition"
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Brain Controls</span>
            </button>
          </div>
        </header>

        {/* Active Workspace View */}
        <main className="flex-1 min-h-0 overflow-hidden">
          {activeTab === "chat" && activeSessionId && (
            <ChatWorkspace
              sessionId={activeSessionId}
              settings={settings}
              brainState={brainState}
              setBrainState={setBrainState}
              onRefreshVitals={refreshVitals}
              onMemoryToast={handleMemoryToast}
              onNavigateTab={navigateTab}
            />
          )}
          {activeTab === "memory" && <MemoryPalace onRefreshVitals={refreshVitals} />}
          {activeTab === "training" && <TrainingStudio onRefreshVitals={refreshVitals} />}
          {activeTab === "lab" && (
            <SandboxTools
              initialCode={sandboxInitCode}
              onRefreshVitals={refreshVitals}
            />
          )}
        </main>

        {/* Floating Memory Consolidation Toast */}
        {memoryToast && memoryToast.length > 0 && (
          <div
            onClick={() => {
              setMemoryToast(null);
              setActiveTab("memory");
            }}
            className="fixed bottom-20 right-6 z-40 max-w-sm glass-panel rounded-2xl p-3.5 border border-emerald-400/45 shadow-2xl cursor-pointer animate-bounce"
          >
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-300 mb-1">
              <Sparkles className="w-3.5 h-3.5" />
              Synaptic Memory Consolidated!
            </div>
            {memoryToast.slice(0, 2).map((m, i) => (
              <div key={i} className="text-[11.5px] font-mono text-slate-200 truncate">
                {m.subject} → {m.relation} → {m.object}
              </div>
            ))}
            <div className="text-[10px] text-slate-400 mt-1">
              Click to inspect in Memory Palace →
            </div>
          </div>
        )}
      </div>

      {/* Slide-Over Brain Modulation Controls */}
      <CognitiveControls
        open={controlsOpen}
        onClose={() => setControlsOpen(false)}
        settings={settings}
        onUpdate={handleUpdateSettings}
      />
    </div>
  );
};

export default App;
