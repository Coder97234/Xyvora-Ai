import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  Bot,
  Code2,
  Cpu,
  Database,
  FlaskConical,
  FolderKanban,
  Globe,
  Menu,
  MessageSquare,
  Moon,
  Pin,
  Plus,
  Search,
  Sliders,
  Sparkles,
  Sun,
  Trash2,
  User,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import { api } from "./api";
import { AdminAnalyticsView } from "./components/AdminAnalyticsView";
import { AgentsWorkspace } from "./components/AgentsWorkspace";
import { AuthAccountModal } from "./components/AuthAccountModal";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { CodingStudio } from "./components/CodingStudio";
import { CognitiveControls } from "./components/CognitiveControls";
import { MemoryPalace } from "./components/MemoryPalace";
import { NeuralOrb } from "./components/NeuralOrb";
import { ResearchStudio } from "./components/ResearchStudio";
import { SandboxTools } from "./components/SandboxTools";
import { TrainingStudio } from "./components/TrainingStudio";
import {
  BrainState,
  ChatSession,
  CognitiveSettings,
  HealthVitals,
  ProjectWorkspace,
  ProviderInfo,
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
  theme: "dark",
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>("chat");
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [projects, setProjects] = useState<ProjectWorkspace[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [vitals, setVitals] = useState<HealthVitals | null>(null);
  const [settings, setSettings] = useState<CognitiveSettings>(DEFAULT_SETTINGS);
  const [brainState, setBrainState] = useState<BrainState>("idle");

  // UI Drawers & Modals
  const [controlsOpen, setControlsOpen] = useState(false);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [sandboxInitCode, setSandboxInitCode] = useState<string | undefined>(undefined);
  const [memoryToast, setMemoryToast] = useState<SemanticMemory[] | null>(null);

  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const refreshVitals = useCallback(async () => {
    try {
      const [h, sessList, projList, provRes] = await Promise.all([
        api.getHealth(),
        api.listSessions({ project_id: activeProjectId, q: searchQuery || undefined }),
        api.listProjects(),
        api.listProviders(),
      ]);
      setVitals(h);
      setSettings(h.settings);
      setSessions(sessList);
      setProjects(projList);
      setProviders(provRes.providers);
      setActiveSessionId((curr) => {
        if (curr && sessList.some((s) => s.id === curr)) return curr;
        return sessList[0]?.id || "";
      });
    } catch (err) {
      console.error("Failed loading Xyvora platform state:", err);
    }
  }, [activeProjectId, searchQuery]);

  useEffect(() => {
    refreshVitals();
  }, [refreshVitals]);

  // Global keyboard shortcut Cmd/Ctrl+K -> conversation search
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSidebarOpen(true);
        setTimeout(() => searchInputRef.current?.focus(), 50);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleUpdateSettings = async (patch: Partial<CognitiveSettings>) => {
    const updated = await api.patchSettings(patch);
    setSettings(updated);
  };

  const toggleTheme = () => {
    const next = settings.theme === "light" ? "dark" : "light";
    handleUpdateSettings({ theme: next });
  };

  const handleNewSession = async () => {
    const created = await api.createSession("New Neural Session", activeProjectId);
    await refreshVitals();
    setActiveSessionId(created.id);
    setActiveTab("chat");
    setSidebarOpen(false);
  };

  const handleTogglePinSession = async (s: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    await api.patchSession(s.id, { pinned: s.pinned ? 0 : 1 });
    await refreshVitals();
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

  const isAdmin = vitals?.user?.role === "admin" || vitals?.user?.role === "owner";

  const NAV_ITEMS: {
    id: WorkspaceTab;
    label: string;
    icon: React.FC<{ className?: string }>;
    badge?: string | number;
  }[] = [
    { id: "chat", label: "Neural Chat", icon: MessageSquare },
    { id: "research", label: "Research Studio", icon: Globe },
    {
      id: "coding",
      label: "Coding Studio IDE",
      icon: Code2,
      badge: vitals?.projects ?? undefined,
    },
    { id: "agents", label: "Agents & Extensions", icon: Bot },
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
      badge: vitals ? `${Math.round((vitals.neural_cv_accuracy ?? vitals.neural_accuracy) * 100)}%` : undefined,
    },
    {
      id: "lab",
      label: "AI Lab & Vault",
      icon: FlaskConical,
      badge: vitals?.documents ?? undefined,
    },
    ...(isAdmin ? [{ id: "admin" as WorkspaceTab, label: "Developer Telemetry", icon: Activity }] : []),
  ];

  const isLight = settings.theme === "light";

  return (
    <div
      className={`h-screen w-screen overflow-hidden flex bg-[#050811] text-slate-100 neural-grid-bg ${
        isLight ? "theme-light" : ""
      }`}
    >
      {/* Mobile Sidebar Backdrop */}
      {sidebarOpen && (
        <div onClick={() => setSidebarOpen(false)} className="fixed inset-0 z-30 bg-black/60 md:hidden" />
      )}

      {/* Left Cognitive Navigation Rail */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 bg-[#070b18]/95 border-r border-slate-800/80 flex flex-col transition-transform duration-200 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
        style={{ width: "276px" }}
      >
        {/* Brand + Interactive Neural Orb */}
        <div className="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <NeuralOrb state={brainState} size={40} onClick={() => setControlsOpen(true)} />
            <div>
              <div className="text-sm font-extrabold tracking-wide text-white flex items-center gap-1.5">
                XYVORA <span className="text-cyan-400">AI</span>
                <span className="text-[9.5px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  v2.0
                </span>
              </div>
              <div className="text-[10.5px] font-mono text-cyan-300/80">
                One intelligence. Endless possibilities.
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

        {/* Project Workspace Filter */}
        <div className="px-3 pt-2.5 pb-2 border-b border-slate-800/80">
          <div className="flex items-center gap-1.5 px-2 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800">
            <FolderKanban className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
            <select
              value={activeProjectId || ""}
              onChange={(e) => setActiveProjectId(e.target.value || null)}
              className="flex-1 bg-transparent text-xs text-slate-200 focus:outline-none truncate"
            >
              <option value="">All Workspaces (Global)</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  Project: {p.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Workspace Switcher */}
        <div className="p-2.5 space-y-0.5 border-b border-slate-800/80">
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
                className={`w-full flex items-center justify-between px-3 py-1.5 rounded-xl text-xs font-medium transition ${
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

        {/* Conversation Search + Persistent Sessions List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search threads (Ctrl+K)..."
              className="w-full pl-8 pr-2.5 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-white placeholder-slate-500 focus:border-cyan-500/40 focus:outline-none"
            />
          </div>

          <div className="flex items-center justify-between px-1 pt-1">
            <span className="text-[10.5px] font-mono uppercase tracking-wider text-slate-500">
              Conversations
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
                  className={`group flex items-center justify-between gap-1.5 px-2.5 py-2 rounded-xl text-xs cursor-pointer transition ${
                    active
                      ? "bg-slate-800/90 text-white border border-slate-700"
                      : "text-slate-400 hover:bg-slate-900/70 hover:text-slate-200"
                  }`}
                >
                  <span className="truncate flex items-center gap-1.5">
                    {s.pinned ? <Pin className="w-3 h-3 text-amber-300 shrink-0" /> : null}
                    <span className="truncate">{s.title}</span>
                  </span>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition shrink-0">
                    <button
                      type="button"
                      onClick={(e) => handleTogglePinSession(s, e)}
                      className={`p-0.5 ${s.pinned ? "text-amber-300" : "text-slate-500 hover:text-amber-300"}`}
                      title={s.pinned ? "Unpin" : "Pin conversation"}
                    >
                      <Pin className="w-3 h-3" />
                    </button>
                    {sessions.length > 1 && (
                      <button
                        type="button"
                        onClick={(e) => handleDeleteSession(s.id, e)}
                        className="p-0.5 text-slate-500 hover:text-rose-400"
                        title="Delete thread"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Account & Cloud Profile Footer */}
        <div className="p-3 border-t border-slate-800/80 bg-[#050914] space-y-2">
          <button
            type="button"
            onClick={() => setAuthModalOpen(true)}
            className="w-full flex items-center justify-between px-3 py-2 rounded-xl bg-slate-900/90 hover:bg-slate-800 border border-slate-800 text-xs transition"
          >
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-6 h-6 rounded-full bg-gradient-to-br from-cyan-500 to-indigo-500 flex items-center justify-center text-slate-950 font-bold shrink-0">
                <User className="w-3.5 h-3.5" />
              </div>
              <div className="text-left min-w-0">
                <div className="font-semibold text-white truncate">
                  {vitals?.user?.is_guest ? "Local Workspace" : vitals?.user?.display_name || "Account"}
                </div>
                <div className="text-[10px] font-mono text-cyan-300 truncate">
                  {vitals?.user?.is_guest ? "Sign in / Create Account" : `@${vitals?.user?.username}`}
                </div>
              </div>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300">
              {vitals?.user?.is_guest ? "Guest" : vitals?.user?.role}
            </span>
          </button>
        </div>
      </aside>

      {/* Main Content Shell */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Cognitive Telemetry Header */}
        <header className="h-13 px-4 md:px-6 border-b border-slate-800/80 bg-[#070b18]/85 backdrop-blur-md flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="md:hidden p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300"
            >
              <Menu className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-2 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  brainState === "idle"
                    ? "bg-emerald-400"
                    : brainState === "listening"
                    ? "bg-rose-400 animate-ping"
                    : "bg-cyan-400 animate-pulse"
                }`}
              />
              <span className="font-semibold text-slate-200 capitalize">{activeTab}</span>
              <span className="hidden sm:inline text-slate-500">•</span>
              <span className="hidden sm:inline text-[11px] font-mono text-cyan-300/90 capitalize">
                Persona: {settings.persona}
              </span>
              {vitals?.active_provider && (
                <span className="hidden lg:inline text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-indigo-300">
                  Engine: {vitals.active_provider} ({vitals.active_model})
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={toggleTheme}
              title={isLight ? "Switch to Dark Futuristic Theme" : "Switch to Light Theme"}
              className="p-1.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 hover:text-cyan-300 transition"
            >
              {isLight ? <Moon className="w-3.5 h-3.5" /> : <Sun className="w-3.5 h-3.5" />}
            </button>

            <button
              type="button"
              onClick={() => handleUpdateSettings({ voice_enabled: !settings.voice_enabled })}
              title="Toggle Auto Voice Readout"
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs border transition ${
                settings.voice_enabled
                  ? "bg-cyan-500/15 border-cyan-500/40 text-cyan-200"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              {settings.voice_enabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">Voice</span>
            </button>

            <button
              type="button"
              onClick={() => setControlsOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs bg-slate-900 hover:bg-slate-800 border border-cyan-500/30 text-cyan-200 transition"
            >
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              <span>Providers & Brain</span>
            </button>
          </div>
        </header>

        {/* Live Memory Consolidation Toast */}
        {memoryToast && (
          <div className="fixed bottom-20 right-6 z-40 max-w-sm rounded-2xl bg-slate-900/95 border border-emerald-400/40 px-4 py-3 shadow-2xl">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-300">
              <Sparkles className="w-3.5 h-3.5" />
              {memoryToast.some((m) => m.verified === 0)
                ? "Memory Queued for Approval"
                : "Consolidated Into Memory Graph"}
            </div>
            <div className="mt-1 space-y-0.5 text-[11px] text-slate-200 font-mono">
              {memoryToast.slice(0, 3).map((m) => (
                <div key={m.id}>
                  {m.subject} → {m.relation} → {m.object}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Active Workspace Body */}
        <main className="flex-1 min-h-0 overflow-hidden">
          {activeTab === "chat" && (
            <ChatWorkspace
              sessionId={activeSessionId}
              activeProjectId={activeProjectId}
              settings={settings}
              providers={providers}
              brainState={brainState}
              setBrainState={setBrainState}
              onRefreshVitals={refreshVitals}
              onMemoryToast={handleMemoryToast}
              onNavigateTab={navigateTab}
              onOpenControls={() => setControlsOpen(true)}
            />
          )}
          {activeTab === "research" && <ResearchStudio activeProjectId={activeProjectId} />}
          {activeTab === "coding" && (
            <CodingStudio activeProjectId={activeProjectId} onSelectProject={(id) => setActiveProjectId(id)} />
          )}
          {activeTab === "agents" && <AgentsWorkspace activeProjectId={activeProjectId} />}
          {activeTab === "memory" && <MemoryPalace onRefreshVitals={refreshVitals} />}
          {activeTab === "training" && <TrainingStudio onRefreshVitals={refreshVitals} />}
          {activeTab === "lab" && <SandboxTools initialCode={sandboxInitCode} onRefreshVitals={refreshVitals} />}
          {activeTab === "admin" && <AdminAnalyticsView />}
        </main>
      </div>

      {/* Drawers & Modals */}
      <CognitiveControls
        open={controlsOpen}
        onClose={() => setControlsOpen(false)}
        settings={settings}
        onUpdate={handleUpdateSettings}
        onProvidersChanged={refreshVitals}
      />

      <AuthAccountModal
        open={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        user={vitals?.user}
        storage={vitals?.storage}
        onAuthChanged={refreshVitals}
      />
    </div>
  );
};

export default App;

