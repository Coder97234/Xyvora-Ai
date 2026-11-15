import React, { useEffect, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Database,
  Plus,
  Sliders,
  Trash2,
  Volume2,
  X,
} from "lucide-react";
import { api } from "../api";
import { CognitiveSettings, CustomPersonality, ProviderInfo } from "../types";

interface CognitiveControlsProps {
  open: boolean;
  onClose: () => void;
  settings: CognitiveSettings;
  onUpdate: (patch: Partial<CognitiveSettings>) => void;
  onProvidersChanged?: () => void;
}

const BUILTIN_PERSONAS = [
  { id: "nexus", label: "Nexus", desc: "Balanced polymath — warm, analytical, clear" },
  { id: "axiom", label: "Axiom", desc: "Rigorous scientist — formal derivations & deep theory" },
  { id: "muse", label: "Muse", desc: "Creative visionary — evocative metaphors & ideation" },
  { id: "forge", label: "Forge", desc: "Systems engineer — code-first & pragmatic" },
  { id: "socrates", label: "Socratic", desc: "Interactive tutor — guides with reflective inquiry" },
];

export const CognitiveControls: React.FC<CognitiveControlsProps> = ({
  open,
  onClose,
  settings,
  onUpdate,
  onProvidersChanged,
}) => {
  const [tab, setTab] = useState<"providers" | "persona" | "memory">("providers");
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [selectedPid, setSelectedPid] = useState<string>("openai");
  const [keyInput, setKeyInput] = useState("");
  const [baseUrlInput, setBaseUrlInput] = useState("");
  const [modelInput, setModelInput] = useState("");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

  // Custom personalities
  const [customPersonas, setCustomPersonas] = useState<CustomPersonality[]>([]);
  const [addingPersona, setAddingPersona] = useState(false);
  const [newPName, setNewPName] = useState("");
  const [newPInst, setNewPInst] = useState("");
  const [newPDomain, setNewPDomain] = useState("");

  const loadData = async () => {
    try {
      const [pRes, perRes] = await Promise.all([api.listProviders(), api.listPersonalities()]);
      setProviders(pRes.providers);
      setCustomPersonas(perRes.custom);
      const cur = pRes.providers.find((x) => x.id === selectedPid) || pRes.providers[1] || pRes.providers[0];
      if (cur) {
        setSelectedPid(cur.id);
        setBaseUrlInput(cur.base_url || "");
        setModelInput(cur.default_model || "");
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (open) {
      setTestResult(null);
      setKeyInput("");
      loadData();
    }
  }, [open]);

  if (!open) return null;

  const activeProvObj = providers.find((p) => p.id === selectedPid) || null;

  const handleSelectProv = (p: ProviderInfo) => {
    setSelectedPid(p.id);
    setKeyInput("");
    setBaseUrlInput(p.base_url || "");
    setModelInput(p.default_model || "");
    setTestResult(null);
  };

  const handleSaveProvider = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeProvObj || activeProvObj.id === "xyvora_local") return;
    const res = await api.updateProvider(activeProvObj.id, {
      api_key: keyInput.trim() ? keyInput.trim() : undefined,
      base_url: baseUrlInput.trim(),
      default_model: modelInput.trim(),
      enabled: true,
    });
    setProviders(res.providers);
    setKeyInput("");
    setTestResult("Saved encrypted provider configuration on server.");
    onProvidersChanged?.();
  };

  const handleClearKey = async () => {
    if (!activeProvObj) return;
    const res = await api.updateProvider(activeProvObj.id, { api_key: "__CLEAR__" });
    setProviders(res.providers);
    setTestResult("Cleared stored API key.");
    onProvidersChanged?.();
  };

  const handleTestConnection = async () => {
    if (!activeProvObj) return;
    setTesting(true);
    setTestResult(null);
    try {
      const res = await api.testProvider(activeProvObj.id);
      setTestResult(
        res.ok
          ? `Connected (${res.latency_ms ?? 0} ms): ${res.detail || res.status}`
          : `Unreachable / Unconfigured: ${res.detail || res.status}`
      );
    } catch (err: any) {
      setTestResult(`Error: ${err.message}`);
    } finally {
      setTesting(false);
    }
  };

  const handleCreatePersona = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newPName.trim() || !newPInst.trim()) return;
    const created = await api.createPersonality({
      name: newPName,
      system_instructions: newPInst,
      domain_focus: newPDomain,
      creativity: settings.creativity,
      verbosity: settings.verbosity,
    });
    setNewPName("");
    setNewPInst("");
    setNewPDomain("");
    setAddingPersona(false);
    const perRes = await api.listPersonalities();
    setCustomPersonas(perRes.custom);
    onUpdate({ persona: created.slug });
  };

  const handleDeletePersona = async (id: string) => {
    await api.deletePersonality(id);
    const perRes = await api.listPersonalities();
    setCustomPersonas(perRes.custom);
    onUpdate({ persona: "nexus" });
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-lg h-full bg-[#070c1a] border-l border-cyan-500/25 p-5 overflow-y-auto space-y-5 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
            <Sliders className="w-4 h-4" />
            Xyvora Intelligence & Provider Settings
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Section Tabs */}
        <div className="grid grid-cols-3 gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
          <button
            type="button"
            onClick={() => setTab("providers")}
            className={`py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 ${
              tab === "providers" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
            }`}
          >
            <Cpu className="w-3.5 h-3.5" /> AI Providers
          </button>
          <button
            type="button"
            onClick={() => setTab("persona")}
            className={`py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 ${
              tab === "persona" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
            }`}
          >
            <BrainCircuit className="w-3.5 h-3.5" /> Personalities
          </button>
          <button
            type="button"
            onClick={() => setTab("memory")}
            className={`py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 ${
              tab === "memory" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
            }`}
          >
            <Database className="w-3.5 h-3.5" /> Memory & Voice
          </button>
        </div>

        {tab === "providers" && (
          <div className="space-y-4">
            <p className="text-[11px] text-slate-400">
              API keys are encrypted at rest on the server and never sent back in plaintext. Unconfigured providers are marked honestly.
            </p>

            <div className="grid grid-cols-2 gap-2">
              {providers.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handleSelectProv(p)}
                  className={`p-2.5 rounded-xl border text-left transition ${
                    selectedPid === p.id
                      ? "bg-cyan-500/15 border-cyan-400/60"
                      : "bg-slate-900/70 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-xs font-bold text-white truncate">{p.name}</span>
                    <span
                      className={`w-2 h-2 rounded-full shrink-0 ${
                        p.configured ? "bg-emerald-400" : "bg-amber-400/70"
                      }`}
                    />
                  </div>
                  <div className="text-[10px] font-mono text-slate-400 mt-0.5">
                    {p.configured ? (p.has_api_key ? p.masked_api_key : "Ready") : "Needs API Key"}
                  </div>
                </button>
              ))}
            </div>

            {activeProvObj && (
              <form onSubmit={handleSaveProvider} className="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xs font-bold text-cyan-200">{activeProvObj.name}</div>
                    <div className="text-[11px] text-slate-400">{activeProvObj.description}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onUpdate({ active_provider: activeProvObj.id, active_model: modelInput || activeProvObj.default_model })}
                    className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold ${
                      settings.active_provider === activeProvObj.id
                        ? "bg-emerald-500/20 border border-emerald-500/40 text-emerald-200"
                        : "bg-cyan-500/20 border border-cyan-500/40 text-cyan-200"
                    }`}
                  >
                    {settings.active_provider === activeProvObj.id ? "Selected Default" : "Set Active"}
                  </button>
                </div>

                {activeProvObj.id !== "xyvora_local" && (
                  <>
                    <div>
                      <label className="block text-[11px] text-slate-400 mb-1">Base URL</label>
                      <input
                        type="text"
                        value={baseUrlInput}
                        onChange={(e) => setBaseUrlInput(e.target.value)}
                        className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-white font-mono"
                      />
                    </div>
                    {activeProvObj.requires_api_key && (
                      <div>
                        <label className="block text-[11px] text-slate-400 mb-1">
                          API Key {activeProvObj.masked_api_key ? `(Stored: ${activeProvObj.masked_api_key})` : "(Not set)"}
                        </label>
                        <div className="flex gap-1.5">
                          <input
                            type="password"
                            value={keyInput}
                            onChange={(e) => setKeyInput(e.target.value)}
                            placeholder="Paste new API key to encrypt & store..."
                            className="flex-1 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-white font-mono"
                          />
                          {activeProvObj.has_api_key && (
                            <button
                              type="button"
                              onClick={handleClearKey}
                              className="px-2.5 py-1.5 rounded-lg bg-rose-500/15 border border-rose-500/35 text-[11px] text-rose-200"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                      </div>
                    )}
                    <div>
                      <label className="block text-[11px] text-slate-400 mb-1">Default Model ID</label>
                      <input
                        type="text"
                        value={modelInput}
                        onChange={(e) => setModelInput(e.target.value)}
                        className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-white font-mono"
                      />
                    </div>
                    <div className="flex gap-2 pt-1">
                      <button
                        type="submit"
                        className="flex-1 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950"
                      >
                        Save Provider Config
                      </button>
                      <button
                        type="button"
                        onClick={handleTestConnection}
                        disabled={testing}
                        className="px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-xs text-cyan-200"
                      >
                        {testing ? "Testing..." : "Test Connection"}
                      </button>
                    </div>
                  </>
                )}

                {testResult && (
                  <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-cyan-200 flex items-center gap-1.5">
                    <CheckCircle2 className="w-3.5 h-3.5 shrink-0 text-cyan-400" /> {testResult}
                  </div>
                )}
              </form>
            )}
          </div>
        )}

        {tab === "persona" && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">Built-In & Custom Personalities</span>
              <button
                type="button"
                onClick={() => setAddingPersona((s) => !s)}
                className="text-[11px] text-cyan-300 flex items-center gap-1"
              >
                <Plus className="w-3.5 h-3.5" /> Create Persona
              </button>
            </div>

            {addingPersona && (
              <form onSubmit={handleCreatePersona} className="p-3 rounded-xl bg-slate-900 border border-cyan-500/35 space-y-2">
                <input
                  type="text"
                  required
                  value={newPName}
                  onChange={(e) => setNewPName(e.target.value)}
                  placeholder="Persona Name (e.g., Rust Kernel Mentor)"
                  className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-white"
                />
                <textarea
                  rows={2}
                  required
                  value={newPInst}
                  onChange={(e) => setNewPInst(e.target.value)}
                  placeholder="System behavior & communication style..."
                  className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-white"
                />
                <input
                  type="text"
                  value={newPDomain}
                  onChange={(e) => setNewPDomain(e.target.value)}
                  placeholder="Domain Focus (optional)"
                  className="w-full px-2.5 py-1.5 rounded bg-slate-950 border border-slate-700 text-xs text-white"
                />
                <button type="submit" className="w-full py-1.5 rounded bg-cyan-500 text-slate-950 text-xs font-bold">
                  Save Custom Personality
                </button>
              </form>
            )}

            <div className="grid grid-cols-1 gap-2">
              {BUILTIN_PERSONAS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => onUpdate({ persona: p.id })}
                  className={`text-left px-3 py-2 rounded-xl border transition ${
                    settings.persona === p.id
                      ? "bg-cyan-500/15 border-cyan-400/60 text-white"
                      : "bg-slate-900/70 border-slate-800 text-slate-300"
                  }`}
                >
                  <div className="text-xs font-bold text-cyan-200">{p.label}</div>
                  <div className="text-[11px] text-slate-400">{p.desc}</div>
                </button>
              ))}
              {customPersonas.map((cp) => (
                <div
                  key={cp.id}
                  onClick={() => onUpdate({ persona: cp.slug })}
                  className={`px-3 py-2 rounded-xl border cursor-pointer flex items-center justify-between ${
                    settings.persona === cp.slug
                      ? "bg-indigo-500/20 border-indigo-400/60 text-white"
                      : "bg-slate-900/70 border-slate-800 text-slate-300"
                  }`}
                >
                  <div>
                    <div className="text-xs font-bold text-indigo-200">{cp.name} (Custom)</div>
                    <div className="text-[11px] text-slate-400 line-clamp-1">{cp.system_instructions}</div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeletePersona(cp.id);
                    }}
                    className="text-slate-500 hover:text-rose-400"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>

            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Creativity / Temperature</span>
                  <span className="font-mono text-cyan-300">{settings.creativity.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={settings.creativity}
                  onChange={(e) => onUpdate({ creativity: Number(e.target.value) })}
                  className="w-full accent-cyan-400"
                />
              </div>

              <div>
                <label className="text-xs text-slate-300 block mb-1">Response Length</label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(["concise", "balanced", "comprehensive"] as const).map((v) => (
                    <button
                      key={v}
                      type="button"
                      onClick={() => onUpdate({ verbosity: v })}
                      className={`py-1.5 rounded-lg text-xs capitalize border ${
                        settings.verbosity === v
                          ? "bg-indigo-500/20 border-indigo-400/50 text-indigo-200 font-semibold"
                          : "bg-slate-900 border-slate-800 text-slate-400"
                      }`}
                    >
                      {v}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Custom System Instructions</label>
                <textarea
                  rows={2}
                  value={settings.custom_instructions || ""}
                  onChange={(e) => onUpdate({ custom_instructions: e.target.value })}
                  placeholder="Global instructions applied across conversations..."
                  className="w-full p-2.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
                />
              </div>
            </div>
          </div>
        )}

        {tab === "memory" && (
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-300">Associative Memory Recall Threshold</span>
                <span className="font-mono text-emerald-300">{settings.memory_threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={0.65}
                step={0.02}
                value={settings.memory_threshold}
                onChange={(e) => onUpdate({ memory_threshold: Number(e.target.value) })}
                className="w-full accent-emerald-400"
              />
            </div>

            <div className="space-y-2">
              {[
                {
                  key: "auto_learn_memory" as const,
                  label: "Auto-Extract Personal Facts From Conversations",
                  val: settings.auto_learn_memory,
                },
                {
                  key: "require_memory_confirmation" as const,
                  label: "Memory Approval Mode (Ask Before Permanently Saving)",
                  val: Boolean(settings.require_memory_confirmation),
                },
                {
                  key: "show_thought_trace" as const,
                  label: "Show Concise Cognitive Reasoning Trace",
                  val: settings.show_thought_trace,
                },
                {
                  key: "enable_tool_calling" as const,
                  label: "Enable Autonomous Multi-Step Tool Calling",
                  val: settings.enable_tool_calling !== false,
                },
              ].map((sw) => (
                <label
                  key={sw.key}
                  className="flex items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-900/70 border border-slate-800 cursor-pointer"
                >
                  <span className="text-xs text-slate-200">{sw.label}</span>
                  <input
                    type="checkbox"
                    checked={sw.val}
                    onChange={(e) => onUpdate({ [sw.key]: e.target.checked })}
                    className="w-4 h-4 accent-cyan-400"
                  />
                </label>
              ))}
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <div className="text-xs font-semibold text-cyan-200 flex items-center gap-1.5">
                <Volume2 className="w-3.5 h-3.5" /> Voice & Speech Engine (Honest Capabilities)
              </div>
              <p className="text-[11px] text-slate-400">
                Browser Mode uses your browser's built-in Web Speech API (STT/TTS) locally—not a real-time neural voice model. Cloud Neural TTS uses OpenAI `tts-1` when an OpenAI API key is configured.
              </p>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => onUpdate({ voice_provider: "browser" })}
                  className={`py-1.5 rounded-lg text-xs border ${
                    (settings.voice_provider || "browser") === "browser"
                      ? "bg-cyan-500/20 border-cyan-500/40 text-cyan-200"
                      : "bg-slate-950 border-slate-800 text-slate-400"
                  }`}
                >
                  Browser Speech API
                </button>
                <button
                  type="button"
                  onClick={() => onUpdate({ voice_provider: "neural" })}
                  className={`py-1.5 rounded-lg text-xs border ${
                    settings.voice_provider === "neural"
                      ? "bg-indigo-500/20 border-indigo-400/40 text-indigo-200"
                      : "bg-slate-950 border-slate-800 text-slate-400"
                  }`}
                >
                  Cloud Neural TTS
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
