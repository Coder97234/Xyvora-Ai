import React from "react";
import { BrainCircuit, Globe, Sliders, X } from "lucide-react";
import { CognitiveSettings } from "../types";

interface CognitiveControlsProps {
  open: boolean;
  onClose: () => void;
  settings: CognitiveSettings;
  onUpdate: (patch: Partial<CognitiveSettings>) => void;
}

const PERSONAS: {
  id: CognitiveSettings["persona"];
  label: string;
  desc: string;
}[] = [
  { id: "nexus", label: "Nexus", desc: "Balanced polymath — warm, analytical, clear" },
  { id: "axiom", label: "Axiom", desc: "Rigorous scientist — formal derivations & deep theory" },
  { id: "muse", label: "Muse", desc: "Creative visionary — evocative metaphors & ideation" },
  { id: "forge", label: "Forge", desc: "Systems engineer — code-first & pragmatic" },
  { id: "socrates", label: "Socrates", desc: "Interactive tutor — guides with reflective inquiry" },
];

export const CognitiveControls: React.FC<CognitiveControlsProps> = ({
  open,
  onClose,
  settings,
  onUpdate,
}) => {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md h-full bg-[#070c1a] border-l border-cyan-500/25 p-5 overflow-y-auto space-y-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
            <Sliders className="w-4 h-4" />
            Xyvora Cognitive Brain Controls
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Persona Selection */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <BrainCircuit className="w-3.5 h-3.5 text-cyan-400" />
            Active Cognitive Persona
          </label>
          <div className="grid grid-cols-1 gap-2">
            {PERSONAS.map((p) => {
              const active = settings.persona === p.id;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => onUpdate({ persona: p.id })}
                  className={`text-left px-3 py-2.5 rounded-xl border transition ${
                    active
                      ? "bg-cyan-500/15 border-cyan-400/60 text-white"
                      : "bg-slate-900/70 border-slate-800 text-slate-300 hover:border-slate-700"
                  }`}
                >
                  <div className="text-xs font-bold text-cyan-200">{p.label}</div>
                  <div className="text-[11px] text-slate-400">{p.desc}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Sliders */}
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-300">Divergent Creativity</span>
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
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-300">Associative Memory Threshold</span>
              <span className="font-mono text-emerald-300">
                {settings.memory_threshold.toFixed(2)}
              </span>
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
            <div className="text-[10.5px] text-slate-500">
              Lower = broader associative recall; Higher = strict exact fact match.
            </div>
          </div>

          <div>
            <label className="text-xs text-slate-300 block mb-1.5">Response Depth</label>
            <div className="grid grid-cols-3 gap-1.5">
              {(["concise", "balanced", "comprehensive"] as const).map((v) => (
                <button
                  key={v}
                  type="button"
                  onClick={() => onUpdate({ verbosity: v })}
                  className={`py-1.5 rounded-lg text-xs capitalize border transition ${
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
        </div>

        {/* Behavioral Switches */}
        <div className="space-y-2.5 pt-1 border-t border-slate-800">
          {[
            {
              key: "auto_learn_memory" as const,
              label: "Auto-Learn Personal Facts From Chat",
              val: settings.auto_learn_memory,
            },
            {
              key: "show_thought_trace" as const,
              label: "Display Live Neural Thought Trace",
              val: settings.show_thought_trace,
            },
            {
              key: "voice_enabled" as const,
              label: "Auto-Speak Replies (Neural TTS)",
              val: settings.voice_enabled,
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

        {/* Optional External OpenAI-Compatible Bridge */}
        <div className="space-y-2.5 pt-2 border-t border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Globe className="w-3.5 h-3.5 text-indigo-400" />
              Optional External LLM Bridge
            </span>
            <input
              type="checkbox"
              checked={settings.external_llm_enabled}
              onChange={(e) => onUpdate({ external_llm_enabled: e.target.checked })}
              className="w-4 h-4 accent-indigo-400"
            />
          </div>
          <p className="text-[11px] text-slate-500">
            Xyvora runs 100% self-contained locally by default. Optionally connect an OpenAI-compatible API augmented by Xyvora's Memory Graph.
          </p>
          {settings.external_llm_enabled && (
            <div className="space-y-2 pt-1">
              <input
                type="text"
                value={settings.external_llm_base_url}
                onChange={(e) => onUpdate({ external_llm_base_url: e.target.value })}
                placeholder="https://api.openai.com/v1"
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs text-white"
              />
              <input
                type="password"
                value={settings.external_llm_api_key}
                onChange={(e) => onUpdate({ external_llm_api_key: e.target.value })}
                placeholder="API Key (sk-...)"
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs text-white"
              />
              <input
                type="text"
                value={settings.external_llm_model}
                onChange={(e) => onUpdate({ external_llm_model: e.target.value })}
                placeholder="Model (e.g., gpt-4o-mini)"
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs text-white"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
