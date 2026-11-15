import React, { useEffect, useState } from "react";
import {
  Activity,
  Cpu,
  GraduationCap,
  Layers,
  Plus,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import { api } from "../api";
import { NeuralStats } from "../types";

interface TrainingStudioProps {
  onRefreshVitals: () => void;
}

export const TrainingStudio: React.FC<TrainingStudioProps> = ({ onRefreshVitals }) => {
  const [stats, setStats] = useState<NeuralStats | null>(null);
  const [probeText, setProbeText] = useState("Differentiate and plot sin(x) * cos(x/2)");
  const [probeDist, setProbeDist] = useState<{ intent: string; probability: number }[]>([]);
  const [retraining, setRetraining] = useState(false);

  // Hyperparameters
  const [h1, setH1] = useState(128);
  const [h2, setH2] = useState(64);
  const [maxIter, setMaxIter] = useState(95);

  // Add training sample form
  const [sampleText, setSampleText] = useState("");
  const [sampleIntent, setSampleIntent] = useState("knowledge_question");

  // Add custom skill form
  const [skillTrigger, setSkillTrigger] = useState("");
  const [skillResponse, setSkillResponse] = useState("");

  const loadStats = async () => {
    const s = await api.getNeuralStats();
    setStats(s);
    onRefreshVitals();
  };

  useEffect(() => {
    loadStats().catch(console.error);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (!probeText.trim()) {
        setProbeDist([]);
        return;
      }
      api
        .predictIntent(probeText)
        .then((res) => setProbeDist(res.distribution))
        .catch(console.error);
    }, 180);
    return () => clearTimeout(timer);
  }, [probeText, stats?.last_trained_at]);

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      await api.retrainNeural(h1, h2, maxIter);
      await loadStats();
    } finally {
      setRetraining(false);
    }
  };

  const handleAddSample = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sampleText.trim() || !sampleIntent.trim()) return;
    await api.addTrainingSample(sampleText, sampleIntent);
    setSampleText("");
    await handleRetrain();
  };

  const handleAddSkill = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!skillTrigger.trim() || !skillResponse.trim()) return;
    await api.addCustomSkill(skillTrigger, skillResponse);
    setSkillTrigger("");
    setSkillResponse("");
    await loadStats();
  };

  const renderLossCurve = (curve: number[]) => {
    if (!curve || curve.length < 2) return null;
    const W = 500;
    const H = 165;
    const PX = 34;
    const PY = 18;
    const maxL = Math.max(...curve, 0.1);
    const minL = Math.min(...curve, 0);
    const span = maxL - minL || 1;
    const pts = curve
      .map((v, i) => {
        const x = PX + (i / (curve.length - 1)) * (W - PX * 2);
        const y = H - PY - ((v - minL) / span) * (H - PY * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-40 rounded-xl bg-[#040812] border border-slate-800">
        <polyline fill="none" stroke="#22d3ee" strokeWidth="2.3" points={pts} />
        <text x={PX} y={14} fill="#94a3b8" fontSize="10" fontFamily="monospace">
          Start Loss: {curve[0].toFixed(3)}
        </text>
        <text x={W - PX} y={H - 5} textAnchor="end" fill="#34d399" fontSize="10" fontFamily="monospace">
          Epoch {curve.length}: {curve[curve.length - 1].toFixed(4)}
        </text>
      </svg>
    );
  };

  return (
    <div className="h-full overflow-y-auto px-4 md:px-8 py-6 space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Live Neural Cortex & Backpropagation Training Studio
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Probe live softmax intent activations, teach new behaviors, and retrain Xyvora's Multi-Layer Perceptron in real time.
          </p>
        </div>
        <button
          type="button"
          disabled={retraining}
          onClick={handleRetrain}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-400 hover:to-cyan-400 disabled:opacity-50 text-slate-950 font-bold text-xs shadow-lg transition"
        >
          <Zap className={`w-4 h-4 ${retraining ? "animate-spin" : ""}`} />
          {retraining ? "Running Backpropagation..." : "Retrain Neural Cortex"}
        </button>
      </div>

      {/* Telemetry Stat Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="glass-panel rounded-xl p-3.5">
            <div className="text-[11px] text-slate-400">Trainable Parameters</div>
            <div className="text-xl font-bold text-cyan-300 font-mono mt-0.5">
              {stats.parameter_count.toLocaleString()}
            </div>
            <div className="text-[10.5px] text-slate-500 mt-0.5">{stats.feature_dim} TF-IDF dims</div>
          </div>
          <div className="glass-panel rounded-xl p-3.5">
            <div className="text-[11px] text-slate-400">Training / CV Accuracy</div>
            <div className="text-xl font-bold text-emerald-300 font-mono mt-0.5">
              {(stats.training_accuracy * 100).toFixed(1)}%{" "}
              <span className="text-xs text-slate-400">/ {(stats.cv_accuracy * 100).toFixed(0)}%</span>
            </div>
            <div className="text-[10.5px] text-slate-500 mt-0.5">
              {stats.sample_count} labeled utterances
            </div>
          </div>
          <div className="glass-panel rounded-xl p-3.5">
            <div className="text-[11px] text-slate-400">Cognitive Intents</div>
            <div className="text-xl font-bold text-indigo-300 font-mono mt-0.5">
              {stats.intent_count}
            </div>
            <div className="text-[10.5px] text-slate-500 mt-0.5">
              {stats.custom_skills.length} custom skills
            </div>
          </div>
          <div className="glass-panel rounded-xl p-3.5">
            <div className="text-[11px] text-slate-400">Convergence Speed</div>
            <div className="text-xl font-bold text-amber-300 font-mono mt-0.5">
              {stats.epochs_run} epochs
            </div>
            <div className="text-[10.5px] text-slate-500 mt-0.5">{stats.training_ms} ms fit time</div>
          </div>
        </div>
      )}

      {/* Loss Curve + Real-Time Neural Softmax Probe */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              Adam Cross-Entropy Loss Curve
            </span>
            <span className="text-[11px] font-mono text-slate-400">
              Layers: ({h1}, {h2})
            </span>
          </div>
          {stats && renderLossCurve(stats.loss_curve)}

          {/* Hyperparameter controls */}
          <div className="grid grid-cols-3 gap-2 pt-1 text-xs">
            <div>
              <label className="text-[10.5px] text-slate-400 block mb-1">Hidden Layer 1</label>
              <input
                type="number"
                min={32}
                max={256}
                value={h1}
                onChange={(e) => setH1(Number(e.target.value))}
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs font-mono text-white"
              />
            </div>
            <div>
              <label className="text-[10.5px] text-slate-400 block mb-1">Hidden Layer 2</label>
              <input
                type="number"
                min={16}
                max={192}
                value={h2}
                onChange={(e) => setH2(Number(e.target.value))}
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs font-mono text-white"
              />
            </div>
            <div>
              <label className="text-[10.5px] text-slate-400 block mb-1">Max Epochs</label>
              <input
                type="number"
                min={30}
                max={200}
                value={maxIter}
                onChange={(e) => setMaxIter(Number(e.target.value))}
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2.5 py-1.5 text-xs font-mono text-white"
              />
            </div>
          </div>
        </div>

        {/* Real-time Neural Intent Probe */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-indigo-200">
            <Layers className="w-4 h-4 text-indigo-400" />
            Live Neural Activation Probe
          </div>
          <input
            type="text"
            value={probeText}
            onChange={(e) => setProbeText(e.target.value)}
            placeholder="Type any utterance to watch live softmax activations..."
            className="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
          />
          <div className="space-y-2 pt-1">
            {probeDist.map((item, idx) => {
              const pct = Math.min(100, Math.max(1, item.probability * 100));
              return (
                <div key={item.intent} className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className={idx === 0 ? "text-cyan-300 font-bold" : "text-slate-300"}>
                      {item.intent}
                    </span>
                    <span className="text-slate-400">{(item.probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                    <div
                      style={{ width: `${pct}%` }}
                      className={`h-full rounded-full transition-all duration-200 ${
                        idx === 0
                          ? "bg-gradient-to-r from-cyan-400 to-indigo-400"
                          : "bg-slate-600"
                      }`}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Teach Custom Skill + Add Supervised Intent Sample */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Custom Skill Teacher */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-emerald-200">
            <GraduationCap className="w-4 h-4 text-emerald-400" />
            Teach Custom 1-Shot Skill / Rule
          </div>
          <form onSubmit={handleAddSkill} className="space-y-2.5">
            <input
              type="text"
              value={skillTrigger}
              onChange={(e) => setSkillTrigger(e.target.value)}
              placeholder="When I say... (e.g., 'status report protocol')"
              className="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <textarea
              rows={2}
              value={skillResponse}
              onChange={(e) => setSkillResponse(e.target.value)}
              placeholder="Xyvora responds... (e.g., 'All orbital systems nominal, {name}!')"
              className="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <button
              type="submit"
              className="w-full rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/40 text-emerald-200 font-semibold text-xs py-2 transition"
            >
              Teach Skill & Retrain
            </button>
          </form>

          {stats && stats.custom_skills.length > 0 && (
            <div className="space-y-1.5 pt-2 max-h-36 overflow-y-auto">
              {stats.custom_skills.map((sk) => (
                <div
                  key={sk.id}
                  className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs"
                >
                  <div className="truncate">
                    <span className="text-cyan-300 font-mono">"{sk.trigger_phrase}"</span> →{" "}
                    <span className="text-slate-300">{sk.response_template}</span>
                  </div>
                  <button
                    type="button"
                    onClick={async () => {
                      await api.deleteCustomSkill(sk.id);
                      await loadStats();
                    }}
                    className="text-slate-500 hover:text-rose-300"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Supervised Sample Teacher */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-cyan-200">
            <Plus className="w-4 h-4 text-cyan-400" />
            Add Labeled Neural Training Utterance
          </div>
          <form onSubmit={handleAddSample} className="space-y-2.5">
            <input
              type="text"
              value={sampleText}
              onChange={(e) => setSampleText(e.target.value)}
              placeholder="Example training phrase..."
              className="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
            />
            <div className="flex gap-2">
              <select
                value={sampleIntent}
                onChange={(e) => setSampleIntent(e.target.value)}
                className="flex-1 rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-slate-200"
              >
                {(stats?.intents || ["knowledge_question", "math_computation", "code_execution", "memory_store", "creative_writing"]).map(
                  (intentName) => (
                    <option key={intentName} value={intentName}>
                      {intentName}
                    </option>
                  )
                )}
              </select>
              <button
                type="submit"
                className="px-4 py-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/40 text-cyan-200 font-semibold text-xs transition"
              >
                Add & Fit
              </button>
            </div>
          </form>

          {stats && stats.user_samples.length > 0 && (
            <div className="space-y-1.5 pt-2 max-h-36 overflow-y-auto">
              {stats.user_samples.map((s) => (
                <div
                  key={s.id}
                  className="flex items-center justify-between gap-2 px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs"
                >
                  <div className="truncate">
                    <span className="text-slate-200">"{s.text}"</span>{" "}
                    <span className="text-indigo-300 font-mono">[{s.intent}]</span>
                  </div>
                  <button
                    type="button"
                    onClick={async () => {
                      await api.deleteTrainingSample(s.id);
                      await handleRetrain();
                    }}
                    className="text-slate-500 hover:text-rose-300"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
