import React, { useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Circle,
  Play,
  Terminal,
  TrendingUp,
} from "lucide-react";
import {
  AgentTask,
  ChartArtifact,
  ChatArtifact,
  CodeExecutionArtifact,
  FunctionPlotArtifact,
} from "../types";
import { api } from "../api";

export const FunctionPlotCard: React.FC<{ artifact: FunctionPlotArtifact }> = ({ artifact }) => {
  const [showDeriv, setShowDeriv] = useState(true);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const W = 540;
  const H = 220;
  const PAD_X = 36;
  const PAD_Y = 22;

  const pts = artifact.points;
  const xs = pts.map((p) => p.x);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);

  const allY: number[] = [];
  for (const p of pts) {
    if (p.y !== null) allY.push(p.y);
    if (showDeriv && p.dy !== null) allY.push(p.dy);
  }
  let yMin = allY.length ? Math.min(...allY) : -5;
  let yMax = allY.length ? Math.max(...allY) : 5;
  if (Math.abs(yMax - yMin) < 1e-4) {
    yMin -= 2;
    yMax += 2;
  }
  const marginY = (yMax - yMin) * 0.1;
  yMin -= marginY;
  yMax += marginY;

  const mapX = (x: number) => PAD_X + ((x - xMin) / (xMax - xMin || 1)) * (W - PAD_X * 2);
  const mapY = (y: number) => H - PAD_Y - ((y - yMin) / (yMax - yMin || 1)) * (H - PAD_Y * 2);

  const buildPath = (key: "y" | "dy") => {
    let d = "";
    let penDown = false;
    for (const p of pts) {
      const val = p[key];
      if (val === null) {
        penDown = false;
        continue;
      }
      const sx = mapX(p.x).toFixed(1);
      const sy = mapY(val).toFixed(1);
      if (!penDown) {
        d += `M ${sx} ${sy} `;
        penDown = true;
      } else {
        d += `L ${sx} ${sy} `;
      }
    }
    return d;
  };

  const zeroY = yMin <= 0 && yMax >= 0 ? mapY(0) : null;
  const zeroX = xMin <= 0 && xMax >= 0 ? mapX(0) : null;
  const hovered = hoverIdx !== null ? pts[hoverIdx] : null;

  return (
    <div className="mt-3 rounded-xl border border-cyan-500/25 bg-slate-950/90 p-3.5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-cyan-100 font-mono">{artifact.title}</span>
        </div>
        <button
          type="button"
          onClick={() => setShowDeriv((s) => !s)}
          className={`text-[11px] px-2.5 py-1 rounded-full border transition ${
            showDeriv
              ? "border-indigo-400/50 bg-indigo-500/20 text-indigo-200"
              : "border-slate-700 bg-slate-900 text-slate-400"
          }`}
        >
          {showDeriv ? "Showing f'(x)" : "Show f'(x)"}
        </button>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-48 rounded-lg bg-[#050914] border border-slate-800/80 cursor-crosshair select-none"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * W;
          const ratio = Math.max(0, Math.min(1, (relX - PAD_X) / (W - PAD_X * 2)));
          const idx = Math.round(ratio * (pts.length - 1));
          setHoverIdx(idx);
        }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Subtle horizontal grid */}
        {[0.25, 0.5, 0.75].map((r) => {
          const gy = PAD_Y + r * (H - PAD_Y * 2);
          return (
            <line
              key={r}
              x1={PAD_X}
              x2={W - PAD_X}
              y1={gy}
              y2={gy}
              stroke="rgba(148, 163, 184, 0.09)"
              strokeWidth="1"
            />
          );
        })}

        {/* Coordinate Zero Axes */}
        {zeroY !== null && (
          <line
            x1={PAD_X}
            x2={W - PAD_X}
            y1={zeroY}
            y2={zeroY}
            stroke="rgba(148, 163, 184, 0.35)"
            strokeWidth="1"
          />
        )}
        {zeroX !== null && (
          <line
            x1={zeroX}
            x2={zeroX}
            y1={PAD_Y}
            y2={H - PAD_Y}
            stroke="rgba(148, 163, 184, 0.35)"
            strokeWidth="1"
          />
        )}

        {/* Derivative curve f'(x) */}
        {showDeriv && (
          <path
            d={buildPath("dy")}
            fill="none"
            stroke="#818cf8"
            strokeWidth="1.75"
            strokeDasharray="4 3"
          />
        )}

        {/* Primary curve f(x) */}
        <path d={buildPath("y")} fill="none" stroke="#22d3ee" strokeWidth="2.4" />

        {/* Interactive hover crosshair */}
        {hovered && (
          <g>
            <line
              x1={mapX(hovered.x)}
              x2={mapX(hovered.x)}
              y1={PAD_Y}
              y2={H - PAD_Y}
              stroke="rgba(34, 211, 238, 0.45)"
              strokeWidth="1"
            />
            {hovered.y !== null && (
              <circle
                cx={mapX(hovered.x)}
                cy={mapY(hovered.y)}
                r="4.5"
                fill="#22d3ee"
                stroke="#050811"
                strokeWidth="1.5"
              />
            )}
            {showDeriv && hovered.dy !== null && (
              <circle
                cx={mapX(hovered.x)}
                cy={mapY(hovered.dy)}
                r="3.8"
                fill="#818cf8"
                stroke="#050811"
                strokeWidth="1.5"
              />
            )}
          </g>
        )}
      </svg>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-cyan-300">
            <span className="w-2.5 h-0.5 bg-cyan-400 inline-block" /> f(x) = {artifact.expression}
          </span>
          {showDeriv && (
            <span className="flex items-center gap-1.5 text-indigo-300">
              <span className="w-2.5 h-0.5 bg-indigo-400 inline-block" /> f'(x) = {artifact.derivative}
            </span>
          )}
        </div>
        {hovered ? (
          <span className="text-emerald-300 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            x = {hovered.x} | f(x) = {hovered.y ?? "undef"}
            {showDeriv ? ` | f'(x) = ${hovered.dy ?? "undef"}` : ""}
          </span>
        ) : (
          <span className="text-slate-500">Hover curve to inspect coordinates</span>
        )}
      </div>
    </div>
  );
};

export const DataChartCard: React.FC<{ artifact: ChartArtifact }> = ({ artifact }) => {
  const [mode, setMode] = useState<"bar" | "line" | "area">(artifact.chart_type || "bar");
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const data = artifact.data || [];
  if (!data.length) return null;

  const vals = data.map((d) => d.value);
  const minVal = Math.min(0, ...vals);
  const maxVal = Math.max(...vals, 1);
  const span = maxVal - minVal || 1;

  const W = 520;
  const H = 195;
  const PAD_X = 34;
  const PAD_Y = 24;

  const mapX = (i: number) =>
    data.length === 1
      ? W / 2
      : PAD_X + (i / (data.length - 1)) * (W - PAD_X * 2);
  const mapY = (v: number) => H - PAD_Y - ((v - minVal) / span) * (H - PAD_Y * 2);

  const linePoints = data.map((d, i) => `${mapX(i).toFixed(1)},${mapY(d.value).toFixed(1)}`).join(" ");
  const areaPolygon = `${mapX(0).toFixed(1)},${mapY(minVal).toFixed(1)} ${linePoints} ${mapX(data.length - 1).toFixed(1)},${mapY(minVal).toFixed(1)}`;

  return (
    <div className="mt-3 rounded-xl border border-indigo-500/25 bg-slate-950/90 p-3.5 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <div className="flex items-center gap-2">
          {mode === "bar" ? (
            <BarChart3 className="w-4 h-4 text-cyan-400" />
          ) : (
            <TrendingUp className="w-4 h-4 text-indigo-400" />
          )}
          <span className="text-xs font-semibold text-slate-100">{artifact.title}</span>
        </div>
        <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800">
          {(["bar", "line", "area"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-md transition ${
                mode === m
                  ? "bg-cyan-500/20 text-cyan-200 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {mode === "bar" ? (
        <div className="grid grid-cols-1 gap-1.5 pt-1">
          <div className="flex items-end gap-2 h-40 px-2 pt-4 pb-1 bg-[#050914] rounded-lg border border-slate-800/80">
            {data.map((item, idx) => {
              const pct = Math.max(4, ((item.value - minVal) / span) * 100);
              const active = hoverIdx === idx;
              return (
                <div
                  key={`${item.label}-${idx}`}
                  onMouseEnter={() => setHoverIdx(idx)}
                  onMouseLeave={() => setHoverIdx(null)}
                  className="flex-1 flex flex-col items-center justify-end h-full group cursor-pointer"
                >
                  <span
                    className={`text-[10px] font-mono mb-1 transition ${
                      active ? "text-cyan-200 opacity-100" : "text-slate-400 opacity-75"
                    }`}
                  >
                    {item.value}
                  </span>
                  <div
                    style={{ height: `${pct}%` }}
                    className={`w-full max-w-[38px] rounded-t-md transition-all duration-200 ${
                      active
                        ? "bg-gradient-to-t from-indigo-500 to-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.45)]"
                        : "bg-gradient-to-t from-indigo-600/75 to-cyan-500/75"
                    }`}
                  />
                  <span className="text-[10px] text-slate-400 truncate max-w-[52px] mt-1">
                    {item.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-44 rounded-lg bg-[#050914] border border-slate-800/80"
        >
          <defs>
            <linearGradient id="xyAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(34, 211, 238, 0.38)" />
              <stop offset="100%" stopColor="rgba(99, 102, 241, 0.02)" />
            </linearGradient>
          </defs>
          {mode === "area" && <polygon points={areaPolygon} fill="url(#xyAreaGrad)" />}
          <polyline
            fill="none"
            stroke="#22d3ee"
            strokeWidth="2.4"
            points={linePoints}
          />
          {data.map((d, i) => (
            <g
              key={i}
              onMouseEnter={() => setHoverIdx(i)}
              onMouseLeave={() => setHoverIdx(null)}
              className="cursor-pointer"
            >
              <circle
                cx={mapX(i)}
                cy={mapY(d.value)}
                r={hoverIdx === i ? 5.2 : 3.4}
                fill={hoverIdx === i ? "#34d399" : "#22d3ee"}
                stroke="#050811"
                strokeWidth="1.5"
              />
              {(data.length <= 12 || i % Math.ceil(data.length / 8) === 0) && (
                <text
                  x={mapX(i)}
                  y={H - 6}
                  textAnchor="middle"
                  fill="#94a3b8"
                  fontSize="9.5"
                >
                  {d.label}
                </text>
              )}
            </g>
          ))}
        </svg>
      )}

      {hoverIdx !== null && data[hoverIdx] && (
        <div className="mt-2 text-[11px] font-mono text-cyan-300">
          Selected: <strong>{data[hoverIdx].label}</strong> → {data[hoverIdx].value}
        </div>
      )}
    </div>
  );
};

export const CodeExecutionCard: React.FC<{
  artifact: CodeExecutionArtifact;
  onOpenInSandbox?: (code: string) => void;
}> = ({ artifact, onOpenInSandbox }) => {
  return (
    <div className="mt-3 rounded-xl border border-emerald-500/25 bg-slate-950/95 p-3 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-semibold text-slate-200">{artifact.title}</span>
          <span
            className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
              artifact.ok
                ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                : "bg-rose-500/15 text-rose-300 border border-rose-500/30"
            }`}
          >
            {artifact.ok ? `SANDBOX OK • ${artifact.duration_ms}ms` : "SANDBOX ERROR"}
          </span>
        </div>
        {onOpenInSandbox && (
          <button
            type="button"
            onClick={() => onOpenInSandbox(artifact.code)}
            className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-200 border border-cyan-500/30 transition"
          >
            <Play className="w-3 h-3" /> Edit & Re-Run in AI Lab
          </button>
        )}
      </div>
      {artifact.stdout && (
        <pre className="text-xs font-mono text-emerald-200/95 bg-[#040812] p-2.5 rounded-lg border border-slate-800 overflow-x-auto max-h-48">
          {artifact.stdout}
        </pre>
      )}
      {artifact.stderr && (
        <pre className="mt-2 text-xs font-mono text-rose-300 bg-rose-950/30 p-2.5 rounded-lg border border-rose-500/30 overflow-x-auto">
          {artifact.stderr}
        </pre>
      )}
    </div>
  );
};

export const TaskBoardWidget: React.FC<{
  tasks: AgentTask[];
  title?: string;
  onTasksChanged?: () => void;
}> = ({ tasks: initialTasks, title = "Agent Task Board", onTasksChanged }) => {
  const [items, setItems] = useState<AgentTask[]>(initialTasks);

  const toggle = async (task: AgentTask) => {
    const nextStatus = task.status === "done" ? "todo" : "done";
    const updated = await api.patchTask(task.id, { status: nextStatus });
    setItems((prev) => prev.map((t) => (t.id === task.id ? updated : t)));
    if (onTasksChanged) onTasksChanged();
  };

  return (
    <div className="mt-3 rounded-xl border border-amber-500/25 bg-slate-950/90 p-3.5">
      <div className="text-xs font-semibold text-amber-200 mb-2">{title}</div>
      {items.length === 0 ? (
        <div className="text-xs text-slate-400">No tasks queued yet.</div>
      ) : (
        <div className="space-y-1.5">
          {items.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => toggle(t)}
              className="w-full flex items-center justify-between gap-2 text-left px-2.5 py-2 rounded-lg bg-slate-900/80 hover:bg-slate-800/80 border border-slate-800 transition"
            >
              <div className="flex items-center gap-2 min-w-0">
                {t.status === "done" ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                ) : (
                  <Circle className="w-4 h-4 text-slate-500 shrink-0" />
                )}
                <span
                  className={`text-xs truncate ${
                    t.status === "done" ? "line-through text-slate-500" : "text-slate-200"
                  }`}
                >
                  {t.title}
                </span>
              </div>
              <span
                className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded-full shrink-0 ${
                  t.priority === "high"
                    ? "bg-rose-500/15 text-rose-300"
                    : t.priority === "low"
                    ? "bg-slate-700/50 text-slate-300"
                    : "bg-amber-500/15 text-amber-300"
                }`}
              >
                {t.priority}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export const ArtifactsList: React.FC<{
  artifacts?: ChatArtifact[];
  onOpenCodeInSandbox?: (code: string) => void;
  onTasksChanged?: () => void;
}> = ({ artifacts, onOpenCodeInSandbox, onTasksChanged }) => {
  if (!artifacts || artifacts.length === 0) return null;

  return (
    <div className="space-y-2.5">
      {artifacts.map((art, i) => {
        if (art.type === "function_plot") {
          return <FunctionPlotCard key={i} artifact={art} />;
        }
        if (art.type === "chart") {
          return <DataChartCard key={i} artifact={art} />;
        }
        if (art.type === "code_execution") {
          return (
            <CodeExecutionCard
              key={i}
              artifact={art}
              onOpenInSandbox={onOpenCodeInSandbox}
            />
          );
        }
        if (art.type === "task_list") {
          return (
            <TaskBoardWidget
              key={i}
              tasks={art.tasks}
              title={art.title}
              onTasksChanged={onTasksChanged}
            />
          );
        }
        return null;
      })}
    </div>
  );
};
