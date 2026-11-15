import React, { useEffect, useState } from "react";
import { Activity, AlertTriangle, Database, RefreshCw, ShieldAlert, Users, Zap } from "lucide-react";
import { api } from "../api";
import { AdminAnalytics } from "../types";

export const AdminAnalyticsView: React.FC = () => {
  const [data, setData] = useState<AdminAnalytics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getAdminAnalytics();
      setData(res);
    } catch (err: any) {
      setError(err.message || "Admin access required");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (loading) {
    return <div className="p-8 text-xs text-slate-400">Loading platform analytics...</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 max-w-xl mx-auto">
        <div className="p-4 rounded-2xl bg-rose-500/15 border border-rose-500/35 text-xs text-rose-200 flex items-center gap-2.5">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <div>
            <div className="font-bold">Role-Protected Admin Interface</div>
            <div className="mt-0.5">{error}</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4 md:p-6 space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" /> Developer Analytics & Performance Telemetry
          </h2>
          <p className="text-xs text-slate-400">
            Role-protected operational metrics across users, latency, token usage, sandbox isolation, and security audit logs
          </p>
        </div>
        <button
          type="button"
          onClick={load}
          className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-cyan-300 flex items-center gap-1.5"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* Top KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="glass-panel rounded-2xl p-4">
          <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5 text-cyan-400" /> Registered Users
          </div>
          <div className="text-xl font-extrabold text-white mt-1 font-mono">{data.total_registered_users}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">{data.active_users_24h} active in 24h</div>
        </div>

        <div className="glass-panel rounded-2xl p-4">
          <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-indigo-400" /> Requests & Latency
          </div>
          <div className="text-xl font-extrabold text-white mt-1 font-mono">{data.requests_24h}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">Avg {data.avg_latency_ms} ms ({data.total_requests} total)</div>
        </div>

        <div className="glass-panel rounded-2xl p-4">
          <div className="text-[11px] text-slate-400">Token Throughput & Errors</div>
          <div className="text-xl font-extrabold text-emerald-300 mt-1 font-mono">
            {data.total_estimated_tokens.toLocaleString()}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            Failed: {data.failed_requests} | Provider Err: {data.provider_errors}
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-4">
          <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-amber-400" /> Sandbox & Storage
          </div>
          <div className="text-xl font-extrabold text-amber-300 mt-1 font-mono">
            {(data.storage.db_size_bytes / 1024).toFixed(1)} KB
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">
            Sandbox runs: {data.sandbox_executions} ({data.sandbox_blocked_or_failed} blocked/err)
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Tool Execution Statistics */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <h3 className="text-xs font-bold text-cyan-300 uppercase tracking-wider">Tool & Extension Execution Stats</h3>
          {data.tool_stats.length === 0 ? (
            <p className="text-xs text-slate-400">No tool invocations logged yet.</p>
          ) : (
            <div className="space-y-2">
              {data.tool_stats.map((t) => (
                <div key={t.tool_name} className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-xs">
                  <span className="font-mono text-slate-200">{t.tool_name}</span>
                  <span className="font-mono text-cyan-300">
                    {t.calls} calls • avg {t.avg_ms} ms
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Security Audit Logs */}
        <div className="glass-panel rounded-2xl p-4 space-y-3">
          <h3 className="text-xs font-bold text-amber-300 uppercase tracking-wider flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4" /> Recent Security & Auth Audit Log
          </h3>
          {data.recent_security_events.length === 0 ? (
            <p className="text-xs text-slate-400">No security audit events recorded yet.</p>
          ) : (
            <div className="space-y-1.5 max-h-72 overflow-y-auto">
              {data.recent_security_events.map((ev, idx) => (
                <div key={idx} className="p-2 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-semibold text-slate-200">{ev.event_type}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        ev.severity === "warn" ? "bg-amber-500/20 text-amber-300" : "bg-cyan-500/15 text-cyan-300"
                      }`}
                    >
                      {ev.severity}
                    </span>
                  </div>
                  <div className="text-slate-400 mt-0.5 truncate">{ev.detail || ev.resource}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
