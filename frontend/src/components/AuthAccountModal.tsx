import React, { useEffect, useState } from "react";
import {
  CheckCircle2,
  Download,
  KeyRound,
  LogIn,
  LogOut,
  Shield,
  Trash2,
  Upload,
  UserCheck,
  UserPlus,
  X,
} from "lucide-react";
import { api } from "../api";
import { StorageDiagnostics, UserProfile } from "../types";

interface AuthAccountModalProps {
  open: boolean;
  onClose: () => void;
  user?: UserProfile;
  storage?: StorageDiagnostics;
  onAuthChanged: () => void;
}

export const AuthAccountModal: React.FC<AuthAccountModalProps> = ({
  open,
  onClose,
  user,
  storage,
  onAuthChanged,
}) => {
  const [mode, setMode] = useState<"login" | "register" | "reset" | "profile">("profile");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [identifier, setIdentifier] = useState("");
  const [resetToken, setResetToken] = useState("");

  // Profile fields
  const [profName, setProfName] = useState("");
  const [profAvatar, setProfAvatar] = useState("");
  const [profBio, setProfBio] = useState("");
  const [currPw, setCurrPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [usage, setUsage] = useState<{
    total_requests: number;
    requests_last_24h: number;
    estimated_tokens_total: number;
    avg_latency_ms: number;
    sessions_count: number;
    memories_count: number;
    documents_count: number;
    projects_count: number;
  } | null>(null);

  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setErrMsg(null);
    setStatusMsg(null);
    if (!user || user.is_guest) {
      setMode("login");
    } else {
      setMode("profile");
      setProfName(user.display_name || "");
      setProfAvatar(user.avatar_url || "");
      setProfBio(user.bio || "");
    }
    api
      .getMe()
      .then((res) => setUsage(res.usage))
      .catch(() => {});
  }, [open, user]);

  if (!open) return null;

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErrMsg(null);
    try {
      await api.register({ email, username, password, display_name: displayName });
      setStatusMsg("Account created! Your private workspace is ready.");
      onAuthChanged();
      setMode("profile");
    } catch (err: any) {
      setErrMsg(err.message || "Registration failed");
    } finally {
      setBusy(false);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErrMsg(null);
    try {
      await api.login({ identifier, password });
      setStatusMsg("Signed in cleanly.");
      onAuthChanged();
      setMode("profile");
    } catch (err: any) {
      setErrMsg(err.message || "Login failed");
    } finally {
      setBusy(false);
    }
  };

  const handleLogout = async () => {
    setBusy(true);
    try {
      await api.logout();
      onAuthChanged();
      setMode("login");
    } finally {
      setBusy(false);
    }
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErrMsg(null);
    try {
      await api.updateProfile({ display_name: profName, avatar_url: profAvatar, bio: profBio });
      setStatusMsg("Profile updated.");
      onAuthChanged();
    } catch (err: any) {
      setErrMsg(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleChangePw = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErrMsg(null);
    try {
      await api.changePassword(currPw, newPw);
      setCurrPw("");
      setNewPw("");
      setStatusMsg("Password updated.");
    } catch (err: any) {
      setErrMsg(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleExportWorkspace = async () => {
    try {
      const bundle = await api.exportAccountWorkspace();
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `xyvora-workspace-${user?.username || "local"}.json`;
      a.click();
      URL.revokeObjectURL(url);
      setStatusMsg("Exported workspace backup JSON (API keys excluded for safety).");
    } catch (err: any) {
      setErrMsg(err.message);
    }
  };

  const handleImportWorkspace = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const res = await api.importAccountWorkspace(parsed);
      setStatusMsg(`Imported backup (${res.memories_imported} memories restored).`);
      onAuthChanged();
    } catch (err: any) {
      setErrMsg(err.message || "Invalid workspace JSON");
    }
  };

  const handleDeleteAccount = async () => {
    if (!window.confirm("Permanently delete your account and all private conversations, projects, and memories?")) return;
    setBusy(true);
    try {
      await api.deleteAccount();
      onAuthChanged();
      onClose();
    } catch (err: any) {
      setErrMsg(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-xl max-h-[90vh] overflow-y-auto rounded-2xl bg-[#080d1c] border border-cyan-500/30 shadow-2xl p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2.5">
            <Shield className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-sm font-bold text-white">Account & Workspace Cloud</h3>
              <p className="text-[11px] text-slate-400">
                Private per-user sessions, memories, projects, encrypted provider keys & portable backups
              </p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="p-1.5 text-slate-400 hover:text-white rounded-lg">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Mode Tabs */}
        <div className="flex gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          {user && !user.is_guest ? (
            <button
              type="button"
              onClick={() => setMode("profile")}
              className={`flex-1 py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 ${
                mode === "profile" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
              }`}
            >
              <UserCheck className="w-3.5 h-3.5" /> Signed In ({user.username})
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => setMode("login")}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 ${
              mode === "login" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
            }`}
          >
            <LogIn className="w-3.5 h-3.5" /> Sign In
          </button>
          <button
            type="button"
            onClick={() => setMode("register")}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 ${
              mode === "register" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
            }`}
          >
            <UserPlus className="w-3.5 h-3.5" /> Create Account
          </button>
          {user?.is_guest && (
            <button
              type="button"
              onClick={() => setMode("profile")}
              className={`flex-1 py-1.5 rounded-lg text-xs font-medium ${
                mode === "profile" ? "bg-cyan-500/20 text-cyan-200 border border-cyan-500/40" : "text-slate-400"
              }`}
            >
              Local Guest Stats
            </button>
          )}
        </div>

        {errMsg && (
          <div className="p-2.5 rounded-xl bg-rose-500/15 border border-rose-500/35 text-xs text-rose-200">{errMsg}</div>
        )}
        {statusMsg && (
          <div className="p-2.5 rounded-xl bg-emerald-500/15 border border-emerald-500/35 text-xs text-emerald-200 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" /> {statusMsg}
          </div>
        )}

        {mode === "login" && (
          <form onSubmit={handleLogin} className="space-y-3">
            <div>
              <label className="block text-[11px] text-slate-400 mb-1">Email or Username</label>
              <input
                type="text"
                required
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                placeholder="you@domain.com or username"
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
              />
            </div>
            <div>
              <label className="block text-[11px] text-slate-400 mb-1">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
              />
            </div>
            <button
              type="submit"
              disabled={busy}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950 hover:opacity-95 disabled:opacity-50"
            >
              {busy ? "Signing in..." : "Sign In to Private Workspace"}
            </button>
            <div className="text-right">
              <button
                type="button"
                onClick={() => setMode("reset")}
                className="text-[11px] text-cyan-300 hover:underline"
              >
                Forgot password? Reset via token
              </button>
            </div>
          </form>
        )}

        {mode === "reset" && (
          <div className="space-y-3">
            <div className="space-y-2">
              <label className="block text-[11px] text-slate-400">1. Request 15-Minute Reset Token</label>
              <div className="flex gap-2">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Account email address"
                  className="flex-1 px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
                />
                <button
                  type="button"
                  disabled={busy || !email.trim()}
                  onClick={async () => {
                    setBusy(true);
                    setErrMsg(null);
                    try {
                      const r = await api.requestPasswordReset(email);
                      setStatusMsg(r.message);
                      if (r.reset_token) setResetToken(r.reset_token);
                    } catch (err: any) {
                      setErrMsg(err.message);
                    } finally {
                      setBusy(false);
                    }
                  }}
                  className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-cyan-200 border border-cyan-500/30"
                >
                  Generate Token
                </button>
              </div>
            </div>
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <label className="block text-[11px] text-slate-400">2. Confirm Password Reset</label>
              <input
                type="text"
                value={resetToken}
                onChange={(e) => setResetToken(e.target.value)}
                placeholder="Paste reset token"
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs font-mono text-cyan-200"
              />
              <input
                type="password"
                value={newPw}
                onChange={(e) => setNewPw(e.target.value)}
                placeholder="New strong password (8+ chars, upper/lower/number)"
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
              />
              <button
                type="button"
                disabled={busy || !resetToken.trim() || !newPw}
                onClick={async () => {
                  setBusy(true);
                  setErrMsg(null);
                  try {
                    await api.confirmPasswordReset(resetToken, newPw);
                    setStatusMsg("Password reset complete! You can now sign in.");
                    setNewPw("");
                    setResetToken("");
                    setMode("login");
                  } catch (err: any) {
                    setErrMsg(err.message);
                  } finally {
                    setBusy(false);
                  }
                }}
                className="w-full py-2 rounded-xl bg-cyan-500 text-slate-950 font-bold text-xs"
              >
                Reset Password
              </button>
            </div>
          </div>
        )}

        {mode === "register" && (
          <form onSubmit={handleRegister} className="space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Email Address</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="alex@xyvora.net"
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Username</label>
                <input
                  type="text"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="alex_dev"
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
                />
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Display Name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="Alex Mercer"
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-400 mb-1">Password (min 8 chars + letter & digit)</label>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="StrongPass99"
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
                />
              </div>
            </div>
            <button
              type="submit"
              disabled={busy}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950 hover:opacity-95 disabled:opacity-50"
            >
              {busy ? "Creating isolated account..." : "Create Private Xyvora Account"}
            </button>
          </form>
        )}

        {mode === "profile" && (
          <div className="space-y-4">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-white">
                  {user?.display_name} <span className="text-cyan-400 font-mono">@{user?.username}</span>
                </div>
                <div className="text-[11px] text-slate-400">
                  Role: <span className="font-mono text-indigo-300">{user?.role}</span> •{" "}
                  {user?.is_guest ? "Shared Local Guest Mode" : user?.email}
                </div>
              </div>
              {!user?.is_guest && (
                <button
                  type="button"
                  onClick={handleLogout}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1.5"
                >
                  <LogOut className="w-3.5 h-3.5" /> Sign Out
                </button>
              )}
            </div>

            {usage && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Requests (24h)</div>
                  <div className="text-sm font-bold text-cyan-300 font-mono">{usage.requests_last_24h}</div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Est. Tokens</div>
                  <div className="text-sm font-bold text-indigo-300 font-mono">{usage.estimated_tokens_total.toLocaleString()}</div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Memories</div>
                  <div className="text-sm font-bold text-emerald-300 font-mono">{usage.memories_count}</div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Projects</div>
                  <div className="text-sm font-bold text-amber-300 font-mono">{usage.projects_count}</div>
                </div>
              </div>
            )}

            {storage && (
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-[11px] text-slate-300 space-y-1">
                <div className="font-semibold text-cyan-300">
                  Storage Engine: `{storage.engine}` ({storage.persistence_mode})
                </div>
                {storage.ephemeral_warning && <div className="text-amber-300/90">{storage.ephemeral_warning}</div>}
                <div className="flex flex-wrap gap-2 pt-1.5">
                  <button
                    type="button"
                    onClick={handleExportWorkspace}
                    className="px-2.5 py-1 rounded-lg bg-cyan-500/15 border border-cyan-500/35 text-cyan-200 flex items-center gap-1"
                  >
                    <Download className="w-3.5 h-3.5" /> Export Backup JSON
                  </button>
                  <label className="px-2.5 py-1 rounded-lg bg-indigo-500/15 border border-indigo-500/35 text-indigo-200 flex items-center gap-1 cursor-pointer">
                    <Upload className="w-3.5 h-3.5" /> Import Backup JSON
                    <input type="file" accept=".json" onChange={handleImportWorkspace} className="hidden" />
                  </label>
                </div>
              </div>
            )}

            {!user?.is_guest && (
              <>
                <form onSubmit={handleSaveProfile} className="space-y-2.5 border-t border-slate-800 pt-3">
                  <div className="text-xs font-semibold text-slate-200">Profile Settings</div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    <input
                      type="text"
                      value={profName}
                      onChange={(e) => setProfName(e.target.value)}
                      placeholder="Display Name"
                      className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
                    />
                    <input
                      type="text"
                      value={profAvatar}
                      onChange={(e) => setProfAvatar(e.target.value)}
                      placeholder="Avatar Image URL (optional)"
                      className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
                    />
                  </div>
                  <input
                    type="text"
                    value={profBio}
                    onChange={(e) => setProfBio(e.target.value)}
                    placeholder="Short bio or engineering focus"
                    className="w-full px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
                  />
                  <button type="submit" className="px-3 py-1.5 rounded-lg bg-cyan-500/20 border border-cyan-500/40 text-xs text-cyan-200">
                    Save Profile
                  </button>
                </form>

                <form onSubmit={handleChangePw} className="space-y-2.5 border-t border-slate-800 pt-3">
                  <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                    <KeyRound className="w-3.5 h-3.5 text-cyan-400" /> Change Password
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    <input
                      type="password"
                      required
                      value={currPw}
                      onChange={(e) => setCurrPw(e.target.value)}
                      placeholder="Current password"
                      className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
                    />
                    <input
                      type="password"
                      required
                      value={newPw}
                      onChange={(e) => setNewPw(e.target.value)}
                      placeholder="New password"
                      className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-white"
                    />
                  </div>
                  <div className="flex items-center justify-between pt-1">
                    <button type="submit" className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200">
                      Update Password
                    </button>
                    <button
                      type="button"
                      onClick={handleDeleteAccount}
                      className="px-3 py-1.5 rounded-lg bg-rose-500/15 border border-rose-500/35 text-xs text-rose-300 flex items-center gap-1"
                    >
                      <Trash2 className="w-3.5 h-3.5" /> Delete Account
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
