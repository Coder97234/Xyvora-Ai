import React, { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Code2,
  FileCode,
  FilePlus,
  FolderKanban,
  GitCompare,
  History,
  Play,
  RotateCcw,
  Save,
  Sparkles,
  Terminal,
  Trash2,
} from "lucide-react";
import { api } from "../api";
import { ProjectFile, ProjectWorkspace } from "../types";
import { MarkdownView } from "./MarkdownView";

interface CodingStudioProps {
  activeProjectId: string | null;
  onSelectProject: (id: string) => void;
}

export const CodingStudio: React.FC<CodingStudioProps> = ({ activeProjectId, onSelectProject }) => {
  const [projects, setProjects] = useState<ProjectWorkspace[]>([]);
  const [files, setFiles] = useState<ProjectFile[]>([]);
  const [selectedPath, setSelectedPath] = useState<string>("");
  const [editorContent, setEditorContent] = useState<string>("");
  const [dirty, setDirty] = useState(false);
  const [viewMode, setViewMode] = useState<"editor" | "diff">("editor");

  // New project / file inputs
  const [creatingProj, setCreatingProj] = useState(false);
  const [newProjName, setNewProjName] = useState("");
  const [newProjInst, setNewProjInst] = useState("");
  const [addingFile, setAddingFile] = useState(false);
  const [newFilePath, setNewFilePath] = useState("");

  // Diagnostics, Terminal & AI Panel
  const [projAnalysis, setProjAnalysis] = useState<{
    file_count: number;
    total_lines: number;
    dependencies: string[];
    diagnostics: { path: string; line: number; severity: string; message: string }[];
  } | null>(null);
  const [terminalOut, setTerminalOut] = useState<string>("Ready. Click 'Run Unit Tests' or 'Run File' to execute in the OS-isolated sandbox.");
  const [terminalOk, setTerminalOk] = useState<boolean>(true);
  const [runningTests, setRunningTests] = useState(false);

  // AI Coding Assistant
  const [aiAction, setAiAction] = useState("review");
  const [aiPrompt, setAiPrompt] = useState("");
  const [aiBusy, setAiBusy] = useState(false);
  const [aiResult, setAiResult] = useState<{
    explanation: string;
    proposed_content?: string | null;
    diff?: string;
  } | null>(null);

  const loadProjects = useCallback(async () => {
    const list = await api.listProjects();
    setProjects(list);
    if (!activeProjectId && list[0]) {
      onSelectProject(list[0].id);
    }
  }, [activeProjectId, onSelectProject]);

  const loadProjectFiles = useCallback(async (pid: string) => {
    const [fList, analysis] = await Promise.all([api.listProjectFiles(pid), api.analyzeProject(pid)]);
    setFiles(fList);
    setProjAnalysis(analysis);
    setSelectedPath((curr) => {
      const next = fList.find((f) => f.path === curr) || fList[0];
      if (next) {
        setEditorContent(next.content);
        setDirty(false);
        return next.path;
      }
      setEditorContent("");
      return "";
    });
  }, []);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    if (activeProjectId) {
      loadProjectFiles(activeProjectId);
    }
  }, [activeProjectId, loadProjectFiles]);

  const currentFile = files.find((f) => f.path === selectedPath) || null;

  const handleSelectFile = (f: ProjectFile) => {
    setSelectedPath(f.path);
    setEditorContent(f.content);
    setDirty(false);
    setAiResult(null);
  };

  const handleSaveFile = async () => {
    if (!activeProjectId || !selectedPath) return;
    const saved = await api.saveProjectFile(activeProjectId, {
      path: selectedPath,
      content: editorContent,
      summary: "Saved in Coding Studio",
    });
    await loadProjectFiles(activeProjectId);
    setSelectedPath(saved.path);
    setDirty(false);
    setTerminalOut(`[Saved] ${saved.path} (v${saved.version})`);
    setTerminalOk(true);
  };

  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjName.trim()) return;
    const created = await api.createProject({ name: newProjName, instructions: newProjInst });
    await api.saveProjectFile(created.id, {
      path: "src/main.py",
      content: '"""Starter module."""\n\ndef greet(name: str) -> str:\n    return f"Hello, {name}!"\n\nif __name__ == "__main__":\n    print(greet("Xyvora"))\n',
      summary: "Initial starter file",
    });
    setNewProjName("");
    setNewProjInst("");
    setCreatingProj(false);
    await loadProjects();
    onSelectProject(created.id);
  };

  const handleCreateFile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeProjectId || !newFilePath.trim()) return;
    const clean = newFilePath.trim();
    await api.saveProjectFile(activeProjectId, {
      path: clean,
      content: clean.endsWith(".py")
        ? "# New Python module\n"
        : clean.endsWith(".luau") || clean.endsWith(".lua")
        ? "-- Luau module\nlocal Module = {}\nreturn Module\n"
        : "// New source file\n",
      summary: "Created file",
    });
    setNewFilePath("");
    setAddingFile(false);
    await loadProjectFiles(activeProjectId);
    setSelectedPath(clean);
  };

  const handleDeleteFile = async (f: ProjectFile) => {
    if (!activeProjectId || !window.confirm(`Delete ${f.path}?`)) return;
    await api.deleteProjectFile(activeProjectId, f.id);
    await loadProjectFiles(activeProjectId);
  };

  const handleRevert = async (revisionId: string) => {
    if (!activeProjectId || !currentFile) return;
    const reverted = await api.revertProjectFile(activeProjectId, currentFile.id, revisionId);
    await loadProjectFiles(activeProjectId);
    setEditorContent(reverted.content);
    setDirty(false);
    setTerminalOut(`[Reverted] Restored ${reverted.path} to selected snapshot (v${reverted.version}).`);
  };

  const handleRunTests = async () => {
    if (!activeProjectId) return;
    setRunningTests(true);
    try {
      if (dirty && selectedPath) await handleSaveFile();
      const res = await api.runProjectTests(activeProjectId);
      setTerminalOk(res.ok);
      setTerminalOut(
        `[Unit Test Runner • ${res.duration_ms} ms]\n` +
          (res.stdout ? `${res.stdout}\n` : "") +
          (res.stderr ? `${res.stderr}` : "")
      );
    } catch (err: any) {
      setTerminalOk(false);
      setTerminalOut(err.message);
    } finally {
      setRunningTests(false);
    }
  };

  const handleRunCurrentPython = async () => {
    if (!editorContent.trim()) return;
    setRunningTests(true);
    try {
      const res = await api.runPython(editorContent);
      setTerminalOk(res.ok);
      setTerminalOut(
        `[Hardened Python Sandbox • ${res.duration_ms} ms]\n` +
          (res.stdout ? `${res.stdout}\n` : "") +
          (res.stderr ? `${res.stderr}` : "(No stdout emitted)")
      );
    } catch (err: any) {
      setTerminalOk(false);
      setTerminalOut(err.message);
    } finally {
      setRunningTests(false);
    }
  };

  const handleRunAiAssist = async () => {
    if (!selectedPath) return;
    setAiBusy(true);
    try {
      const res = await api.codingAssist({
        project_id: activeProjectId,
        action: aiAction,
        prompt: aiPrompt,
        path: selectedPath,
        language: currentFile?.language || "python",
        content: editorContent,
      });
      setAiResult(res);
    } catch (err: any) {
      setAiResult({ explanation: `Error: ${err.message}` });
    } finally {
      setAiBusy(false);
    }
  };

  const applyProposedPatch = async () => {
    if (!aiResult?.proposed_content || !activeProjectId || !selectedPath) return;
    setEditorContent(aiResult.proposed_content);
    await api.saveProjectFile(activeProjectId, {
      path: selectedPath,
      content: aiResult.proposed_content,
      summary: `Applied AI ${aiAction} patch`,
    });
    await loadProjectFiles(activeProjectId);
    setViewMode("diff");
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Top IDE Bar */}
      <div className="px-4 py-2.5 border-b border-slate-800/80 bg-[#070b18]/90 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <FolderKanban className="w-4 h-4 text-cyan-400" />
          <select
            value={activeProjectId || ""}
            onChange={(e) => onSelectProject(e.target.value)}
            className="px-2.5 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs font-semibold text-white"
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setCreatingProj((s) => !s)}
            className="px-2.5 py-1 rounded-lg bg-cyan-500/15 border border-cyan-500/35 text-[11px] text-cyan-200"
          >
            + Project
          </button>
          {projAnalysis && (
            <span className="hidden lg:inline text-[11px] font-mono text-slate-400">
              {projAnalysis.file_count} files • {projAnalysis.total_lines} LOC • deps:{" "}
              {projAnalysis.dependencies.slice(0, 4).join(", ") || "stdlib"}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setViewMode(viewMode === "editor" ? "diff" : "editor")}
            className={`px-2.5 py-1 rounded-lg border text-xs flex items-center gap-1.5 ${
              viewMode === "diff"
                ? "bg-indigo-500/20 border-indigo-400/50 text-indigo-200"
                : "bg-slate-900 border-slate-700 text-slate-300"
            }`}
          >
            <GitCompare className="w-3.5 h-3.5" /> {viewMode === "diff" ? "Back to Editor" : "Diff & Revisions"}
          </button>
          <button
            type="button"
            onClick={handleSaveFile}
            disabled={!dirty}
            className="px-2.5 py-1 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-xs text-emerald-200 disabled:opacity-40 flex items-center gap-1"
          >
            <Save className="w-3.5 h-3.5" /> Save
          </button>
          {selectedPath.endsWith(".py") && (
            <button
              type="button"
              onClick={handleRunCurrentPython}
              disabled={runningTests}
              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-cyan-200 flex items-center gap-1"
            >
              <Play className="w-3.5 h-3.5" /> Run File
            </button>
          )}
          <button
            type="button"
            onClick={handleRunTests}
            disabled={runningTests}
            className="px-3 py-1 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950 flex items-center gap-1"
          >
            <Play className="w-3.5 h-3.5" /> Run Unit Tests
          </button>
        </div>
      </div>

      {creatingProj && (
        <form onSubmit={handleCreateProject} className="px-4 py-2.5 bg-slate-900/95 border-b border-slate-800 flex flex-wrap gap-2 items-center">
          <input
            type="text"
            required
            value={newProjName}
            onChange={(e) => setNewProjName(e.target.value)}
            placeholder="Project Name (e.g., Quantum Game Engine)"
            className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-white"
          />
          <input
            type="text"
            value={newProjInst}
            onChange={(e) => setNewProjInst(e.target.value)}
            placeholder="Project AI Instructions (stack, conventions)"
            className="flex-1 min-w-[220px] px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-700 text-xs text-white"
          />
          <button type="submit" className="px-3 py-1.5 rounded-lg bg-cyan-500 text-slate-950 text-xs font-bold">
            Create
          </button>
        </form>
      )}

      {/* Main 3-Pane IDE Body */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
        {/* Left: File Explorer & Diagnostics */}
        <div className="lg:col-span-3 border-r border-slate-800/80 bg-[#060a15] flex flex-col overflow-y-auto p-3 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400">Workspace Files</span>
            <button
              type="button"
              onClick={() => setAddingFile((s) => !s)}
              className="text-[11px] text-cyan-300 flex items-center gap-1"
            >
              <FilePlus className="w-3.5 h-3.5" /> New File
            </button>
          </div>

          {addingFile && (
            <form onSubmit={handleCreateFile} className="flex gap-1.5">
              <input
                type="text"
                required
                value={newFilePath}
                onChange={(e) => setNewFilePath(e.target.value)}
                placeholder="src/utils.ts or script.luau"
                className="flex-1 px-2 py-1 rounded bg-slate-900 border border-slate-700 text-xs text-white font-mono"
              />
              <button type="submit" className="px-2 py-1 rounded bg-cyan-500/20 text-cyan-200 text-xs">
                Add
              </button>
            </form>
          )}

          <div className="space-y-1">
            {files.map((f) => {
              const active = f.path === selectedPath;
              return (
                <div
                  key={f.id}
                  onClick={() => handleSelectFile(f)}
                  className={`group flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-mono cursor-pointer ${
                    active ? "bg-cyan-500/15 border border-cyan-500/35 text-cyan-200" : "text-slate-300 hover:bg-slate-900"
                  }`}
                >
                  <span className="flex items-center gap-1.5 truncate">
                    <FileCode className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                    <span className="truncate">{f.path}</span>
                  </span>
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] text-slate-500">v{f.version}</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteFile(f);
                      }}
                      className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Static & Security Diagnostics */}
          <div className="pt-2 border-t border-slate-800 space-y-1.5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-amber-400" /> Static & Security Diagnostics
            </div>
            {!projAnalysis?.diagnostics?.length ? (
              <div className="text-[11px] text-emerald-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" /> Zero syntax or static warnings
              </div>
            ) : (
              projAnalysis.diagnostics.slice(0, 8).map((d, i) => (
                <div key={i} className="p-2 rounded-lg bg-slate-900/90 border border-amber-500/25 text-[11px]">
                  <div className="font-mono text-amber-300">
                    {d.path}:{d.line} ({d.severity})
                  </div>
                  <div className="text-slate-300">{d.message}</div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Center: Code Editor / Diff Viewer + Terminal */}
        <div className="lg:col-span-5 flex flex-col overflow-hidden border-r border-slate-800/80">
          {viewMode === "editor" ? (
            <div className="flex-1 flex flex-col overflow-hidden">
              <div className="px-3 py-1.5 bg-slate-900/80 border-b border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span>
                  {selectedPath || "Select a file"} {dirty && <span className="text-amber-300">● unsaved</span>}
                </span>
                <span>{currentFile?.language || "code"}</span>
              </div>
              <textarea
                value={editorContent}
                onChange={(e) => {
                  setEditorContent(e.target.value);
                  setDirty(true);
                }}
                onKeyDown={(e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
                    e.preventDefault();
                    handleSaveFile();
                  }
                }}
                spellCheck={false}
                className="flex-1 w-full p-3.5 bg-[#050811] text-slate-100 font-mono text-xs leading-relaxed focus:outline-none resize-none"
              />
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-3.5 space-y-3 bg-[#050811]">
              <h4 className="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
                <History className="w-4 h-4" /> Unified Diff & Reversible Revisions ({selectedPath})
              </h4>
              {currentFile?.diff ? (
                <pre className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono overflow-x-auto text-slate-200">
                  {currentFile.diff}
                </pre>
              ) : (
                <p className="text-xs text-slate-400">Make an edit or apply an AI patch to inspect unified diffs.</p>
              )}

              <div className="space-y-2">
                {(currentFile?.revisions || []).map((rev) => (
                  <div key={rev.id} className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs">
                    <div>
                      <div className="font-mono text-cyan-300">
                        Snapshot v{rev.version} — {rev.summary}
                      </div>
                      <div className="text-[10px] text-slate-400">{new Date(rev.created_at * 1000).toLocaleString()}</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleRevert(rev.id)}
                      className="px-2.5 py-1 rounded-lg bg-amber-500/15 border border-amber-500/35 text-amber-200 text-[11px] flex items-center gap-1"
                    >
                      <RotateCcw className="w-3 h-3" /> Revert
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Bottom Terminal Output */}
          <div className="h-44 border-t border-slate-800 bg-[#040710] flex flex-col">
            <div className="px-3 py-1.5 border-b border-slate-800/80 flex items-center justify-between text-[11px] font-mono">
              <span className="flex items-center gap-1.5 text-slate-300">
                <Terminal className="w-3.5 h-3.5 text-cyan-400" /> Sandbox Terminal & Test Runner
              </span>
              <span className={terminalOk ? "text-emerald-400" : "text-rose-400"}>{terminalOk ? "OK" : "ERROR"}</span>
            </div>
            <pre className="flex-1 p-3 overflow-y-auto text-[11px] font-mono text-slate-200 whitespace-pre-wrap">
              {terminalOut}
            </pre>
          </div>
        </div>

        {/* Right: AI Coding Copilot */}
        <div className="lg:col-span-4 flex flex-col overflow-y-auto p-3.5 space-y-3 bg-[#060a15]">
          <div className="flex items-center gap-2">
            <Code2 className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold text-white">Project-Aware AI Coding Assistant</h3>
          </div>
          <p className="text-[11px] text-slate-400">
            Supports Python, JS/TS, React, HTML/CSS, Java, C++, C#, and Luau with reviewable patches.
          </p>

          <div className="grid grid-cols-3 gap-1.5">
            {[
              { id: "explain", label: "Explain" },
              { id: "debug", label: "Debug" },
              { id: "review", label: "Code Review" },
              { id: "refactor", label: "Refactor" },
              { id: "generate_patch", label: "Gen Patch" },
            ].map((act) => (
              <button
                key={act.id}
                type="button"
                onClick={() => setAiAction(act.id)}
                className={`py-1.5 px-2 rounded-lg text-[11px] font-medium border ${
                  aiAction === act.id
                    ? "bg-cyan-500/20 border-cyan-500/45 text-cyan-200"
                    : "bg-slate-900 border-slate-800 text-slate-400"
                }`}
              >
                {act.label}
              </button>
            ))}
          </div>

          <textarea
            rows={3}
            value={aiPrompt}
            onChange={(e) => setAiPrompt(e.target.value)}
            placeholder="Optional instruction (e.g., 'Add docstrings and edge-case bounds checking')"
            className="w-full p-2.5 rounded-xl bg-slate-900 border border-slate-700 text-xs text-white"
          />
          <button
            type="button"
            onClick={handleRunAiAssist}
            disabled={aiBusy || !selectedPath}
            className="w-full py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 text-xs font-bold text-slate-950 flex items-center justify-center gap-1.5 disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5" /> {aiBusy ? "Analyzing codebase..." : `Run ${aiAction.replace("_", " ")}`}
          </button>

          {aiResult && (
            <div className="space-y-3 pt-2 border-t border-slate-800">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs">
                <MarkdownView content={aiResult.explanation} />
              </div>
              {aiResult.diff && (
                <div className="space-y-2">
                  <div className="text-[11px] font-mono text-indigo-300">Proposed Reviewable Diff:</div>
                  <pre className="p-2.5 rounded-xl bg-slate-950 border border-indigo-500/30 text-[10.5px] font-mono overflow-x-auto text-slate-200 max-h-48">
                    {aiResult.diff}
                  </pre>
                  <button
                    type="button"
                    onClick={applyProposedPatch}
                    className="w-full py-2 rounded-xl bg-emerald-500/20 border border-emerald-500/45 text-xs font-bold text-emerald-200"
                  >
                    Approve & Apply Patch (Reversible)
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
