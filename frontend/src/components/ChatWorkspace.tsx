import React, { useEffect, useRef, useState } from "react";
import {
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Cpu,
  Database,
  Edit3,
  FileText,
  GraduationCap,
  Mic,
  MicOff,
  Paperclip,
  RefreshCw,
  Send,
  Sparkles,
  ThumbsUp,
  Volume2,
  VolumeX,
  Wrench,
  X,
} from "lucide-react";
import { api } from "../api";
import {
  BrainState,
  ChatArtifact,
  ChatAttachment,
  ChatMessage,
  CitationItem,
  CognitiveSettings,
  ProviderInfo,
  SemanticMemory,
  ThoughtStep,
  ToolActivity,
  WorkspaceTab,
} from "../types";
import { ArtifactsList } from "./ArtifactsRenderer";
import { MarkdownView } from "./MarkdownView";

interface ChatWorkspaceProps {
  sessionId: string;
  activeProjectId?: string | null;
  settings: CognitiveSettings;
  providers: ProviderInfo[];
  brainState: BrainState;
  setBrainState: (s: BrainState) => void;
  onRefreshVitals: () => void;
  onMemoryToast: (memories: SemanticMemory[]) => void;
  onNavigateTab: (tab: WorkspaceTab, codePayload?: string) => void;
  onOpenControls: () => void;
}

const SLASH_COMMANDS = [
  { cmd: "/remember ", template: "Remember that ", desc: "Store a fact or preference in your Memory Palace" },
  { cmd: "/recall", template: "What do you remember about me so far?", desc: "Query your Semantic Knowledge Graph" },
  { cmd: "/research ", template: "Research the tradeoffs between Rust and C++ for systems programming", desc: "Run grounded multi-source Web Research" },
  { cmd: "/math ", template: "Differentiate and plot sin(x) * cos(x/2)", desc: "Solve symbolic calculus/algebra & plot f(x)" },
  { cmd: "/simulate", template: "Simulate 1000 dice rolls in Python and chart the results", desc: "Synthesize & execute Python simulation" },
  { cmd: "/chart ", template: "Chart Jan 120, Feb 165, Mar 210, Apr 195, May 290", desc: "Render interactive data visualization" },
  { cmd: "/task ", template: "Add a task to ", desc: "Create or manage persistent tasks" },
];

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  sessionId,
  activeProjectId,
  settings,
  providers,
  brainState,
  setBrainState,
  onRefreshVitals,
  onMemoryToast,
  onNavigateTab,
  onOpenControls,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streamingContent, setStreamingContent] = useState("");
  const [liveThoughts, setLiveThoughts] = useState<ThoughtStep[]>([]);
  const [liveTools, setLiveTools] = useState<ToolActivity[]>([]);
  const [expandedTraces, setExpandedTraces] = useState<Record<string, boolean>>({});
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [statusBanner, setStatusBanner] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Provider/Model quick selector
  const [selectedProvider, setSelectedProvider] = useState<string>(settings.active_provider || "auto");
  const [selectedModel, setSelectedModel] = useState<string>(settings.active_model || "");

  // Attachments & Drag-and-Drop
  const [pendingAttachments, setPendingAttachments] = useState<ChatAttachment[]>([]);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  // Edit user message inline
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");

  // Quick 1-click "Teach Xyvora" modal
  const [teachModal, setTeachModal] = useState<{ trigger: string; reply: string } | null>(null);
  const [teachSaved, setTeachSaved] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!sessionId) return;
    api.getMessages(sessionId).then(setMessages).catch(console.error);
  }, [sessionId]);

  useEffect(() => {
    setSelectedProvider(settings.active_provider || "auto");
    setSelectedModel(settings.active_model || "");
  }, [settings.active_provider, settings.active_model]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, liveThoughts, liveTools]);

  const speakText = async (msgId: string, rawMarkdown: string) => {
    if (speakingId === msgId) {
      window.speechSynthesis?.cancel();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      setSpeakingId(null);
      return;
    }
    window.speechSynthesis?.cancel();
    const clean = rawMarkdown
      .replace(/```[\s\S]*?```/g, " Code block omitted. ")
      .replace(/[#*_`~>|]/g, "")
      .slice(0, 950);

    if (settings.voice_provider === "neural") {
      setSpeakingId(msgId);
      try {
        const tts = await api.synthesizeSpeech(clean);
        if (tts.ok && tts.audio_data_url) {
          const audio = new Audio(tts.audio_data_url);
          audioRef.current = audio;
          audio.onended = () => setSpeakingId(null);
          audio.onerror = () => setSpeakingId(null);
          await audio.play();
          return;
        }
        setStatusBanner(tts.error || "Cloud Neural TTS unconfigured; using Browser SpeechSynthesis.");
        setTimeout(() => setStatusBanner(null), 4000);
      } catch {
        // fallback to browser TTS
      }
    }

    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      const utter = new SpeechSynthesisUtterance(clean);
      utter.rate = settings.voice_rate || 1.03;
      utter.onend = () => setSpeakingId(null);
      utter.onerror = () => setSpeakingId(null);
      setSpeakingId(msgId);
      window.speechSynthesis.speak(utter);
    }
  };

  const toggleVoiceMic = () => {
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      setStatusBanner("Browser SpeechRecognition API is unavailable in this browser.");
      setTimeout(() => setStatusBanner(null), 4000);
      return;
    }
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
      setBrainState("idle");
      return;
    }
    const rec = new SpeechRec();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.onstart = () => {
      setIsListening(true);
      setBrainState("listening");
    };
    rec.onresult = (event: any) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      if (transcript) setInput((prev) => (prev ? `${prev} ${transcript}` : transcript));
    };
    rec.onend = () => {
      setIsListening(false);
      setBrainState("idle");
    };
    rec.onerror = () => {
      setIsListening(false);
      setBrainState("idle");
    };
    recognitionRef.current = rec;
    rec.start();
  };

  const handleFilesSelected = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    setUploadingFile(true);
    try {
      const uploaded: ChatAttachment[] = [];
      for (const f of Array.from(fileList).slice(0, 4)) {
        const att = await api.uploadAttachment(f, sessionId, activeProjectId);
        uploaded.push(att);
      }
      setPendingAttachments((prev) => [...prev, ...uploaded]);
    } catch (err: any) {
      setStatusBanner(err.message || "Attachment upload failed");
      setTimeout(() => setStatusBanner(null), 4500);
    } finally {
      setUploadingFile(false);
    }
  };

  const buildStreamHandlers = () => {
    let acc = "";
    return {
      onThought: (step: ThoughtStep) => setLiveThoughts((prev) => [...prev, step]),
      onToolActivity: (act: ToolActivity) => {
        setLiveTools((prev) => {
          const filtered = prev.filter((x) => x.tool !== act.tool);
          return [...filtered, act];
        });
      },
      onMemoryConsolidated: (mems: SemanticMemory[]) => onMemoryToast(mems),
      onToken: (delta: string) => {
        setBrainState("streaming");
        acc += delta;
        setStreamingContent(acc);
      },
      onDone: (donePayload: {
        message_id: string;
        intent: string;
        confidence: number;
        provider_used?: string;
        model_used?: string;
        thought_trace: ThoughtStep[];
        tool_calls?: ToolActivity[];
        citations?: CitationItem[];
        artifacts: ChatArtifact[];
        memories_added: SemanticMemory[];
        suggestions: string[];
      }) => {
        const aiMsg: ChatMessage = {
          id: donePayload.message_id,
          session_id: sessionId,
          role: "assistant",
          content: acc,
          intent: donePayload.intent,
          confidence: donePayload.confidence,
          provider_used: donePayload.provider_used,
          model_used: donePayload.model_used,
          thought_trace: donePayload.thought_trace,
          tool_calls: donePayload.tool_calls,
          citations: donePayload.citations,
          artifacts: donePayload.artifacts,
          memories_added: donePayload.memories_added,
          suggestions: donePayload.suggestions,
          created_at: Date.now() / 1000,
        };
        setMessages((prev) => [...prev, aiMsg]);
        setStreamingContent("");
        setLiveThoughts([]);
        setLiveTools([]);
        setBrainState("idle");
        onRefreshVitals();
        if (settings.voice_enabled) speakText(aiMsg.id, acc);
      },
    };
  };

  const sendMessage = async (textOverride?: string) => {
    const clean = (textOverride ?? input).trim();
    if ((!clean && pendingAttachments.length === 0) || brainState === "thinking" || brainState === "streaming") return;

    const attsToSend = [...pendingAttachments];
    const optimisticUser: ChatMessage = {
      id: `local-${Date.now()}`,
      session_id: sessionId,
      role: "user",
      content: clean || "(Attached file analysis)",
      attachments: attsToSend,
      created_at: Date.now() / 1000,
    };

    setMessages((prev) => [...prev, optimisticUser]);
    setInput("");
    setPendingAttachments([]);
    setStreamingContent("");
    setLiveThoughts([]);
    setLiveTools([]);
    setBrainState("thinking");

    try {
      await api.streamChat(sessionId, clean, buildStreamHandlers(), {
        project_id: activeProjectId,
        attachment_ids: attsToSend.map((a) => a.id),
        provider: selectedProvider !== "auto" ? selectedProvider : undefined,
        model: selectedModel || undefined,
      });
      // Sync canonical message IDs so Edit & Resend works on newly created messages
      const fresh = await api.getMessages(sessionId);
      setMessages(fresh);
    } catch (err: any) {
      setBrainState("idle");
      setStreamingContent("");
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          session_id: sessionId,
          role: "assistant",
          content: `Request error: ${err.message || "Unable to complete stream."}`,
          created_at: Date.now() / 1000,
        },
      ]);
    }
  };

  const handleRegenerate = async () => {
    if (brainState !== "idle") return;
    setMessages((prev) => {
      const copy = [...prev];
      if (copy.length > 0 && copy[copy.length - 1].role === "assistant") copy.pop();
      return copy;
    });
    setBrainState("thinking");
    setLiveThoughts([]);
    setLiveTools([]);
    try {
      await api.regenerateChat(sessionId, buildStreamHandlers());
      setMessages(await api.getMessages(sessionId));
    } catch (err: any) {
      setBrainState("idle");
      setStatusBanner(err.message);
    }
  };

  const handleSaveEditAndResend = async (msgId: string) => {
    if (!editingText.trim() || brainState !== "idle") return;
    const idx = messages.findIndex((m) => m.id === msgId);
    if (idx >= 0) {
      const updated = messages.slice(0, idx + 1);
      updated[idx] = { ...updated[idx], content: editingText.trim(), edited_at: Date.now() / 1000 };
      setMessages(updated);
    }
    setEditingMsgId(null);
    setBrainState("thinking");
    setLiveThoughts([]);
    setLiveTools([]);
    try {
      await api.editAndResendMessage(msgId, editingText.trim(), buildStreamHandlers());
      setMessages(await api.getMessages(sessionId));
    } catch (err: any) {
      setBrainState("idle");
      setStatusBanner(err.message);
    }
  };

  const handleApproveMemoryInline = async (memId: string) => {
    await api.approveMemoryFact(memId);
    onRefreshVitals();
    setMessages(await api.getMessages(sessionId));
  };

  const copyMessage = (id: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const showSlashMenu = input.startsWith("/") && !input.includes(" ");
  const matchingSlash = showSlashMenu
    ? SLASH_COMMANDS.filter((s) => s.cmd.startsWith(input.toLowerCase()))
    : [];

  return (
    <div
      className="flex flex-col h-full overflow-hidden relative"
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        handleFilesSelected(e.dataTransfer.files);
      }}
    >
      {dragActive && (
        <div className="absolute inset-3 z-30 rounded-2xl border-2 border-dashed border-cyan-400 bg-cyan-950/75 backdrop-blur-sm flex items-center justify-center text-sm font-bold text-cyan-200 pointer-events-none">
          Drop images, PDFs, DOCX, XLSX, CSV, JSON, or source code to attach
        </div>
      )}

      {/* Top Honest Model & Context Control Bar */}
      <div className="px-4 py-2 border-b border-slate-800/80 bg-[#070b18]/75 flex flex-wrap items-center justify-between gap-2 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <select
            value={selectedProvider}
            onChange={(e) => {
              const pid = e.target.value;
              setSelectedProvider(pid);
              const found = providers.find((p) => p.id === pid);
              if (found) setSelectedModel(found.default_model);
            }}
            className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-100"
          >
            <option value="auto">Auto Router (Configured LLM → Local Fallback)</option>
            {providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} {p.configured ? "✓" : "(Needs API Key)"}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={onOpenControls}
            className="text-[11px] px-2.5 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/20"
          >
            Configure Providers & Persona
          </button>
        </div>

        <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono">
          <span>Voice: {(settings.voice_provider || "browser") === "browser" ? "Browser Web Speech" : "Cloud Neural TTS"}</span>
        </div>
      </div>

      {/* Messages Feed */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-5 space-y-5">
        {messages.map((msg, idx) => {
          const isUser = msg.role === "user";
          const isLastAssistant = !isUser && idx === messages.length - 1;
          const traceOpen = !!expandedTraces[msg.id];

          return (
            <div key={msg.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-3xl w-full rounded-2xl px-4 py-3.5 shadow-md ${
                  isUser
                    ? "md:w-auto bg-gradient-to-br from-cyan-600/25 to-indigo-600/25 border border-cyan-400/35 text-slate-100"
                    : "glass-panel text-slate-100"
                }`}
              >
                {/* Attachments inside Message */}
                {msg.attachments && msg.attachments.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2.5">
                    {msg.attachments.map((att) =>
                      att.kind === "image" && att.data_url ? (
                        <div key={att.id} className="relative rounded-xl overflow-hidden border border-slate-700 max-w-[220px]">
                          <img src={att.data_url} alt={att.filename} className="max-h-36 object-cover" />
                          <span className="block px-2 py-0.5 bg-slate-950/90 text-[10px] font-mono text-cyan-300 truncate">
                            Uploaded: {att.filename}
                          </span>
                        </div>
                      ) : (
                        <div
                          key={att.id}
                          className="px-2.5 py-1.5 rounded-xl bg-slate-900/90 border border-slate-700 text-[11px] font-mono text-cyan-200 flex items-center gap-1.5"
                        >
                          <FileText className="w-3.5 h-3.5 text-cyan-400" />
                          {att.filename}
                        </div>
                      )
                    )}
                  </div>
                )}

                {/* Concise Tool Activity Badges & Optional Reasoning Trace */}
                {!isUser && ((msg.tool_calls && msg.tool_calls.length > 0) || (settings.show_thought_trace && msg.thought_trace && msg.thought_trace.length > 0)) && (
                  <div className="mb-2.5 flex flex-wrap items-center gap-1.5">
                    {(msg.tool_calls || []).map((tc, tIdx) => (
                      <span
                        key={tIdx}
                        className="inline-flex items-center gap-1 text-[10.5px] font-mono px-2 py-0.5 rounded-full bg-indigo-500/15 border border-indigo-400/30 text-indigo-200"
                      >
                        <Wrench className="w-3 h-3 text-indigo-400" />
                        {tc.label}
                      </span>
                    ))}

                    {settings.show_thought_trace && msg.thought_trace && msg.thought_trace.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setExpandedTraces((p) => ({ ...p, [msg.id]: !p[msg.id] }))}
                        className="inline-flex items-center gap-1 text-[10.5px] font-mono px-2 py-0.5 rounded-full bg-slate-900/90 border border-cyan-500/25 text-cyan-300"
                      >
                        {traceOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                        {msg.provider_used || "xyvora_local"}
                        {msg.intent && msg.intent !== "welcome" ? ` • ${msg.intent}` : ""}
                      </button>
                    )}
                  </div>
                )}

                {traceOpen && msg.thought_trace && (
                  <div className="mb-3 p-2.5 rounded-xl bg-slate-950/90 border border-slate-800 space-y-1.5 text-[11px]">
                    {msg.thought_trace.map((st, sIdx) => (
                      <div key={sIdx} className="flex items-start justify-between gap-2">
                        <div>
                          <span className="font-semibold text-cyan-300">{st.step}:</span>{" "}
                          <span className="text-slate-300">{st.detail}</span>
                        </div>
                        <span className="font-mono text-[10px] text-slate-500 shrink-0">{st.ms} ms</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Inline Edit Mode for User Message vs Normal Markdown */}
                {editingMsgId === msg.id ? (
                  <div className="space-y-2 min-w-[260px]">
                    <textarea
                      rows={3}
                      value={editingText}
                      onChange={(e) => setEditingText(e.target.value)}
                      className="w-full p-2.5 rounded-xl bg-slate-950 border border-cyan-500/40 text-xs text-white"
                    />
                    <div className="flex justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => setEditingMsgId(null)}
                        className="px-2.5 py-1 rounded-lg bg-slate-800 text-xs text-slate-300"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSaveEditAndResend(msg.id)}
                        className="px-3 py-1 rounded-lg bg-cyan-500 text-slate-950 text-xs font-bold"
                      >
                        Save & Regenerate
                      </button>
                    </div>
                  </div>
                ) : isUser ? (
                  <div className="space-y-1.5">
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                    {!msg.id.startsWith("local-") && (
                      <div className="flex justify-end items-center gap-2 pt-0.5">
                        {msg.edited_at && <span className="text-[10px] text-slate-400 italic">edited</span>}
                        <button
                          type="button"
                          onClick={() => {
                            setEditingMsgId(msg.id);
                            setEditingText(msg.content);
                          }}
                          className="text-[11px] text-cyan-300/80 hover:text-cyan-200 flex items-center gap-1"
                        >
                          <Edit3 className="w-3 h-3" /> Edit
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <MarkdownView content={msg.content} />
                )}

                {/* Memory Consolidation / Approval Badges */}
                {!isUser && msg.memories_added && msg.memories_added.length > 0 && (
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {msg.memories_added.map((m) => (
                      <span
                        key={m.id}
                        className={`inline-flex items-center gap-1.5 text-[11px] px-2.5 py-0.5 rounded-full border ${
                          m.verified === 0
                            ? "bg-amber-500/15 border-amber-400/40 text-amber-200"
                            : "bg-emerald-500/10 border-emerald-500/30 text-emerald-200"
                        }`}
                      >
                        <Database className="w-3 h-3" />
                        {m.verified === 0 ? "Pending Approval:" : "Saved:"} {m.subject} → {m.relation} → {m.object}
                        {m.verified === 0 && (
                          <button
                            type="button"
                            onClick={() => handleApproveMemoryInline(m.id)}
                            className="ml-1 px-1.5 py-0.5 rounded bg-emerald-500 text-slate-950 text-[10px] font-bold"
                          >
                            Approve
                          </button>
                        )}
                      </span>
                    ))}
                  </div>
                )}

                {/* Source Attribution Citations */}
                {!isUser && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 pt-2.5 border-t border-slate-800/80 space-y-1.5">
                    <div className="text-[10.5px] font-mono uppercase text-cyan-300 flex items-center gap-1">
                      <BookOpen className="w-3 h-3" /> Grounded Source Attribution
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {msg.citations.map((c, ci) => (
                        <span
                          key={ci}
                          title={c.snippet || ""}
                          className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-300"
                        >
                          <span className="font-mono text-cyan-300">[{c.source_type}]</span> {c.title}
                          {c.section ? ` (${c.section})` : ""}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Interactive Artifacts */}
                {!isUser && (
                  <ArtifactsList
                    artifacts={msg.artifacts}
                    onOpenCodeInSandbox={(code) => onNavigateTab("lab", code)}
                    onTasksChanged={onRefreshVitals}
                  />
                )}

                {/* Assistant Action Bar */}
                {!isUser && (
                  <div className="mt-3 pt-2 border-t border-slate-800/70 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => copyMessage(msg.id, msg.content)}
                        className="text-[11px] text-slate-400 hover:text-cyan-300 flex items-center gap-1 px-2 py-1 rounded-md hover:bg-slate-800/60"
                      >
                        {copiedId === msg.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        {copiedId === msg.id ? "Copied" : "Copy"}
                      </button>

                      {isLastAssistant && msg.intent !== "welcome" && (
                        <button
                          type="button"
                          onClick={handleRegenerate}
                          className="text-[11px] text-slate-400 hover:text-cyan-300 flex items-center gap-1 px-2 py-1 rounded-md hover:bg-slate-800/60"
                        >
                          <RefreshCw className="w-3.5 h-3.5" /> Regenerate
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => speakText(msg.id, msg.content)}
                        className="text-[11px] text-slate-400 hover:text-cyan-300 flex items-center gap-1 px-2 py-1 rounded-md hover:bg-slate-800/60"
                      >
                        {speakingId === msg.id ? <VolumeX className="w-3.5 h-3.5 text-amber-400" /> : <Volume2 className="w-3.5 h-3.5" />}
                        {speakingId === msg.id ? "Stop" : "Speak"}
                      </button>

                      <button
                        type="button"
                        onClick={async () => {
                          const next = msg.feedback === 1 ? 0 : 1;
                          await api.sendFeedback(msg.id, next);
                          setMessages((prev) => prev.map((m) => (m.id === msg.id ? { ...m, feedback: next } : m)));
                        }}
                        className={`text-[11px] flex items-center gap-1 px-2 py-1 rounded-md ${
                          msg.feedback === 1 ? "text-emerald-400 bg-emerald-500/10" : "text-slate-400 hover:text-slate-200"
                        }`}
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>

                      {idx > 0 && messages[idx - 1]?.role === "user" && (
                        <button
                          type="button"
                          onClick={() => {
                            setTeachSaved(false);
                            setTeachModal({ trigger: messages[idx - 1].content, reply: msg.content });
                          }}
                          className="text-[11px] text-slate-400 hover:text-indigo-300 flex items-center gap-1 px-2 py-1 rounded-md hover:bg-slate-800/60"
                        >
                          <GraduationCap className="w-3.5 h-3.5" /> Teach Skill
                        </button>
                      )}
                    </div>

                    {msg.suggestions && msg.suggestions.length > 0 && idx === messages.length - 1 && (
                      <div className="flex flex-wrap gap-1.5">
                        {msg.suggestions.map((sug, i) => (
                          <button
                            key={i}
                            type="button"
                            onClick={() => sendMessage(sug)}
                            className="text-[11px] px-2.5 py-1 rounded-full bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-200 border border-cyan-500/25 flex items-center gap-1"
                          >
                            <Sparkles className="w-3 h-3 text-cyan-400" />
                            {sug}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Live Streaming Assistant Card */}
        {(brainState === "thinking" || brainState === "streaming") && (
          <div className="flex justify-start">
            <div className="max-w-3xl w-full glass-panel rounded-2xl px-4 py-3.5 border border-cyan-400/35">
              <div className="flex flex-wrap items-center gap-2 mb-2 text-xs text-cyan-300 font-mono">
                <Cpu className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                <span>{brainState === "thinking" ? "Synthesizing context..." : "Streaming response..."}</span>
                {liveTools.map((lt) => (
                  <span
                    key={lt.tool}
                    className="px-2 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-400/40 text-[10.5px] text-indigo-200"
                  >
                    {lt.label} {lt.status === "done" ? "✓" : "…"}
                  </span>
                ))}
              </div>
              {streamingContent && <MarkdownView content={streamingContent} />}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Composer Area */}
      <div className="px-4 md:px-8 pb-4 pt-2 border-t border-slate-800/80 bg-[#070b18]/90 backdrop-blur-md">
        {statusBanner && (
          <div className="mb-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-1.5">
            {statusBanner}
          </div>
        )}

        {/* Pending Attachments Preview Strip */}
        {(pendingAttachments.length > 0 || uploadingFile) && (
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {pendingAttachments.map((att) => (
              <div
                key={att.id}
                className="flex items-center gap-2 px-2.5 py-1 rounded-xl bg-slate-900 border border-cyan-500/35 text-xs text-cyan-200"
              >
                {att.kind === "image" && att.data_url ? (
                  <img src={att.data_url} alt={att.filename} className="w-6 h-6 rounded object-cover" />
                ) : (
                  <FileText className="w-3.5 h-3.5 text-cyan-400" />
                )}
                <span className="truncate max-w-[160px]">{att.filename}</span>
                <button
                  type="button"
                  onClick={() => setPendingAttachments((p) => p.filter((x) => x.id !== att.id))}
                  className="text-slate-400 hover:text-rose-400"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
            {uploadingFile && <span className="text-xs font-mono text-cyan-300">Parsing attachment...</span>}
          </div>
        )}

        {matchingSlash.length > 0 && (
          <div className="mb-2 rounded-xl bg-slate-900/95 border border-cyan-500/30 p-1.5 shadow-xl space-y-1">
            {matchingSlash.map((item) => (
              <button
                key={item.cmd}
                type="button"
                onClick={() => setInput(item.template)}
                className="w-full flex items-center justify-between text-left px-3 py-1.5 rounded-lg hover:bg-slate-800"
              >
                <span className="text-xs font-mono text-cyan-300 font-semibold">{item.cmd}</span>
                <span className="text-[11px] text-slate-400">{item.desc}</span>
              </button>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2 rounded-2xl bg-slate-900/90 border border-slate-700/90 focus-within:border-cyan-400/60 px-3 py-2 shadow-inner">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.docx,.xlsx,.pptx,.csv,.json,.py,.js,.ts,.tsx,.jsx,.html,.css,.java,.cpp,.cs,.luau,.lua,.png,.jpg,.jpeg,.webp,.gif,.svg"
            onChange={(e) => handleFilesSelected(e.target.files)}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            title="Attach Image or Document (PDF, DOCX, XLSX, PPTX, CSV, Code, PNG/JPG)"
            className="p-2 rounded-xl text-slate-400 hover:text-cyan-300 hover:bg-slate-800 transition"
          >
            <Paperclip className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={toggleVoiceMic}
            title="Browser Voice Dictation"
            className={`p-2 rounded-xl transition ${
              isListening
                ? "bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse"
                : "text-slate-400 hover:text-cyan-300 hover:bg-slate-800"
            }`}
          >
            {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          </button>

          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Message Xyvora, attach files/images, or type / for quick commands..."
            className="flex-1 bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none max-h-32 py-1.5"
          />

          <button
            type="button"
            disabled={(!input.trim() && pendingAttachments.length === 0) || brainState === "thinking" || brainState === "streaming"}
            onClick={() => sendMessage()}
            className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 hover:from-cyan-400 hover:to-indigo-400 disabled:opacity-40 text-white shadow-md transition"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Teach Xyvora Skill Modal */}
      {teachModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl bg-slate-900 border border-cyan-500/35 p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-cyan-300 font-semibold text-sm">
                <GraduationCap className="w-4 h-4" /> Teach Xyvora a Custom Skill
              </div>
              <button type="button" onClick={() => setTeachModal(null)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="space-y-2.5">
              <input
                type="text"
                value={teachModal.trigger}
                onChange={(e) => setTeachModal({ ...teachModal, trigger: e.target.value })}
                className="w-full rounded-lg bg-slate-950 border border-slate-700 px-3 py-2 text-xs text-white"
              />
              <textarea
                rows={4}
                value={teachModal.reply}
                onChange={(e) => setTeachModal({ ...teachModal, reply: e.target.value })}
                className="w-full rounded-lg bg-slate-950 border border-slate-700 px-3 py-2 text-xs text-white"
              />
            </div>
            <div className="flex justify-end gap-2">
              <button type="button" onClick={() => setTeachModal(null)} className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300">
                Close
              </button>
              <button
                type="button"
                onClick={async () => {
                  await api.addCustomSkill(teachModal.trigger, teachModal.reply);
                  setTeachSaved(true);
                  onRefreshVitals();
                  setTimeout(() => setTeachModal(null), 900);
                }}
                className="text-xs px-4 py-1.5 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-500 text-white font-medium"
              >
                {teachSaved ? "Trained!" : "Save & Retrain Classifier"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
