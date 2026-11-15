import React, { useEffect, useRef, useState } from "react";
import {
  Brain,
  ChevronDown,
  ChevronRight,
  Cpu,
  Database,
  GraduationCap,
  Mic,
  MicOff,
  Send,
  Sparkles,
  ThumbsUp,
  Volume2,
  VolumeX,
  Wand2,
  X,
} from "lucide-react";
import { api } from "../api";
import {
  BrainState,
  ChatArtifact,
  ChatMessage,
  CognitiveSettings,
  SemanticMemory,
  ThoughtStep,
  WorkspaceTab,
} from "../types";
import { ArtifactsList } from "./ArtifactsRenderer";
import { MarkdownView } from "./MarkdownView";

interface ChatWorkspaceProps {
  sessionId: string;
  settings: CognitiveSettings;
  brainState: BrainState;
  setBrainState: (s: BrainState) => void;
  onRefreshVitals: () => void;
  onMemoryToast: (memories: SemanticMemory[]) => void;
  onNavigateTab: (tab: WorkspaceTab, codePayload?: string) => void;
}

const SLASH_COMMANDS = [
  { cmd: "/remember ", template: "Remember that ", desc: "Store a permanent fact or preference in Memory Palace" },
  { cmd: "/recall", template: "What do you remember about me so far?", desc: "Query your Semantic Knowledge Graph" },
  { cmd: "/math ", template: "Differentiate and plot sin(x) * cos(x/2)", desc: "Solve symbolic calculus/algebra & plot f(x)" },
  { cmd: "/simulate", template: "Simulate 1000 dice rolls in Python and chart the results", desc: "Synthesize & execute Python simulation" },
  { cmd: "/chart ", template: "Chart Jan 120, Feb 165, Mar 210, Apr 195, May 290", desc: "Render interactive data visualization" },
  { cmd: "/summarize ", template: "Summarize this text: ", desc: "Run TextRank centrality summarization" },
  { cmd: "/task ", template: "Add a task to ", desc: "Create or manage autonomous tasks" },
];

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  sessionId,
  settings,
  brainState,
  setBrainState,
  onRefreshVitals,
  onMemoryToast,
  onNavigateTab,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streamingContent, setStreamingContent] = useState("");
  const [liveThoughts, setLiveThoughts] = useState<ThoughtStep[]>([]);
  const [liveMemories, setLiveMemories] = useState<SemanticMemory[]>([]);
  const [expandedTraces, setExpandedTraces] = useState<Record<string, boolean>>({});
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [micNote, setMicNote] = useState<string | null>(null);

  // Quick 1-click "Teach Xyvora" modal state
  const [teachModal, setTeachModal] = useState<{
    trigger: string;
    reply: string;
  } | null>(null);
  const [teachSaved, setTeachSaved] = useState(false);

  const bottomRef = useRef<HTMLDivElement | null>(null);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!sessionId) return;
    api.getMessages(sessionId).then(setMessages).catch(console.error);
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, liveThoughts]);

  const speakText = (msgId: string, rawMarkdown: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    if (speakingId === msgId) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
      return;
    }
    window.speechSynthesis.cancel();
    const clean = rawMarkdown
      .replace(/```[\s\S]*?```/g, " Code block omitted. ")
      .replace(/[#*_`~>|]/g, "")
      .slice(0, 900);
    const utter = new SpeechSynthesisUtterance(clean);
    utter.rate = 1.03;
    utter.pitch = 1.0;
    utter.onend = () => setSpeakingId(null);
    utter.onerror = () => setSpeakingId(null);
    setSpeakingId(msgId);
    window.speechSynthesis.speak(utter);
  };

  const toggleVoiceMic = () => {
    const SpeechRec =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRec) {
      setMicNote("Voice dictation isn't supported in this browser; type or use neural TTS speaker icons!");
      setTimeout(() => setMicNote(null), 4000);
      return;
    }

    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
      setBrainState("idle");
      return;
    }

    try {
      const rec = new SpeechRec();
      rec.continuous = false;
      rec.interimResults = true;
      rec.lang = "en-US";
      rec.onstart = () => {
        setIsListening(true);
        setBrainState("listening");
      };
      rec.onresult = (event: any) => {
        let transcript = "";
        for (let i = 0; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setInput(transcript);
      };
      rec.onerror = () => {
        setIsListening(false);
        setBrainState("idle");
        setMicNote("Microphone permission blocked by browser/iframe — type directly below!");
        setTimeout(() => setMicNote(null), 4000);
      };
      rec.onend = () => {
        setIsListening(false);
        setBrainState("idle");
      };
      recognitionRef.current = rec;
      rec.start();
    } catch {
      setIsListening(false);
      setBrainState("idle");
    }
  };

  const sendMessage = async (overrideText?: string) => {
    const raw = (overrideText ?? input).trim();
    if (!raw || brainState === "thinking" || brainState === "streaming") return;

    setInput("");
    const optimisticUser: ChatMessage = {
      id: `temp-u-${Date.now()}`,
      session_id: sessionId,
      role: "user",
      content: raw,
      created_at: Date.now() / 1000,
    };
    setMessages((prev) => [...prev, optimisticUser]);
    setStreamingContent("");
    setLiveThoughts([]);
    setLiveMemories([]);
    setBrainState("thinking");

    let accumulated = "";
    try {
      await api.streamChat(sessionId, raw, {
        onThought: (step) => {
          setLiveThoughts((prev) => [...prev, step]);
        },
        onMemoryConsolidated: (mems) => {
          setLiveMemories(mems);
          onMemoryToast(mems);
        },
        onToken: (delta) => {
          setBrainState("streaming");
          accumulated += delta;
          setStreamingContent(accumulated);
        },
        onDone: (donePayload) => {
          const aiMsg: ChatMessage = {
            id: donePayload.message_id,
            session_id: sessionId,
            role: "assistant",
            content: accumulated,
            intent: donePayload.intent,
            confidence: donePayload.confidence,
            thought_trace: donePayload.thought_trace,
            artifacts: donePayload.artifacts as ChatArtifact[],
            memories_added: donePayload.memories_added,
            suggestions: donePayload.suggestions,
            created_at: Date.now() / 1000,
          };
          setMessages((prev) => [...prev, aiMsg]);
          setStreamingContent("");
          setLiveThoughts([]);
          setLiveMemories([]);
          setBrainState("idle");
          onRefreshVitals();

          if (settings.voice_enabled && accumulated) {
            speakText(donePayload.message_id, accumulated);
          }
        },
      });
    } catch (err) {
      console.error(err);
      setBrainState("idle");
      setStreamingContent("");
    }
  };

  const handleFeedback = async (msg: ChatMessage, val: number) => {
    const next = msg.feedback === val ? 0 : val;
    await api.sendFeedback(msg.id, next);
    setMessages((prev) =>
      prev.map((m) => (m.id === msg.id ? { ...m, feedback: next } : m))
    );
  };

  const submitTeachSkill = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!teachModal || !teachModal.trigger.trim() || !teachModal.reply.trim()) return;
    await api.addCustomSkill(teachModal.trigger, teachModal.reply);
    setTeachSaved(true);
    onRefreshVitals();
    setTimeout(() => {
      setTeachSaved(false);
      setTeachModal(null);
    }, 1000);
  };

  const showSlashMenu = input.startsWith("/") && !input.includes(" ");
  const lastAssistantMsg = [...messages].reverse().find((m) => m.role === "assistant");

  return (
    <div className="flex flex-col h-full relative">
      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-5 space-y-5">
        {messages.map((msg, idx) => {
          const isUser = msg.role === "user";
          const traceOpen = !!expandedTraces[msg.id];
          const prevUserText =
            !isUser && idx > 0 && messages[idx - 1].role === "user"
              ? messages[idx - 1].content
              : "";

          return (
            <div
              key={msg.id}
              className={`flex ${isUser ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-3xl w-full ${
                  isUser
                    ? "md:w-auto md:max-w-xl bg-gradient-to-br from-cyan-600/25 to-indigo-600/25 border border-cyan-400/30 rounded-2xl rounded-tr-sm px-4 py-3 text-slate-100 shadow-md"
                    : "glass-panel rounded-2xl rounded-tl-sm px-4 md:px-5 py-4 shadow-xl"
                }`}
              >
                {/* Assistant Cognitive Header & Thought Trace */}
                {!isUser && (
                  <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-bold tracking-wide text-cyan-300 flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                        XYVORA
                      </span>
                      {msg.intent && msg.intent !== "welcome" && (
                        <span className="text-[10.5px] font-mono px-2 py-0.5 rounded-full bg-slate-900/90 border border-slate-700/80 text-slate-300">
                          {msg.intent} • {Math.round((msg.confidence || 0) * 100)}%
                        </span>
                      )}
                      {settings.show_thought_trace &&
                        msg.thought_trace &&
                        msg.thought_trace.length > 0 && (
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedTraces((p) => ({ ...p, [msg.id]: !p[msg.id] }))
                            }
                            className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-indigo-500/15 hover:bg-indigo-500/25 text-indigo-200 border border-indigo-400/25 transition"
                          >
                            <Cpu className="w-3 h-3" />
                            {msg.thought_trace.length} Neural Steps
                            {traceOpen ? (
                              <ChevronDown className="w-3 h-3" />
                            ) : (
                              <ChevronRight className="w-3 h-3" />
                            )}
                          </button>
                        )}
                    </div>

                    {/* Memory Consolidated Pills */}
                    {msg.memories_added && msg.memories_added.length > 0 && (
                      <button
                        type="button"
                        onClick={() => onNavigateTab("memory")}
                        className="flex items-center gap-1.5 text-[11px] px-2.5 py-0.5 rounded-full bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 transition"
                        title="View consolidated facts in Memory Palace"
                      >
                        <Database className="w-3 h-3" />+{msg.memories_added.length} Memory Saved
                      </button>
                    )}
                  </div>
                )}

                {/* Expanded Reasoning Trace */}
                {!isUser && traceOpen && msg.thought_trace && (
                  <div className="mb-3 rounded-xl bg-[#040812]/95 border border-indigo-500/25 p-3 space-y-2 text-xs">
                    {msg.thought_trace.map((st, sIdx) => (
                      <div key={sIdx} className="space-y-1">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-semibold text-indigo-300 font-mono text-[11px]">
                            {sIdx + 1}. {st.step}
                          </span>
                          <span className="text-[10px] font-mono text-slate-500">
                            {st.ms} ms
                          </span>
                        </div>
                        <div className="text-slate-300 text-[11.5px]">{st.detail}</div>
                        {st.distribution && (
                          <div className="flex flex-wrap gap-1.5 pt-0.5">
                            {st.distribution.map((d) => (
                              <span
                                key={d.intent}
                                className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 text-cyan-200/90 border border-slate-800"
                              >
                                {d.intent}: {(d.probability * 100).toFixed(1)}%
                              </span>
                            ))}
                          </div>
                        )}
                        {st.recalled_nodes && st.recalled_nodes.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 pt-0.5">
                            {st.recalled_nodes.map((rn, ri) => (
                              <span
                                key={ri}
                                className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/50 text-emerald-300 border border-emerald-500/25"
                              >
                                {rn.triple} ({rn.similarity})
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Message Body */}
                {isUser ? (
                  <p className="text-[14px] whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                ) : (
                  <MarkdownView
                    content={msg.content}
                    onRunPython={(code) => onNavigateTab("lab", code)}
                  />
                )}

                {/* Interactive Embedded Artifacts */}
                {!isUser && (
                  <ArtifactsList
                    artifacts={msg.artifacts}
                    onOpenCodeInSandbox={(code) => onNavigateTab("lab", code)}
                    onTasksChanged={onRefreshVitals}
                  />
                )}

                {/* Assistant Action Bar */}
                {!isUser && msg.intent !== "welcome" && (
                  <div className="mt-3 pt-2 border-t border-slate-800/70 flex items-center justify-between gap-2 text-slate-400">
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => speakText(msg.id, msg.content)}
                        className={`p-1.5 rounded-lg hover:bg-slate-800/80 transition ${
                          speakingId === msg.id ? "text-cyan-300 bg-cyan-500/15" : ""
                        }`}
                        title="Read aloud with Neural Voice"
                      >
                        {speakingId === msg.id ? (
                          <VolumeX className="w-3.5 h-3.5" />
                        ) : (
                          <Volume2 className="w-3.5 h-3.5" />
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleFeedback(msg, 1)}
                        className={`p-1.5 rounded-lg hover:bg-slate-800/80 transition ${
                          msg.feedback === 1 ? "text-emerald-400 bg-emerald-500/15" : ""
                        }`}
                        title="Helpful response"
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>
                      {prevUserText && (
                        <button
                          type="button"
                          onClick={() =>
                            setTeachModal({
                              trigger: prevUserText,
                              reply: msg.content,
                            })
                          }
                          className="flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg hover:bg-slate-800/80 hover:text-cyan-200 transition"
                          title="Teach Xyvora how to respond to this prompt"
                        >
                          <GraduationCap className="w-3.5 h-3.5" /> Teach Xyvora
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Active Live Streaming Turn */}
        {(brainState === "thinking" || brainState === "streaming") && (
          <div className="flex justify-start">
            <div className="max-w-3xl w-full glass-panel rounded-2xl rounded-tl-sm px-5 py-4 shadow-xl border-cyan-400/30">
              <div className="flex items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-cyan-400 animate-pulse" />
                  <span className="text-xs font-bold text-cyan-300">
                    {brainState === "thinking" ? "XYVORA PROCESSING..." : "XYVORA SYNTHESIZING..."}
                  </span>
                </div>
                {liveMemories.length > 0 && (
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    +{liveMemories.length} Memory Consolidated
                  </span>
                )}
              </div>

              {settings.show_thought_trace && liveThoughts.length > 0 && (
                <div className="mb-3 rounded-lg bg-[#040812]/90 border border-slate-800 p-2.5 space-y-1">
                  {liveThoughts.map((t, i) => (
                    <div key={i} className="text-[11px] font-mono text-indigo-200 flex justify-between">
                      <span>
                        ▸ <strong>{t.step}:</strong> {t.detail}
                      </span>
                      <span className="text-slate-500">{t.ms}ms</span>
                    </div>
                  ))}
                </div>
              )}

              {streamingContent ? (
                <MarkdownView content={streamingContent} />
              ) : (
                <div className="flex items-center gap-1.5 py-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" />
                  <span className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce [animation-delay:120ms]" />
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce [animation-delay:240ms]" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Contextual Follow-up Suggestion Chips */}
        {brainState === "idle" &&
          lastAssistantMsg?.suggestions &&
          lastAssistantMsg.suggestions.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-1">
              {lastAssistantMsg.suggestions.map((sug, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => sendMessage(sug)}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full bg-slate-900/85 hover:bg-cyan-500/15 text-slate-300 hover:text-cyan-200 border border-slate-800 hover:border-cyan-500/40 transition"
                >
                  <Wand2 className="w-3 h-3 text-cyan-400 shrink-0" />
                  <span>{sug}</span>
                </button>
              ))}
            </div>
          )}

        <div ref={bottomRef} />
      </div>

      {/* Input Composer Area */}
      <div className="px-4 md:px-8 pb-4 pt-2 border-t border-slate-800/60 bg-[#060a16]/90 backdrop-blur-md relative">
        {micNote && (
          <div className="mb-2 text-xs text-amber-200 bg-amber-500/15 border border-amber-500/30 rounded-lg px-3 py-1.5">
            {micNote}
          </div>
        )}

        {/* Slash Command Palette */}
        {showSlashMenu && (
          <div className="mb-2 rounded-xl border border-cyan-500/30 bg-slate-950/95 p-2 shadow-2xl space-y-1">
            <div className="text-[10px] font-mono uppercase tracking-wider text-cyan-400 px-2 py-0.5">
              Xyvora Cognitive Slash Commands
            </div>
            {SLASH_COMMANDS.filter((c) => c.cmd.startsWith(input.toLowerCase())).map((item) => (
              <button
                key={item.cmd}
                type="button"
                onClick={() => setInput(item.template)}
                className="w-full flex items-center justify-between gap-3 px-2.5 py-1.5 rounded-lg hover:bg-cyan-500/15 text-left transition"
              >
                <span className="text-xs font-mono text-cyan-200 font-semibold">{item.cmd}</span>
                <span className="text-xs text-slate-400 truncate">{item.desc}</span>
              </button>
            ))}
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          className="flex items-end gap-2 rounded-2xl border border-slate-700/80 focus-within:border-cyan-400/60 bg-slate-900/80 px-3 py-2 shadow-inner transition"
        >
          <button
            type="button"
            onClick={toggleVoiceMic}
            className={`p-2 rounded-xl transition shrink-0 ${
              isListening
                ? "bg-emerald-500/25 text-emerald-300 border border-emerald-400/50 animate-pulse"
                : "text-slate-400 hover:text-cyan-300 hover:bg-slate-800"
            }`}
            title="Voice input dictation"
          >
            {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
            rows={Math.min(4, Math.max(1, input.split("\n").length))}
            placeholder="Ask Xyvora anything, share facts to remember, solve calculus, simulate Python, or type '/' for commands..."
            className="flex-1 bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none py-1.5 max-h-32"
          />

          <button
            type="submit"
            disabled={!input.trim() || brainState === "thinking" || brainState === "streaming"}
            className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-500 hover:from-cyan-400 hover:to-indigo-400 disabled:opacity-40 text-slate-950 font-semibold transition shrink-0 shadow-md"
            title="Transmit to Xyvora Cortex"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>

      {/* Quick 1-Click Teach Xyvora Modal */}
      {teachModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <form
            onSubmit={submitTeachSkill}
            className="max-w-lg w-full glass-panel rounded-2xl p-5 space-y-4 shadow-2xl border border-cyan-400/30"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-cyan-300 font-semibold text-sm">
                <GraduationCap className="w-4 h-4" />
                Teach Xyvora Custom Response Skill
              </div>
              <button
                type="button"
                onClick={() => setTeachModal(null)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-slate-400">
              Train Xyvora on how to answer this trigger phrase in the future (supports{" "}
              <code className="text-cyan-300">{"{name}"}</code> placeholder!).
            </p>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Trigger Utterance</label>
              <input
                type="text"
                value={teachModal.trigger}
                onChange={(e) => setTeachModal({ ...teachModal, trigger: e.target.value })}
                className="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Ideal Xyvora Response</label>
              <textarea
                rows={4}
                value={teachModal.reply}
                onChange={(e) => setTeachModal({ ...teachModal, reply: e.target.value })}
                className="w-full rounded-xl bg-slate-900 border border-slate-700 px-3 py-2 text-xs text-white focus:border-cyan-400 focus:outline-none"
              />
            </div>
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setTeachModal(null)}
                className="px-3 py-1.5 rounded-lg text-xs text-slate-300 hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition"
              >
                {teachSaved ? "Trained & Saved!" : "Save & Retrain Neural Cortex"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
};
