import {
  AgentTask,
  ChatArtifact,
  ChatMessage,
  ChatSession,
  CognitiveSettings,
  HealthVitals,
  KnowledgeDoc,
  MemoryGraphData,
  NeuralStats,
  SemanticMemory,
  ThoughtStep,
} from "./types";

const MEM_STORAGE_KEY = "xyvora_memories_v1";
const SKILLS_STORAGE_KEY = "xyvora_skills_v1";
const SETTINGS_STORAGE_KEY = "xyvora_settings_v1";

function loadLocal<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function saveLocal<T>(key: string, val: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(val));
  } catch {
    // ignore storage quota errors
  }
}

function cacheUserMemories(memories: SemanticMemory[]): void {
  const userOnly = memories.filter((m) => m.category !== "system");
  saveLocal(MEM_STORAGE_KEY, userOnly);
}

async function jsonReq<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || `Request failed (${res.status})`);
  }
  return (await res.json()) as T;
}

export const api = {
  getHealth: async (): Promise<HealthVitals> => {
    const cachedMems = loadLocal<SemanticMemory[]>(MEM_STORAGE_KEY, []);
    const cachedSkills = loadLocal<{ trigger_phrase: string; response_template: string }[]>(SKILLS_STORAGE_KEY, []);
    const cachedSettings = loadLocal<Partial<CognitiveSettings> | null>(SETTINGS_STORAGE_KEY, null);

    let health = await jsonReq<HealthVitals>("/api/health");
    if ((cachedMems.length > 0 && health.semantic_memories <= 3) || cachedSkills.length > health.custom_skills) {
      await jsonReq("/api/sync", {
        method: "POST",
        body: JSON.stringify({
          memories: cachedMems,
          skills: cachedSkills,
          settings: cachedSettings,
        }),
      });
      health = await jsonReq<HealthVitals>("/api/health");
    }
    return health;
  },

  // Sessions & Chat
  listSessions: () => jsonReq<ChatSession[]>("/api/sessions"),
  createSession: (title?: string) =>
    jsonReq<ChatSession>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title: title || "New Neural Session" }),
    }),
  deleteSession: (id: string) =>
    jsonReq<{ deleted: boolean }>(`/api/sessions/${id}`, { method: "DELETE" }),
  getMessages: (sessionId: string) =>
    jsonReq<ChatMessage[]>(`/api/sessions/${sessionId}/messages`),
  sendFeedback: (messageId: string, feedback: number) =>
    jsonReq<{ ok: boolean }>(`/api/messages/${messageId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ feedback }),
    }),

  streamChat: async (
    sessionId: string,
    message: string,
    handlers: {
      onThought?: (step: ThoughtStep) => void;
      onMemoryConsolidated?: (memories: SemanticMemory[]) => void;
      onToken?: (delta: string) => void;
      onDone?: (payload: {
        message_id: string;
        intent: string;
        confidence: number;
        thought_trace: ThoughtStep[];
        artifacts: ChatArtifact[];
        memories_added: SemanticMemory[];
        suggestions: string[];
        total_ms: number;
      }) => void;
    }
  ): Promise<void> => {
    const client_memories = loadLocal<SemanticMemory[]>(MEM_STORAGE_KEY, []);
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message, client_memories }),
    });
    if (!res.ok || !res.body) {
      throw new Error(`Stream error: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        const lines = block.split("\n");
        let evtType = "message";
        let dataStr = "";
        for (const line of lines) {
          if (line.startsWith("event:")) {
            evtType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataStr += line.slice(5).trim();
          }
        }
        if (!dataStr) continue;
        try {
          const parsed = JSON.parse(dataStr);
          if (evtType === "thought" && handlers.onThought) {
            handlers.onThought(parsed);
          } else if (evtType === "memory_consolidated") {
            if (Array.isArray(parsed) && parsed.length > 0) {
              const existing = loadLocal<SemanticMemory[]>(MEM_STORAGE_KEY, []);
              const mergedMap = new Map(
                existing.map((m) => [`${m.subject.toLowerCase()}::${m.relation.toLowerCase()}::${m.object.toLowerCase()}`, m])
              );
              for (const nm of parsed as SemanticMemory[]) {
                if (nm.status === "updated") {
                  for (const [k, v] of mergedMap.entries()) {
                    if (v.subject.toLowerCase() === nm.subject.toLowerCase() && v.relation.toLowerCase() === nm.relation.toLowerCase()) {
                      mergedMap.delete(k);
                    }
                  }
                }
                mergedMap.set(`${nm.subject.toLowerCase()}::${nm.relation.toLowerCase()}::${nm.object.toLowerCase()}`, nm);
              }
              cacheUserMemories(Array.from(mergedMap.values()));
            }
            if (handlers.onMemoryConsolidated) handlers.onMemoryConsolidated(parsed);
          } else if (evtType === "token" && handlers.onToken) {
            handlers.onToken(parsed.delta || "");
          } else if (evtType === "done" && handlers.onDone) {
            handlers.onDone(parsed);
          }
        } catch {
          // ignore malformed partial block
        }
      }
    }
  },

  // Memory Palace
  getMemoryGraph: async () => {
    const data = await jsonReq<MemoryGraphData>("/api/memory/graph");
    cacheUserMemories(data.memories);
    return data;
  },
  probeMemoryRecall: (query: string, threshold = 0.16) =>
    jsonReq<{
      semantic: SemanticMemory[];
      episodic: { id: string; user_text: string; ai_summary: string; similarity?: number; created_at: number }[];
      user_profile: Record<string, unknown>;
    }>("/api/memory/recall", {
      method: "POST",
      body: JSON.stringify({ query, threshold }),
    }),
  addMemoryFact: async (payload: {
    subject: string;
    relation: string;
    object: string;
    category?: string;
    pinned?: number;
  }) => {
    const rec = await jsonReq<SemanticMemory>("/api/memory/facts", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const existing = loadLocal<SemanticMemory[]>(MEM_STORAGE_KEY, []);
    cacheUserMemories([rec, ...existing.filter((m) => m.id !== rec.id)]);
    return rec;
  },
  patchMemoryFact: async (
    id: string,
    patch: Partial<{ subject: string; relation: string; object: string; pinned: number; importance: number }>
  ) => {
    const updated = await jsonReq<SemanticMemory>(`/api/memory/facts/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    const existing = loadLocal<SemanticMemory[]>(MEM_STORAGE_KEY, []);
    cacheUserMemories(existing.map((m) => (m.id === id ? updated : m)));
    return updated;
  },
  deleteMemoryFact: async (id: string) => {
    const res = await jsonReq<{ deleted: boolean }>(`/api/memory/facts/${id}`, { method: "DELETE" });
    const existing = loadLocal<SemanticMemory[]>(MEM_STORAGE_KEY, []);
    cacheUserMemories(existing.filter((m) => m.id !== id));
    return res;
  },
  resetUserMemories: async () => {
    const res = await jsonReq<{ reset: boolean }>("/api/memory/reset", { method: "POST" });
    saveLocal(MEM_STORAGE_KEY, []);
    return res;
  },

  // Neural Training Studio
  getNeuralStats: async () => {
    const s = await jsonReq<NeuralStats>("/api/neural/stats");
    saveLocal(SKILLS_STORAGE_KEY, s.custom_skills);
    return s;
  },
  predictIntent: (text: string) =>
    jsonReq<{
      intent: string;
      confidence: number;
      distribution: { intent: string; probability: number }[];
    }>("/api/neural/predict", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  addTrainingSample: (text: string, intent: string) =>
    jsonReq<{ id: string; text: string; intent: string }>("/api/neural/samples", {
      method: "POST",
      body: JSON.stringify({ text, intent }),
    }),
  deleteTrainingSample: (id: string) =>
    jsonReq<{ deleted: boolean }>(`/api/neural/samples/${id}`, { method: "DELETE" }),
  addCustomSkill: async (trigger_phrase: string, response_template: string) => {
    const rec = await jsonReq<{ id: string; trigger_phrase: string; response_template: string }>("/api/neural/skills", {
      method: "POST",
      body: JSON.stringify({ trigger_phrase, response_template }),
    });
    const existing = loadLocal<{ id?: string; trigger_phrase: string; response_template: string }[]>(SKILLS_STORAGE_KEY, []);
    saveLocal(SKILLS_STORAGE_KEY, [rec, ...existing]);
    return rec;
  },
  deleteCustomSkill: async (id: string) => {
    const res = await jsonReq<{ deleted: boolean }>(`/api/neural/skills/${id}`, { method: "DELETE" });
    const existing = loadLocal<{ id?: string; trigger_phrase: string; response_template: string }[]>(SKILLS_STORAGE_KEY, []);
    saveLocal(SKILLS_STORAGE_KEY, existing.filter((s) => s.id !== id));
    return res;
  },
  retrainNeural: (hidden_1 = 128, hidden_2 = 64, max_iter = 85) =>
    jsonReq<NeuralStats>("/api/neural/retrain", {
      method: "POST",
      body: JSON.stringify({ hidden_1, hidden_2, max_iter }),
    }),

  // AI Lab Tools
  runPython: (code: string) =>
    jsonReq<{
      ok: boolean;
      stdout: string;
      stderr: string;
      duration_ms: number;
      artifacts: ChatArtifact[];
    }>("/api/tools/python", {
      method: "POST",
      body: JSON.stringify({ code }),
    }),
  solveMath: (query: string) =>
    jsonReq<{
      ok: boolean;
      operation?: string;
      input?: string;
      result?: string;
      steps?: string[];
      error?: string;
      artifacts: ChatArtifact[];
    }>("/api/tools/math", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  // RAG Documents
  listDocuments: () => jsonReq<KnowledgeDoc[]>("/api/documents"),
  createDocument: (title: string, content: string) =>
    jsonReq<KnowledgeDoc>("/api/documents", {
      method: "POST",
      body: JSON.stringify({ title, content, source_type: "note" }),
    }),
  uploadDocumentFile: async (file: File): Promise<KnowledgeDoc> => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/documents/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return (await res.json()) as KnowledgeDoc;
  },
  deleteDocument: (id: string) =>
    jsonReq<{ deleted: boolean }>(`/api/documents/${id}`, { method: "DELETE" }),

  // Tasks
  listTasks: () => jsonReq<AgentTask[]>("/api/tasks"),
  createTask: (title: string, priority: "high" | "medium" | "low" = "medium", due_note = "") =>
    jsonReq<AgentTask>("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ title, priority, due_note }),
    }),
  patchTask: (id: string, patch: Partial<Pick<AgentTask, "title" | "status" | "priority">>) =>
    jsonReq<AgentTask>(`/api/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  deleteTask: (id: string) =>
    jsonReq<{ deleted: boolean }>(`/api/tasks/${id}`, { method: "DELETE" }),

  // Cognitive Settings
  getSettings: () => jsonReq<CognitiveSettings>("/api/settings"),
  patchSettings: async (patch: Partial<CognitiveSettings>) => {
    const updated = await jsonReq<CognitiveSettings>("/api/settings", {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    saveLocal(SETTINGS_STORAGE_KEY, updated);
    return updated;
  },
};
