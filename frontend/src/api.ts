import {
  AdminAnalytics,
  AgentRun,
  AgentTask,
  ChatArtifact,
  ChatAttachment,
  ChatMessage,
  ChatSession,
  CitationItem,
  CognitiveSettings,
  CustomPersonality,
  ExtensionItem,
  HealthVitals,
  KnowledgeDoc,
  MemoryGraphData,
  NeuralStats,
  ProjectFile,
  ProjectWorkspace,
  ProviderInfo,
  ResearchReport,
  SemanticMemory,
  ThoughtStep,
  ToolActivity,
  UserProfile,
} from "./types";

const AUTH_TOKEN_KEY = "xyvora_auth_token_v2";
const ACTIVE_UID_KEY = "xyvora_active_uid_v2";

function getAuthToken(): string {
  try {
    return localStorage.getItem(AUTH_TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function setAuthToken(token: string, userId = "usr_local_workspace"): void {
  try {
    if (token) localStorage.setItem(AUTH_TOKEN_KEY, token);
    else localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.setItem(ACTIVE_UID_KEY, userId);
  } catch {
    // ignore storage quota errors
  }
}

function activeUid(): string {
  try {
    return localStorage.getItem(ACTIVE_UID_KEY) || "usr_local_workspace";
  } catch {
    return "usr_local_workspace";
  }
}

function memKey(): string {
  return `xyvora_memories_${activeUid()}`;
}

function skillsKey(): string {
  return `xyvora_skills_${activeUid()}`;
}

function settingsKey(): string {
  return `xyvora_settings_${activeUid()}`;
}

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
    // ignore
  }
}

function cacheUserMemories(memories: SemanticMemory[]): void {
  const userOnly = memories.filter((m) => m.category !== "system");
  saveLocal(memKey(), userOnly);
}

function authHeaders(extra?: HeadersInit): HeadersInit {
  const token = getAuthToken();
  const base: Record<string, string> = {};
  if (token) base["Authorization"] = `Bearer ${token}`;
  return { ...base, ...(extra as Record<string, string>) };
}

async function jsonReq<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    ...init,
    headers: authHeaders({ "Content-Type": "application/json", ...(init?.headers || {}) }),
  });
  if (!res.ok) {
    let errMsg = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      errMsg = body.detail || JSON.stringify(body);
    } catch {
      const txt = await res.text();
      if (txt) errMsg = txt;
    }
    throw new Error(errMsg);
  }
  return (await res.json()) as T;
}

export interface StreamHandlers {
  onThought?: (step: ThoughtStep) => void;
  onToolActivity?: (activity: ToolActivity) => void;
  onMemoryConsolidated?: (memories: SemanticMemory[]) => void;
  onToken?: (delta: string) => void;
  onDone?: (payload: {
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
    total_ms: number;
  }) => void;
}

async function consumeSseResponse(res: Response, handlers: StreamHandlers): Promise<void> {
  if (!res.ok || !res.body) {
    let detail = `Stream error (${res.status})`;
    try {
      const b = await res.json();
      detail = b.detail || detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
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
        } else if (evtType === "tool_activity" && handlers.onToolActivity) {
          handlers.onToolActivity(parsed);
        } else if (evtType === "memory_consolidated") {
          if (Array.isArray(parsed) && parsed.length > 0) {
            const existing = loadLocal<SemanticMemory[]>(memKey(), []);
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
        // ignore malformed partial frame
      }
    }
  }
}

export const api = {
  getHealth: async (): Promise<HealthVitals> => {
    let health: HealthVitals;
    try {
      health = await jsonReq<HealthVitals>("/api/health");
    } catch {
      setAuthToken("", "usr_local_workspace");
      health = await jsonReq<HealthVitals>("/api/health");
    }
    if (health.user?.id) {
      localStorage.setItem(ACTIVE_UID_KEY, health.user.id);
    }
    const cachedMems = loadLocal<SemanticMemory[]>(memKey(), []);
    const cachedSkills = loadLocal<{ trigger_phrase: string; response_template: string }[]>(skillsKey(), []);
    const cachedSettings = loadLocal<Partial<CognitiveSettings> | null>(settingsKey(), null);

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

  // Authentication & User Account
  register: async (payload: { email: string; username: string; password: string; display_name?: string }) => {
    const res = await jsonReq<{ token: string; user: UserProfile; settings: CognitiveSettings }>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setAuthToken(res.token, res.user.id);
    return res;
  },
  login: async (payload: { identifier: string; password: string }) => {
    const res = await jsonReq<{ token: string; user: UserProfile; settings: CognitiveSettings }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setAuthToken(res.token, res.user.id);
    return res;
  },
  logout: async () => {
    await jsonReq("/api/auth/logout", { method: "POST" });
    setAuthToken("", "usr_local_workspace");
  },
  refreshSession: async () => {
    const res = await jsonReq<{ token: string; user: UserProfile }>("/api/auth/refresh", { method: "POST" });
    setAuthToken(res.token, res.user.id);
    return res;
  },
  requestPasswordReset: (email: string) =>
    jsonReq<{ ok: boolean; delivery_mode: string; reset_token?: string | null; message: string }>(
      "/api/auth/password-reset/request",
      { method: "POST", body: JSON.stringify({ email }) }
    ),
  confirmPasswordReset: (token: string, new_password: string) =>
    jsonReq<{ reset: boolean }>("/api/auth/password-reset/confirm", {
      method: "POST",
      body: JSON.stringify({ token, new_password }),
    }),
  getMe: () =>
    jsonReq<{
      user: UserProfile;
      usage: {
        total_requests: number;
        requests_last_24h: number;
        estimated_tokens_total: number;
        avg_latency_ms: number;
        sessions_count: number;
        memories_count: number;
        documents_count: number;
        projects_count: number;
      };
      settings: CognitiveSettings;
    }>("/api/auth/me"),
  updateProfile: (patch: { display_name?: string; avatar_url?: string; bio?: string }) =>
    jsonReq<{ user: UserProfile }>("/api/auth/profile", { method: "PATCH", body: JSON.stringify(patch) }),
  changePassword: (current_password: string, new_password: string) =>
    jsonReq<{ ok: boolean }>("/api/auth/change-password", {
      method: "POST",
      body: JSON.stringify({ current_password, new_password }),
    }),
  deleteAccount: async () => {
    const res = await jsonReq<{ deleted: boolean }>("/api/auth/account", { method: "DELETE" });
    setAuthToken("", "usr_local_workspace");
    return res;
  },
  exportAccountWorkspace: () => jsonReq<Record<string, unknown>>("/api/account/export"),
  importAccountWorkspace: (bundle: Record<string, unknown>) =>
    jsonReq<{ imported: boolean; memories_imported: number }>("/api/account/import", {
      method: "POST",
      body: JSON.stringify(bundle),
    }),

  // Providers
  listProviders: () =>
    jsonReq<{ providers: ProviderInfo[]; active: { provider_id: string; model: string; available: boolean } }>("/api/providers"),
  updateProvider: (
    providerId: string,
    payload: { api_key?: string; base_url?: string; default_model?: string; enabled?: boolean }
  ) =>
    jsonReq<{ provider: ProviderInfo; providers: ProviderInfo[] }>(`/api/providers/${providerId}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  testProvider: (providerId: string) =>
    jsonReq<{ ok: boolean; status: string; latency_ms?: number; detail?: string }>(`/api/providers/${providerId}/test`, {
      method: "POST",
    }),

  // Projects & Coding Studio
  listProjects: () => jsonReq<ProjectWorkspace[]>("/api/projects"),
  createProject: (payload: { name: string; description?: string; instructions?: string; color?: string }) =>
    jsonReq<ProjectWorkspace>("/api/projects", { method: "POST", body: JSON.stringify(payload) }),
  patchProject: (id: string, patch: Partial<ProjectWorkspace>) =>
    jsonReq<ProjectWorkspace>(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  deleteProject: (id: string) => jsonReq<{ deleted: boolean }>(`/api/projects/${id}`, { method: "DELETE" }),
  getProjectDetail: (id: string) =>
    jsonReq<{
      project: ProjectWorkspace;
      files: ProjectFile[];
      activity: { id: string; action: string; summary: string; created_at: number }[];
      sessions: ChatSession[];
      tasks: AgentTask[];
      documents: KnowledgeDoc[];
    }>(`/api/projects/${id}/detail`),
  listProjectFiles: (projectId: string) => jsonReq<ProjectFile[]>(`/api/projects/${projectId}/files`),
  saveProjectFile: (projectId: string, payload: { path: string; content: string; language?: string; summary?: string }) =>
    jsonReq<ProjectFile>(`/api/projects/${projectId}/files`, { method: "PUT", body: JSON.stringify(payload) }),
  deleteProjectFile: (projectId: string, fileId: string) =>
    jsonReq<{ deleted: boolean }>(`/api/projects/${projectId}/files/${fileId}`, { method: "DELETE" }),
  revertProjectFile: (projectId: string, fileId: string, revisionId: string) =>
    jsonReq<ProjectFile>(`/api/projects/${projectId}/files/${fileId}/revert`, {
      method: "POST",
      body: JSON.stringify({ revision_id: revisionId }),
    }),
  analyzeProject: (projectId: string) =>
    jsonReq<{
      file_count: number;
      total_lines: number;
      languages: Record<string, number>;
      dependencies: string[];
      diagnostics: { path: string; line: number; severity: string; message: string }[];
    }>(`/api/projects/${projectId}/analyze`),
  runProjectTests: (projectId: string) =>
    jsonReq<{ ok: boolean; stdout: string; stderr: string; duration_ms: number; test_files?: string[] }>(
      `/api/projects/${projectId}/run-tests`,
      { method: "POST" }
    ),
  codingAssist: (payload: {
    project_id?: string | null;
    action: string;
    prompt: string;
    path: string;
    language: string;
    content: string;
  }) =>
    jsonReq<{
      ok: boolean;
      explanation: string;
      proposed_content?: string | null;
      diff?: string;
      diagnostics?: { line: number; severity: string; message: string }[];
    }>("/api/coding/assist", { method: "POST", body: JSON.stringify(payload) }),

  // Sessions, Search, Attachments & Chat
  listSessions: (params?: { project_id?: string | null; q?: string }) => {
    const qs = new URLSearchParams();
    if (params?.project_id) qs.set("project_id", params.project_id);
    if (params?.q) qs.set("q", params.q);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return jsonReq<ChatSession[]>(`/api/sessions${suffix}`);
  },
  createSession: (title?: string, project_id?: string | null) =>
    jsonReq<ChatSession>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title: title || "New Neural Session", project_id: project_id || null }),
    }),
  patchSession: (id: string, patch: { title?: string; pinned?: number; project_id?: string | null }) =>
    jsonReq<ChatSession>(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  deleteSession: (id: string) => jsonReq<{ deleted: boolean }>(`/api/sessions/${id}`, { method: "DELETE" }),
  summarizeSession: (id: string) =>
    jsonReq<{ summary: string; message_count: number }>(`/api/sessions/${id}/summarize`, { method: "POST" }),
  getMessages: (sessionId: string) => jsonReq<ChatMessage[]>(`/api/sessions/${sessionId}/messages`),
  sendFeedback: (messageId: string, feedback: number) =>
    jsonReq<{ ok: boolean }>(`/api/messages/${messageId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ feedback }),
    }),
  uploadAttachment: async (file: File, sessionId?: string, projectId?: string | null): Promise<ChatAttachment> => {
    const fd = new FormData();
    fd.append("file", file);
    if (sessionId) fd.append("session_id", sessionId);
    if (projectId) fd.append("project_id", projectId);
    const res = await fetch("/api/attachments", {
      method: "POST",
      credentials: "include",
      headers: authHeaders(),
      body: fd,
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `Attachment upload failed (${res.status})`);
    }
    return (await res.json()) as ChatAttachment;
  },

  streamChat: async (
    sessionId: string,
    message: string,
    handlers: StreamHandlers,
    opts?: {
      project_id?: string | null;
      attachment_ids?: string[];
      provider?: string;
      model?: string;
    }
  ): Promise<void> => {
    const client_memories = loadLocal<SemanticMemory[]>(memKey(), []);
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      credentials: "include",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        session_id: sessionId,
        message,
        project_id: opts?.project_id || null,
        attachment_ids: opts?.attachment_ids || [],
        provider: opts?.provider,
        model: opts?.model,
        client_memories,
      }),
    });
    await consumeSseResponse(res, handlers);
  },

  regenerateChat: async (sessionId: string, handlers: StreamHandlers): Promise<void> => {
    const res = await fetch(`/api/sessions/${sessionId}/regenerate`, {
      method: "POST",
      credentials: "include",
      headers: authHeaders({ "Content-Type": "application/json" }),
    });
    await consumeSseResponse(res, handlers);
  },

  editAndResendMessage: async (messageId: string, content: string, handlers: StreamHandlers): Promise<void> => {
    const res = await fetch(`/api/messages/${messageId}`, {
      method: "PUT",
      credentials: "include",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ content }),
    });
    await consumeSseResponse(res, handlers);
  },

  // Web Research Studio
  getResearchStatus: () => jsonReq<Record<string, unknown>>("/api/research/status"),
  runResearch: (payload: { query: string; mode?: "quick" | "deep"; max_results?: number; follow_links?: boolean; project_id?: string | null }) =>
    jsonReq<ResearchReport>("/api/research/query", { method: "POST", body: JSON.stringify(payload) }),
  listResearchHistory: () => jsonReq<ResearchReport[]>("/api/research/history"),
  deleteResearchReport: (id: string) => jsonReq<{ deleted: boolean }>(`/api/research/history/${id}`, { method: "DELETE" }),

  // Autonomous Agents & Extensions
  startAgentRun: (payload: { goal: string; session_id?: string; project_id?: string | null }) =>
    jsonReq<AgentRun>("/api/agents/runs", { method: "POST", body: JSON.stringify(payload) }),
  listAgentRuns: () => jsonReq<AgentRun[]>("/api/agents/runs"),
  getAgentRun: (runId: string) => jsonReq<AgentRun>(`/api/agents/runs/${runId}`),
  approveAgentRun: (runId: string, approved: boolean) =>
    jsonReq<AgentRun>(`/api/agents/runs/${runId}/approve`, { method: "POST", body: JSON.stringify({ approved }) }),
  cancelAgentRun: (runId: string) => jsonReq<AgentRun>(`/api/agents/runs/${runId}/cancel`, { method: "POST" }),

  listExtensions: (category?: string) =>
    jsonReq<ExtensionItem[]>(`/api/extensions${category ? `?category=${encodeURIComponent(category)}` : ""}`),
  installExtension: (id: string) => jsonReq<ExtensionItem>(`/api/extensions/${id}/install`, { method: "POST" }),
  uninstallExtension: (id: string) => jsonReq<{ id: string; installed: boolean }>(`/api/extensions/${id}/uninstall`, { method: "POST" }),
  createCustomExtension: (payload: { name: string; description: string; category: string; prompt_template: string }) =>
    jsonReq<ExtensionItem>("/api/extensions/custom", { method: "POST", body: JSON.stringify(payload) }),
  runExtension: (id: string, inputs: Record<string, unknown>) =>
    jsonReq<{ ok: boolean; output?: Record<string, unknown>; error?: string; duration_ms?: number }>(
      `/api/extensions/${id}/run`,
      { method: "POST", body: JSON.stringify({ inputs }) }
    ),

  // Personalities
  listPersonalities: () =>
    jsonReq<{ builtin: { slug: string; name: string }[]; custom: CustomPersonality[] }>("/api/personalities"),
  createPersonality: (payload: {
    name: string;
    description?: string;
    system_instructions: string;
    tone?: string;
    creativity?: number;
    verbosity?: string;
    domain_focus?: string;
  }) => jsonReq<CustomPersonality>("/api/personalities", { method: "POST", body: JSON.stringify(payload) }),
  deletePersonality: (id: string) => jsonReq<{ deleted: boolean }>(`/api/personalities/${id}`, { method: "DELETE" }),

  // Memory Palace
  getMemoryGraph: async (projectId?: string | null) => {
    const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
    const data = await jsonReq<MemoryGraphData>(`/api/memory/graph${qs}`);
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
    ttl_days?: number;
    project_id?: string | null;
  }) => {
    const rec = await jsonReq<SemanticMemory>("/api/memory/facts", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const existing = loadLocal<SemanticMemory[]>(memKey(), []);
    cacheUserMemories([rec, ...existing.filter((m) => m.id !== rec.id)]);
    return rec;
  },
  patchMemoryFact: async (
    id: string,
    patch: Partial<{ subject: string; relation: string; object: string; pinned: number; importance: number; verified: number }>
  ) => {
    const updated = await jsonReq<SemanticMemory>(`/api/memory/facts/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    const existing = loadLocal<SemanticMemory[]>(memKey(), []);
    cacheUserMemories(existing.map((m) => (m.id === id ? updated : m)));
    return updated;
  },
  approveMemoryFact: async (id: string) => {
    const updated = await jsonReq<SemanticMemory>(`/api/memory/facts/${id}/approve`, { method: "POST" });
    const existing = loadLocal<SemanticMemory[]>(memKey(), []);
    cacheUserMemories(existing.map((m) => (m.id === id ? updated : m)));
    return updated;
  },
  deleteMemoryFact: async (id: string) => {
    const res = await jsonReq<{ deleted: boolean }>(`/api/memory/facts/${id}`, { method: "DELETE" });
    const existing = loadLocal<SemanticMemory[]>(memKey(), []);
    cacheUserMemories(existing.filter((m) => m.id !== id));
    return res;
  },
  resetUserMemories: async () => {
    const res = await jsonReq<{ reset: boolean }>("/api/memory/reset", { method: "POST" });
    saveLocal(memKey(), []);
    return res;
  },

  // Neural Training Studio
  getNeuralStats: async () => {
    const s = await jsonReq<NeuralStats>("/api/neural/stats");
    saveLocal(skillsKey(), s.custom_skills);
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
  validateDataset: () =>
    jsonReq<{
      valid: boolean;
      total_samples: number;
      intent_count: number;
      duplicate_count: number;
      imbalanced_intents: string[];
      warnings: string[];
    }>("/api/neural/dataset/validate"),
  exportDataset: () =>
    jsonReq<{ samples: { text: string; intent: string; source?: string }[]; intent_count: number }>("/api/neural/dataset/export"),
  importDataset: (samples: { text: string; intent: string }[], replace_existing = false) =>
    jsonReq<{ imported: number; stats: NeuralStats }>("/api/neural/dataset/import", {
      method: "POST",
      body: JSON.stringify({ samples, replace_existing }),
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
    const existing = loadLocal<{ id?: string; trigger_phrase: string; response_template: string }[]>(skillsKey(), []);
    saveLocal(skillsKey(), [rec, ...existing]);
    return rec;
  },
  deleteCustomSkill: async (id: string) => {
    const res = await jsonReq<{ deleted: boolean }>(`/api/neural/skills/${id}`, { method: "DELETE" });
    const existing = loadLocal<{ id?: string; trigger_phrase: string; response_template: string }[]>(skillsKey(), []);
    saveLocal(skillsKey(), existing.filter((s) => s.id !== id));
    return res;
  },
  retrainNeural: (hidden_1 = 128, hidden_2 = 64, max_iter = 85) =>
    jsonReq<NeuralStats>("/api/neural/retrain", {
      method: "POST",
      body: JSON.stringify({ hidden_1, hidden_2, max_iter }),
    }),

  // AI Lab Tools & Vault
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
  getSandboxHistory: () =>
    jsonReq<{ id: string; code_sha256: string; ok: number; denied_reason: string; duration_ms: number; created_at: number }[]>(
      "/api/sandbox/history"
    ),
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
  generateImage: (prompt: string, size = "1024x1024") =>
    jsonReq<{ ok: boolean; image_url?: string; revised_prompt?: string; error?: string; provider?: string }>(
      "/api/images/generate",
      { method: "POST", body: JSON.stringify({ prompt, size }) }
    ),
  synthesizeSpeech: (text: string, voice = "alloy") =>
    jsonReq<{ ok: boolean; audio_data_url?: string; error?: string }>("/api/voice/tts", {
      method: "POST",
      body: JSON.stringify({ text, voice }),
    }),

  // Knowledge Vault Documents
  listDocuments: (params?: { collection?: string; project_id?: string | null; q?: string }) => {
    const qs = new URLSearchParams();
    if (params?.collection) qs.set("collection", params.collection);
    if (params?.project_id) qs.set("project_id", params.project_id);
    if (params?.q) qs.set("q", params.q);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return jsonReq<KnowledgeDoc[]>(`/api/documents${suffix}`);
  },
  getDocumentDetail: (docId: string) =>
    jsonReq<KnowledgeDoc & { content: string; chunks: { id: string; chunk_index: number; section_label: string; content: string }[] }>(
      `/api/documents/${docId}`
    ),
  createDocument: (title: string, content: string, collection = "general", project_id?: string | null) =>
    jsonReq<KnowledgeDoc>("/api/documents", {
      method: "POST",
      body: JSON.stringify({ title, content, source_type: "note", collection, project_id: project_id || null }),
    }),
  uploadDocumentFile: async (file: File, collection = "general", projectId?: string | null): Promise<KnowledgeDoc> => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("collection", collection);
    if (projectId) fd.append("project_id", projectId);
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      credentials: "include",
      headers: authHeaders(),
      body: fd,
    });
    if (!res.ok) {
      let msg = `Upload failed (${res.status})`;
      try {
        const j = await res.json();
        msg = j.detail || msg;
      } catch {
        // ignore
      }
      throw new Error(msg);
    }
    return (await res.json()) as KnowledgeDoc;
  },
  reindexDocument: (docId: string) =>
    jsonReq<KnowledgeDoc>(`/api/documents/${docId}/reindex`, { method: "POST" }),
  searchDocuments: (query: string, collection?: string, project_id?: string | null) =>
    jsonReq<{
      rag_chunks: { doc_id: string; title: string; section_label: string; chunk_index: number; content: string; score: number }[];
      encyclopedia: { title: string; summary: string }[];
    }>("/api/documents/search", {
      method: "POST",
      body: JSON.stringify({ query, collection, project_id: project_id || null }),
    }),
  deleteDocument: (id: string) => jsonReq<{ deleted: boolean }>(`/api/documents/${id}`, { method: "DELETE" }),

  // Tasks
  listTasks: (projectId?: string | null) =>
    jsonReq<AgentTask[]>(`/api/tasks${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`),
  createTask: (title: string, priority: "high" | "medium" | "low" = "medium", due_note = "", project_id?: string | null) =>
    jsonReq<AgentTask>("/api/tasks", {
      method: "POST",
      body: JSON.stringify({ title, priority, due_note, project_id: project_id || null }),
    }),
  patchTask: (id: string, patch: Partial<Pick<AgentTask, "title" | "status" | "priority">>) =>
    jsonReq<AgentTask>(`/api/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),
  deleteTask: (id: string) => jsonReq<{ deleted: boolean }>(`/api/tasks/${id}`, { method: "DELETE" }),

  // Settings & Admin Analytics
  getSettings: () => jsonReq<CognitiveSettings>("/api/settings"),
  patchSettings: async (patch: Partial<CognitiveSettings>) => {
    const updated = await jsonReq<CognitiveSettings>("/api/settings", {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    saveLocal(settingsKey(), updated);
    return updated;
  },
  getAdminAnalytics: () => jsonReq<AdminAnalytics>("/api/admin/analytics"),
};
