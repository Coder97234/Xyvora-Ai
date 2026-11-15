export type BrainState = "idle" | "listening" | "thinking" | "streaming";

export type WorkspaceTab =
  | "chat"
  | "research"
  | "coding"
  | "agents"
  | "memory"
  | "training"
  | "lab"
  | "admin";

export interface UserProfile {
  id: string;
  email: string;
  username: string;
  display_name: string;
  role: string;
  is_guest: boolean;
  avatar_url?: string;
  bio?: string;
  created_at: number;
  last_login_at: number;
}

export interface CognitiveSettings {
  persona: string;
  tone?: string;
  creativity: number;
  verbosity: "concise" | "balanced" | "comprehensive";
  custom_instructions?: string;
  domain_instructions?: string;
  memory_threshold: number;
  auto_learn_memory: boolean;
  require_memory_confirmation?: boolean;
  memory_default_ttl_days?: number;
  show_thought_trace: boolean;
  voice_enabled: boolean;
  voice_provider?: "browser" | "neural";
  voice_name?: string;
  voice_rate?: number;
  active_provider?: string;
  active_model?: string;
  request_timeout_sec?: number;
  max_context_tokens?: number;
  enable_web_research?: boolean;
  enable_tool_calling?: boolean;
  theme?: "dark" | "light";
}

export interface StorageDiagnostics {
  engine: string;
  persistence_mode: string;
  postgres_configured: boolean;
  ephemeral_warning: string;
  db_size_bytes: number;
}

export interface HealthVitals {
  status: string;
  engine: string;
  active_provider?: string;
  active_model?: string;
  semantic_memories: number;
  episodic_memories: number;
  documents: number;
  open_tasks: number;
  custom_skills: number;
  projects?: number;
  neural_accuracy: number;
  neural_cv_accuracy?: number;
  neural_params: number;
  storage?: StorageDiagnostics;
  user?: UserProfile;
  settings: CognitiveSettings;
}

export interface ProviderModelItem {
  id: string;
  label: string;
  vision?: boolean;
  tools?: boolean;
}

export interface ProviderInfo {
  id: string;
  name: string;
  description: string;
  requires_api_key: boolean;
  configured: boolean;
  enabled: boolean;
  status: "ready" | "configured" | "missing_credentials" | "disabled";
  has_api_key: boolean;
  masked_api_key: string;
  key_source: string;
  base_url: string;
  default_model: string;
  models: ProviderModelItem[];
}

export interface ProjectWorkspace {
  id: string;
  user_id: string;
  name: string;
  description: string;
  instructions: string;
  color: string;
  file_count?: number;
  session_count?: number;
  open_task_count?: number;
  created_at: number;
  updated_at: number;
}

export interface ProjectFileRevision {
  id: string;
  version: number;
  summary: string;
  created_at: number;
  diff?: string;
}

export interface ProjectFile {
  id: string;
  project_id: string;
  path: string;
  language: string;
  content: string;
  version: number;
  created_at: number;
  updated_at: number;
  diff?: string;
  revisions?: ProjectFileRevision[];
  diagnostics?: CodeDiagnostic[];
}

export interface CodeDiagnostic {
  line: number;
  severity: "error" | "warning" | "info";
  message: string;
  rule?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  persona: string;
  project_id?: string | null;
  summary?: string;
  pinned?: number;
  created_at: number;
  updated_at: number;
  message_count?: number;
}

export interface ThoughtStep {
  step: string;
  detail: string;
  ms: number;
  distribution?: { intent: string; probability: number }[];
  recalled_nodes?: { triple: string; similarity: number }[];
}

export interface ToolActivity {
  tool: string;
  label: string;
  status?: "running" | "done";
  ok?: boolean;
  duration_ms?: number;
}

export interface CitationItem {
  source_type: string;
  title: string;
  section?: string;
  url?: string;
  snippet?: string;
  score?: number;
}

export interface ChatAttachment {
  id: string;
  filename: string;
  mime_type: string;
  kind: "image" | "document" | string;
  size_bytes: number;
  width?: number | null;
  height?: number | null;
  data_url?: string;
  extracted_preview?: string;
  is_generated?: boolean;
}

export interface FunctionPlotPoint {
  x: number;
  y: number | null;
  dy: number | null;
}

export interface FunctionPlotArtifact {
  type: "function_plot";
  title: string;
  expression: string;
  derivative: string;
  points: FunctionPlotPoint[];
  y_min: number;
  y_max: number;
}

export interface ChartArtifact {
  type: "chart";
  chart_type: "bar" | "line" | "area";
  title: string;
  data: { label: string; value: number }[];
}

export interface CodeExecutionArtifact {
  type: "code_execution";
  title: string;
  code: string;
  stdout: string;
  stderr: string;
  ok: boolean;
  duration_ms: number;
}

export interface AgentTask {
  id: string;
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
  status: "todo" | "done";
  due_note: string;
  project_id?: string | null;
  created_at: number;
  updated_at: number;
}

export interface TaskListArtifact {
  type: "task_list";
  title: string;
  tasks: AgentTask[];
}

export interface GeneratedImageArtifact {
  type: "generated_image";
  title: string;
  image_url: string;
  prompt: string;
  revised_prompt?: string;
  provider: string;
  is_generated: boolean;
}

export type ChatArtifact =
  | FunctionPlotArtifact
  | ChartArtifact
  | CodeExecutionArtifact
  | TaskListArtifact
  | GeneratedImageArtifact;

export interface SemanticMemory {
  id: string;
  subject: string;
  relation: string;
  object: string;
  category: string;
  confidence: number;
  importance: number;
  pinned: number;
  verified?: number;
  expires_at?: number | null;
  source_text: string;
  access_count: number;
  created_at: number;
  updated_at: number;
  similarity?: number;
  status?: string;
  previous_object?: string;
}

export interface EpisodicMemory {
  id: string;
  session_id: string;
  user_text: string;
  ai_summary: string;
  topics: string;
  sentiment: number;
  importance: number;
  created_at: number;
  similarity?: number;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  intent?: string;
  confidence?: number;
  provider_used?: string;
  model_used?: string;
  thought_trace?: ThoughtStep[];
  tool_calls?: ToolActivity[];
  citations?: CitationItem[];
  attachments?: ChatAttachment[];
  artifacts?: ChatArtifact[];
  memories_added?: SemanticMemory[];
  suggestions?: string[];
  feedback?: number;
  edited_at?: number | null;
  created_at: number;
}

export interface GraphNode {
  id: string;
  label: string;
  type: "user" | "ai" | "entity" | "concept" | "document";
  category: string;
  weight: number;
  meta?: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  category: string;
  confidence: number;
  pinned: boolean;
  access_count: number;
}

export interface MemoryGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  memories: SemanticMemory[];
  episodic: EpisodicMemory[];
  history: {
    id: string;
    memory_id: string;
    action: string;
    previous_value: string;
    new_value: string;
    timestamp: number;
  }[];
}

export interface TrainingRunRecord {
  id: string;
  architecture: string;
  sample_count: number;
  intent_count: number;
  train_accuracy: number;
  cv_accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  duration_ms: number;
  created_at: number;
}

export interface NeuralStats {
  model_scope?: string;
  scope_disclaimer?: string;
  architecture: string;
  feature_dim: number;
  parameter_count: number;
  sample_count: number;
  intent_count: number;
  intents: string[];
  intent_counts: Record<string, number>;
  epochs_run: number;
  training_accuracy: number;
  cv_accuracy: number;
  precision_macro?: number;
  recall_macro?: number;
  f1_macro?: number;
  per_class_metrics?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix?: { labels: string[]; matrix: number[][] };
  training_history?: TrainingRunRecord[];
  loss_curve: number[];
  training_ms: number;
  last_trained_at: number;
  user_samples: { id: string; text: string; intent: string; created_at: number }[];
  custom_skills: { id: string; trigger_phrase: string; response_template: string; created_at: number }[];
}

export interface KnowledgeDoc {
  id: string;
  title: string;
  filename?: string;
  mime_type?: string;
  file_size?: number;
  collection?: string;
  project_id?: string | null;
  source_type: string;
  preview?: string;
  char_count?: number;
  chunk_count: number;
  created_at: number;
}

export interface ResearchSource {
  title: string;
  url: string;
  snippet: string;
  provider: string;
  published_date?: string;
  extracted_excerpt?: string;
  credibility?: { tier: "high" | "medium" | "low"; label: string };
}

export interface ResearchReport {
  report_id?: string;
  id?: string;
  ok?: boolean;
  query: string;
  mode: string;
  summary: string;
  report_markdown?: string;
  sources: ResearchSource[];
  follow_up_questions?: string[];
  diagnostics?: Record<string, unknown>;
  created_at?: number;
}

export interface AgentStep {
  id: string;
  step_index: number;
  title: string;
  tool_name: string;
  tool_args: Record<string, unknown>;
  status: "pending" | "running" | "awaiting_approval" | "completed" | "failed" | "skipped";
  requires_approval: boolean;
  retries: number;
  output_summary: string;
  duration_ms: number;
}

export interface AgentRun {
  id: string;
  goal: string;
  status: "planning" | "running" | "awaiting_approval" | "completed" | "failed" | "cancelled";
  current_step: number;
  max_steps: number;
  pending_approval?: { step_index: number; title: string; tool_name: string; reason?: string } | null;
  final_output: string;
  error_message?: string;
  steps: AgentStep[];
  created_at: number;
  updated_at: number;
}

export interface ExtensionItem {
  id: string;
  name: string;
  description: string;
  category: string;
  version: string;
  author: string;
  permissions: string[];
  compatibility: string;
  input_schema: {
    type: string;
    required?: string[];
    properties?: Record<string, { type: string; description?: string }>;
  };
  builtin: boolean;
  installed: boolean;
}

export interface CustomPersonality {
  id: string;
  slug: string;
  name: string;
  description: string;
  system_instructions: string;
  tone: string;
  creativity: number;
  verbosity: string;
  domain_focus: string;
}

export interface AdminAnalytics {
  total_registered_users: number;
  total_workspace_accounts: number;
  active_users_24h: number;
  requests_24h: number;
  total_requests: number;
  failed_requests: number;
  provider_errors: number;
  avg_latency_ms: number;
  total_estimated_tokens: number;
  sandbox_executions: number;
  sandbox_blocked_or_failed: number;
  tool_stats: { tool_name: string; calls: number; avg_ms: number }[];
  recent_security_events: { event_type: string; severity: string; resource: string; detail: string; created_at: number }[];
  storage: StorageDiagnostics;
}
