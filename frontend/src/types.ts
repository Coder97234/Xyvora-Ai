export type BrainState = "idle" | "listening" | "thinking" | "streaming";

export type WorkspaceTab = "chat" | "memory" | "training" | "lab";

export interface CognitiveSettings {
  persona: "nexus" | "axiom" | "muse" | "forge" | "socrates";
  creativity: number;
  verbosity: "concise" | "balanced" | "comprehensive";
  memory_threshold: number;
  auto_learn_memory: boolean;
  show_thought_trace: boolean;
  voice_enabled: boolean;
  external_llm_enabled: boolean;
  external_llm_base_url: string;
  external_llm_api_key: string;
  external_llm_model: string;
}

export interface HealthVitals {
  status: string;
  engine: string;
  semantic_memories: number;
  episodic_memories: number;
  documents: number;
  open_tasks: number;
  custom_skills: number;
  neural_accuracy: number;
  neural_params: number;
  settings: CognitiveSettings;
}

export interface ChatSession {
  id: string;
  title: string;
  persona: string;
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
  created_at: number;
  updated_at: number;
}

export interface TaskListArtifact {
  type: "task_list";
  title: string;
  tasks: AgentTask[];
}

export type ChatArtifact =
  | FunctionPlotArtifact
  | ChartArtifact
  | CodeExecutionArtifact
  | TaskListArtifact;

export interface SemanticMemory {
  id: string;
  subject: string;
  relation: string;
  object: string;
  category: string;
  confidence: number;
  importance: number;
  pinned: number;
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
  thought_trace?: ThoughtStep[];
  artifacts?: ChatArtifact[];
  memories_added?: SemanticMemory[];
  suggestions?: string[];
  feedback?: number;
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

export interface NeuralStats {
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
  loss_curve: number[];
  training_ms: number;
  last_trained_at: number;
  user_samples: { id: string; text: string; intent: string; created_at: number }[];
  custom_skills: { id: string; trigger_phrase: string; response_template: string; created_at: number }[];
}

export interface KnowledgeDoc {
  id: string;
  title: string;
  source_type: string;
  preview?: string;
  char_count?: number;
  chunk_count: number;
  created_at: number;
}
