import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from security import decrypt_secret, encrypt_secret, hash_password, logger, mask_secret

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("XYVORA_DATA_DIR"):
    DATA_DIR = os.environ["XYVORA_DATA_DIR"]
elif os.environ.get("VERCEL") or not os.access(_BASE_DIR, os.W_OK):
    DATA_DIR = "/tmp/xyvora_data"
else:
    DATA_DIR = os.path.join(_BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.environ.get("XYVORA_DB_PATH") or os.path.join(DATA_DIR, "xyvora.db")

DEFAULT_USER_ID = "usr_local_workspace"

_db_lock = threading.RLock()

DEFAULT_USER_SETTINGS: Dict[str, Any] = {
    "persona": "nexus",
    "tone": "balanced",
    "creativity": 0.65,
    "verbosity": "balanced",
    "custom_instructions": "",
    "domain_instructions": "",
    "memory_threshold": 0.24,
    "auto_learn_memory": True,
    "memory_Graph_enabled": True,
    "require_memory_confirmation": False,
    "memory_default_ttl_days": 0,
    "show_thought_trace": True,
    "voice_enabled": False,
    "voice_name": "",
    "voice_rate": 1.02,
    "voice_provider": "browser",
    "active_provider": "auto",
    "active_model": "",
    "request_timeout_sec": 30,
    "max_context_tokens": 8000,
    "enable_web_research": True,
    "enable_tool_calling": True,
    "theme": "dark",
}


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r["name"] for r in rows}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
    cols = _table_columns(conn, table)
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")


def init_db() -> None:
    with _db_lock:
        conn = get_conn()
        try:
            # 1. Core + Upgraded Schema Creation
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_guest INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    last_login_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ip_address TEXT DEFAULT '',
                    user_agent TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    settings_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS provider_configs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    base_url TEXT NOT NULL DEFAULT '',
                    encrypted_api_key TEXT NOT NULL DEFAULT '',
                    default_model TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, provider_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    instructions TEXT DEFAULT '',
                    color TEXT DEFAULT '#22d3ee',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS project_files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'python',
                    content TEXT NOT NULL DEFAULT '',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(project_id, path),
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS project_file_revisions (
                    id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    previous_content TEXT NOT NULL,
                    new_content TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES project_files(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS project_activity (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS custom_personalities (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    system_instructions TEXT NOT NULL,
                    tone TEXT DEFAULT 'balanced',
                    creativity REAL DEFAULT 0.65,
                    verbosity TEXT DEFAULT 'balanced',
                    domain_focus TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    UNIQUE(user_id, slug),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    project_id TEXT DEFAULT NULL,
                    title TEXT NOT NULL,
                    persona TEXT NOT NULL DEFAULT 'nexus',
                    summary TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    provider_used TEXT DEFAULT '',
                    model_used TEXT DEFAULT '',
                    thought_trace TEXT DEFAULT '[]',
                    tool_calls TEXT DEFAULT '[]',
                    citations TEXT DEFAULT '[]',
                    attachments TEXT DEFAULT '[]',
                    artifacts TEXT DEFAULT '[]',
                    memories_added TEXT DEFAULT '[]',
                    suggestions TEXT DEFAULT '[]',
                    feedback INTEGER DEFAULT 0,
                    edited_at REAL DEFAULT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS attachments (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT DEFAULT NULL,
                    project_id TEXT DEFAULT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'file',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    width INTEGER DEFAULT NULL,
                    height INTEGER DEFAULT NULL,
                    extracted_text TEXT DEFAULT '',
                    data_url TEXT DEFAULT '',
                    is_generated INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    project_id TEXT DEFAULT NULL,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    confidence REAL NOT NULL DEFAULT 0.92,
                    importance REAL NOT NULL DEFAULT 0.75,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    verified INTEGER NOT NULL DEFAULT 1,
                    expires_at REAL DEFAULT NULL,
                    source_text TEXT DEFAULT '',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    previous_value TEXT DEFAULT '',
                    new_value TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    project_id TEXT DEFAULT NULL,
                    session_id TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    ai_summary TEXT NOT NULL,
                    topics TEXT NOT NULL DEFAULT '[]',
                    sentiment REAL NOT NULL DEFAULT 0.0,
                    importance REAL NOT NULL DEFAULT 0.6,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    project_id TEXT DEFAULT NULL,
                    collection TEXT NOT NULL DEFAULT 'general',
                    title TEXT NOT NULL,
                    filename TEXT DEFAULT '',
                    mime_type TEXT DEFAULT 'text/plain',
                    file_size INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT 'note',
                    content TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL DEFAULT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    section_label TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    project_id TEXT DEFAULT NULL,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'todo',
                    due_note TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT DEFAULT NULL,
                    project_id TEXT DEFAULT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'planning',
                    plan_json TEXT NOT NULL DEFAULT '[]',
                    current_step INTEGER NOT NULL DEFAULT 0,
                    max_steps INTEGER NOT NULL DEFAULT 8,
                    pending_approval_json TEXT DEFAULT NULL,
                    final_output TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    cancelled INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_steps (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    tool_name TEXT DEFAULT '',
                    tool_args_json TEXT DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    requires_approval INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    output_summary TEXT DEFAULT '',
                    artifacts_json TEXT DEFAULT '[]',
                    duration_ms REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS custom_skills (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    trigger_phrase TEXT NOT NULL,
                    intent_label TEXT NOT NULL DEFAULT 'custom_skill',
                    response_template TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_samples (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    text TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'usr_local_workspace',
                    architecture TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    intent_count INTEGER NOT NULL,
                    train_accuracy REAL NOT NULL,
                    cv_accuracy REAL NOT NULL,
                    precision_macro REAL NOT NULL DEFAULT 0.0,
                    recall_macro REAL NOT NULL DEFAULT 0.0,
                    f1_macro REAL NOT NULL DEFAULT 0.0,
                    confusion_matrix_json TEXT DEFAULT '{}',
                    per_class_json TEXT DEFAULT '{}',
                    loss_curve_json TEXT DEFAULT '[]',
                    duration_ms REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS security_audit_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT DEFAULT '',
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'info',
                    ip_address TEXT DEFAULT '',
                    resource TEXT DEFAULT '',
                    detail TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sandbox_audit_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    code_sha256 TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    denied_reason TEXT DEFAULT '',
                    duration_ms REAL NOT NULL DEFAULT 0.0,
                    stdout_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_reports (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT DEFAULT NULL,
                    query TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'quick',
                    summary TEXT NOT NULL DEFAULT '',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    follow_up_json TEXT NOT NULL DEFAULT '[]',
                    diagnostics_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS installed_extensions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    extension_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    installed_at REAL NOT NULL,
                    UNIQUE(user_id, extension_id)
                );

                CREATE TABLE IF NOT EXISTS custom_extensions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    category TEXT NOT NULL DEFAULT 'productivity',
                    version TEXT NOT NULL DEFAULT '1.0.0',
                    permissions_json TEXT NOT NULL DEFAULT '[]',
                    input_schema_json TEXT NOT NULL DEFAULT '{}',
                    prompt_template TEXT NOT NULL DEFAULT '',
                    tool_name TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    UNIQUE(user_id, slug)
                );

                CREATE TABLE IF NOT EXISTS usage_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    provider_id TEXT DEFAULT '',
                    model TEXT DEFAULT '',
                    tool_name TEXT DEFAULT '',
                    estimated_tokens INTEGER NOT NULL DEFAULT 0,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'ok',
                    error_detail TEXT DEFAULT '',
                    created_at REAL NOT NULL
                );
                """
            )

            # 2. Non-destructive migrations for pre-existing v1.0 SQLite databases
            _ensure_column(conn, "users", "avatar_url", "TEXT DEFAULT ''")
            _ensure_column(conn, "users", "bio", "TEXT DEFAULT ''")
            _ensure_column(conn, "users", "email_verified", "INTEGER NOT NULL DEFAULT 1")

            _ensure_column(conn, "sessions", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "sessions", "project_id", "TEXT DEFAULT NULL")
            _ensure_column(conn, "sessions", "summary", "TEXT DEFAULT ''")
            _ensure_column(conn, "sessions", "pinned", "INTEGER NOT NULL DEFAULT 0")

            _ensure_column(conn, "messages", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "messages", "provider_used", "TEXT DEFAULT ''")
            _ensure_column(conn, "messages", "model_used", "TEXT DEFAULT ''")
            _ensure_column(conn, "messages", "tool_calls", "TEXT DEFAULT '[]'")
            _ensure_column(conn, "messages", "citations", "TEXT DEFAULT '[]'")
            _ensure_column(conn, "messages", "attachments", "TEXT DEFAULT '[]'")
            _ensure_column(conn, "messages", "edited_at", "REAL DEFAULT NULL")

            _ensure_column(conn, "semantic_memories", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "semantic_memories", "project_id", "TEXT DEFAULT NULL")
            _ensure_column(conn, "semantic_memories", "verified", "INTEGER NOT NULL DEFAULT 1")
            _ensure_column(conn, "semantic_memories", "expires_at", "REAL DEFAULT NULL")

            _ensure_column(conn, "memory_history", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "episodic_memories", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "episodic_memories", "project_id", "TEXT DEFAULT NULL")

            _ensure_column(conn, "documents", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "documents", "project_id", "TEXT DEFAULT NULL")
            _ensure_column(conn, "documents", "collection", "TEXT NOT NULL DEFAULT 'general'")
            _ensure_column(conn, "documents", "filename", "TEXT DEFAULT ''")
            _ensure_column(conn, "documents", "mime_type", "TEXT DEFAULT 'text/plain'")
            _ensure_column(conn, "documents", "file_size", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "documents", "content_hash", "TEXT DEFAULT ''")
            _ensure_column(conn, "documents", "metadata_json", "TEXT DEFAULT '{}'")
            _ensure_column(conn, "documents", "updated_at", "REAL DEFAULT NULL")

            _ensure_column(conn, "document_chunks", "section_label", "TEXT DEFAULT ''")

            _ensure_column(conn, "tasks", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "tasks", "project_id", "TEXT DEFAULT NULL")
            _ensure_column(conn, "custom_skills", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")
            _ensure_column(conn, "training_samples", "user_id", f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_ID}'")

            # Helpful indexes for fast multi-user isolation queries
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_sem_mem_user ON semantic_memories(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_epi_mem_user ON episodic_memories(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_user ON tasks(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id, updated_at DESC);
                """
            )

            now = time.time()

            # 3. Ensure default local workspace account exists so pre-existing history & frictionless local mode work seamlessly
            existing_default_user = conn.execute("SELECT id FROM users WHERE id = ?", (DEFAULT_USER_ID,)).fetchone()
            if not existing_default_user:
                conn.execute(
                    """
                    INSERT INTO users (id, email, username, display_name, password_hash, role, is_guest, created_at, last_login_at)
                    VALUES (?, 'workspace@xyvora.local', 'workspace', 'Local Workspace', ?, 'owner', 1, ?, ?)
                    """,
                    (DEFAULT_USER_ID, hash_password(os.urandom(24).hex()), now, now),
                )

            # 4. Migrate legacy global settings -> per-user settings AND scrub any plaintext API key!
            legacy_rows = conn.execute("SELECT key, value FROM settings").fetchall()
            legacy_map: Dict[str, Any] = {}
            for lr in legacy_rows:
                try:
                    legacy_map[lr["key"]] = json.loads(lr["value"])
                except Exception:
                    legacy_map[lr["key"]] = lr["value"]

            existing_user_settings = conn.execute(
                "SELECT user_id FROM user_settings WHERE user_id = ?", (DEFAULT_USER_ID,)
            ).fetchone()
            if not existing_user_settings:
                merged = dict(DEFAULT_USER_SETTINGS)
                for k in ("persona", "creativity", "verbosity", "memory_threshold", "auto_learn_memory", "show_thought_trace", "voice_enabled"):
                    if k in legacy_map:
                        merged[k] = legacy_map[k]
                conn.execute(
                    "INSERT INTO user_settings (user_id, settings_json, updated_at) VALUES (?, ?, ?)",
                    (DEFAULT_USER_ID, json.dumps(merged), now),
                )

            # Migrate any legacy plaintext external_llm_api_key into encrypted provider_configs and delete plaintext!
            legacy_key = str(legacy_map.get("external_llm_api_key") or "").strip()
            if legacy_key:
                enc = encrypt_secret(legacy_key, DATA_DIR)
                base_url = str(legacy_map.get("external_llm_base_url") or "https://api.openai.com/v1")
                model = str(legacy_map.get("external_llm_model") or "gpt-4o-mini")
                enabled = 1 if legacy_map.get("external_llm_enabled") else 0
                conn.execute(
                    """
                    INSERT INTO provider_configs (id, user_id, provider_id, base_url, encrypted_api_key, default_model, enabled, extra_json, updated_at)
                    VALUES (?, ?, 'openai_compatible', ?, ?, ?, ?, '{}', ?)
                    ON CONFLICT(user_id, provider_id) DO UPDATE SET
                        base_url = excluded.base_url,
                        encrypted_api_key = excluded.encrypted_api_key,
                        default_model = excluded.default_model,
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    (str(uuid.uuid4()), DEFAULT_USER_ID, base_url, enc, model, enabled, now),
                )
            # Always scrub legacy plaintext key from global settings table
            conn.execute("DELETE FROM settings WHERE key = 'external_llm_api_key'")

            # 5. Seed core system facts, starter specification doc, starter project, and welcome session for default workspace if empty
            _seed_user_starter_state(conn, DEFAULT_USER_ID, include_starter_project=True)

            conn.commit()
        finally:
            conn.close()


def _seed_user_starter_state(conn: sqlite3.Connection, user_id: str, include_starter_project: bool = False) -> None:
    now = time.time()

    # Settings
    cur = conn.execute("SELECT COUNT(*) AS c FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
    if cur["c"] == 0:
        conn.execute(
            "INSERT INTO user_settings (user_id, settings_json, updated_at) VALUES (?, ?, ?)",
            (user_id, json.dumps(DEFAULT_USER_SETTINGS), now),
        )

    # Core system memories
    cur = conn.execute("SELECT COUNT(*) AS c FROM semantic_memories WHERE user_id = ?", (user_id,)).fetchone()
    if cur["c"] == 0:
        seed_facts = [
            ("Xyvora AI", "architecture", "Multi-Provider LLM + Neuro-Symbolic Cortex + Tool Orchestrator", "system", 1.0, 0.95, 1, "Core initialization"),
            ("Xyvora AI", "memory_engine", "User-Isolated Tri-Layer Working + Episodic + Semantic Graph", "system", 1.0, 0.90, 1, "Core initialization"),
            ("Xyvora AI", "capabilities", "LLM Streaming, Web Research, Multimodal Docs/Vision, Coding IDE, Autonomous Agents, Hardened Python Sandbox", "system", 1.0, 0.90, 0, "Core initialization"),
        ]
        for subj, rel, obj, cat, conf, imp, pin, src in seed_facts:
            conn.execute(
                """
                INSERT INTO semantic_memories
                (id, user_id, subject, relation, object, category, confidence, importance, pinned, verified, source_text, access_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0, ?, ?)
                """,
                (str(uuid.uuid4()), user_id, subj, rel, obj, cat, conf, imp, pin, src, now, now),
            )

    # Starter RAG document
    cur = conn.execute("SELECT COUNT(*) AS c FROM documents WHERE user_id = ?", (user_id,)).fetchone()
    if cur["c"] == 0:
        doc_id = str(uuid.uuid4())
        starter_doc = (
            "Xyvora Next-Generation Intelligence Platform Specification (v2.0)\n\n"
            "Xyvora AI is a unified multimodal, tool-using artificial intelligence platform combining a multi-provider "
            "Language Model orchestration layer (OpenAI, Anthropic, Gemini, Ollama, OpenAI-compatible endpoints, plus an offline Neuro-Symbolic engine) "
            "with persistent user-isolated Tri-Layer Memory, grounded Web Research, multi-format Document Intelligence, a Project-Aware Coding Workspace, "
            "and Autonomous Multi-Step Agents.\n\n"
            "1. Central Intelligence & Tool Calling: Requests flow through Provider Manager -> Model Adapter -> Context Builder -> "
            "Multi-Step Tool Orchestrator -> Streaming SSE Response -> Memory Palace. The trainable TF-IDF + MLP Neural Classifier provides sub-5ms intent routing "
            "and live experimentation in the Training Studio.\n\n"
            "2. Security & Sandbox Isolation: User accounts use PBKDF2-HMAC-SHA256 (260k iterations), SHA-256 hashed session tokens, "
            "server-side authenticated encryption for provider API keys, explicit CORS origin allowlists, rate limiting, and OS resource-limited Python execution.\n\n"
            "3. Multimodal & Knowledge Vault: Supports PDF, DOCX, XLSX, PPTX, CSV, JSON, Markdown, TXT, source code, and image analysis with chunk-level section citations."
        )
        paragraphs = [p.strip() for p in starter_doc.split("\n\n") if p.strip()]
        conn.execute(
            """
            INSERT INTO documents (id, user_id, collection, title, filename, mime_type, file_size, source_type, content, chunk_count, created_at, updated_at)
            VALUES (?, ?, 'architecture', ?, 'xyvora_spec_v2.md', 'text/markdown', ?, 'system_doc', ?, ?, ?, ?)
            """,
            (doc_id, user_id, "Xyvora Platform Architecture Spec v2.0", len(starter_doc), starter_doc, len(paragraphs), now, now),
        )
        for idx, chunk in enumerate(paragraphs):
            conn.execute(
                "INSERT INTO document_chunks (id, doc_id, chunk_index, section_label, content) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), doc_id, idx, f"Section {idx + 1}", chunk),
            )

    # Starter Coding Project so Coding Workspace works out of the box with sample multi-file code + unit tests
    if include_starter_project:
        cur = conn.execute("SELECT COUNT(*) AS c FROM projects WHERE user_id = ?", (user_id,)).fetchone()
        if cur["c"] == 0:
            proj_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO projects (id, user_id, name, description, instructions, color, created_at, updated_at)
                VALUES (?, ?, 'Orbital Telemetry Toolkit', 'Sample multi-file engineering workspace with unit tests', 'Write clean, well-typed, modular code with deterministic unit tests.', '#22d3ee', ?, ?)
                """,
                (proj_id, user_id, now, now),
            )
            starter_files = [
                (
                    "telemetry/analyzer.py",
                    "python",
                    '''"""Orbital sensor telemetry statistical analyzer."""
from typing import Dict, List
import math


def compute_signal_metrics(samples: List[float]) -> Dict[str, float]:
    """Computes mean, standard deviation, and peak-to-peak amplitude."""
    if not samples:
        raise ValueError("samples cannot be empty")
    n = len(samples)
    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / n
    std_dev = math.sqrt(variance)
    return {
        "count": float(n),
        "mean": round(mean, 4),
        "std_dev": round(std_dev, 4),
        "peak_to_peak": round(max(samples) - min(samples), 4),
    }


def detect_anomalies(samples: List[float], z_threshold: float = 2.0) -> List[int]:
    """Returns indices of readings exceeding z_threshold standard deviations."""
    if len(samples) < 3:
        return []
    metrics = compute_signal_metrics(samples)
    std = metrics["std_dev"]
    if std == 0:
        return []
    mean = metrics["mean"]
    return [idx for idx, val in enumerate(samples) if abs(val - mean) / std >= z_threshold]
''',
                ),
                (
                    "tests/test_analyzer.py",
                    "python",
                    '''import unittest
from telemetry.analyzer import compute_signal_metrics, detect_anomalies


class TestTelemetryAnalyzer(unittest.TestCase):
    def test_metrics_computation(self):
        res = compute_signal_metrics([10.0, 12.0, 14.0, 16.0])
        self.assertEqual(res["count"], 4.0)
        self.assertAlmostEqual(res["mean"], 13.0)
        self.assertAlmostEqual(res["peak_to_peak"], 6.0)

    def test_anomaly_detection(self):
        series = [10.0, 10.2, 9.9, 10.1, 10.0, 9.8, 45.0]
        outliers = detect_anomalies(series, z_threshold=2.0)
        self.assertEqual(outliers, [6])


if __name__ == "__main__":
    unittest.main()
''',
                ),
                (
                    "ui/TelemetryCard.tsx",
                    "typescript",
                    '''import React from "react";

export interface TelemetryReading {
  sensorId: string;
  mean: number;
  anomalies: number;
}

export const TelemetryCard: React.FC<{ reading: TelemetryReading }> = ({ reading }) => {
  return (
    <div className="p-4 rounded-xl bg-slate-900 border border-cyan-500/30">
      <h4 className="text-sm font-bold text-cyan-300">{reading.sensorId}</h4>
      <p className="text-xs text-slate-300">Mean Signal: {reading.mean.toFixed(2)}</p>
      <span className="text-xs font-mono text-amber-300">Anomalies: {reading.anomalies}</span>
    </div>
  );
};
''',
                ),
            ]
            for path, lang, content in starter_files:
                conn.execute(
                    """
                    INSERT INTO project_files (id, project_id, user_id, path, language, content, version, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (str(uuid.uuid4()), proj_id, user_id, path, lang, content, now, now),
                )
            conn.execute(
                """
                INSERT INTO project_activity (id, project_id, user_id, action, summary, created_at)
                VALUES (?, ?, ?, 'project_created', 'Initialized Orbital Telemetry Toolkit workspace with 3 files', ?)
                """,
                (str(uuid.uuid4()), proj_id, user_id, now),
            )

    # Starter welcome session
    cur = conn.execute("SELECT COUNT(*) AS c FROM sessions WHERE user_id = ?", (user_id,)).fetchone()
    if cur["c"] == 0:
        sess_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (id, user_id, title, persona, created_at, updated_at) VALUES (?, ?, ?, 'nexus', ?, ?)",
            (sess_id, user_id, "Neural Nexus Session", now, now),
        )
        welcome_msg = (
            "Hello! I'm **Xyvora AI** — your multimodal, tool-using intelligence platform.\n\n"
            "My cognitive architecture unifies **multi-provider LLM orchestration** (OpenAI, Anthropic, Gemini, Ollama, or Local Neuro-Symbolic mode), "
            "**autonomous multi-step tool calling**, **grounded Web Research**, a **user-isolated Tri-Layer Memory Palace**, "
            "**multi-format Document & Image Intelligence**, a **project-aware Coding IDE**, and **hardened Python execution**.\n\n"
            "Try any capability right away:\n"
            "- **Tool Calling & Math**: *\"Differentiate and plot sin(x) * cos(x/2) + x/4\"* or *\"Simulate 2000 dice rolls in Python and chart the distribution\"*\n"
            "- **Persistent Memory**: *\"My name is Jordan, I live in Seattle, and I prefer concise TypeScript examples.\"* — then ask what I remember!\n"
            "- **Documents & Vision**: Attach an image, PDF, DOCX, XLSX, CSV, or code file right in the composer below.\n"
            "- **Workspaces**: Explore **Projects**, **Web Research**, **Coding Studio**, **Autonomous Agents**, **Memory Palace**, and **Training Studio** in the sidebar!"
        )
        suggestions = [
            "My name is Alex and I'm building an AI platform in Python & TypeScript",
            "Differentiate and plot sin(x) * x",
            "Simulate Monte Carlo Pi estimation in Python and chart convergence",
            "Search the Knowledge Vault for Xyvora v2.0 security architecture",
        ]
        conn.execute(
            """
            INSERT INTO messages
            (id, session_id, user_id, role, content, intent, confidence, thought_trace, artifacts, memories_added, suggestions, created_at)
            VALUES (?, ?, ?, 'assistant', ?, 'welcome', 1.0, '[]', '[]', '[]', ?, ?)
            """,
            (str(uuid.uuid4()), sess_id, user_id, welcome_msg, json.dumps(suggestions), now),
        )


def ensure_user_initialized(user_id: str) -> None:
    with _db_lock:
        conn = get_conn()
        try:
            _seed_user_starter_state(conn, user_id, include_starter_project=True)
            conn.commit()
        finally:
            conn.close()


# ----------------- User Settings & Secure Provider Config Helpers -----------------

def get_user_settings(user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT settings_json FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
            merged = dict(DEFAULT_USER_SETTINGS)
            if row:
                try:
                    stored = json.loads(row["settings_json"])
                    if isinstance(stored, dict):
                        for k, v in stored.items():
                            # Never allow secret keys inside frontend-visible settings
                            if "api_key" not in k.lower() and "secret" not in k.lower() and "password" not in k.lower():
                                merged[k] = v
                except Exception:
                    pass
            return merged
        finally:
            conn.close()


def update_user_settings(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    current = get_user_settings(user_id)
    now = time.time()

    # Intercept legacy external_llm_* fields safely if sent by older clients, routing secrets to encrypted vault!
    legacy_api_key = patch.pop("external_llm_api_key", None)
    legacy_base_url = patch.get("external_llm_base_url")
    legacy_model = patch.get("external_llm_model")
    legacy_enabled = patch.get("external_llm_enabled")

    if legacy_api_key or legacy_base_url or legacy_model or legacy_enabled is not None:
        upsert_provider_config(
            user_id=user_id,
            provider_id="openai_compatible",
            api_key=str(legacy_api_key).strip() if legacy_api_key else None,
            base_url=str(legacy_base_url) if legacy_base_url else None,
            default_model=str(legacy_model) if legacy_model else None,
            enabled=bool(legacy_enabled) if legacy_enabled is not None else None,
        )

    for k, v in patch.items():
        if "api_key" in k.lower() or "secret" in k.lower() or "password" in k.lower():
            continue
        current[k] = v

    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, settings_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET settings_json = excluded.settings_json, updated_at = excluded.updated_at
                """,
                (user_id, json.dumps(current), now),
            )
            conn.commit()
        finally:
            conn.close()
    return get_user_settings(user_id)


#Backward-compatible aliases for default workspace
def get_settings(user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    return get_user_settings(user_id)


def update_settings(patch: Dict[str, Any], user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    return update_user_settings(user_id, patch)


def upsert_provider_config(
    user_id: str,
    provider_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_model: Optional[str] = None,
    enabled: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute(
                "SELECT * FROM provider_configs WHERE user_id = ? AND provider_id = ?",
                (user_id, provider_id),
            ).fetchone()

            if existing:
                enc_key = existing["encrypted_api_key"]
                if api_key is not None:
                    if api_key == "__CLEAR__":
                        enc_key = ""
                    elif api_key.strip() and "••••" not in api_key:
                        enc_key = encrypt_secret(api_key.strip(), DATA_DIR)
                new_base = base_url if base_url is not None else existing["base_url"]
                new_model = default_model if default_model is not None else existing["default_model"]
                new_enabled = (1 if enabled else 0) if enabled is not None else existing["enabled"]
                new_extra = json.dumps(extra) if extra is not None else existing["extra_json"]
                conn.execute(
                    """
                    UPDATE provider_configs
                    SET base_url = ?, encrypted_api_key = ?, default_model = ?, enabled = ?, extra_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_base, enc_key, new_model, new_enabled, new_extra, now, existing["id"]),
                )
            else:
                enc_key = ""
                if api_key and api_key != "__CLEAR__" and "••••" not in api_key:
                    enc_key = encrypt_secret(api_key.strip(), DATA_DIR)
                conn.execute(
                    """
                    INSERT INTO provider_configs (id, user_id, provider_id, base_url, encrypted_api_key, default_model, enabled, extra_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        provider_id,
                        base_url or "",
                        enc_key,
                        default_model or "",
                        1 if (enabled is None or enabled) else 0,
                        json.dumps(extra or {}),
                        now,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
    return get_provider_config_public(user_id, provider_id)


def get_provider_secret_internal(user_id: str, provider_id: str) -> Dict[str, Any]:
    """Server-internal only: resolves credentials from user vault or server environment variables."""
    env_key_map = {
        "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1", "gpt-4o-mini"),
        "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "https://api.anthropic.com", "claude-3-5-haiku-latest"),
        "gemini": ("GEMINI_API_KEY", "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
        "groq": ("GROQ_API_KEY", "GROQ_BASE_URL", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
        "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
        "ollama": ("OLLAMA_API_KEY", "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1", "llama3.2"),
        "openai_compatible": ("CUSTOM_LLM_API_KEY", "CUSTOM_LLM_BASE_URL", "https://api.openai.com/v1", "gpt-4o-mini"),
    }
    env_key_name, env_url_name, default_url, default_mdl = env_key_map.get(
        provider_id, ("", "", "", "")
    )
    env_api_key = os.environ.get(env_key_name, "").strip() if env_key_name else ""
    env_base_url = os.environ.get(env_url_name, "").strip() if env_url_name else ""

    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM provider_configs WHERE user_id = ? AND provider_id = ?",
                (user_id, provider_id),
            ).fetchone()
        finally:
            conn.close()

    db_key = decrypt_secret(row["encrypted_api_key"], DATA_DIR) if row and row["encrypted_api_key"] else ""
    api_key = db_key or env_api_key
    base_url = (row["base_url"] if row and row["base_url"] else "") or env_base_url or default_url
    model = (row["default_model"] if row and row["default_model"] else "") or default_mdl
    enabled = bool(row["enabled"]) if row else True

    return {
        "provider_id": provider_id,
        "api_key": api_key,
        "key_source": "user_vault" if db_key else ("environment" if env_api_key else "none"),
        "base_url": base_url.rstrip("/"),
        "default_model": model,
        "enabled": enabled,
    }


def get_provider_config_public(user_id: str, provider_id: str) -> Dict[str, Any]:
    """Safe public representation: NEVER contains raw API keys."""
    internal = get_provider_secret_internal(user_id, provider_id)
    raw_key = internal.pop("api_key", "")
    internal["has_api_key"] = bool(raw_key)
    internal["masked_api_key"] = mask_secret(raw_key) if raw_key else ""
    return internal


def log_security_event(
    event_type: str,
    detail: str = "",
    user_id: str = "",
    severity: str = "info",
    ip_address: str = "",
    resource: str = "",
) -> None:
    from security import redact_sensitive
    safe_detail = redact_sensitive(detail)[:500]
    logger.info("security_event=%s severity=%s user=%s resource=%s detail=%s", event_type, severity, user_id or "anon", resource, safe_detail)
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO security_audit_logs (id, user_id, event_type, severity, ip_address, resource, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), user_id, event_type, severity, ip_address, resource, safe_detail, time.time()),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()


def record_usage_event(
    user_id: str,
    endpoint: str,
    provider_id: str = "",
    model: str = "",
    tool_name: str = "",
    estimated_tokens: int = 0,
    latency_ms: float = 0.0,
    status: str = "ok",
    error_detail: str = "",
) -> None:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO usage_logs
                (id, user_id, endpoint, provider_id, model, tool_name, estimated_tokens, latency_ms, status, error_detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    user_id or DEFAULT_USER_ID,
                    endpoint,
                    provider_id,
                    model,
                    tool_name,
                    max(0, int(estimated_tokens)),
                    round(float(latency_ms), 2),
                    status,
                    error_detail[:300],
                    time.time(),
                ),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()


def get_storage_diagnostics() -> Dict[str, Any]:
    pg_url = (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or "").strip()
    is_ephemeral_tmp = DATA_DIR.startswith("/tmp")
    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    return {
        "engine": "postgresql_configured" if pg_url else "sqlite_wal",
        "persistence_mode": (
            "cloud_postgres_env_detected"
            if pg_url
            else ("serverless_ephemeral_tmp" if is_ephemeral_tmp else "persistent_local_disk")
        ),
        "postgres_configured": bool(pg_url),
        "ephemeral_warning": (
            "Running on serverless ephemeral /tmp storage without DATABASE_URL; use Account Backup Export/Import or attach PostgreSQL for permanent cloud multi-instance persistence."
            if (is_ephemeral_tmp and not pg_url)
            else ""
        ),
        "db_size_bytes": db_size_bytes,
    }


def get_user_usage_summary(user_id: str) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            since_24h = time.time() - 86400
            total_reqs = conn.execute("SELECT COUNT(*) AS c FROM usage_logs WHERE user_id = ?", (user_id,)).fetchone()["c"]
            reqs_24h = conn.execute("SELECT COUNT(*) AS c FROM usage_logs WHERE user_id = ? AND created_at >= ?", (user_id, since_24h)).fetchone()["c"]
            tok_row = conn.execute("SELECT COALESCE(SUM(estimated_tokens), 0) AS t, COALESCE(AVG(latency_ms), 0) AS lat FROM usage_logs WHERE user_id = ?", (user_id,)).fetchone()
            sessions_cnt = conn.execute("SELECT COUNT(*) AS c FROM sessions WHERE user_id = ?", (user_id,)).fetchone()["c"]
            memories_cnt = conn.execute("SELECT COUNT(*) AS c FROM semantic_memories WHERE user_id = ?", (user_id,)).fetchone()["c"]
            docs_cnt = conn.execute("SELECT COUNT(*) AS c FROM documents WHERE user_id = ?", (user_id,)).fetchone()["c"]
            projects_cnt = conn.execute("SELECT COUNT(*) AS c FROM projects WHERE user_id = ?", (user_id,)).fetchone()["c"]
            return {
                "total_requests": total_reqs,
                "requests_last_24h": reqs_24h,
                "estimated_tokens_total": int(tok_row["t"] or 0),
                "avg_latency_ms": round(float(tok_row["lat"] or 0.0), 1),
                "sessions_count": sessions_cnt,
                "memories_count": memories_cnt,
                "documents_count": docs_cnt,
                "projects_count": projects_cnt,
            }
        finally:
            conn.close()


def get_admin_analytics() -> Dict[str, Any]:
    now = time.time()
    since_24h = now - 86400
    with _db_lock:
        conn = get_conn()
        try:
            total_users = conn.execute("SELECT COUNT(*) AS c FROM users WHERE is_guest = 0").fetchone()["c"]
            total_accounts = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            active_24h = conn.execute("SELECT COUNT(DISTINCT user_id) AS c FROM usage_logs WHERE created_at >= ?", (since_24h,)).fetchone()["c"]
            reqs_24h = conn.execute("SELECT COUNT(*) AS c FROM usage_logs WHERE created_at >= ?", (since_24h,)).fetchone()["c"]
            total_reqs = conn.execute("SELECT COUNT(*) AS c FROM usage_logs").fetchone()["c"]
            failed_reqs = conn.execute("SELECT COUNT(*) AS c FROM usage_logs WHERE status != 'ok'").fetchone()["c"]
            provider_errors = conn.execute("SELECT COUNT(*) AS c FROM usage_logs WHERE status = 'provider_error'").fetchone()["c"]
            agg = conn.execute(
                "SELECT COALESCE(AVG(latency_ms), 0) AS avg_lat, COALESCE(SUM(estimated_tokens), 0) AS tot_tok FROM usage_logs"
            ).fetchone()
            tool_rows = conn.execute(
                """
                SELECT tool_name, COUNT(*) AS calls, ROUND(AVG(latency_ms), 1) AS avg_ms
                FROM usage_logs
                WHERE tool_name != ''
                GROUP BY tool_name
                ORDER BY calls DESC
                LIMIT 12
                """
            ).fetchall()
            recent_sec = conn.execute(
                "SELECT event_type, severity, resource, detail, created_at FROM security_audit_logs ORDER BY created_at DESC LIMIT 15"
            ).fetchall()
            sandbox_runs = conn.execute("SELECT COUNT(*) AS c FROM sandbox_audit_logs").fetchone()["c"]
            sandbox_denied = conn.execute("SELECT COUNT(*) AS c FROM sandbox_audit_logs WHERE ok = 0").fetchone()["c"]
        finally:
            conn.close()

    return {
        "total_registered_users": total_users,
        "total_workspace_accounts": total_accounts,
        "active_users_24h": max(1, active_24h) if total_reqs > 0 else active_24h,
        "requests_24h": reqs_24h,
        "total_requests": total_reqs,
        "failed_requests": failed_reqs,
        "provider_errors": provider_errors,
        "avg_latency_ms": round(float(agg["avg_lat"] or 0.0), 1),
        "total_estimated_tokens": int(agg["tot_tok"] or 0),
        "sandbox_executions": sandbox_runs,
        "sandbox_blocked_or_failed": sandbox_denied,
        "tool_stats": [dict(r) for r in tool_rows],
        "recent_security_events": [dict(r) for r in recent_sec],
        "storage": get_storage_diagnostics(),
    }


def export_user_workspace(user_id: str) -> Dict[str, Any]:
    """Exports user-owned settings, memories, projects, tasks, custom skills, and personalities (NEVER API keys)."""
    with _db_lock:
        conn = get_conn()
        try:
            memories = [
                dict(r)
                for r in conn.execute(
                    "SELECT subject, relation, object, category, confidence, importance, pinned, verified FROM semantic_memories WHERE user_id = ? AND category != 'system'",
                    (user_id,),
                ).fetchall()
            ]
            projects = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, name, description, instructions, color FROM projects WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
            tasks = [
                dict(r)
                for r in conn.execute(
                    "SELECT title, description, priority, status, due_note FROM tasks WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
            skills = [
                dict(r)
                for r in conn.execute(
                    "SELECT trigger_phrase, intent_label, response_template FROM custom_skills WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
            personalities = [
                dict(r)
                for r in conn.execute(
                    "SELECT slug, name, description, system_instructions, tone, creativity, verbosity, domain_focus FROM custom_personalities WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            ]
        finally:
            conn.close()
    return {
        "schema_version": "xyvora-workspace-v2",
        "exported_at": time.time(),
        "settings": get_user_settings(user_id),
        "memories": memories,
        "projects": projects,
        "tasks": tasks,
        "custom_skills": skills,
        "custom_personalities": personalities,
    }

