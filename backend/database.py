import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL") or not os.access(_BASE_DIR, os.W_OK):
    DATA_DIR = "/tmp/xyvora_data"
else:
    DATA_DIR = os.path.join(_BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "xyvora.db")

_db_lock = threading.RLock()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    with _db_lock:
        conn = get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    persona TEXT NOT NULL DEFAULT 'nexus',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    intent TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    thought_trace TEXT DEFAULT '[]',
                    artifacts TEXT DEFAULT '[]',
                    memories_added TEXT DEFAULT '[]',
                    suggestions TEXT DEFAULT '[]',
                    feedback INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    confidence REAL NOT NULL DEFAULT 0.92,
                    importance REAL NOT NULL DEFAULT 0.75,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    source_text TEXT DEFAULT '',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_history (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    previous_value TEXT DEFAULT '',
                    new_value TEXT DEFAULT '',
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id TEXT PRIMARY KEY,
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
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'note',
                    content TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'medium',
                    status TEXT NOT NULL DEFAULT 'todo',
                    due_note TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS custom_skills (
                    id TEXT PRIMARY KEY,
                    trigger_phrase TEXT NOT NULL,
                    intent_label TEXT NOT NULL DEFAULT 'custom_skill',
                    response_template TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_samples (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

            # Seed default cognitive settings if empty
            cur = conn.execute("SELECT COUNT(*) AS c FROM settings")
            if cur.fetchone()["c"] == 0:
                default_settings = {
                    "persona": "nexus",
                    "creativity": 0.65,
                    "verbosity": "balanced",
                    "memory_threshold": 0.24,
                    "auto_learn_memory": True,
                    "show_thought_trace": True,
                    "voice_enabled": False,
                    "external_llm_enabled": False,
                    "external_llm_base_url": "https://api.openai.com/v1",
                    "external_llm_api_key": "",
                    "external_llm_model": "gpt-4o-mini",
                }
                for k, v in default_settings.items():
                    conn.execute(
                        "INSERT INTO settings (key, value) VALUES (?, ?)",
                        (k, json.dumps(v)),
                    )

            # Seed core Xyvora self-architecture facts in Knowledge Graph if completely empty
            cur = conn.execute("SELECT COUNT(*) AS c FROM semantic_memories")
            if cur.fetchone()["c"] == 0:
                now = time.time()
                seed_facts = [
                    ("Xyvora AI", "architecture", "Neuro-Symbolic + MLP Neural Cortex", "system", 1.0, 0.95, 1, "Core initialization"),
                    ("Xyvora AI", "memory_engine", "Tri-Layer Episodic + Semantic Graph + Working Memory", "system", 1.0, 0.90, 1, "Core initialization"),
                    ("Xyvora AI", "capabilities", "Symbolic Math, Sandboxed Python, RAG Vault, Live Neural Training", "system", 1.0, 0.90, 0, "Core initialization"),
                ]
                for subj, rel, obj, cat, conf, imp, pin, src in seed_facts:
                    conn.execute(
                        """
                        INSERT INTO semantic_memories
                        (id, subject, relation, object, category, confidence, importance, pinned, source_text, access_count, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (str(uuid.uuid4()), subj, rel, obj, cat, conf, imp, pin, src, now, now),
                    )

            # Seed starter RAG document if empty so RAG search works out of the box
            cur = conn.execute("SELECT COUNT(*) AS c FROM documents")
            if cur.fetchone()["c"] == 0:
                now = time.time()
                doc_id = str(uuid.uuid4())
                starter_doc = (
                    "Xyvora Cognitive Architecture Specification (v1.0)\n\n"
                    "Xyvora AI is a hybrid neuro-symbolic artificial intelligence engineered with persistent tri-layer memory, "
                    "real-time neural intent classification, symbolic mathematical reasoning, and sandboxed code execution.\n\n"
                    "1. Neural Perception Layer: Uses character and word n-gram TF-IDF embeddings projected through Latent Semantic Analysis (SVD) "
                    "into a Multi-Layer Perceptron (MLP) neural network trained via Adam optimization. Users can inspect intent probability distributions "
                    "and retrain the neural network live in the Training Studio.\n\n"
                    "2. Tri-Layer Memory Palace: Combines short-term Working Memory (active dialogue state and pronoun coreference), "
                    "Episodic Memory (vector-indexed autobiographical conversation history), and a Semantic Knowledge Graph (subject-relation-object triples "
                    "automatically extracted from user statements, updating dynamically when preferences change).\n\n"
                    "3. Autonomous Tool Execution: Xyvora dynamically invokes SymPy for exact calculus, algebra, and linear algebra; "
                    "an AST-validated isolated Python subprocess for live computation and simulations; TextRank graph centrality for summarizing long articles; "
                    "and hybrid BM25 + LSA vector retrieval across uploaded documents in the Knowledge Vault."
                )
                paragraphs = [p.strip() for p in starter_doc.split("\n\n") if p.strip()]
                conn.execute(
                    "INSERT INTO documents (id, title, source_type, content, chunk_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, "Xyvora Architecture Spec v1.0", "system_doc", starter_doc, len(paragraphs), now),
                )
                for idx, chunk in enumerate(paragraphs):
                    conn.execute(
                        "INSERT INTO document_chunks (id, doc_id, chunk_index, content) VALUES (?, ?, ?, ?)",
                        (str(uuid.uuid4()), doc_id, idx, chunk),
                    )

            # Seed default session if none exists
            cur = conn.execute("SELECT COUNT(*) AS c FROM sessions")
            if cur.fetchone()["c"] == 0:
                now = time.time()
                sess_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO sessions (id, title, persona, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (sess_id, "Neural Nexus Session", "nexus", now, now),
                )
                welcome_msg = (
                    "Hello! I'm **Xyvora AI** — your interactive neuro-symbolic artificial intelligence.\n\n"
                    "Unlike static chatbots, my cognitive core combines a **trainable neural classifier**, a **persistent Tri-Layer Memory Graph**, "
                    "**symbolic mathematics & function plotting**, **sandboxed Python execution**, and a **local RAG Knowledge Vault**.\n\n"
                    "Here are a few things you can try right now:\n"
                    "- **Test my memory**: *\"My name is Jordan, I live in Seattle, and I love astrophysics and TypeScript.\"* — then ask me about yourself anytime!\n"
                    "- **Symbolic math & live plots**: *\"Differentiate and plot sin(x) * cos(x/2) + x/4\"* or *\"Solve x^2 - 5x + 6 = 0\"*\n"
                    "- **Run live code & simulations**: *\"Write and run Python code to simulate 1000 rolls of two dice and chart the distribution\"*\n"
                    "- **Explore my brain**: Open the **Memory Palace**, **Training Studio**, or **AI Lab** tabs on the left to visually inspect, edit, and train me live!"
                )
                suggestions = [
                    "My name is Alex and I love building AI tools in Python",
                    "Differentiate and plot sin(x) * x",
                    "Simulate 1000 dice rolls in Python and chart the results",
                    "Explain how quantum entanglement works like I'm 12",
                ]
                conn.execute(
                    """
                    INSERT INTO messages
                    (id, session_id, role, content, intent, confidence, thought_trace, artifacts, memories_added, suggestions, created_at)
                    VALUES (?, ?, 'assistant', ?, 'welcome', 1.0, '[]', '[]', '[]', ?, ?)
                    """,
                    (str(uuid.uuid4()), sess_id, welcome_msg, json.dumps(suggestions), now),
                )

            conn.commit()
        finally:
            conn.close()


def get_settings() -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            result: Dict[str, Any] = {}
            for r in rows:
                try:
                    result[r["key"]] = json.loads(r["value"])
                except Exception:
                    result[r["key"]] = r["value"]
            return result
        finally:
            conn.close()


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            for k, v in patch.items():
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, json.dumps(v)),
                )
            conn.commit()
        finally:
            conn.close()
    return get_settings()
