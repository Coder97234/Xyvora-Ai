# Xyvora AI v2.0 — Next-Generation Unified Intelligence Platform

> *"One intelligence. Endless possibilities."*

**Xyvora AI** is a full-stack, multi-user, multimodal, tool-using artificial intelligence platform combining an OpenAI-compatible multi-provider LLM orchestration engine with a persistent user-isolated **Tri-Layer Memory Palace**, **Web Research Studio**, **Coding Studio IDE**, **Autonomous Multi-Step Agents**, **Declarative Extension Marketplace**, **OS-Hardened Python Sandbox**, **Symbolic Math Explorer**, and a live **Neural Intent Router Training Studio**.

---

## Layered Architecture (`backend/`)

1. **Security, Accounts & Multi-User Isolation (`backend/security.py`, `backend/database.py`)**
   - Email/username/password registration & login (`PBKDF2-HMAC-SHA256`, 260,000 iterations), SHA-256 hashed `HttpOnly` + Bearer session tokens, token rotation (`/api/auth/refresh`), email verification, password reset flows, profile customization, account deletion, and portable workspace export/import.
   - Server-side authenticated secret encryption for user AI provider keys — `/api/health`, `/api/settings`, and `/api/providers` **never** expose plaintext secrets.
   - Role-based access control (`owner` / `admin` / `user`) protecting `/api/admin/analytics`, explicit CORS allowlists, sliding-window rate limiting, SSRF outbound URL guards, and constructive safety-policy enforcement.

2. **Multi-Provider LLM Adapter & Context Builder (`backend/engine/providers.py`, `backend/engine/cortex.py`)**
   - Supports **OpenAI**, **Anthropic**, **Google Gemini**, **Groq**, **OpenRouter**, **Together AI**, **Ollama**, custom OpenAI-compatible endpoints, and the built-in **Xyvora Local Neuro-Symbolic Engine** (`xyvora_local`) with honest capability badges and live `/test` connectivity diagnostics.
   - Token-budgeted `ContextBuilder` assembling persona directives, project instructions, recalled semantic/episodic memories, grounded Knowledge Vault chunks, multimodal attachments, and compressed history summaries.

3. **Formal Tool Registry, Autonomous Agents & Safe Extensions (`backend/engine/tool_registry.py`, `backend/engine/agent_system.py`, `backend/engine/extensions.py`)**
   - 11 JSON-Schema tools (`calculator`, `python_sandbox`, `document_search`, `memory_retrieve`, `memory_store`, `web_research`, `project_file_read`, `project_file_write`, `data_profiler`, `task_manager`, `image_generate`).
   - Multi-step `AgentOrchestrator` with bounded step planning, approval checkpoints for workspace writes, and cancellation.
   - **Extension Marketplace** running declarative/structured extensions with scoped permissions (`compute:math`, `compute:text`, `compute:data`) — never executing arbitrary untrusted third-party code inside the backend process.

4. **Smarter Tri-Layer Memory Palace (`backend/engine/memory_system.py`)**
   - Working + Episodic + Semantic Knowledge Graph with confidence scores, importance ranking, optional TTL expiration (`7/30/90` days), contradiction superseding, inline editing, project-scoped memories, and **Memory Approval Mode** (requiring user confirmation before committing auto-extracted facts).

5. **Web Research, Multimodal Vault & Coding Studio IDE**
   - **Research Studio (`backend/engine/web_research.py`)**: Multi-source search & SSRF-protected HTML extraction, credibility scoring, contradiction detection, persistent research history, and Markdown/PDF export.
   - **Multimodal & Knowledge Vault (`backend/engine/multimodal.py`, `backend/engine/knowledge_base.py`)**: Ingests `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.csv`, `.json`, `.md`, `.txt`, source code, and images into collections with duplicate detection, section-level citations, and re-indexing.
   - **Coding Studio IDE (`backend/engine/coding_assistant.py`, `backend/engine/code_engine.py`)**: Multi-file project workspaces, unified diff & 1-click revision rollback, static AST/security diagnostics, AI patch proposals, and `RLIMIT`-isolated unit test execution.

6. **Neural Intent Router & Training Studio (`backend/engine/neural_classifier.py`)**
   - Sub-5ms TF-IDF + MLP (`128 -> 64 -> Softmax`) intent router with dataset validation, JSON import/export, Macro & Per-Intent Out-of-Fold Precision/Recall/F1 metrics, live probability probing, and run history.

---

## Quick Start & Automated Verification

```bash
# 1. Python virtual environment & dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run automated security & platform test suite
.venv/bin/pytest -v

# 3. Build production frontend
cd frontend && npm install && npm run build && cd ..

# 4. Start full-stack server
./start.sh
```
