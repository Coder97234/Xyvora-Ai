import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import get_conn, get_settings, init_db, update_settings, _db_lock
from engine.cortex import XyvoraCortex

init_db()
cortex = XyvoraCortex()

app = FastAPI(title="Xyvora AI Cognitive Engine", version="1.0.0")


@app.middleware("http")
async def normalize_vercel_api_prefix(request, call_next):
    path = request.scope.get("path", "")
    # If Vercel rewrites /api/foo -> /foo or /api/index.py, normalize cleanly
    if path == "/api/index.py" or path == "/api/index":
        matched = request.headers.get("x-matched-path") or request.url.path
        if matched and matched not in ("/api/index.py", "/api/index"):
            request.scope["path"] = matched if matched.startswith("/api") else f"/api{matched}"
    elif path and not path.startswith("/api/") and path != "/api":
        # Only rewrite when running on Vercel or when matching known API prefixes
        if os.environ.get("VERCEL"):
            request.scope["path"] = f"/api{path}" if path.startswith("/") else f"/api/{path}"
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Pydantic Schemas -----------------

class CreateSessionReq(BaseModel):
    title: Optional[str] = "New Neural Session"
    persona: Optional[str] = "nexus"


class ChatStreamReq(BaseModel):
    session_id: str
    message: str
    client_memories: Optional[List[Dict[str, Any]]] = None


class ClientVaultSyncReq(BaseModel):
    memories: Optional[List[Dict[str, Any]]] = None
    skills: Optional[List[Dict[str, Any]]] = None
    settings: Optional[Dict[str, Any]] = None


class FeedbackReq(BaseModel):
    feedback: int


class MemoryRecallReq(BaseModel):
    query: str
    threshold: Optional[float] = 0.18


class CreateMemoryReq(BaseModel):
    subject: str
    relation: str
    object: str
    category: Optional[str] = "fact"
    pinned: Optional[int] = 0


class PatchMemoryReq(BaseModel):
    subject: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    pinned: Optional[int] = None
    importance: Optional[float] = None


class NeuralPredictReq(BaseModel):
    text: str


class TrainingSampleReq(BaseModel):
    text: str
    intent: str


class CustomSkillReq(BaseModel):
    trigger_phrase: str
    response_template: str


class RetrainReq(BaseModel):
    hidden_1: Optional[int] = 128
    hidden_2: Optional[int] = 64
    max_iter: Optional[int] = 95


class PythonExecReq(BaseModel):
    code: str


class MathQueryReq(BaseModel):
    query: str


class CreateDocReq(BaseModel):
    title: str
    content: str
    source_type: Optional[str] = "note"


class CreateTaskReq(BaseModel):
    title: str
    description: Optional[str] = ""
    priority: Optional[str] = "medium"
    due_note: Optional[str] = ""


class PatchTaskReq(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


# ----------------- Endpoints -----------------

@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            mem_count = conn.execute("SELECT COUNT(*) AS c FROM semantic_memories").fetchone()["c"]
            epi_count = conn.execute("SELECT COUNT(*) AS c FROM episodic_memories").fetchone()["c"]
            doc_count = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
            task_open = conn.execute("SELECT COUNT(*) AS c FROM tasks WHERE status != 'done'").fetchone()["c"]
            skill_count = conn.execute("SELECT COUNT(*) AS c FROM custom_skills").fetchone()["c"]
        finally:
            conn.close()

    n_stats = cortex.neural.get_stats()
    return {
        "status": "online",
        "engine": "Xyvora Cortex v1.0",
        "semantic_memories": mem_count,
        "episodic_memories": epi_count,
        "documents": doc_count,
        "open_tasks": task_open,
        "custom_skills": skill_count,
        "neural_accuracy": n_stats["training_accuracy"],
        "neural_params": n_stats["parameter_count"],
        "settings": get_settings(),
    }


# ---- Sessions & Streaming Chat ----

@app.get("/api/sessions")
def list_sessions() -> List[Dict[str, Any]]:
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT s.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.post("/api/sessions")
def create_session(req: CreateSessionReq) -> Dict[str, Any]:
    now = time.time()
    sess_id = str(uuid.uuid4())
    settings = get_settings()
    persona = req.persona or settings.get("persona", "nexus")

    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, title, persona, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (sess_id, req.title or "New Neural Session", persona, now, now),
            )
            # Check if we already know user's name to greet them personally in a fresh session!
            name_row = conn.execute(
                "SELECT object FROM semantic_memories WHERE lower(subject)='user' AND lower(relation)='name' LIMIT 1"
            ).fetchone()
            user_name = name_row["object"] if name_row else ""
            greeting = (
                f"Welcome back{f', **{user_name}**' if user_name else ''}! This is a fresh conversation thread, "
                f"and my **Semantic Knowledge Graph** retains everything I've learned across sessions.\n\n"
                f"What would you like to explore, build, or analyze next?"
            )
            suggestions = [
                "What do you remember about me?",
                "Differentiate and plot cos(x) * x",
                "Simulate Monte Carlo Pi in Python",
                "Explain CAP theorem in distributed systems",
            ]
            conn.execute(
                """
                INSERT INTO messages
                (id, session_id, role, content, intent, confidence, thought_trace, artifacts, memories_added, suggestions, created_at)
                VALUES (?, ?, 'assistant', ?, 'welcome', 1.0, '[]', '[]', '[]', ?, ?)
                """,
                (str(uuid.uuid4()), sess_id, greeting, json.dumps(suggestions), now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sess_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                for field in ("thought_trace", "artifacts", "memories_added", "suggestions"):
                    try:
                        d[field] = json.loads(d.get(field) or "[]")
                    except Exception:
                        d[field] = []
                out.append(d)
            return out
        finally:
            conn.close()


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamReq):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    async def event_generator():
        async for evt in cortex.process_message_stream(req.session_id, req.message, req.client_memories):
            payload = json.dumps(evt["data"])
            yield f"event: {evt['event']}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/sync")
def sync_client_vault(req: ClientVaultSyncReq) -> Dict[str, Any]:
    if req.settings:
        update_settings(req.settings)
    if req.memories:
        for cm in req.memories[:80]:
            subj = (cm.get("subject") or "").strip()
            rel = (cm.get("relation") or "").strip()
            obj = (cm.get("object") or "").strip()
            cat = (cm.get("category") or "fact").strip()
            if subj and rel and obj and cat != "system":
                cortex.memory.upsert_semantic_memory(
                    subject=subj,
                    relation=rel,
                    obj=obj,
                    category=cat,
                    confidence=float(cm.get("confidence", 0.92)),
                    pinned=int(cm.get("pinned", 0)),
                )
    if req.skills:
        now = time.time()
        retrain_needed = False
        with _db_lock:
            conn = get_conn()
            try:
                for sk in req.skills[:40]:
                    trig = (sk.get("trigger_phrase") or "").strip()
                    tmpl = (sk.get("response_template") or "").strip()
                    if not trig or not tmpl:
                        continue
                    exists = conn.execute(
                        "SELECT id FROM custom_skills WHERE lower(trigger_phrase) = lower(?)",
                        (trig,),
                    ).fetchone()
                    if not exists:
                        conn.execute(
                            "INSERT INTO custom_skills (id, trigger_phrase, intent_label, response_template, created_at) VALUES (?, ?, 'custom_skill', ?, ?)",
                            (str(uuid.uuid4()), trig, tmpl, now),
                        )
                        retrain_needed = True
                conn.commit()
            finally:
                conn.close()
        if retrain_needed:
            cortex.neural.train()

    all_mems = [m for m in cortex.memory.get_knowledge_graph()["memories"] if m["category"] != "system"]
    return {"synced": True, "memories": all_mems, "settings": get_settings()}


@app.post("/api/messages/{message_id}/feedback")
def set_message_feedback(message_id: str, req: FeedbackReq) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("UPDATE messages SET feedback = ? WHERE id = ?", (req.feedback, message_id))
            conn.commit()
        finally:
            conn.close()
    return {"ok": True, "feedback": req.feedback}


# ---- Memory Palace & Knowledge Graph ----

@app.get("/api/memory/graph")
def get_memory_graph() -> Dict[str, Any]:
    return cortex.memory.get_knowledge_graph()


@app.post("/api/memory/recall")
def test_memory_recall(req: MemoryRecallReq) -> Dict[str, Any]:
    return cortex.memory.recall(req.query, threshold=req.threshold or 0.15, top_k=10)


@app.post("/api/memory/facts")
def add_memory_fact(req: CreateMemoryReq) -> Dict[str, Any]:
    return cortex.memory.upsert_semantic_memory(
        subject=req.subject,
        relation=req.relation,
        obj=req.object,
        category=req.category or "fact",
        pinned=req.pinned or 0,
        source_text="Manual Memory Palace entry",
    )


@app.patch("/api/memory/facts/{memory_id}")
def patch_memory_fact(memory_id: str, req: PatchMemoryReq) -> Dict[str, Any]:
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (memory_id,)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Memory node not found")
            subj = req.subject if req.subject is not None else existing["subject"]
            rel = req.relation if req.relation is not None else existing["relation"]
            obj = req.object if req.object is not None else existing["object"]
            pin = req.pinned if req.pinned is not None else existing["pinned"]
            imp = req.importance if req.importance is not None else existing["importance"]
            conn.execute(
                """
                UPDATE semantic_memories
                SET subject = ?, relation = ?, object = ?, pinned = ?, importance = ?, updated_at = ?
                WHERE id = ?
                """,
                (subj, rel, obj, pin, imp, now, memory_id),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (memory_id,)).fetchone()
            return dict(updated)
        finally:
            conn.close()


@app.delete("/api/memory/facts/{memory_id}")
def delete_memory_fact(memory_id: str) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM semantic_memories WHERE id = ?", (memory_id,))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.post("/api/memory/reset")
def reset_user_memories() -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM semantic_memories WHERE category != 'system'")
            conn.execute("DELETE FROM episodic_memories")
            conn.commit()
        finally:
            conn.close()
    return {"reset": True}


# ---- Neural Training Studio ----

@app.get("/api/neural/stats")
def get_neural_stats() -> Dict[str, Any]:
    stats = cortex.neural.get_stats()
    with _db_lock:
        conn = get_conn()
        try:
            samples = [dict(r) for r in conn.execute("SELECT * FROM training_samples ORDER BY created_at DESC LIMIT 50").fetchall()]
            skills = [dict(r) for r in conn.execute("SELECT * FROM custom_skills ORDER BY created_at DESC LIMIT 50").fetchall()]
        finally:
            conn.close()
    stats["user_samples"] = samples
    stats["custom_skills"] = skills
    return stats


@app.post("/api/neural/predict")
def predict_neural_intent(req: NeuralPredictReq) -> Dict[str, Any]:
    if not req.text.strip():
        return {"intent": "idle", "confidence": 0.0, "distribution": []}
    return cortex.neural.predict(req.text)


@app.post("/api/neural/samples")
def add_training_sample(req: TrainingSampleReq) -> Dict[str, Any]:
    if not req.text.strip() or not req.intent.strip():
        raise HTTPException(status_code=400, detail="Text and intent are required.")
    sample_id = str(uuid.uuid4())
    now = time.time()
    clean_intent = req.intent.strip().lower().replace(" ", "_")
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO training_samples (id, text, intent, source, created_at) VALUES (?, ?, ?, 'user', ?)",
                (sample_id, req.text.strip(), clean_intent, now),
            )
            conn.commit()
        finally:
            conn.close()
    return {"id": sample_id, "text": req.text.strip(), "intent": clean_intent, "created_at": now}


@app.delete("/api/neural/samples/{sample_id}")
def delete_training_sample(sample_id: str) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM training_samples WHERE id = ?", (sample_id,))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.post("/api/neural/skills")
def add_custom_skill(req: CustomSkillReq) -> Dict[str, Any]:
    if not req.trigger_phrase.strip() or not req.response_template.strip():
        raise HTTPException(status_code=400, detail="Trigger phrase and response template are required.")
    skill_id = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO custom_skills (id, trigger_phrase, intent_label, response_template, created_at) VALUES (?, ?, 'custom_skill', ?, ?)",
                (skill_id, req.trigger_phrase.strip(), req.response_template.strip(), now),
            )
            conn.commit()
        finally:
            conn.close()
    cortex.neural.train()
    return {"id": skill_id, "trigger_phrase": req.trigger_phrase.strip(), "response_template": req.response_template.strip()}


@app.delete("/api/neural/skills/{skill_id}")
def delete_custom_skill(skill_id: str) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM custom_skills WHERE id = ?", (skill_id,))
            conn.commit()
        finally:
            conn.close()
    cortex.neural.train()
    return {"deleted": True}


@app.post("/api/neural/retrain")
def retrain_neural_cortex(req: RetrainReq) -> Dict[str, Any]:
    h1 = max(32, min(256, req.hidden_1 or 128))
    h2 = max(16, min(192, req.hidden_2 or 64))
    iters = max(30, min(200, req.max_iter or 95))
    return cortex.neural.train(hidden_layers=(h1, h2), max_iter=iters)


# ---- AI Lab Tools: Python, Math, Documents, Tasks ----

@app.post("/api/tools/python")
def run_python_sandbox(req: PythonExecReq) -> Dict[str, Any]:
    return cortex.code.execute_python(req.code)


@app.post("/api/tools/math")
def run_symbolic_math(req: MathQueryReq) -> Dict[str, Any]:
    return cortex.math.solve_query(req.query)


@app.get("/api/documents")
def list_docs() -> List[Dict[str, Any]]:
    return cortex.knowledge.list_documents()


@app.post("/api/documents")
def create_doc(req: CreateDocReq) -> Dict[str, Any]:
    try:
        return cortex.knowledge.ingest_document(req.title, req.content, req.source_type or "note")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/documents/upload")
async def upload_doc_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw_bytes = await file.read()
    if len(raw_bytes) > 2_000_000:
        raise HTTPException(status_code=400, detail="File exceeds 2 MB limit.")
    try:
        content = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode text file.")
    title = file.filename or "Uploaded Document"
    return cortex.knowledge.ingest_document(title, content, source_type="file_upload")


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: str) -> Dict[str, Any]:
    ok = cortex.knowledge.delete_document(doc_id)
    return {"deleted": ok}


@app.get("/api/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.post("/api/tasks")
def create_task(req: CreateTaskReq) -> Dict[str, Any]:
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Task title required")
    tid = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO tasks (id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, ?, ?, 'todo', ?, ?, ?)",
                (tid, req.title.strip(), req.description or "", req.priority or "medium", req.due_note or "", now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.patch("/api/tasks/{task_id}")
def patch_task(task_id: str, req: PatchTaskReq) -> Dict[str, Any]:
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Task not found")
            title = req.title if req.title is not None else existing["title"]
            status = req.status if req.status is not None else existing["status"]
            priority = req.priority if req.priority is not None else existing["priority"]
            conn.execute(
                "UPDATE tasks SET title = ?, status = ?, priority = ?, updated_at = ? WHERE id = ?",
                (title, status, priority, now, task_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


# ---- Cognitive Settings ----

@app.get("/api/settings")
def read_settings() -> Dict[str, Any]:
    return get_settings()


@app.patch("/api/settings")
def patch_settings_route(patch: Dict[str, Any]) -> Dict[str, Any]:
    return update_settings(patch)


# Serve built SPA if frontend/dist exists in non-Vercel standalone mode
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if not os.environ.get("VERCEL") and os.path.isdir(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        candidate = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
