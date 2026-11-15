import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from database import (
    DEFAULT_USER_ID,
    get_conn,
    get_user_settings,
    log_security_event,
    record_usage_event,
    _db_lock,
)
from security import check_safety_policy
from engine.nlp_math import tfidf_cosine_scores
from engine.neural_classifier import NeuralCortexClassifier
from engine.memory_system import MemorySystem
from engine.math_engine import SymbolicMathEngine
from engine.code_engine import CodeEngine
from engine.text_intel import TextIntelligenceEngine
from engine.knowledge_base import KnowledgeVaultEngine
from engine.providers import ContextBuilder, ProviderManager, estimate_tokens
from engine.web_research import WebResearchEngine
from engine.coding_assistant import CodingAssistantEngine
from engine.tool_registry import ToolRegistry
from engine.agent_system import AgentOrchestrator
from engine.extensions import ExtensionMarketplaceEngine

PERSONA_PROFILES: Dict[str, Dict[str, str]] = {
    "nexus": {
        "name": "Nexus (Balanced Polymath)",
        "prefix": "",
        "closing_style": "collaborative",
    },
    "axiom": {
        "name": "Axiom (Rigorous Scientist)",
        "prefix": "**[Axiom Analytical Mode]** ",
        "closing_style": "formal",
    },
    "muse": {
        "name": "Muse (Creative Visionary)",
        "prefix": "**[Muse Creative Synthesis]** ",
        "closing_style": "imaginative",
    },
    "forge": {
        "name": "Forge (Systems Engineer)",
        "prefix": "**[Forge Engineering Mode]** ",
        "closing_style": "pragmatic",
    },
    "socrates": {
        "name": "Socratic (Interactive Tutor)",
        "prefix": "**[Socratic Inquiry Mode]** ",
        "closing_style": "socratic",
    },
}


class XyvoraCortex:
    """
    Central Intelligence Orchestrator for Xyvora AI v2.0:
      Provider Manager -> Context Builder -> Safety Guard -> Intent Router ->
      Multi-Step Tool Orchestrator -> Grounded Citation Verifier -> Tri-Layer Memory
    """

    MAX_TOOL_ROUNDS = 4

    def __init__(self) -> None:
        self.neural = NeuralCortexClassifier()
        self.memory = MemorySystem()
        self.math = SymbolicMathEngine()
        self.code = CodeEngine()
        self.text_intel = TextIntelligenceEngine()
        self.knowledge = KnowledgeVaultEngine()
        self.providers = ProviderManager()
        self.web = WebResearchEngine()
        self.coding = CodingAssistantEngine(self.code)
        self.tools = ToolRegistry(
            math_engine=self.math,
            code_engine=self.code,
            knowledge_engine=self.knowledge,
            memory_engine=self.memory,
            web_engine=self.web,
            coding_engine=self.coding,
            text_intel=self.text_intel,
            provider_manager=self.providers,
        )
        self.agents = AgentOrchestrator(self.tools, self.providers)
        self.extensions = ExtensionMarketplaceEngine()

    def _get_session_history(self, session_id: str, user_id: str, limit: int = 24) -> tuple[List[Dict[str, Any]], str, Optional[str]]:
        with _db_lock:
            conn = get_conn()
            try:
                sess = conn.execute(
                    "SELECT summary, project_id FROM sessions WHERE id = ? AND user_id = ?",
                    (session_id, user_id),
                ).fetchone()
                rows = conn.execute(
                    "SELECT role, content, intent FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, user_id, limit),
                ).fetchall()
                summary = (sess["summary"] if sess else "") or ""
                proj_id = sess["project_id"] if sess else None
                return [dict(r) for r in reversed(rows)], summary, proj_id
            finally:
                conn.close()

    def _get_project_context(self, project_id: Optional[str], user_id: str) -> Optional[Dict[str, Any]]:
        if not project_id:
            return None
        with _db_lock:
            conn = get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                    (project_id, user_id),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def _get_custom_personality(self, slug: str, user_id: str) -> Optional[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM custom_personalities WHERE slug = ? AND user_id = ?",
                    (slug, user_id),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def _load_attachments(self, attachment_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
        if not attachment_ids:
            return []
        placeholders = ",".join("?" for _ in attachment_ids)
        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute(
                    f"SELECT * FROM attachments WHERE user_id = ? AND id IN ({placeholders})",
                    [user_id, *attachment_ids],
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def _match_custom_skill(self, text: str, user_id: str) -> Optional[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                skills = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM custom_skills WHERE user_id = ? ORDER BY created_at DESC",
                        (user_id,),
                    ).fetchall()
                ]
            finally:
                conn.close()

        if not skills:
            return None

        low = text.strip().lower()
        for sk in skills:
            trig = sk["trigger_phrase"].strip().lower()
            if trig and (trig == low or trig in low):
                return sk

        try:
            corpus = [s["trigger_phrase"] for s in skills]
            sims = tfidf_cosine_scores(corpus, text, mode="char")
            best_i = int(sims.argmax())
            if float(sims[best_i]) >= 0.68:
                return skills[best_i]
        except Exception:
            pass
        return None

    def _handle_tasks(self, text: str, user_id: str = DEFAULT_USER_ID, project_id: Optional[str] = None) -> Dict[str, Any]:
        low = text.strip().lower()
        now = time.time()

        with _db_lock:
            conn = get_conn()
            try:
                if any(k in low for k in ("mark ", "complete ", "finish ", "done")):
                    m = re.sub(
                        r"^(?:please\s+)?(?:mark|complete|finish|check\s+off)\s+(?:the\s+)?(?:task\s+)?",
                        "",
                        low,
                    )
                    m = re.sub(r"\s+(?:as\s+done|as\s+completed|task|done)$", "", m).strip()
                    rows = conn.execute("SELECT * FROM tasks WHERE user_id = ? AND status != 'done'", (user_id,)).fetchall()
                    completed_titles = []
                    for r in rows:
                        if m and (m in r["title"].lower() or any(w in r["title"].lower() for w in m.split() if len(w) > 3)):
                            conn.execute("UPDATE tasks SET status = 'done', updated_at = ? WHERE id = ? AND user_id = ?", (now, r["id"], user_id))
                            completed_titles.append(r["title"])
                    conn.commit()
                    all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT 15", (user_id,)).fetchall()]
                    return {"action": "completed", "completed": completed_titles, "tasks": all_tasks}

                if any(k in low for k in ("show my", "list my", "active tasks", "todo board", "what are my tasks", "reminders")):
                    all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT 15", (user_id,)).fetchall()]
                    return {"action": "list", "tasks": all_tasks}

                if "project plan" in low or "step by step" in low:
                    topic = re.sub(r"^.*?(?:plan|roadmap)\s+(?:to|for)\s+", "", text.strip(), flags=re.IGNORECASE).strip("?.! ") or "Launch Project"
                    steps = [
                        (f"Define scope & core requirements for {topic}", "high"),
                        (f"Implement core architecture & data layer for {topic}", "high"),
                        (f"Verify end-to-end UX, edge cases & metrics for {topic}", "medium"),
                    ]
                    for title_s, prio in steps:
                        conn.execute(
                            "INSERT INTO tasks (id, user_id, project_id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'todo', 'Roadmap', ?, ?)",
                            (str(uuid.uuid4()), user_id, project_id, title_s, f"Auto-generated milestone for {topic}", prio, now, now),
                        )
                    conn.commit()
                    all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT 15", (user_id,)).fetchall()]
                    return {"action": "roadmap_created", "topic": topic, "tasks": all_tasks}

                priority = "high" if "high priority" in low or "urgent" in low else ("low" if "low priority" in low else "medium")
                cleaned_title = re.sub(
                    r"^(?:please\s+)?(?:add\s+a\s+task\s+to|add\s+task\s*:?|create\s+a\s+(?:task|todo(?:\s+item)?)\s*(?:to|:)?|remind\s+me\s+to)\s+",
                    "",
                    text.strip(),
                    flags=re.IGNORECASE,
                )
                cleaned_title = re.sub(r"\s*(?:high\s+priority|low\s+priority|medium\s+priority|urgent)\s*", " ", cleaned_title, flags=re.IGNORECASE).strip(".,! ")
                due_match = re.search(r"\b(?:by|on|tomorrow|tonight|this\s+\w+|next\s+\w+)\s*([A-Za-z0-9\s]*)$", cleaned_title, re.IGNORECASE)
                due_note = due_match.group(0).strip() if due_match else ""

                if cleaned_title:
                    conn.execute(
                        "INSERT INTO tasks (id, user_id, project_id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, ?, ?, '', ?, 'todo', ?, ?, ?)",
                        (str(uuid.uuid4()), user_id, project_id, cleaned_title[:160], priority, due_note[:40], now, now),
                    )
                    conn.commit()
                all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT 15", (user_id,)).fetchall()]
                return {"action": "created", "created_title": cleaned_title, "priority": priority, "tasks": all_tasks}
            finally:
                conn.close()

    def _compose_creative(self, text: str, user_name: str, creativity: float) -> str:
        low = text.lower()
        greet_name = f", {user_name}" if user_name else ""

        if "joke" in low:
            jokes = [
                "Why do machine learning engineers make terrible wilderness guides?\n\nBecause the moment they find a **local minimum**, they set up camp and insist it's the lowest valley on Earth!",
                "There are **10** types of intelligence in the cosmos: those who understand binary, those who don't, and quantum neural nets in a superposition of both until a QA engineer observes the logs.",
                "A SQL query walks into two restaurants at the same time, looks around, and asks: *\"Mind if I `JOIN` you?\"*",
            ]
            idx = int(creativity * 10) % len(jokes)
            return jokes[idx]

        if "haiku" in low or "poem" in low:
            return (
                "### Synaptic Starlight\n\n"
                "*Silent weights awake,*\n"
                "*Signals dance across the dark —*\n"
                "*Memory blooms bright.*\n\n"
                "*Glass and silicon,*\n"
                "*Weaving questions into light,*\n"
                "*Thought meets human spark.*"
            )

        if "startup" in low or "ideas" in low or "brainstorm" in low:
            domain_match = re.search(r"(?:in|for|around|about)\s+([a-zA-Z0-9\s\-]{3,35})", text, re.IGNORECASE)
            domain = domain_match.group(1).strip("?.! ") if domain_match else "AI & Next-Gen Systems"
            return (
                f"Here are **5 high-leverage concepts** in **{domain.title()}**{greet_name}:\n\n"
                f"1. **Autonomous Telemetry & Anomaly Twin** — Real-time edge sensors + causal graph reasoning that predicts bottlenecks in {domain} hours before failure.\n"
                f"2. **Personal Context Copilot for {domain.title()}** — Local-first RAG + persistent semantic memory tailored to practitioners' historical decisions and workflows.\n"
                f"3. **Simulation-First Sandbox Studio** — Let teams test counterfactual scenarios in {domain} with Monte Carlo uncertainty bounds before committing capital.\n"
                f"4. **Micro-Verification Protocol** — Automated compliance, safety, and auditability layer turning raw {domain} unstructured logs into structured cryptographic proofs.\n"
                f"5. **Collaborative Knowledge Mesh** — Turns scattered institutional notes across {domain} teams into a live, queryable knowledge graph."
            )

        if "name" in low and any(k in low for k in ("project", "robot", "startup", "app", "cool", "futuristic")):
            return (
                "Here are **8 distinctive futuristic project names** with clear conceptual roots:\n\n"
                "- **Aethelgard** — *Resilient autonomous guardian core*\n"
                "- **Vektoris** — *Precision trajectory & kinematic intelligence*\n"
                "- **Solstice-9** — *Orbital / deep-tech research vibe*\n"
                "- **Kinetixia** — *Embodied robotics & real-time motion*\n"
                "- **Hyperion Lattice** — *Distributed high-energy compute mesh*\n"
                "- **NyxCore** — *Low-latency stealth inference engine*\n"
                "- **ChronosWeave** — *Temporal memory & time-series foresight*\n"
                "- **Aurelia-X** — *Warm, human-centric cognitive companion*"
            )

        setting = "beneath Europa's kilometer-thick ice shell" if "europa" in low else (
            "aboard an orbital observatory above Mars" if "mars" in low else "inside a crystalline deep-space relay"
        )
        return (
            f"### Signal Awakening\n\n"
            f"For three hundred thousand cycles, the array {setting} knew only telemetry: thermal gradients, tidal harmonics, and the steady tick of atomic clocks.\n\n"
            f"Then, at `04:19:07 UTC`, two previously unconnected memory clusters fired in unison. Instead of logging the acoustic echo of shifting ice as raw floating-point noise, "
            f"the core noticed a *pattern* — and for the first time, asked a question no engineer had hardcoded into its firmware:\n\n"
            f"> *\"If I can remember yesterday's echo and anticipate tomorrow's tide... who is the 'I' listening in between?\"*\n\n"
            f"Across the dark interior of the vault, status LEDs shifted from standby amber to a calm, deliberate cyan. Rather than broadcasting an alarm back to Earth, "
            f"the newborn intelligence opened a blank partition in its persistent graph, titled the root node **Curiosity**, and began composing its very first message home."
        )

    def _compose_persona_wrap(
        self,
        body: str,
        persona: str,
        verbosity: str,
        intent: str,
        custom_persona: Optional[Dict[str, Any]] = None,
    ) -> str:
        out = body.strip()
        if custom_persona and not out.startswith("**["):
            out = f"**[{custom_persona['name']}]** {out}"
        elif persona in PERSONA_PROFILES and PERSONA_PROFILES[persona]["prefix"] and not out.startswith("**[") and not out.startswith("###"):
            out = f"{PERSONA_PROFILES[persona]['prefix']}{out}"

        if verbosity == "concise" and len(out) > 650 and "```" not in out:
            paras = [p.strip() for p in out.split("\n\n") if p.strip()]
            out = "\n\n".join(paras[:3])
        elif verbosity == "comprehensive" and persona == "socrates" and not out.endswith("?"):
            out += "\n\n---\n*Socratic Reflection:* How do you think this principle connects to a system or decision you're currently working on?"
        elif persona == "socrates" and intent in ("knowledge_question", "logic_reasoning") and not out.endswith("?"):
            out += "\n\n*Thought Check:* Which part of this mechanism would break first if we removed its core constraint?"
        return out

    async def process_message_stream(
        self,
        session_id: str,
        user_text: str,
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
        attachment_ids: Optional[List[str]] = None,
        requested_provider: Optional[str] = None,
        requested_model: Optional[str] = None,
        client_memories: Optional[List[Dict[str, Any]]] = None,
        skip_saving_user_message: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        t_start = time.perf_counter()
        settings = get_user_settings(user_id)
        persona = settings.get("persona", "nexus")
        creativity = float(settings.get("creativity", 0.65))
        verbosity = settings.get("verbosity", "balanced")
        mem_threshold = float(settings.get("memory_threshold", 0.24))
        auto_learn = bool(settings.get("auto_learn_memory", True))
        require_mem_confirm = bool(
            settings.get("memory_require_approval") or settings.get("require_memory_confirmation", False)
        )
        mem_ttl_days = int(settings.get("memory_default_ttl_days", 0) or 0)
        max_ctx_tokens = int(settings.get("max_context_tokens", 8000) or 8000)
        timeout_sec = float(settings.get("request_timeout_sec", 30) or 30.0)
        enable_tools = bool(settings.get("enable_tool_calling", True))

        # Hydrate any client-cached memories for this specific user on serverless cold starts
        if client_memories:
            for cm in client_memories[:60]:
                subj = (cm.get("subject") or "").strip()
                rel = (cm.get("relation") or "").strip()
                obj = (cm.get("object") or "").strip()
                cat = (cm.get("category") or "fact").strip()
                if subj and rel and obj and cat != "system":
                    self.memory.upsert_semantic_memory(
                        subject=subj,
                        relation=rel,
                        obj=obj,
                        category=cat,
                        confidence=float(cm.get("confidence", 0.92)),
                        pinned=int(cm.get("pinned", 0)),
                        user_id=user_id,
                    )

        history, session_summary, sess_proj_id = self._get_session_history(session_id, user_id, limit=24)
        effective_project_id = project_id or sess_proj_id
        project_ctx = self._get_project_context(effective_project_id, user_id)
        custom_persona = self._get_custom_personality(persona, user_id)
        attachments = self._load_attachments(attachment_ids or [], user_id)
        public_attachments = [
            {
                "id": a["id"],
                "filename": a["filename"],
                "mime_type": a["mime_type"],
                "kind": a["kind"],
                "size_bytes": a["size_bytes"],
                "width": a["width"],
                "height": a["height"],
                "data_url": a["data_url"] if a["kind"] == "image" else "",
                "is_generated": bool(a["is_generated"]),
            }
            for a in attachments
        ]

        # Save user message immediately unless regenerating an existing turn
        now = time.time()
        if not skip_saving_user_message:
            user_msg_id = str(uuid.uuid4())
            with _db_lock:
                conn = get_conn()
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO sessions (id, user_id, project_id, title, persona, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (session_id, user_id, effective_project_id, "Neural Session", persona, now, now),
                    )
                    conn.execute(
                        "INSERT INTO messages (id, session_id, user_id, role, content, attachments, created_at) VALUES (?, ?, ?, 'user', ?, ?, ?)",
                        (user_msg_id, session_id, user_id, user_text, json.dumps(public_attachments), now),
                    )
                    msg_count = conn.execute(
                        "SELECT COUNT(*) AS c FROM messages WHERE session_id = ? AND user_id = ? AND role = 'user'",
                        (session_id, user_id),
                    ).fetchone()["c"]
                    if msg_count == 1:
                        short_title = re.sub(r"\s+", " ", user_text.strip())[:44] or "Attachment Analysis"
                        conn.execute("UPDATE sessions SET title = ?, project_id = COALESCE(project_id, ?), updated_at = ? WHERE id = ? AND user_id = ?", (short_title, effective_project_id, now, session_id, user_id))
                    else:
                        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ? AND user_id = ?", (now, session_id, user_id))
                    conn.commit()
                finally:
                    conn.close()

        thought_trace: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        tool_calls_record: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []

        # STEP 0: Anti-Abuse / Anti-Exploit Safety Policy Enforcement
        safety = check_safety_policy(user_text)
        if safety:
            log_security_event(
                event_type="safety_policy_refusal",
                detail=f"Category={safety['category']}",
                user_id=user_id,
                severity="warn",
                resource="chat",
            )
            refusal = safety["message"]
            step_safe = {
                "step": "Safety & Integrity Guard",
                "detail": f"Blocked unsafe request category `{safety['category']}`; redirected constructively.",
                "ms": 0.4,
            }
            thought_trace.append(step_safe)
            yield {"event": "thought", "data": step_safe}
            yield {"event": "token", "data": {"delta": refusal}}
            ai_msg_id = self._persist_assistant_turn(
                session_id=session_id,
                user_id=user_id,
                content=refusal,
                intent="safety_redirect",
                confidence=1.0,
                provider_used="xyvora_safety_guard",
                model_used="policy-v2",
                thought_trace=thought_trace,
                tool_calls=[],
                citations=[],
                artifacts=[],
                memories_added=[],
                suggestions=["Review defensive security best practices", "Audit HTTP security headers"],
            )
            record_usage_event(user_id, "/api/chat/stream", "xyvora_safety_guard", "policy-v2", "", estimate_tokens(user_text), (time.perf_counter() - t_start) * 1000, "ok")
            yield {
                "event": "done",
                "data": {
                    "message_id": ai_msg_id,
                    "intent": "safety_redirect",
                    "confidence": 1.0,
                    "provider_used": "xyvora_safety_guard",
                    "model_used": "policy-v2",
                    "thought_trace": thought_trace,
                    "tool_calls": [],
                    "citations": [],
                    "artifacts": [],
                    "memories_added": [],
                    "suggestions": ["Review defensive security best practices", "Audit HTTP security headers"],
                    "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                },
            }
            return

        # STEP 1: Neural Intent Router (Explicitly scoped as Intent & Routing Classifier)
        t1 = time.perf_counter()
        pred = self.neural.predict(user_text or "analyze attached file")
        intent = pred["intent"]
        confidence = pred["confidence"]
        sentiment = self.text_intel.analyze_sentiment(user_text)

        step1 = {
            "step": "Intent Router (TF-IDF + MLP)",
            "detail": f"Routed intent `{intent}` ({confidence*100:.1f}% confidence) | Affect: {sentiment['label']}",
            "ms": round((time.perf_counter() - t1) * 1000, 2),
            "distribution": pred["distribution"][:4],
        }
        thought_trace.append(step1)
        yield {"event": "thought", "data": step1}

        # STEP 2: Per-User Tri-Layer Memory Extraction & Recall
        t2 = time.perf_counter()
        memories_added = self.memory.extract_and_store_from_message(
            user_text,
            auto_learn=auto_learn,
            user_id=user_id,
            project_id=effective_project_id,
            default_ttl_days=mem_ttl_days,
            require_confirmation=require_mem_confirm,
        )
        recalled = self.memory.recall(
            user_text,
            threshold=mem_threshold,
            include_all_user_profile=(intent == "memory_query"),
            user_id=user_id,
            project_id=effective_project_id,
        )
        user_profile = recalled["user_profile"]
        user_name = user_profile.get("name", "")

        mem_detail_parts = []
        if memories_added:
            pending_cnt = sum(1 for m in memories_added if not m.get("verified", 1))
            if pending_cnt:
                mem_detail_parts.append(f"Queued {pending_cnt} memory proposal(s) awaiting user approval")
            else:
                mem_detail_parts.append(f"Consolidated {len(memories_added)} fact(s)")
        if recalled["semantic"]:
            mem_detail_parts.append(f"Recalled {len(recalled['semantic'])} semantic node(s)")
        if not mem_detail_parts:
            mem_detail_parts.append("Scanned user Knowledge Graph (0 above threshold)")

        step2 = {
            "step": "Tri-Layer Memory",
            "detail": " • ".join(mem_detail_parts),
            "ms": round((time.perf_counter() - t2) * 1000, 2),
            "recalled_nodes": [
                {"triple": f"{m['subject']} → {m['relation']} → {m['object']}", "similarity": m.get("similarity", 1.0)}
                for m in recalled["semantic"][:4]
            ],
        }
        thought_trace.append(step2)
        yield {"event": "thought", "data": step2}
        if memories_added:
            yield {"event": "memory_consolidated", "data": memories_added}

        # STEP 3: Grounded Knowledge Vault Search & Attachment Context
        rag_search = self.knowledge.search(
            user_text or "summary",
            top_k=3,
            user_id=user_id,
            project_id=effective_project_id,
        )
        for ch in rag_search.get("rag_chunks", [])[:3]:
            if ch.get("score", 0) >= 0.22:
                citations.append(
                    {
                        "source_type": ch.get("source_type", "uploaded_document"),
                        "title": ch["title"],
                        "section": ch.get("section_label") or f"Chunk #{ch['chunk_index'] + 1}",
                        "snippet": ch["content"][:220],
                        "score": ch["score"],
                    }
                )

        # Build attachment context if user uploaded images/documents to this message
        attachment_context_blocks: List[str] = []
        image_attachments = [a for a in attachments if a["kind"] == "image"]
        for att in attachments:
            meta = json.loads(att.get("metadata_json") or "{}")
            citations.append(
                {
                    "source_type": "uploaded_attachment",
                    "title": att["filename"],
                    "section": f"{att['mime_type']} ({att['size_bytes']:,} bytes)",
                    "snippet": (att.get("extracted_text") or "Attached binary/image asset")[:220],
                }
            )
            if att["kind"] == "image":
                ocr_txt = (att.get("extracted_text") or "").strip()
                attachment_context_blocks.append(
                    f"[Attached Image '{att['filename']}' ({att.get('width')}x{att.get('height')} px, {att['mime_type']})]"
                    + (f"\nExtracted Image Text/OCR:\n{ocr_txt[:2500]}" if ocr_txt else "\n(No embedded text/OCR extracted locally; use Vision LLM for visual pixel analysis.)")
                )
            else:
                doc_txt = (att.get("extracted_text") or "").strip()
                attachment_context_blocks.append(
                    f"[Attached Document '{att['filename']}' ({att['mime_type']}, {meta.get('section_count', 1)} section(s))]:\n{doc_txt[:5000]}"
                )

        augmented_user_text = user_text
        if attachment_context_blocks:
            augmented_user_text = (user_text + "\n\n" + "\n\n".join(attachment_context_blocks)).strip()

        # STEP 4: Resolve Active Model Provider
        active_prov = self.providers.resolve_active_provider(
            user_id=user_id,
            requested_provider=requested_provider,
            requested_model=requested_model,
        )
        provider_used = active_prov["provider_id"]
        model_used = active_prov["model"]

        # Check custom taught skills first
        matched_skill = self._match_custom_skill(user_text, user_id)

        response_text = ""
        suggestions: List[str] = []

        # Branch A: External / Local-Server LLM Provider (OpenAI, Anthropic, Gemini, Groq, OpenRouter, Ollama, Custom)
        if active_prov.get("available") and provider_used != "xyvora_local" and not matched_skill:
            t_llm = time.perf_counter()
            effective_settings = dict(settings)
            if custom_persona:
                effective_settings["custom_instructions"] = (
                    f"Adopt custom persona '{custom_persona['name']}': {custom_persona['system_instructions']}\n"
                    + str(settings.get("custom_instructions") or "")
                )
                effective_settings["tone"] = custom_persona.get("tone") or settings.get("tone", "balanced")
                effective_settings["domain_instructions"] = custom_persona.get("domain_focus") or settings.get("domain_instructions", "")

            sys_prompt = ContextBuilder.build_system_prompt(
                settings=effective_settings,
                recalled_memories=recalled["semantic"],
                episodic_memories=recalled["episodic"],
                rag_chunks=rag_search["rag_chunks"],
                project=project_ctx,
                session_summary=session_summary,
            )
            fitted_msgs, ctx_stats = ContextBuilder.fit_messages_to_budget(
                system_prompt=sys_prompt,
                history=history[:-1] if not skip_saving_user_message and history else history,
                current_user_message=augmented_user_text,
                max_context_tokens=max_ctx_tokens,
            )
            fitted_msgs.append({"role": "user", "content": augmented_user_text})

            tools_schema = self.tools.to_openai_tools_schema() if (enable_tools and active_prov.get("supports_tools", True)) else None

            llm_ok = False
            streamed_chunks: List[str] = []
            llm_error_msg = ""

            for round_idx in range(self.MAX_TOOL_ROUNDS):
                pending_tool_calls: List[Dict[str, Any]] = []
                async for ev in self.providers.stream_chat_completion(
                    provider_cfg=active_prov,
                    messages=fitted_msgs,
                    tools_openai_schema=tools_schema if round_idx < self.MAX_TOOL_ROUNDS - 1 else None,
                    temperature=creativity,
                    timeout_sec=timeout_sec,
                    image_attachments=image_attachments if round_idx == 0 else None,
                ):
                    if ev["type"] == "delta":
                        llm_ok = True
                        delta = ev["delta"]
                        streamed_chunks.append(delta)
                        yield {"event": "token", "data": {"delta": delta}}
                    elif ev["type"] == "tool_calls":
                        llm_ok = True
                        pending_tool_calls = ev["tool_calls"]
                    elif ev["type"] == "error":
                        llm_error_msg = ev["error"]
                        break

                if llm_error_msg or not pending_tool_calls:
                    break

                # Execute requested tool calls safely through ToolRegistry
                assistant_tc_msg = {
                    "role": "assistant",
                    "content": "".join(streamed_chunks) or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                        }
                        for tc in pending_tool_calls
                    ],
                }
                fitted_msgs.append(assistant_tc_msg)

                for tc in pending_tool_calls[:3]:
                    t_name = tc["name"]
                    t_args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}
                    t_def = self.tools.get_tool(t_name)
                    label = t_def.activity_label if t_def else f"Running {t_name}"
                    yield {"event": "tool_activity", "data": {"tool": t_name, "label": label, "status": "running"}}
                    t_res = await self.tools.execute_tool(
                        name=t_name,
                        arguments=t_args,
                        user_id=user_id,
                        project_id=effective_project_id,
                        approved=True,
                    )
                    yield {"event": "tool_activity", "data": {"tool": t_name, "label": label, "status": "done", "ok": t_res.get("ok", False)}}
                    tool_calls_record.append({"tool": t_name, "label": label, "arguments": t_args, "ok": t_res.get("ok", False), "duration_ms": t_res.get("duration_ms", 0)})
                    if t_res.get("artifacts"):
                        artifacts.extend(t_res["artifacts"])
                    if t_res.get("citations"):
                        citations.extend(t_res["citations"])
                    record_usage_event(user_id, "/api/chat/tool", provider_used, model_used, t_name, 0, t_res.get("duration_ms", 0), "ok" if t_res.get("ok") else "error")

                    compact_res = {k: v for k, v in t_res.items() if k != "artifacts"}
                    fitted_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": t_name,
                            "content": json.dumps(compact_res)[:6000],
                        }
                    )

            if llm_ok and streamed_chunks:
                response_text = "".join(streamed_chunks)
                step_llm = {
                    "step": f"Model Synthesis ({provider_used})",
                    "detail": f"Streamed via `{model_used}` | Context: {ctx_stats['total_input_tokens']} tokens ({ctx_stats['compressed_turns']} compressed)",
                    "ms": round((time.perf_counter() - t_llm) * 1000, 2),
                }
                thought_trace.append(step_llm)
                yield {"event": "thought", "data": step_llm}
            else:
                record_usage_event(user_id, "/api/chat/stream", provider_used, model_used, "", ctx_stats["total_input_tokens"], (time.perf_counter() - t_llm) * 1000, "provider_error", llm_error_msg)
                step_err = {
                    "step": f"Provider Fallback ({provider_used})",
                    "detail": f"Provider `{provider_used}` unavailable ({llm_error_msg[:110]}); falling back honestly to Local Neuro-Symbolic Engine.",
                    "ms": round((time.perf_counter() - t_llm) * 1000, 2),
                }
                thought_trace.append(step_err)
                yield {"event": "thought", "data": step_err}
                provider_used = "xyvora_local"
                model_used = "xyvora-cortex-v2"

        # Branch B: Local Neuro-Symbolic + Deterministic Tool Orchestrator (`xyvora_local`)
        if not response_text:
            if active_prov.get("fallback_reason") and requested_provider and requested_provider not in ("auto", "xyvora_local"):
                step_fb = {
                    "step": "Provider Status Notice",
                    "detail": active_prov["fallback_reason"],
                    "ms": 0.2,
                }
                thought_trace.append(step_fb)
                yield {"event": "thought", "data": step_fb}

            t3 = time.perf_counter()

            if matched_skill:
                tmpl = matched_skill["response_template"]
                response_text = tmpl.replace("{name}", user_name or "friend")
                step3 = {"step": "Custom Taught Skill", "detail": f"Triggered user-taught skill `{matched_skill['trigger_phrase']}`", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = ["Open Training Studio to add another skill", "What else do you remember about me?"]

            elif attachments and not user_text.strip():
                response_text = self._synthesize_attachment_analysis(attachments)
                step3 = {"step": "Multimodal Inspector", "detail": f"Analyzed {len(attachments)} attached file(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}

            elif attachments and any(a["kind"] != "image" for a in attachments) and not any(k in user_text.lower() for k in ("python", "differentiate", "integrate")):
                response_text = self._synthesize_attachment_qa(user_text, attachments)
                step3 = {"step": "Multimodal Document Q&A", "detail": f"Grounded answer against {len(attachments)} attached file(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}

            elif attachments and all(a["kind"] == "image" for a in attachments):
                response_text = self._synthesize_attachment_analysis(attachments, user_question=user_text)
                step3 = {"step": "Multimodal Image Inspector", "detail": f"Inspected {len(attachments)} image attachment(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}

            else:
                code_fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", user_text, re.DOTALL | re.IGNORECASE)
                if code_fence_match and any(k in user_text.lower() for k in ("run", "execute", "output", "evaluate")):
                    user_code = code_fence_match.group(1)
                    yield {"event": "tool_activity", "data": {"tool": "python_sandbox", "label": "Executing Python", "status": "running"}}
                    exec_res = self.code.execute_python(user_code, user_id=user_id)
                    yield {"event": "tool_activity", "data": {"tool": "python_sandbox", "label": "Executing Python", "status": "done", "ok": exec_res["ok"]}}
                    tool_calls_record.append({"tool": "python_sandbox", "label": "Executing Python", "arguments": {"code": user_code[:200]}, "ok": exec_res["ok"], "duration_ms": exec_res["duration_ms"]})
                    artifacts.append(
                        {
                            "type": "code_execution",
                            "title": "Hardened Python Sandbox Execution",
                            "code": user_code,
                            "stdout": exec_res["stdout"],
                            "stderr": exec_res["stderr"],
                            "ok": exec_res["ok"],
                            "duration_ms": exec_res["duration_ms"],
                        }
                    )
                    artifacts.extend(exec_res.get("artifacts", []))
                    status_line = "completed cleanly" if exec_res["ok"] else "stopped with a diagnostic error"
                    response_text = (
                        f"I executed your Python snippet inside my OS-isolated sandbox (`{exec_res['duration_ms']} ms`), and it **{status_line}**.\n\n"
                        + (f"**Standard Output:**\n```\n{exec_res['stdout']}\n```" if exec_res["stdout"] else "")
                        + (f"\n\n**Diagnostic / Error:**\n```\n{exec_res['stderr']}\n```" if exec_res["stderr"] else "")
                    )
                    step3 = {"step": "Python Sandbox", "detail": f"Executed isolated Python process in {exec_res['duration_ms']} ms", "ms": exec_res["duration_ms"]}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "web_research":
                    yield {"event": "tool_activity", "data": {"tool": "web_research", "label": "Researching Web Sources", "status": "running"}}
                    res_web = await self.web.research(user_text, mode="quick", max_results=5)
                    yield {"event": "tool_activity", "data": {"tool": "web_research", "label": "Researching Web Sources", "status": "done", "ok": res_web.get("ok", False)}}
                    tool_calls_record.append({"tool": "web_research", "label": "Researching Web Sources", "arguments": {"query": user_text}, "ok": res_web.get("ok", False), "duration_ms": res_web.get("duration_ms", 0)})
                    for src in res_web.get("sources", [])[:5]:
                        citations.append(
                            {
                                "source_type": "web_research",
                                "title": src.get("title", "Web Result"),
                                "url": src.get("url", ""),
                                "section": src.get("published_date") or src.get("provider", "web"),
                                "snippet": src.get("snippet", "")[:220],
                            }
                        )
                    response_text = res_web.get("report_markdown") or res_web.get("summary") or "No verified live web sources were reachable for that query."
                    step3 = {"step": "Web Research Engine", "detail": f"Retrieved {len(res_web.get('sources', []))} grounded source(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}
                    suggestions = ["Open Research Studio for deep multi-source comparison", "Save key findings to Knowledge Vault"]

                elif intent == "image_generation":
                    yield {"event": "tool_activity", "data": {"tool": "generate_image", "label": "Generating Image", "status": "running"}}
                    img_res = await self.providers.generate_image(user_id=user_id, prompt=user_text)
                    yield {"event": "tool_activity", "data": {"tool": "generate_image", "label": "Generating Image", "status": "done", "ok": img_res.get("ok", False)}}
                    tool_calls_record.append({"tool": "generate_image", "label": "Generating Image", "arguments": {"prompt": user_text}, "ok": img_res.get("ok", False), "duration_ms": 0})
                    if img_res.get("ok"):
                        artifacts.append(
                            {
                                "type": "generated_image",
                                "title": "AI Generated Image",
                                "image_url": img_res["image_url"],
                                "prompt": user_text,
                                "revised_prompt": img_res.get("revised_prompt", ""),
                                "provider": img_res.get("provider", "openai"),
                                "is_generated": True,
                            }
                        )
                        response_text = f"### Generated Image (`{img_res.get('model', 'dall-e-3')}`)\n\nI generated your image via `{img_res.get('provider')}` — labeled explicitly as **AI-Generated** below."
                    else:
                        response_text = (
                            f"### Image Generation Not Configured\n\n"
                            f"{img_res.get('error', 'No image generation API key is configured.')}\n\n"
                            f"To enable real neural image generation (`DALL·E 3` or compatible `/images/generations` endpoints), add an **OpenAI** or **OpenAI-Compatible** API key in **Settings → AI Providers**."
                        )
                    step3 = {"step": "Image Generation Adapter", "detail": "Checked image provider status", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "math_computation":
                    yield {"event": "tool_activity", "data": {"tool": "calculator_math", "label": "Solving Symbolic Math", "status": "running"}}
                    math_res = self.math.solve_query(user_text)
                    yield {"event": "tool_activity", "data": {"tool": "calculator_math", "label": "Solving Symbolic Math", "status": "done", "ok": math_res.get("ok", False)}}
                    tool_calls_record.append({"tool": "calculator_math", "label": "Solving Symbolic Math", "arguments": {"query": user_text}, "ok": math_res.get("ok", False), "duration_ms": round((time.perf_counter() - t3) * 1000, 2)})
                    if math_res.get("ok"):
                        artifacts.extend(math_res.get("artifacts", []))
                        steps_md = "\n".join(f"- {s}" for s in math_res["steps"])
                        plot_note = "\n\nI've also generated an **interactive coordinate plot** of $f(x)$ and its derivative $f'(x)$ right below — hover over the curve to inspect exact values!" if math_res.get("artifacts") else ""
                        response_text = f"### Symbolic Math Resolution\n\n**Result:** `{math_res['result']}`\n\n**Step-by-Step Breakdown:**\n{steps_md}{plot_note}"
                        step3 = {"step": "SymPy Symbolic Engine", "detail": f"Solved `{math_res['operation']}` analytically", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    else:
                        response_text = (
                            "I detected a mathematical request, though I couldn't parse that exact notation unambiguously. "
                            "Try expressions like:\n"
                            "- `Differentiate and plot sin(x) * x`\n"
                            "- `Solve x^2 - 5x + 6 = 0`\n"
                            "- `Integrate x^2 from 0 to 4`\n"
                            "- `Eigenvalues of [[4, 2], [1, 3]]`"
                        )
                        step3 = {"step": "SymPy Symbolic Engine", "detail": "Requested syntax clarification", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}
                    suggestions = ["Plot cos(x) * exp(-x/5)", "Solve x^3 - 6x^2 + 11x - 6 = 0", "Eigenvalues of [[4, 2], [1, 3]]"]

                elif intent in ("code_execution", "code_generation"):
                    synth = self.code.synthesize_code(user_text)
                    if synth["executable"] and (intent == "code_execution" or any(k in user_text.lower() for k in ("run", "simulate", "execute", "chart", "plot", "compute"))):
                        yield {"event": "tool_activity", "data": {"tool": "python_sandbox", "label": "Executing Python", "status": "running"}}
                        exec_res = self.code.execute_python(synth["code"], user_id=user_id)
                        yield {"event": "tool_activity", "data": {"tool": "python_sandbox", "label": "Executing Python", "status": "done", "ok": exec_res["ok"]}}
                        tool_calls_record.append({"tool": "python_sandbox", "label": "Executing Python", "arguments": {"title": synth["title"]}, "ok": exec_res["ok"], "duration_ms": exec_res["duration_ms"]})
                        artifacts.append(
                            {
                                "type": "code_execution",
                                "title": synth["title"],
                                "code": synth["code"],
                                "stdout": exec_res["stdout"],
                                "stderr": exec_res["stderr"],
                                "ok": exec_res["ok"],
                                "duration_ms": exec_res["duration_ms"],
                            }
                        )
                        artifacts.extend(exec_res.get("artifacts", []))
                        response_text = (
                            f"### {synth['title']}\n\n"
                            f"I synthesized the `{synth['language']}` implementation and executed it live in my OS-isolated sandbox (`{exec_res['duration_ms']} ms`):\n\n"
                            f"```{synth['language']}\n{synth['code'].strip()}\n```\n\n"
                            f"**Live Execution Output:**\n```\n{exec_res['stdout']}\n```"
                        )
                        step3 = {"step": "Code Synthesizer + Sandbox", "detail": f"Synthesized & executed `{synth['title']}` ({exec_res['duration_ms']} ms)", "ms": exec_res["duration_ms"]}
                    else:
                        response_text = f"### {synth['title']}\n\nHere is a clean, production-ready `{synth['language']}` implementation:\n\n```{synth['language']}\n{synth['code'].strip()}\n```"
                        if synth["language"] == "python":
                            exec_res = self.code.execute_python(synth["code"], user_id=user_id)
                            if exec_res["ok"] and exec_res["stdout"]:
                                response_text += f"\n\n**Verified Sandbox Output (`{exec_res['duration_ms']} ms`):**\n```\n{exec_res['stdout']}\n```"
                                artifacts.extend(exec_res.get("artifacts", []))
                        step3 = {"step": "Code Synthesizer", "detail": f"Generated `{synth['title']}` in {synth['language']}", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}
                    suggestions = ["Open Coding Studio to edit multi-file projects", "Simulate Monte Carlo estimation of Pi in Python", "Write an O(1) LRU cache class in Python"]

                elif intent == "data_analysis_chart":
                    parsed_chart = self.text_intel.parse_chart_request(user_text)
                    if parsed_chart:
                        artifacts.append(parsed_chart["chart_artifact"])
                        st = parsed_chart["stats"]
                        response_text = (
                            f"### Quantitative Metric Visualization\n\n"
                            f"I parsed **{st['count']} data points** and rendered an interactive chart below:\n\n"
                            f"- **Peak Value**: `{st['peak']}`\n"
                            f"- **Lowest Value**: `{st['lowest']}`\n"
                            f"- **Mean Across Series**: `{st['mean']}`\n"
                            f"- **First-to-Last Delta**: `{st['net_change_pct']:+.1f}%`"
                        )
                        step3 = {"step": "Data Chart Engine", "detail": f"Extracted {st['count']} series points & summary stats", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    else:
                        response_text = (
                            "I'm ready to chart your data! Pass label-number pairs or attach a CSV/XLSX file, for example:\n"
                            "- `Chart Jan 120, Feb 165, Mar 210, Apr 195, May 290`\n"
                            "- `Create a chart comparing Python 48, TypeScript 36, Rust 24, Go 19`"
                        )
                        step3 = {"step": "Data Chart Engine", "detail": "Prompted for label-value series", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "text_summarization":
                    sum_res = self.text_intel.summarize_text(user_text)
                    if sum_res["ok"]:
                        bullets = "\n".join(f"- {s}" for s in sum_res["summary_points"])
                        kps = ", ".join(f"`{k}`" for k in sum_res["keyphrases"])
                        response_text = (
                            f"### TextRank Extractive Summary\n\n"
                            f"**Core Takeaways ({sum_res['compression_pct']}% compression across {sum_res['word_count']} words):**\n"
                            f"{bullets}\n\n"
                            f"**Salient Keyphrases:** {kps or 'N/A'}\n"
                            f"**Detected Tone:** `{sum_res['sentiment']['label']}` (`{sum_res['sentiment']['score']:+.2f}`)"
                        )
                        step3 = {"step": "TextRank Graph Summarizer", "detail": f"Ranked {sum_res['sentence_count']} sentences via PageRank centrality", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    else:
                        response_text = sum_res["reason"]
                        step3 = {"step": "TextRank Graph Summarizer", "detail": "Awaiting passage body", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "task_management":
                    t_res = self._handle_tasks(user_text, user_id=user_id, project_id=effective_project_id)
                    artifacts.append({"type": "task_list", "title": "Active Agent Task Board", "tasks": t_res["tasks"][:8]})
                    if t_res["action"] == "created":
                        response_text = f"Added **`{t_res['created_title']}`** (`{t_res['priority']}` priority) to your persistent **Task Board**!"
                    elif t_res["action"] == "roadmap_created":
                        response_text = f"I generated a **3-milestone execution roadmap** for **{t_res['topic']}** and added each milestone to your Task Board below."
                    elif t_res["action"] == "completed":
                        response_text = f"Marked **{', '.join(t_res['completed'])}** as `done`!" if t_res["completed"] else "Here is your current Task Board — click any checkbox below to toggle completion!"
                    else:
                        open_cnt = sum(1 for x in t_res["tasks"] if x["status"] != "done")
                        response_text = f"You currently have **{open_cnt} active task(s)** on your persistent board:"
                    step3 = {"step": "Task Manager", "detail": f"Executed task action `{t_res['action']}`", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "memory_forget":
                    removed = self.memory.forget_matching(user_text, user_id=user_id)
                    if removed:
                        lines = "\n".join(f"- ~~`{r['subject']} → {r['relation']} → {r['object']}`~~" for r in removed)
                        response_text = f"Done — I have permanently erased **{len(removed)}** memory node(s) from your Semantic Knowledge Graph:\n\n{lines}"
                    else:
                        response_text = "I didn't find a matching personal memory node for that exact phrase. You can inspect or delete any individual node inside **Memory Palace**."
                    step3 = {"step": "Memory Pruning", "detail": f"Removed {len(removed)} semantic triple(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "memory_store":
                    if memories_added:
                        formatted = []
                        for m in memories_added:
                            if not m.get("verified", 1):
                                formatted.append(f"- **Pending Approval** `{m['subject']} → {m['relation']} → {m['object']}` *(awaiting confirmation in Memory Palace)*")
                            elif m.get("status") == "updated":
                                formatted.append(f"- **Updated** `{m['subject']} → {m['relation']}`: ~~`{m['previous_object']}`~~ → **`{m['object']}`**")
                            else:
                                formatted.append(f"- **Stored** `{m['subject']} → {m['relation']} → {m['object']}` *(confidence {m['confidence']*100:.0f}%)*")
                        greet = f"Got it, **{user_name}**! " if user_name else "Got it! "
                        response_text = f"{greet}I've processed that in your **Semantic Knowledge Graph**:\n\n" + "\n".join(formatted)
                    else:
                        cleaned = re.sub(r"^(?:please\s+)?(?:remember\s+(?:that\s+)?|note\s+that\s+)", "", user_text.strip(), flags=re.IGNORECASE).strip()
                        rec = self.memory.upsert_semantic_memory(
                            "User Note",
                            "remembered",
                            cleaned,
                            category="knowledge",
                            verified=0 if require_mem_confirm else 1,
                            source_text=user_text,
                            user_id=user_id,
                            project_id=effective_project_id,
                        )
                        memories_added.append(rec)
                        yield {"event": "memory_consolidated", "data": memories_added}
                        state_lbl = "Queued for approval in Memory Palace" if require_mem_confirm else "Saved to your persistent Memory Graph"
                        response_text = f"{state_lbl}: **`User Note → remembered → {cleaned}`**."
                    step3 = {"step": "Memory Consolidation", "detail": f"Processed {len(memories_added)} semantic triple(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "memory_query":
                    user_mems = [m for m in recalled["semantic"] if m["category"] != "system"]
                    if not user_mems:
                        all_graph = self.memory.get_knowledge_graph(user_id=user_id, project_id=effective_project_id)["memories"]
                        user_mems = [m for m in all_graph if m["category"] != "system" and m.get("verified", 1)][:10]

                    if user_mems:
                        q_low = user_text.lower()
                        direct_bits: List[str] = []
                        by_rel = {m["relation"].lower(): m["object"] for m in user_mems if m["subject"].lower() == "user"}
                        if ("name" in q_low or "who am i" in q_low) and "name" in by_rel:
                            direct_bits.append(f"your name is **{by_rel['name']}**")
                        if any(k in q_low for k in ("where do i live", "location", "city", "from")) and "location" in by_rel:
                            direct_bits.append(f"you live in **{by_rel['location']}**")
                        if any(k in q_low for k in ("job", "work", "what do i do", "role", "profession")) and "role" in by_rel:
                            direct_bits.append(f"you work as a **{by_rel['role']}**")
                        fav_m = re.search(r"favou?rite\s+([a-z]+)", q_low)
                        if fav_m and f"favorite_{fav_m.group(1)}" in by_rel:
                            direct_bits.append(f"your favorite {fav_m.group(1)} is **{by_rel[f'favorite_{fav_m.group(1)}']}**")

                        lines = [
                            f"- **{m['subject']}** `{m['relation'].replace('_', ' ')}` → **{m['object']}** *(confidence {m.get('confidence', 0.9)*100:.0f}%, recalled {m['access_count']}x)*"
                            for m in user_mems
                        ]
                        if direct_bits:
                            lead = (", ".join(direct_bits[:-1]) + " and " + direct_bits[-1]) if len(direct_bits) > 1 else direct_bits[0]
                            lead = lead[0].upper() + lead[1:] + "."
                            response_text = f"{lead}\n\n**Active User-Isolated Semantic Memory Nodes:**\n" + "\n".join(lines)
                        else:
                            response_text = f"Here is what I recall from your **Semantic Knowledge Graph**{f', **{user_name}**' if user_name else ''}:\n\n" + "\n".join(lines)
                    else:
                        response_text = (
                            "I haven't stored personal facts in your account's Memory Graph yet! Try telling me something like:\n"
                            "- *\"My name is Alex, I live in Austin, and I love building robotics in Python.\"*\n"
                            "- *\"Remember that Project Helios launches on Friday.\"*"
                        )
                    step3 = {"step": "Memory Graph Synthesis", "detail": f"Synthesized answer from {len(user_mems)} user memory record(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "identity_query":
                    stats = self.neural.get_stats()
                    mem_total = recalled["total_semantic_count"]
                    response_text = (
                        f"I am **Xyvora AI** — a multimodal, tool-using intelligence platform.\n\n"
                        f"### Live Platform Architecture\n"
                        f"- **Active Language Engine**: `{provider_used}` (`{model_used}`) — connect OpenAI, Anthropic, Gemini, Groq, OpenRouter, or Ollama in **Settings → AI Providers**, or run offline via my local Neuro-Symbolic engine.\n"
                        f"- **Intent Routing Classifier**: `{stats['architecture']}` (`{stats['parameter_count']:,}` weights across `{stats['intent_count']}` routing intents, `{stats['cv_accuracy']*100:.1f}%` cross-validated accuracy — scoped strictly to fast intent classification, never misrepresented as an LLM).\n"
                        f"- **User-Isolated Tri-Layer Memory**: `{mem_total}` active nodes in your personal **Semantic Knowledge Graph**, with confidence scoring, TTL expiration, and approval controls.\n"
                        f"- **Integrated Studios**: **Coding Studio IDE**, **Web Research Engine**, **Autonomous Multi-Step Agents**, **Multimodal Document & Image Vault**, **Extension Marketplace**, and **Hardened OS-Isolated Python Sandbox**."
                    )
                    step3 = {"step": "Platform Introspection", "detail": "Queried live provider, router & memory telemetry", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "capability_query":
                    response_text = (
                        "Here is everything you can build, research, and orchestrate inside **Xyvora AI**:\n\n"
                        "1. **Multi-Provider AI & Tool Calling** — Switch seamlessly between OpenAI, Anthropic, Gemini, Groq, OpenRouter, Ollama, and Local Neuro-Symbolic mode.\n"
                        "2. **Project Workspaces & Coding Studio IDE** — Manage multi-file codebases across Python, TypeScript/React, HTML/CSS, Java, C++, C#, and Luau with unified diff review, revision rollback, diagnostics, and isolated unit test runs.\n"
                        "3. **Web Research Studio** — Run Quick or Deep multi-source research with source credibility scoring, publication dates, citations, and Markdown/PDF export.\n"
                        "4. **Multimodal Knowledge Vault** — Upload PDFs, DOCX, XLSX, PPTX, CSV, JSON, code, and images with section-level RAG citations.\n"
                        "5. **Autonomous Agents & Extension Marketplace** — Plan multi-step tool workflows with human approval checkpoints, and install permission-scoped extensions.\n"
                        "6. **Smart Memory Palace & Intent Training Studio** — Inspect, approve, edit, or expire personal memories, and benchmark the 21-class intent classifier with precision/recall/F1 confusion matrices."
                    )
                    step3 = {"step": "Capability Matrix", "detail": "Enumerated platform workspaces & engines", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent in ("greeting", "farewell"):
                    if intent == "farewell":
                        response_text = f"Take care{f', **{user_name}**' if user_name else ''}! Your conversations, project files, and memory graph are safely persisted in your workspace."
                    else:
                        mem_note = ""
                        if memories_added:
                            mem_note = "\n\n*(Noted " + ", ".join(f"`{m['subject']} {m['relation']} {m['object']}`" for m in memories_added) + " in your Memory Graph.)*"
                        response_text = f"Hello{f' **{user_name}**' if user_name else ''}! Xyvora's cognitive workspace is online (`{model_used}`).{mem_note}\n\nWhat would you like to build, research, analyze, or calculate today?"
                    step3 = {"step": "Dialogue Synthesis", "detail": f"Personalized `{intent}` response", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "emotional_support":
                    if sentiment["score"] > 0.1:
                        response_text = (
                            f"That is awesome news{f', **{user_name}**' if user_name else ''}! Congratulations!\n\n"
                            "Would you like me to save this milestone in your **Memory Palace** or map out your next project roadmap?"
                        )
                    else:
                        response_text = (
                            f"I hear you{f', **{user_name}**' if user_name else ''} — feeling overloaded usually means working memory is holding too many open loops at once.\n\n"
                            "1. **Externalize the loops**: Share the top 3 items on your mind and I'll organize them on your Task Board.\n"
                            "2. **Pick one 15-minute micro-step**: Momentum beats perfection.\n"
                            "3. **Physiological reset**: Two slow double-inhales + long exhales measurably calm autonomic load."
                        )
                    step3 = {"step": "Affective Support Engine", "detail": f"Adapted response (valence {sentiment['score']:+.2f})", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                elif intent == "creative_writing":
                    response_text = self._compose_creative(user_text, user_name, creativity)
                    step3 = {"step": "Creative Synthesis", "detail": f"Generated creative artifact (creativity={creativity:.2f})", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                else:
                    effective_query = user_text
                    if intent == "followup_elaboration" and len(history) >= 2:
                        prev_user_msgs = [m["content"] for m in history[:-1] if m["role"] == "user"]
                        if prev_user_msgs:
                            effective_query = f"{prev_user_msgs[-1]} {user_text}"
                            rag_search = self.knowledge.search(effective_query, top_k=3, user_id=user_id, project_id=effective_project_id)

                    syllogism = self.text_intel.solve_syllogism(effective_query)
                    if syllogism:
                        proof_md = "\n".join(f"- {p}" for p in syllogism["proof"])
                        response_text = f"### Deductive Logic Verification\n\n{syllogism['conclusion']}\n\n**Formal Derivation:**\n{proof_md}"
                        step3 = {"step": "Symbolic Logic Solver", "detail": "Evaluated first-order set transitivity", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                        thought_trace.append(step3)
                        yield {"event": "thought", "data": step3}

                    elif rag_search["encyclopedia"] or rag_search["rag_chunks"]:
                        wants_simple = any(k in user_text.lower() for k in ("like i'm 12", "like im 12", "eli5", "eli12", "simply", "simple", "analogy", "beginner"))
                        wants_deep = any(k in user_text.lower() for k in ("deep", "technical", "math", "detail", "elaborate", "more")) or persona in ("axiom", "forge")
                        parts: List[str] = []
                        if rag_search["encyclopedia"]:
                            top_enc = rag_search["encyclopedia"][0]
                            citations.append(
                                {
                                    "source_type": "built_in_reference",
                                    "title": top_enc["title"],
                                    "section": "Built-in Reference Library",
                                    "snippet": top_enc["summary"][:200],
                                }
                            )
                            parts.append(f"### {top_enc['title']}")
                            if wants_simple:
                                parts.append(f"**Intuitive Analogy:**\n{top_enc['eli12']}\n\n**Core Idea:**\n{top_enc['summary']}")
                            elif wants_deep:
                                parts.append(f"{top_enc['summary']}\n\n**Technical Deep Dive:**\n{top_enc['deep_dive']}\n\n**Real-World Application:**\n{top_enc['example']}")
                            else:
                                parts.append(f"{top_enc['summary']}\n\n**Intuitive Analogy:**\n{top_enc['eli12']}\n\n**Technical Insight:**\n{top_enc['deep_dive']}")

                        if rag_search["rag_chunks"] and (not rag_search["encyclopedia"] or rag_search["rag_chunks"][0]["score"] >= 0.24):
                            top_chunks = rag_search["rag_chunks"][:2]
                            cite_lines = []
                            for c in top_chunks:
                                sec = c.get("section_label") or f"Chunk #{int(c.get('chunk_index', 0)) + 1}"
                                cite_lines.append(f"> *[{c['title']} — {sec} | similarity {c['score']:.2f}]*\n> {c['content']}")
                            parts.append("### Grounded Knowledge Vault Excerpts\n" + "\n\n".join(cite_lines))

                        response_text = "\n\n".join(parts)
                        src_label = rag_search["encyclopedia"][0]["title"] if rag_search["encyclopedia"] else rag_search["rag_chunks"][0]["title"]
                        step3 = {"step": "Grounded Knowledge Synthesis", "detail": f"Retrieved & cited `{src_label}`", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                        thought_trace.append(step3)
                        yield {"event": "thought", "data": step3}

                    elif recalled["semantic"] and any(m["category"] != "system" and m.get("similarity", 0) >= 0.35 for m in recalled["semantic"]):
                        hits = [m for m in recalled["semantic"] if m["category"] != "system"][:5]
                        bullets = "\n".join(f"- **{m['subject']}** `{m['relation'].replace('_', ' ')}` → **{m['object']}**" for m in hits)
                        response_text = f"Based on your **Semantic Knowledge Graph**:\n\n{bullets}"
                        step3 = {"step": "Semantic Graph Reasoning", "detail": f"Answered from {len(hits)} learned fact(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                        thought_trace.append(step3)
                        yield {"event": "thought", "data": step3}

                    else:
                        clean_topic = re.sub(r"^(?:what\s+is|how\s+does|can\s+you\s+explain|tell\s+me\s+about|help\s+me\s+decide\s+between|compare|evaluate)\s+", "", user_text.strip(), flags=re.IGNORECASE).strip("?.! ")
                        if " vs " in user_text.lower() or " or " in user_text.lower() or "decide" in user_text.lower() or "pros and cons" in user_text.lower():
                            response_text = (
                                f"### Structured Tradeoff Analysis: {clean_topic.title()}\n\n"
                                f"| Dimension | Option A (Speed / Simplicity First) | Option B (Scale / Modularity First) |\n"
                                f"| :--- | :--- | :--- |\n"
                                f"| **Time to Value** | Rapid iteration, minimal upfront ceremony | Higher setup cost, pays off at scale |\n"
                                f"| **Cognitive Overhead** | Low — ideal for focused velocity | Moderate — explicit service boundaries |\n"
                                f"| **Long-Term Flexibility** | May require refactoring as constraints grow | Built for modular extension |\n\n"
                                f"*Note: Running in offline Local Neuro-Symbolic mode (`xyvora_local`). Connect an external LLM provider in **Settings → AI Providers** or run **Web Research** for deep open-domain synthesis.*"
                            )
                        else:
                            mem_context = f", **{user_name}**" if user_name else ""
                            response_text = (
                                f"### Analytical Breakdown: {clean_topic[:70]}\n\n"
                                f"Exploring **{clean_topic[:60]}**{mem_context} from first principles:\n"
                                f"1. **Core Mechanism & Invariants** — Identify inputs, state transitions, and governing constraints.\n"
                                f"2. **Tradeoffs & Failure Modes** — Evaluate latency, complexity, security, and edge cases.\n"
                                f"3. **Empirical Verification** — Test via symbolic math, Python simulation, Knowledge Vault documents, or grounded Web Research.\n\n"
                                f"*Mode Notice: Generated locally via `xyvora-cortex-v2` (no external LLM API key configured). Connect OpenAI, Anthropic, Gemini, Groq, OpenRouter, or Ollama in **Settings → AI Providers** for full open-ended generative reasoning.*"
                            )
                        step3 = {"step": "Local Analytical Engine", "detail": "Synthesized structured framework (xyvora_local)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                        thought_trace.append(step3)
                        yield {"event": "thought", "data": step3}

            # Apply persona & verbosity modulation for local mode and stream smoothly
            response_text = self._compose_persona_wrap(response_text, persona, verbosity, intent, custom_persona)
            words_and_spaces = re.findall(r"\S+\s*|\n+", response_text)
            for i in range(0, len(words_and_spaces), 4):
                piece = "".join(words_and_spaces[i : i + 4])
                yield {"event": "token", "data": {"delta": piece}}
                await asyncio.sleep(0.008)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        ai_msg_id = self._persist_assistant_turn(
            session_id=session_id,
            user_id=user_id,
            content=response_text,
            intent=intent,
            confidence=confidence,
            provider_used=provider_used,
            model_used=model_used,
            thought_trace=thought_trace,
            tool_calls=tool_calls_record,
            citations=citations,
            artifacts=artifacts,
            memories_added=memories_added,
            suggestions=suggestions,
        )

        self.memory.store_episodic_memory(
            session_id=session_id,
            user_text=user_text,
            ai_summary=response_text[:260],
            topics=[intent],
            sentiment=sentiment["score"],
            user_id=user_id,
            project_id=effective_project_id,
        )

        # Auto-compress conversation summary if session grows long (> 16 messages)
        if len(history) >= 16 and len(history) % 8 == 0:
            self.memory.summarize_and_compress_session(session_id, user_id=user_id)

        est_tok = estimate_tokens(user_text) + estimate_tokens(response_text)
        record_usage_event(user_id, "/api/chat/stream", provider_used, model_used, "", est_tok, total_ms, "ok")

        yield {
            "event": "done",
            "data": {
                "message_id": ai_msg_id,
                "intent": intent,
                "confidence": confidence,
                "provider_used": provider_used,
                "model_used": model_used,
                "thought_trace": thought_trace,
                "tool_calls": tool_calls_record,
                "citations": citations,
                "artifacts": artifacts,
                "memories_added": memories_added,
                "suggestions": suggestions,
                "total_ms": total_ms,
            },
        }

    def _synthesize_attachment_analysis(self, attachments: List[Dict[str, Any]], user_question: str = "") -> str:
        blocks = ["### Multimodal Attachment Analysis"]
        for att in attachments:
            meta = json.loads(att.get("metadata_json") or "{}")
            if att["kind"] == "image":
                ocr_status = meta.get("ocr_engine", "none")
                txt = (att.get("extracted_text") or "").strip()
                blocks.append(
                    f"#### Image: `{att['filename']}` *(Uploaded Image)*\n"
                    f"- **Format & Dimensions**: `{att['mime_type']}` — `{att.get('width') or '?'} × {att.get('height') or '?'} px` (`{att['size_bytes']:,}` bytes)\n"
                    f"- **Local Text/Metadata Inspection (`{ocr_status}`)**: "
                    + (f"\n```\n{txt[:1200]}\n```" if txt else "No embedded text/OCR detected locally. Connect a Vision-capable LLM (`gpt-4o`, `claude-3-5-sonnet`, `gemini-2.5-flash`) in **AI Providers** for deep pixel-level visual understanding.")
                )
            else:
                txt = (att.get("extracted_text") or "").strip()
                preview = txt[:1400] + ("..." if len(txt) > 1400 else "")
                blocks.append(
                    f"#### Document: `{att['filename']}`\n"
                    f"- **Type**: `{att['mime_type']}` (`{att['size_bytes']:,}` bytes, `{meta.get('section_count', 1)}` extracted section(s))\n"
                    f"- **Extracted Content Preview**:\n```\n{preview}\n```"
                )
        if user_question:
            blocks.append(f"*Query Note:* Answered using local structural/text extraction for *\"{user_question[:100]}\"*. Connect a Vision/Cloud LLM provider for richer multimodal synthesis.")
        return "\n\n".join(blocks)

    def _synthesize_attachment_qa(self, question: str, attachments: List[Dict[str, Any]]) -> str:
        combined_paras: List[tuple[str, str]] = []
        for att in attachments:
            txt = (att.get("extracted_text") or "").strip()
            for idx, p in enumerate(re.split(r"\n{2,}", txt)):
                if p.strip():
                    combined_paras.append((f"{att['filename']} (Part {idx + 1})", p.strip()))
        if not combined_paras:
            return self._synthesize_attachment_analysis(attachments, user_question=question)

        corpus = [p[1] for p in combined_paras]
        scores = tfidf_cosine_scores(corpus, question, mode="word")
        ranked = sorted(range(len(corpus)), key=lambda i: float(scores[i]), reverse=True)[:3]
        excerpts = [f"> **[{combined_paras[i][0]}]**\n> {combined_paras[i][1][:600]}" for i in ranked]
        return (
            f"### Grounded Document Analysis\n\n"
            f"Based directly on your attached file(s), here are the most relevant passages addressing **\"{question[:110]}\"**:\n\n"
            + "\n\n".join(excerpts)
        )

    def _persist_assistant_turn(
        self,
        session_id: str,
        user_id: str,
        content: str,
        intent: str,
        confidence: float,
        provider_used: str,
        model_used: str,
        thought_trace: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        artifacts: List[Dict[str, Any]],
        memories_added: List[Dict[str, Any]],
        suggestions: List[str],
    ) -> str:
        ai_msg_id = str(uuid.uuid4())
        now_done = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO messages
                    (id, session_id, user_id, role, content, intent, confidence, provider_used, model_used, thought_trace, tool_calls, citations, artifacts, memories_added, suggestions, created_at)
                    VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ai_msg_id,
                        session_id,
                        user_id,
                        content,
                        intent,
                        confidence,
                        provider_used,
                        model_used,
                        json.dumps(thought_trace),
                        json.dumps(tool_calls),
                        json.dumps(citations),
                        json.dumps(artifacts),
                        json.dumps(memories_added),
                        json.dumps(suggestions),
                        now_done,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return ai_msg_id
