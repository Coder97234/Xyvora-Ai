import json
import random
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from database import get_conn, get_settings, _db_lock
from engine.nlp_math import tfidf_cosine_scores
from engine.neural_classifier import NeuralCortexClassifier
from engine.memory_system import MemorySystem
from engine.math_engine import SymbolicMathEngine
from engine.code_engine import CodeEngine
from engine.text_intel import TextIntelligenceEngine
from engine.knowledge_base import KnowledgeVaultEngine

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
        "name": "Socrates (Interactive Tutor)",
        "prefix": "**[Socratic Inquiry Mode]** ",
        "closing_style": "socratic",
    },
}


class XyvoraCortex:
    def __init__(self) -> None:
        self.neural = NeuralCortexClassifier()
        self.memory = MemorySystem()
        self.math = SymbolicMathEngine()
        self.code = CodeEngine()
        self.text_intel = TextIntelligenceEngine()
        self.knowledge = KnowledgeVaultEngine()

    def _get_recent_messages(self, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT role, content, intent FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
                return [dict(r) for r in reversed(rows)]
            finally:
                conn.close()

    def _match_custom_skill(self, text: str) -> Optional[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                skills = [dict(r) for r in conn.execute("SELECT * FROM custom_skills ORDER BY created_at DESC").fetchall()]
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

    def _handle_tasks(self, text: str) -> Dict[str, Any]:
        low = text.strip().lower()
        now = time.time()

        with _db_lock:
            conn = get_conn()
            try:
                # Complete task
                if any(k in low for k in ("mark ", "complete ", "finish ", "done")):
                    m = re.sub(
                        r"^(?:please\s+)?(?:mark|complete|finish|check\s+off)\s+(?:the\s+)?(?:task\s+)?",
                        "",
                        low,
                    )
                    m = re.sub(r"\s+(?:as\s+done|as\s+completed|task|done)$", "", m).strip()
                    rows = conn.execute("SELECT * FROM tasks WHERE status != 'done'").fetchall()
                    completed_titles = []
                    for r in rows:
                        if m and (m in r["title"].lower() or any(w in r["title"].lower() for w in m.split() if len(w) > 3)):
                            conn.execute("UPDATE tasks SET status = 'done', updated_at = ? WHERE id = ?", (now, r["id"]))
                            completed_titles.append(r["title"])
                    conn.commit()
                    all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 15").fetchall()]
                    return {
                        "action": "completed",
                        "completed": completed_titles,
                        "tasks": all_tasks,
                    }

                # List tasks
                if any(k in low for k in ("show my", "list my", "active tasks", "todo board", "what are my tasks", "reminders")):
                    all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 15").fetchall()]
                    return {"action": "list", "tasks": all_tasks}

                # Create new task or project roadmap
                if "project plan" in low or "step by step" in low:
                    topic = re.sub(r"^.*?(?:plan|roadmap)\s+(?:to|for)\s+", "", text.strip(), flags=re.IGNORECASE).strip("?.! ") or "Launch Project"
                    steps = [
                        (f"Define scope & core requirements for {topic}", "high"),
                        (f"Implement core architecture & data layer for {topic}", "high"),
                        (f"Verify end-to-end UX, edge cases & metrics for {topic}", "medium"),
                    ]
                    for title_s, prio in steps:
                        conn.execute(
                            "INSERT INTO tasks (id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, ?, ?, 'todo', 'Roadmap', ?, ?)",
                            (str(uuid.uuid4()), title_s, f"Auto-generated milestone for {topic}", prio, now, now),
                        )
                    conn.commit()
                    all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 15").fetchall()]
                    return {"action": "roadmap_created", "topic": topic, "tasks": all_tasks}

                # Standard single task creation
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
                        "INSERT INTO tasks (id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, '', ?, 'todo', ?, ?, ?)",
                        (str(uuid.uuid4()), cleaned_title[:160], priority, due_note[:40], now, now),
                    )
                    conn.commit()
                all_tasks = [dict(r) for r in conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 15").fetchall()]
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

        # Sci-fi / narrative story synthesis
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
        user_profile: Dict[str, Any],
        intent: str,
    ) -> str:
        out = body.strip()
        if verbosity == "concise" and len(out) > 650 and "```" not in out:
            paras = [p.strip() for p in out.split("\n\n") if p.strip()]
            out = "\n\n".join(paras[:3])
        elif verbosity == "comprehensive" and persona == "socrates" and not out.endswith("?"):
            out += "\n\n---\n*Socratic Reflection:* How do you think this principle connects to a system or decision you're currently working on?"
        elif persona == "socrates" and intent in ("knowledge_question", "logic_reasoning") and not out.endswith("?"):
            out += "\n\n*Thought Check:* Which part of this mechanism would break first if we removed its core constraint?"
        return out

    async def _try_external_llm(
        self,
        user_text: str,
        history: List[Dict[str, Any]],
        recalled: Dict[str, Any],
        rag_hits: List[Dict[str, Any]],
        settings: Dict[str, Any],
    ) -> Optional[str]:
        base_url = (settings.get("external_llm_base_url") or "").rstrip("/")
        api_key = (settings.get("external_llm_api_key") or "").strip()
        model = (settings.get("external_llm_model") or "gpt-4o-mini").strip()
        if not settings.get("external_llm_enabled") or not base_url or not api_key:
            return None

        mem_lines = [f"- {m['subject']} {m['relation']} {m['object']}" for m in recalled.get("semantic", [])[:10]]
        rag_lines = [f"[{c['title']}]: {c['content']}" for c in rag_hits[:3]]
        sys_prompt = (
            "You are Xyvora AI, an advanced interactive artificial intelligence with persistent Tri-Layer Memory and autonomous tools.\n"
            f"Active Persona: {settings.get('persona', 'nexus')}.\n"
            + ("Recalled User & Knowledge Graph Memories:\n" + "\n".join(mem_lines) + "\n" if mem_lines else "")
            + ("Relevant Knowledge Vault Excerpts:\n" + "\n".join(rag_lines) + "\n" if rag_lines else "")
        )
        msgs = [{"role": "system", "content": sys_prompt}]
        for h in history[-6:]:
            if h["role"] in ("user", "assistant"):
                msgs.append({"role": h["role"], "content": h["content"][:1500]})
        msgs.append({"role": "user", "content": user_text})

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": msgs, "temperature": float(settings.get("creativity", 0.65))},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception:
            return None
        return None

    async def process_message_stream(
        self,
        session_id: str,
        user_text: str,
        client_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        t_start = time.perf_counter()
        settings = get_settings()
        persona = settings.get("persona", "nexus")
        creativity = float(settings.get("creativity", 0.65))
        verbosity = settings.get("verbosity", "balanced")
        mem_threshold = float(settings.get("memory_threshold", 0.24))
        auto_learn = bool(settings.get("auto_learn_memory", True))

        # Hydrate any client-cached memories on serverless cold starts
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
                    )

        history = self._get_recent_messages(session_id, limit=8)

        # Save user message immediately in DB (ensuring session row exists on serverless instances)
        user_msg_id = str(uuid.uuid4())
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (id, title, persona, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, "Neural Session", persona, now, now),
                )
                conn.execute(
                    "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
                    (user_msg_id, session_id, user_text, now),
                )
                # Update session title if this is the first user turn
                msg_count = conn.execute("SELECT COUNT(*) AS c FROM messages WHERE session_id = ? AND role = 'user'", (session_id,)).fetchone()["c"]
                if msg_count == 1:
                    short_title = re.sub(r"\s+", " ", user_text.strip())[:42]
                    conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (short_title, now, session_id))
                else:
                    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
                conn.commit()
            finally:
                conn.close()

        thought_trace: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []

        # STEP 1: Neural Perception & Intent Classification
        t1 = time.perf_counter()
        pred = self.neural.predict(user_text)
        intent = pred["intent"]
        confidence = pred["confidence"]
        sentiment = self.text_intel.analyze_sentiment(user_text)

        step1 = {
            "step": "Neural Perception",
            "detail": f"Classified intent `{intent}` ({confidence*100:.1f}% confidence) | Affect: {sentiment['label']}",
            "ms": round((time.perf_counter() - t1) * 1000, 2),
            "distribution": pred["distribution"][:4],
        }
        thought_trace.append(step1)
        yield {"event": "thought", "data": step1}

        # STEP 2: Memory Extraction & Associative Recall
        t2 = time.perf_counter()
        memories_added = self.memory.extract_and_store_from_message(user_text, auto_learn=auto_learn)
        recalled = self.memory.recall(
            user_text,
            threshold=mem_threshold,
            include_all_user_profile=(intent == "memory_query"),
        )
        user_profile = recalled["user_profile"]
        user_name = user_profile.get("name", "")

        mem_detail_parts = []
        if memories_added:
            mem_detail_parts.append(f"Consolidated {len(memories_added)} fact(s)")
        if recalled["semantic"]:
            mem_detail_parts.append(f"Recalled {len(recalled['semantic'])} semantic node(s)")
        if not mem_detail_parts:
            mem_detail_parts.append("Scanned Knowledge Graph (0 above threshold)")

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

        # STEP 3: Cognitive Reasoning, Tool Execution & Response Synthesis
        t3 = time.perf_counter()
        response_text = ""
        suggestions: List[str] = []

        # Check custom taught skills first
        matched_skill = self._match_custom_skill(user_text)
        rag_search = self.knowledge.search(user_text, top_k=3)

        # Optional External Cloud LLM Bridge if user enabled it
        ext_reply = await self._try_external_llm(user_text, history, recalled, rag_search["rag_chunks"], settings)

        if ext_reply:
            response_text = ext_reply
            step3 = {
                "step": "External Neural Bridge",
                "detail": f"Generated via external model `{settings.get('external_llm_model')}` with Xyvora Memory augmentation",
                "ms": round((time.perf_counter() - t3) * 1000, 2),
            }
            thought_trace.append(step3)
            yield {"event": "thought", "data": step3}

        elif matched_skill:
            tmpl = matched_skill["response_template"]
            response_text = tmpl.replace("{name}", user_name or "friend")
            step3 = {
                "step": "Custom Taught Skill",
                "detail": f"Triggered user-taught skill `{matched_skill['trigger_phrase']}`",
                "ms": round((time.perf_counter() - t3) * 1000, 2),
            }
            thought_trace.append(step3)
            yield {"event": "thought", "data": step3}
            suggestions = ["Open Training Studio to add another skill", "What else do you remember about me?"]

        else:
            # Handle code fences directly if user pasted ```python ... ``` and asked to run
            code_fence_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", user_text, re.DOTALL | re.IGNORECASE)
            if code_fence_match and any(k in user_text.lower() for k in ("run", "execute", "output", "evaluate")):
                user_code = code_fence_match.group(1)
                exec_res = self.code.execute_python(user_code)
                artifacts.append(
                    {
                        "type": "code_execution",
                        "title": "Sandboxed Python Execution",
                        "code": user_code,
                        "stdout": exec_res["stdout"],
                        "stderr": exec_res["stderr"],
                        "ok": exec_res["ok"],
                        "duration_ms": exec_res["duration_ms"],
                    }
                )
                artifacts.extend(exec_res.get("artifacts", []))
                status_line = "completed cleanly" if exec_res["ok"] else "raised an error"
                response_text = (
                    f"I executed your Python snippet inside my AST-isolated sandbox (`{exec_res['duration_ms']} ms`), and it **{status_line}**.\n\n"
                    + (f"**Standard Output:**\n```\n{exec_res['stdout']}\n```" if exec_res["stdout"] else "")
                    + (f"\n\n**Diagnostic / Error:**\n```\n{exec_res['stderr']}\n```" if exec_res["stderr"] else "")
                )
                step3 = {"step": "Python Sandbox", "detail": f"Executed user Python block in {exec_res['duration_ms']} ms", "ms": exec_res["duration_ms"]}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}

            elif intent == "math_computation":
                math_res = self.math.solve_query(user_text)
                if math_res.get("ok"):
                    artifacts.extend(math_res.get("artifacts", []))
                    steps_md = "\n".join(f"- {s}" for s in math_res["steps"])
                    plot_note = "\n\nI've also generated an **interactive coordinate plot** of $f(x)$ and its derivative $f'(x)$ right below — hover over the curve to inspect exact values!" if math_res.get("artifacts") else ""
                    response_text = (
                        f"### Symbolic Math Resolution\n\n"
                        f"**Result:** `{math_res['result']}`\n\n"
                        f"**Step-by-Step Breakdown:**\n{steps_md}{plot_note}"
                    )
                    step3 = {"step": "SymPy Symbolic Engine", "detail": f"Solved `{math_res['operation']}` analytically", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                else:
                    response_text = (
                        "I detected a mathematical request, though I couldn't parse that exact notation unambiguously. "
                        "Try writing expressions like:\n"
                        "- `Differentiate and plot sin(x) * x`\n"
                        "- `Solve x^2 - 5x + 6 = 0`\n"
                        "- `Integrate x^2 from 0 to 4`\n"
                        "- `Eigenvalues of [[4, 2], [1, 3]]`"
                    )
                    step3 = {"step": "SymPy Symbolic Engine", "detail": "Requested syntax clarification", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "Plot cos(x) * exp(-x/5)",
                    "Solve x^3 - 6x^2 + 11x - 6 = 0",
                    "Eigenvalues of [[4, 2], [1, 3]]",
                ]

            elif intent in ("code_execution", "code_generation"):
                synth = self.code.synthesize_code(user_text)
                if synth["executable"] and (intent == "code_execution" or any(k in user_text.lower() for k in ("run", "simulate", "execute", "chart", "plot", "compute"))):
                    exec_res = self.code.execute_python(synth["code"])
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
                        f"I synthesized the `{synth['language']}` implementation and executed it live in my isolated sandbox (`{exec_res['duration_ms']} ms`):\n\n"
                        f"```{synth['language']}\n{synth['code'].strip()}\n```\n\n"
                        f"**Live Execution Output:**\n```\n{exec_res['stdout']}\n```"
                    )
                    step3 = {"step": "Code Synthesizer + Sandbox", "detail": f"Synthesized & executed `{synth['title']}` ({exec_res['duration_ms']} ms)", "ms": exec_res["duration_ms"]}
                else:
                    response_text = (
                        f"### {synth['title']}\n\n"
                        f"Here is a clean, production-ready `{synth['language']}` implementation:\n\n"
                        f"```{synth['language']}\n{synth['code'].strip()}\n```"
                    )
                    if synth["language"] == "python":
                        exec_res = self.code.execute_python(synth["code"])
                        if exec_res["ok"] and exec_res["stdout"]:
                            response_text += f"\n\n**Verified Sandbox Output (`{exec_res['duration_ms']} ms`):**\n```\n{exec_res['stdout']}\n```"
                            artifacts.extend(exec_res.get("artifacts", []))
                    step3 = {"step": "Code Synthesizer", "detail": f"Generated `{synth['title']}` in {synth['language']}", "ms": round((time.perf_counter() - t3) * 1000, 2)}

                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "Simulate Monte Carlo estimation of Pi in Python",
                    "Calculate compound interest over 20 years in Python and chart it",
                    "Write an O(1) LRU cache class in Python",
                ]

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
                    step3 = {"step": "Data Chart Engine", "detail": f"Extracted {st['count']} series points & computed summary stats", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                else:
                    response_text = (
                        "I'm ready to chart your data! Pass label-number pairs in any natural format, for example:\n"
                        "- `Chart Jan 120, Feb 165, Mar 210, Apr 195, May 290`\n"
                        "- `Create a chart comparing Python 48, TypeScript 36, Rust 24, Go 19`"
                    )
                    step3 = {"step": "Data Chart Engine", "detail": "Prompted for label-value series", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = ["Chart Jan 120, Feb 165, Mar 210, Apr 195, May 290"]

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
                t_res = self._handle_tasks(user_text)
                artifacts.append({"type": "task_list", "title": "Active Agent Task Board", "tasks": t_res["tasks"][:8]})
                if t_res["action"] == "created":
                    response_text = (
                        f"Added **`{t_res['created_title']}`** (`{t_res['priority']}` priority) to your persistent **Agent Task Board**!"
                    )
                elif t_res["action"] == "roadmap_created":
                    response_text = (
                        f"I generated a **3-milestone execution roadmap** for **{t_res['topic']}** and added each milestone to your Task Board below."
                    )
                elif t_res["action"] == "completed":
                    if t_res["completed"]:
                        response_text = f"Marked **{', '.join(t_res['completed'])}** as `done`!"
                    else:
                        response_text = "Here is your current Task Board — click any checkbox below or in the **AI Lab** tab to toggle completion!"
                else:
                    open_cnt = sum(1 for x in t_res["tasks"] if x["status"] != "done")
                    response_text = f"You currently have **{open_cnt} active task(s)** on your persistent board:"
                step3 = {"step": "Autonomous Task Planner", "detail": f"Executed task action `{t_res['action']}`", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = ["Show my active tasks", "Add a task to benchmark neural accuracy high priority"]

            elif intent == "memory_forget":
                removed = self.memory.forget_matching(user_text)
                if removed:
                    lines = "\n".join(f"- ~~`{r['subject']} → {r['relation']} → {r['object']}`~~" for r in removed)
                    response_text = f"Done — I have permanently erased **{len(removed)}** memory node(s) from my Semantic Knowledge Graph:\n\n{lines}"
                else:
                    response_text = (
                        "I didn't find a matching personal memory node for that exact phrase. "
                        "You can also view or delete any individual node visually inside the **Memory Palace** tab!"
                    )
                step3 = {"step": "Memory Pruning", "detail": f"Removed {len(removed)} semantic triple(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = ["What do you remember about me now?"]

            elif intent == "memory_store":
                if memories_added:
                    formatted = []
                    for m in memories_added:
                        if m.get("status") == "updated":
                            formatted.append(
                                f"- **Updated** `{m['subject']} → {m['relation']}`: ~~`{m['previous_object']}`~~ → **`{m['object']}`**"
                            )
                        else:
                            formatted.append(f"- **Stored** `{m['subject']} → {m['relation']} → {m['object']}` *(confidence {m['confidence']*100:.0f}%)*")
                    greet = f"Got it, **{user_name}**! " if user_name else "Got it! "
                    response_text = (
                        f"{greet}I've consolidated that into my persistent **Semantic Knowledge Graph**:\n\n"
                        + "\n".join(formatted)
                        + "\n\nYou can ask me about this in any future session, or inspect the live synaptic connection inside the **Memory Palace** tab."
                    )
                else:
                    # Fallback store raw note if explicit remember didn't match structured slot
                    cleaned = re.sub(r"^(?:please\s+)?(?:remember\s+(?:that\s+)?|note\s+that\s+)", "", user_text.strip(), flags=re.IGNORECASE).strip()
                    rec = self.memory.upsert_semantic_memory("User Note", "remembered", cleaned, category="knowledge", source_text=user_text)
                    memories_added.append(rec)
                    yield {"event": "memory_consolidated", "data": memories_added}
                    response_text = f"Saved to my persistent Memory Graph: **`User Note → remembered → {cleaned}`**."
                step3 = {"step": "Memory Consolidation", "detail": f"Persisted {len(memories_added)} semantic triple(s) to SQLite", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "What do you know about me so far?",
                    "Remember that Project Orion launches on December 12",
                ]

            elif intent == "memory_query":
                user_mems = [m for m in recalled["semantic"] if m["category"] != "system"]
                if not user_mems:
                    all_graph = self.memory.get_knowledge_graph()["memories"]
                    user_mems = [m for m in all_graph if m["category"] != "system"][:10]

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
                        attr_lbl = fav_m.group(1)
                        direct_bits.append(f"your favorite {attr_lbl} is **{by_rel[f'favorite_{attr_lbl}']}**")

                    lines = [
                        f"- **{m['subject']}** `{m['relation'].replace('_', ' ')}` → **{m['object']}** *(recalled {m['access_count']}x)*"
                        for m in user_mems
                    ]
                    if direct_bits:
                        lead = (", ".join(direct_bits[:-1]) + " and " + direct_bits[-1]) if len(direct_bits) > 1 else direct_bits[0]
                        lead = lead[0].upper() + lead[1:] + "."
                        response_text = f"{lead}\n\n**Active Semantic Knowledge Graph Nodes:**\n" + "\n".join(lines)
                    else:
                        intro = f"Here is what I recall from my **Semantic Knowledge Graph**{f', **{user_name}**' if user_name else ''}:\n\n"
                        response_text = intro + "\n".join(lines)

                    if recalled["episodic"]:
                        ep = recalled["episodic"][0]
                        response_text += f"\n\n**Related Episodic Trace:** *\"{ep['user_text'][:120]}\"*"
                else:
                    response_text = (
                        "I haven't stored personal facts about you yet! Try telling me something like:\n"
                        "- *\"My name is Alex, I live in Austin, and I love building robotics in Python.\"*\n"
                        "- *\"Remember that Project Helios launches on Friday.\"*"
                    )
                step3 = {"step": "Memory Graph Synthesis", "detail": f"Synthesized answer from {len(user_mems)} memory record(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "My favorite color is cyan and I love sci-fi",
                    "Forget where I live",
                ]

            elif intent == "identity_query":
                stats = self.neural.get_stats()
                mem_total = recalled["total_semantic_count"]
                response_text = (
                    f"I am **Xyvora AI** — a hybrid **neuro-symbolic artificial intelligence** running right inside this workspace.\n\n"
                    f"### Live Internal Telemetry\n"
                    f"- **Neural Perception Layer**: `{stats['architecture']}` (`{stats['parameter_count']:,}` trainable weights across `{stats['intent_count']}` cognitive intents, `{stats['training_accuracy']*100:.1f}%` accuracy)\n"
                    f"- **Tri-Layer Memory System**: `{mem_total}` active nodes in my **Semantic Knowledge Graph**, plus vector-indexed Episodic conversation traces and Working Memory\n"
                    f"- **Autonomous Engines**: Exact Symbolic Mathematics (`SymPy`), AST-Isolated Python Execution Sandbox, TextRank Graph Summarizer, and Hybrid BM25/TF-IDF RAG Document Vault\n"
                    f"- **Active Persona**: `{PERSONA_PROFILES.get(persona, PERSONA_PROFILES['nexus'])['name']}`"
                )
                step3 = {"step": "Self-Introspection", "detail": "Queried live neural & memory telemetry", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "What can you do?",
                    "Differentiate and plot sin(x) * cos(x/2)",
                    "Simulate 1000 dice rolls in Python",
                ]

            elif intent == "capability_query":
                response_text = (
                    "Here is everything you can interact with across **Xyvora AI**:\n\n"
                    "1. **Persistent Associative Memory** — Tell me your name, work, preferences, or project facts; I automatically build a live **Knowledge Graph** you can explore and edit in **Memory Palace**.\n"
                    "2. **Symbolic Calculus, Algebra & Interactive Function Plots** — Ask *\"Differentiate and plot sin(x) * x\"* or *\"Solve x^2 - 5x + 6 = 0\"*.\n"
                    "3. **Live Python Code Synthesis & Execution** — Ask *\"Simulate 1000 dice rolls in Python and chart it\"* or run custom code in the **AI Lab**.\n"
                    "4. **Local RAG Knowledge Vault** — Upload or paste documents in **AI Lab → Knowledge Vault**, then ask questions grounded in your files.\n"
                    "5. **Live Neural Training Studio** — Probe my neural network's probability bars in real time, teach me new custom skills, and click **Retrain Neural Cortex** to watch Adam backpropagation run live!\n"
                    "6. **Voice & Brain Modulation** — Use the microphone button, toggle Neural Voice TTS, or shift my **Persona**, **Creativity**, and **Memory Sensitivity** in the Brain Controls."
                )
                step3 = {"step": "Capability Matrix", "detail": "Enumerated 6 interactive subsystems", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "My name is Sam and I'm building an AI startup",
                    "Plot sin(x) + cos(2*x)",
                    "Explain how transformer self-attention works",
                ]

            elif intent in ("greeting", "farewell"):
                if intent == "farewell":
                    response_text = (
                        f"Take care{f', **{user_name}**' if user_name else ''}! Every memory node and task we worked on is safely persisted in my Knowledge Graph whenever you return."
                    )
                else:
                    mem_note = ""
                    if memories_added:
                        mem_note = "\n\n*(I also noted " + ", ".join(f"`{m['subject']} {m['relation']} {m['object']}`" for m in memories_added) + " in my Memory Graph!)*"
                    elif user_profile.get("interests"):
                        top_int = user_profile["interests"][0]
                        mem_note = f" I still remember your profile (`{top_int}`) — want to dive into something related today?"
                    response_text = (
                        f"Hello{f' **{user_name}**' if user_name else ''}! My neural cortex and memory graph are online and ready.{mem_note}\n\n"
                        f"What would you like to explore, calculate, build, or teach me today?"
                    )
                step3 = {"step": "Dialogue Synthesis", "detail": f"Personalized `{intent}` response", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "What do you know about me so far?",
                    "Differentiate and plot sin(x) * x",
                    "Simulate Monte Carlo Pi in Python",
                ]

            elif intent == "emotional_support":
                is_positive = sentiment["score"] > 0.1
                if is_positive:
                    response_text = (
                        f"That is awesome news{f', **{user_name}**' if user_name else ''}! Congratulations! 🎉\n\n"
                        "Moments like that deserve to be savored — take a minute to appreciate the effort that compounded to get you here. "
                        "Would you like me to save this milestone into your persistent **Memory Palace** or help map out your next exciting goal?"
                    )
                else:
                    response_text = (
                        f"I hear you{f', **{user_name}**' if user_name else ''} — feeling overloaded is usually a sign that your working memory is juggling too many open loops at once, not a reflection of your capability.\n\n"
                        "Here is a quick **3-step cognitive reset** that genuinely helps:\n"
                        "1. **Externalize the loops**: Dump the top 3 things weighing on you right here — I can turn them into bite-sized items on your Task Board so your brain stops rehearsing them.\n"
                        "2. **Shrink the horizon**: Pick **one 15-minute micro-step** with zero pressure for perfection.\n"
                        "3. **Physiological reset**: Two slow physiological sighs (double inhale through the nose, long exhale) measurably lower autonomic stress within 30 seconds.\n\n"
                        "Want to talk through what's on your plate right now, or break it into small steps together?"
                    )
                step3 = {"step": "Affective Empathy Engine", "detail": f"Adapted supportive response (valence {sentiment['score']:+.2f})", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = ["Help me create a step by step plan", "Tell me about Stoicism philosophy"]

            elif intent == "creative_writing":
                response_text = self._compose_creative(user_text, user_name, creativity)
                step3 = {"step": "Creative Synthesis", "detail": f"Generated creative artifact (divergence={creativity:.2f})", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                thought_trace.append(step3)
                yield {"event": "thought", "data": step3}
                suggestions = [
                    "Compose a poem about neural networks and starlight",
                    "Brainstorm startup ideas in robotics",
                    "Tell me a clever programming joke",
                ]

            else:
                # Handles: knowledge_question, logic_reasoning, followup_elaboration, general_conversation
                effective_query = user_text
                if intent == "followup_elaboration" and len(history) >= 2:
                    prev_user_msgs = [m["content"] for m in history[:-1] if m["role"] == "user"]
                    if prev_user_msgs:
                        effective_query = f"{prev_user_msgs[-1]} {user_text}"
                        rag_search = self.knowledge.search(effective_query, top_k=3)

                # 1. Check deductive syllogism solver first
                syllogism = self.text_intel.solve_syllogism(effective_query)
                if syllogism:
                    proof_md = "\n".join(f"- {p}" for p in syllogism["proof"])
                    response_text = f"### Deductive Logic Verification\n\n{syllogism['conclusion']}\n\n**Formal Derivation:**\n{proof_md}"
                    step3 = {"step": "Symbolic Logic Solver", "detail": "Evaluated first-order set transitivity", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                # 2. Check Encyclopedia & Knowledge Vault RAG matches
                elif rag_search["encyclopedia"] or rag_search["rag_chunks"]:
                    wants_simple = any(k in user_text.lower() for k in ("like i'm 12", "like im 12", "eli5", "eli12", "simply", "simple", "analogy", "beginner"))
                    wants_deep = any(k in user_text.lower() for k in ("deep", "technical", "math", "detail", "elaborate", "more")) or persona in ("axiom", "forge")

                    parts: List[str] = []
                    if rag_search["encyclopedia"]:
                        top_enc = rag_search["encyclopedia"][0]
                        parts.append(f"### {top_enc['title']}")
                        if wants_simple:
                            parts.append(f"**Intuitive Analogy:**\n{top_enc['eli12']}\n\n**Core Idea:**\n{top_enc['summary']}")
                        elif wants_deep:
                            parts.append(f"{top_enc['summary']}\n\n**Technical Deep Dive:**\n{top_enc['deep_dive']}\n\n**Real-World Application:**\n{top_enc['example']}")
                        else:
                            parts.append(f"{top_enc['summary']}\n\n**Intuitive Analogy:**\n{top_enc['eli12']}\n\n**Technical Insight:**\n{top_enc['deep_dive']}")

                    if rag_search["rag_chunks"] and (not rag_search["encyclopedia"] or rag_search["rag_chunks"][0]["score"] >= 0.24):
                        top_chunks = rag_search["rag_chunks"][:2]
                        cite_lines = [f"> *[{c['title']} — Chunk #{c['chunk_index']+1} | similarity {c['score']:.2f}]*\n> {c['content']}" for c in top_chunks]
                        parts.append("### Grounded Knowledge Vault Citations\n" + "\n\n".join(cite_lines))

                    # Weave subtle personal connection if relevant memory exists
                    if memories_added:
                        parts.append("*(Also saved to Memory Graph: " + ", ".join(f"`{m['subject']} → {m['relation']} → {m['object']}`" for m in memories_added) + ")*")

                    response_text = "\n\n".join(parts)
                    src_label = rag_search["encyclopedia"][0]["title"] if rag_search["encyclopedia"] else rag_search["rag_chunks"][0]["title"]
                    step3 = {"step": "Semantic Knowledge Synthesis", "detail": f"Retrieved & synthesized `{src_label}`", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}
                    suggestions = [
                        "Explain that more simply",
                        "Go deeper into the technical details",
                        "Compare SQL and NoSQL databases",
                    ]

                # 3. Check recalled semantic knowledge if user asked about a taught entity (e.g. "When does Project Orion launch?")
                elif recalled["semantic"] and any(m["category"] != "system" and m.get("similarity", 0) >= 0.35 for m in recalled["semantic"]):
                    hits = [m for m in recalled["semantic"] if m["category"] != "system"][:5]
                    bullets = "\n".join(f"- **{m['subject']}** `{m['relation'].replace('_', ' ')}` → **{m['object']}**" for m in hits)
                    response_text = f"Based on what I have stored in my **Semantic Knowledge Graph**:\n\n{bullets}"
                    step3 = {"step": "Semantic Graph Reasoning", "detail": f"Answered directly from {len(hits)} learned fact(s)", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}

                # 4. Structured analytical / conversational synthesis for open-ended decisions or general topics
                else:
                    clean_topic = re.sub(r"^(?:what\s+is|how\s+does|can\s+you\s+explain|tell\s+me\s+about|help\s+me\s+decide\s+between|compare|evaluate)\s+", "", user_text.strip(), flags=re.IGNORECASE).strip("?.! ")
                    if " vs " in user_text.lower() or " or " in user_text.lower() or "decide" in user_text.lower() or "pros and cons" in user_text.lower():
                        response_text = (
                            f"### Structured Tradeoff Analysis: {clean_topic.title()}\n\n"
                            f"When evaluating **{clean_topic}**, the cleanest way to decide is across four orthogonal axes:\n\n"
                            f"| Dimension | Option A (Speed / Simplicity First) | Option B (Scale / Flexibility First) |\n"
                            f"| :--- | :--- | :--- |\n"
                            f"| **Time to Value** | Rapid iteration, minimal upfront ceremony | Higher setup cost, pays off at scale |\n"
                            f"| **Cognitive Overhead** | Low — easy for one person or small team | Moderate/High — clearer separation of concerns |\n"
                            f"| **Long-Term Flexibility** | May require refactoring as constraints grow | Built for modular extension |\n"
                            f"| **Failure Blast Radius** | Shared runtime / coupled state | Isolated boundaries |\n\n"
                            f"**Pragmatic Recommendation:** Optimize for **iteration velocity and reversible decisions** early on, while keeping clean interfaces so you can evolve without a rewrite."
                        )
                    else:
                        mem_context = f", **{user_name}**" if user_name else ""
                        added_ack = ""
                        if memories_added:
                            added_ack = (
                                "I've noted "
                                + ", ".join(f"**`{m['subject']} → {m['relation']} → {m['object']}`**" for m in memories_added)
                                + " in my Memory Graph!\n\n"
                            )
                        response_text = (
                            f"{added_ack}That's an interesting prompt{mem_context} around **{clean_topic[:70]}**.\n\n"
                            f"From a first-principles perspective, breaking **{clean_topic[:55]}** down works best across three layers:\n"
                            f"1. **Foundational Mechanism** — Identify the core invariants, inputs, and constraints driving the behavior.\n"
                            f"2. **Feedback Loops & Tradeoffs** — Observe how small changes propagate (linear vs. compounding effects).\n"
                            f"3. **Actionable Experiment** — Test a concrete hypothesis, mathematical model, or simulation.\n\n"
                            f"*Tip:* You can also **teach me domain documents** about `{clean_topic[:35]}` in the **Knowledge Vault**, or train custom responses in the **Training Studio**!"
                        )
                    step3 = {"step": "Analytical Discourse Engine", "detail": "Synthesized first-principles framework", "ms": round((time.perf_counter() - t3) * 1000, 2)}
                    thought_trace.append(step3)
                    yield {"event": "thought", "data": step3}
                    suggestions = [
                        "Explain quantum entanglement like I'm 12",
                        "Differentiate and plot sin(x) * x",
                        "Simulate 1000 dice rolls in Python",
                    ]

        # Apply persona & verbosity modulation
        final_text = self._compose_persona_wrap(response_text, persona, verbosity, user_profile, intent)

        # Stream chunks smoothly so the UI shows live token generation + Neural Core speaking animation!
        words_and_spaces = re.findall(r"\S+\s*|\n+", final_text)
        chunk_size = 3
        for i in range(0, len(words_and_spaces), chunk_size):
            piece = "".join(words_and_spaces[i : i + chunk_size])
            yield {"event": "token", "data": {"delta": piece}}
            # Tiny 12ms pacing for smooth visual streaming without feeling slow
            await _async_tiny_sleep(0.012)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        # Persist assistant message & episodic memory
        ai_msg_id = str(uuid.uuid4())
        now_done = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO messages
                    (id, session_id, role, content, intent, confidence, thought_trace, artifacts, memories_added, suggestions, created_at)
                    VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ai_msg_id,
                        session_id,
                        final_text,
                        intent,
                        confidence,
                        json.dumps(thought_trace),
                        json.dumps(artifacts),
                        json.dumps(memories_added),
                        json.dumps(suggestions),
                        now_done,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        self.memory.store_episodic_memory(
            session_id=session_id,
            user_text=user_text,
            ai_summary=final_text[:260],
            topics=[intent],
            sentiment=sentiment["score"],
        )

        yield {
            "event": "done",
            "data": {
                "message_id": ai_msg_id,
                "intent": intent,
                "confidence": confidence,
                "thought_trace": thought_trace,
                "artifacts": artifacts,
                "memories_added": memories_added,
                "suggestions": suggestions,
                "total_ms": total_ms,
            },
        }


async def _async_tiny_sleep(sec: float) -> None:
    import asyncio
    await asyncio.sleep(sec)
