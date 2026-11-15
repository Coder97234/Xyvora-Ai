import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from database import DEFAULT_USER_ID, get_conn, _db_lock
from security import check_safety_policy, redact_sensitive


CONSEQUENTIAL_KEYWORDS = (
    "delete",
    "overwrite",
    "publish",
    "deploy",
    "purchase",
    "buy",
    "erase",
    "drop",
    "remove",
    "account",
    "send email",
)


def _step_requires_approval(tool_name: str, tool_args: Dict[str, Any], goal: str) -> bool:
    low_goal = goal.lower()
    if tool_name == "file_create":
        return True
    if tool_name == "memory_update" and str(tool_args.get("operation", "")).lower() == "forget":
        return True
    if tool_name == "task_manager" and str(tool_args.get("action", "")).lower() == "delete":
        return True
    if any(k in low_goal for k in CONSEQUENTIAL_KEYWORDS) and tool_name in ("web_research", "python_sandbox"):
        return True
    return False


class AgentOrchestrator:
    """
    Autonomous Multi-Step Agent Architecture:
      - Decomposes complex goals into ordered tool steps
      - Enforces permission & human-in-the-loop approval checkpoints
      - Recovers from ordinary failures with bounded retries (max_retries=2)
      - Supports live progress tracking, cancellation, and full audit history
    """

    MAX_STEPS = 8
    MAX_RETRIES = 2

    def __init__(self, tool_registry: Any, provider_manager: Any) -> None:
        self.tools = tool_registry
        self.providers = provider_manager
        self._cancelled_runs: set[str] = set()

    async def build_plan(
        self,
        goal: str,
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Decomposes a user goal into 2–6 structured tool steps."""
        prov = self.providers.resolve_active_provider(user_id)
        if prov.get("available") and prov.get("provider_id") != "xyvora_local":
            tool_names = [t["name"] for t in self.tools.list_tools()]
            sys_prompt = (
                "You are Xyvora's Autonomous Task Planner. Break the user's goal into 2 to 5 concrete sequential steps.\n"
                f"Available tools: {', '.join(tool_names)}.\n"
                "Return ONLY a valid JSON array of objects with keys: "
                '{"title": "...", "tool_name": "...", "tool_args": {...}}.'
            )
            chunks: List[str] = []
            async for evt in self.providers.stream_chat_completion(
                prov,
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": goal}],
                temperature=0.2,
                timeout_sec=20.0,
            ):
                if evt["type"] == "token":
                    chunks.append(evt["delta"])
            raw = "".join(chunks).strip()
            m_arr = re.search(r"\[[\s\S]*\]", raw)
            if m_arr:
                try:
                    parsed = json.loads(m_arr.group(0))
                    if isinstance(parsed, list) and parsed:
                        steps_out = []
                        for item in parsed[: self.MAX_STEPS]:
                            t_name = str(item.get("tool_name") or "")
                            if not self.tools.get_tool(t_name):
                                continue
                            t_args = item.get("tool_args") if isinstance(item.get("tool_args"), dict) else {}
                            steps_out.append(
                                {
                                    "title": str(item.get("title") or f"Execute {t_name}")[:140],
                                    "tool_name": t_name,
                                    "tool_args": t_args,
                                    "requires_approval": _step_requires_approval(t_name, t_args, goal),
                                }
                            )
                        if steps_out:
                            return steps_out
                except Exception:
                    pass

        # Deterministic Multi-Step Decomposition Engine (100% reliable offline & online)
        low = goal.lower()
        steps: List[Dict[str, Any]] = []

        # 1. Context & Memory check if personal/project context may apply
        steps.append(
            {
                "title": "Retrieve relevant user & project context from Memory Palace",
                "tool_name": "memory_retrieve",
                "tool_args": {"query": goal[:160]},
                "requires_approval": False,
            }
        )

        # 2. Knowledge Vault document search if research / spec / architecture / doc mentioned
        if any(k in low for k in ("document", "vault", "spec", "architecture", "analyze", "report", "compare", "research", "study")):
            steps.append(
                {
                    "title": "Search Knowledge Vault documents for grounded evidence",
                    "tool_name": "document_search",
                    "tool_args": {"query": goal[:160]},
                    "requires_approval": False,
                }
            )

        # 3. Web research if web / search / online / github / url / news mentioned
        if any(k in low for k in ("web", "internet", "online", "github", "http://", "https://", "latest", "research", "sources")):
            steps.append(
                {
                    "title": "Search external web & extract verified source citations",
                    "tool_name": "web_research",
                    "tool_args": {"query": goal[:180], "mode": "deep" if "deep" in low else "quick"},
                    "requires_approval": False,
                }
            )

        # 4. Math / Calculus step if equations or expressions present
        if any(k in low for k in ("solve", "differentiate", "derivative", "integral", "integrate", "eigenvalue", "equation", "sin(", "cos(")):
            expr_part = goal
            if " and " in goal:
                for clause in goal.split(" and "):
                    if any(w in clause.lower() for w in ("solve", "differentiate", "derivative", "integral", "sin", "cos", "=")):
                        expr_part = clause.strip()
                        break
            steps.append(
                {
                    "title": "Compute exact symbolic solution via SymPy engine",
                    "tool_name": "calculator_math",
                    "tool_args": {"expression": expr_part},
                    "requires_approval": False,
                }
            )

        # 5. Python simulation / computation step
        if any(k in low for k in ("python", "simulate", "simulation", "monte carlo", "dice", "prime", "fibonacci", "benchmark", "compute", "script")):
            synth = self.tools.code.synthesize_code(goal)
            steps.append(
                {
                    "title": f"Execute sandboxed Python verification: {synth['title']}",
                    "tool_name": "python_sandbox",
                    "tool_args": {"code": synth["code"]},
                    "requires_approval": False,
                }
            )

        # 6. Quantitative data analysis step
        if any(k in low for k in ("chart", "dataset", "metrics", "csv", "statistics", "trend")) and re.search(r"\d", goal):
            steps.append(
                {
                    "title": "Profile dataset statistics and generate chart artifact",
                    "tool_name": "data_analysis",
                    "tool_args": {"data_input": goal},
                    "requires_approval": False,
                }
            )

        # 7. File creation checkpoint if user asks to save/write/create a file!
        file_match = re.search(r"(?:create|write|save)\s+(?:a\s+)?file\s+(?:named|called)?\s*([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)", goal, re.IGNORECASE)
        if file_match:
            fpath = file_match.group(1)
            synth = self.tools.code.synthesize_code(goal)
            steps.append(
                {
                    "title": f"Write generated module to project workspace (`{fpath}`)",
                    "tool_name": "file_create",
                    "tool_args": {
                        "path": fpath,
                        "content": synth["code"],
                        "summary": f"Agent created {fpath}",
                    },
                    "requires_approval": True,  # Requires explicit user confirmation!
                }
            )

        # 8. Task tracking / milestone creation if requested
        if any(k in low for k in ("task", "todo", "roadmap", "milestone", "plan", "remind")):
            clean_t = re.sub(r"^(?:create|build|make)\s+(?:a\s+)?", "", goal.strip(), flags=re.IGNORECASE)[:100]
            steps.append(
                {
                    "title": "Record actionable follow-up item on persistent Task Board",
                    "tool_name": "task_manager",
                    "tool_args": {"action": "create", "title": f"Milestone: {clean_t}", "priority": "high"},
                    "requires_approval": False,
                }
            )

        # Ensure at least 2 meaningful steps even for general analytical goals
        if len(steps) == 1:
            steps.append(
                {
                    "title": "Search Knowledge Vault & reference encyclopedia for domain grounding",
                    "tool_name": "document_search",
                    "tool_args": {"query": goal[:160]},
                    "requires_approval": False,
                }
            )

        return steps[: self.MAX_STEPS]

    async def start_run(
        self,
        goal: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        auto_execute: bool = True,
    ) -> Dict[str, Any]:
        safety = check_safety_policy(goal)
        if safety:
            raise ValueError(safety["message"])

        plan = await self.build_plan(goal, user_id=user_id, project_id=project_id)
        run_id = str(uuid.uuid4())
        now = time.time()

        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO agent_runs
                    (id, user_id, session_id, project_id, goal, status, plan_json, current_step, max_steps, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'running', ?, 0, ?, ?, ?)
                    """,
                    (run_id, user_id, session_id, project_id, goal.strip(), json.dumps(plan), self.MAX_STEPS, now, now),
                )
                for idx, st in enumerate(plan):
                    conn.execute(
                        """
                        INSERT INTO agent_steps
                        (id, run_id, step_index, title, tool_name, tool_args_json, status, requires_approval, retries, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            run_id,
                            idx,
                            st["title"],
                            st["tool_name"],
                            json.dumps(st["tool_args"]),
                            1 if st.get("requires_approval") else 0,
                            now,
                            now,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

        if auto_execute:
            return await self.execute_until_Pause_or_done(run_id, user_id=user_id)
        return self.get_run(run_id, user_id=user_id)  # type: ignore

    def _is_cancelled(self, run_id: str) -> bool:
        if run_id in self._cancelled_runs:
            return True
        with _db_lock:
            conn = get_conn()
            try:
                row = conn.execute("SELECT cancelled FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
                return bool(row and row["cancelled"])
            finally:
                conn.close()

    def cancel_run(self, run_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        self._cancelled_runs.add(run_id)
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                cur = conn.execute(
                    """
                    UPDATE agent_runs
                    SET cancelled = 1, status = 'cancelled', pending_approval_json = NULL, updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (now, run_id, user_id),
                )
                if cur.rowcount == 0:
                    return None
                conn.execute(
                    "UPDATE agent_steps SET status = 'cancelled', updated_at = ? WHERE run_id = ? AND status IN ('pending', 'running', 'awaiting_approval')",
                    (now, run_id),
                )
                conn.commit()
            finally:
                conn.close()
        return self.get_run(run_id, user_id=user_id)

    async def resolve_approval(
        self, run_id: str, approved: bool, user_id: str = DEFAULT_USER_ID
    ) -> Optional[Dict[str, Any]]:
        run = self.get_run(run_id, user_id=user_id)
        if not run:
            return None
        if run["status"] != "awaiting_approval":
            return run

        now = time.time()
        cur_idx = int(run["current_step"])
        with _db_lock:
            conn = get_conn()
            try:
                if approved:
                    # Mark checkpoint approved so execute_until_pause_or_done runs it with approved=True
                    conn.execute(
                        "UPDATE agent_steps SET requires_approval = 0, status = 'pending', updated_at = ? WHERE run_id = ? AND step_index = ?",
                        (now, run_id, cur_idx),
                    )
                    conn.execute(
                        "UPDATE agent_runs SET status = 'running', pending_approval_json = NULL, updated_at = ? WHERE id = ? AND user_id = ?",
                        (now, run_id, user_id),
                    )
                else:
                    conn.execute(
                        "UPDATE agent_steps SET status = 'skipped', output_summary = 'Skipped: User declined approval checkpoint.', updated_at = ? WHERE run_id = ? AND step_index = ?",
                        (now, run_id, cur_idx),
                    )
                    conn.execute(
                        "UPDATE agent_runs SET status = 'running', current_step = ?, pending_approval_json = NULL, updated_at = ? WHERE id = ? AND user_id = ?",
                        (cur_idx + 1, now, run_id, user_id),
                    )
                conn.commit()
            finally:
                conn.close()

        return await self.execute_until_Pause_or_done(run_id, user_id=user_id, force_approved_step=cur_idx if approved else None)

    async def execute_until_Pause_or_done(
        self,
        run_id: str,
        user_id: str = DEFAULT_USER_ID,
        force_approved_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        run = self.get_run(run_id, user_id=user_id)
        if not run:
            raise ValueError("Agent run not found.")

        project_id = run.get("project_id")
        steps = run["steps"]
        step_observations: List[str] = [
            f"Step {s['step_index']+1} ({s['title']}): {s['output_summary']}"
            for s in steps
            if s["status"] == "completed" and s.get("output_summary")
        ]

        for st in steps:
            idx = int(st["step_index"])
            if st["status"] in ("completed", "skipped", "cancelled"):
                continue

            if self._is_cancelled(run_id):
                return self.cancel_run(run_id, user_id=user_id) or run

            # Permission / Confirmation Gate
            if st["requires_approval"] and force_approved_step != idx:
                now = time.time()
                pending_info = {
                    "step_index": idx,
                    "step_id": st["id"],
                    "title": st["title"],
                    "tool_name": st["tool_name"],
                    "tool_args": st["tool_args"],
                    "reason": "This action modifies workspace files or deletes state and requires explicit user confirmation.",
                }
                with _db_lock:
                    conn = get_conn()
                    try:
                        conn.execute(
                            "UPDATE agent_steps SET status = 'awaiting_approval', updated_at = ? WHERE id = ?",
                            (now, st["id"]),
                        )
                        conn.execute(
                            "UPDATE agent_runs SET status = 'awaiting_approval', current_step = ?, pending_approval_json = ?, updated_at = ? WHERE id = ?",
                            (idx, json.dumps(pending_info), now, run_id),
                        )
                        conn.commit()
                    finally:
                        conn.close()
                return self.get_run(run_id, user_id=user_id)  # type: ignore

            # Execute Step with bounded retry recovery
            with _db_lock:
                conn = get_conn()
                try:
                    conn.execute(
                        "UPDATE agent_steps SET status = 'running', updated_at = ? WHERE id = ?",
                        (time.time(), st["id"]),
                    )
                    conn.execute(
                        "UPDATE agent_runs SET current_step = ?, status = 'running', updated_at = ? WHERE id = ?",
                        (idx, time.time(), run_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

            attempt = 0
            res: Dict[str, Any] = {"ok": False, "error": "Not run"}
            while attempt <= self.MAX_RETRIES:
                res = await self.tools.execute_tool(
                    name=st["tool_name"],
                    arguments=st["tool_args"],
                    user_id=user_id,
                    project_id=project_id,
                    approved=True,
                )
                if res.get("ok"):
                    break
                attempt += 1
                if attempt <= self.MAX_RETRIES:
                    await asyncio.sleep(0.15)

            summary_txt = self._summarize_tool_output(st["tool_name"], res)
            artifacts = res.get("artifacts") or []
            step_status = "completed" if res.get("ok") else "failed"

            with _db_lock:
                conn = get_conn()
                try:
                    conn.execute(
                        """
                        UPDATE agent_steps
                        SET status = ?, retries = ?, output_summary = ?, artifacts_json = ?, duration_ms = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            step_status,
                            min(attempt, self.MAX_RETRIES),
                            summary_txt[:1500],
                            json.dumps(artifacts),
                            float(res.get("duration_ms") or 0.0),
                            time.time(),
                            st["id"],
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

            step_observations.append(f"**Step {idx + 1} — {st['title']}** (`{st['tool_name']}`): {summary_txt}")

        # Synthesize final agent report
        final_report = await self._synthesize_final_report(run["goal"], step_observations, user_id)
        now_done = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    UPDATE agent_runs
                    SET status = 'completed', current_step = ?, final_output = ?, pending_approval_json = NULL, updated_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (len(steps), final_report, now_done, run_id, user_id),
                )
                conn.commit()
            finally:
                conn.close()

        return self.get_run(run_id, user_id=user_id)  # type: ignore

    def _summarize_tool_output(self, tool_name: str, res: Dict[str, Any]) -> str:
        if not res.get("ok"):
            return f"Warning/Error: {res.get('error') or res.get('stderr') or 'Tool reported failure'}"
        if tool_name == "calculator_math":
            return f"Result: {res.get('result', '')}"
        if tool_name == "python_sandbox":
            out = (res.get("stdout") or "").strip().replace("\n", " | ")
            return f"Sandbox OK ({res.get('duration_ms')}ms): {out[:260] or 'Completed with chart artifact'}"
        if tool_name == "document_search":
            chunks = res.get("rag_chunks") or []
            enc = res.get("encyclopedia") or []
            titles = [f"{c['title']} ({c['section_label']})" for c in chunks[:3]] + [e["title"] for e in enc[:1]]
            return f"Retrieved {len(chunks)} Vault chunk(s) & {len(enc)} reference(s): {', '.join(titles) or '0 matches'}"
        if tool_name == "memory_retrieve":
            sems = [m for m in (res.get("semantic") or []) if m.get("category") != "system"]
            return f"Recalled {len(sems)} user memory triple(s)."
        if tool_name == "web_research":
            cits = res.get("citations") or []
            return f"Verified {len(cits)} web source(s): " + ", ".join(f"[{c['title']}]({c['url']})" for c in cits[:3])
        if tool_name == "file_create":
            return f"Saved `{res.get('path')}` (v{res.get('version')}) with reversible revision snapshot."
        if tool_name == "data_analysis":
            st = res.get("stats") or {}
            return f"Profiled n={st.get('count')} points: mean={st.get('mean')}, median={st.get('median')}, std={st.get('std_dev')}."
        if tool_name == "task_manager":
            return f"Task action `{res.get('action')}` completed ({len(res.get('tasks') or [])} total tasks)."
        return redact_sensitive(json.dumps(res)[:240])

    async def _synthesize_final_report(
        self, goal: str, observations: List[str], user_id: str
    ) -> str:
        prov = self.providers.resolve_active_provider(user_id)
        obs_block = "\n".join(f"- {o}" for o in observations)
        if prov.get("available") and prov.get("provider_id") != "xyvora_local":
            tokens: List[str] = []
            async for evt in self.providers.stream_chat_completion(
                prov,
                [
                    {
                        "role": "system",
                        "content": "You are Xyvora Autonomous Agent. Produce a clear, structured final deliverable grounded strictly in the executed step results below. Never invent results.",
                    },
                    {
                        "role": "user",
                        "content": f"Goal: {goal}\n\nExecuted Step Results:\n{obs_block}",
                    },
                ],
                temperature=0.35,
                timeout_sec=25.0,
            ):
                if evt["type"] == "token":
                    tokens.append(evt["delta"])
            if tokens:
                return "".join(tokens).strip()

        return (
            f"### Autonomous Agent Execution Summary\n\n"
            f"**Objective:** {goal}\n\n"
            f"**Completed Workflow Steps ({len(observations)}):**\n"
            + "\n".join(f"{i+1}. {o}" for i, o in enumerate(observations))
        )

    def get_run(self, run_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                r = conn.execute(
                    "SELECT * FROM agent_runs WHERE id = ? AND user_id = ?", (run_id, user_id)
                ).fetchone()
                if not r:
                    return None
                steps_rows = conn.execute(
                    "SELECT * FROM agent_steps WHERE run_id = ? ORDER BY step_index ASC",
                    (run_id,),
                ).fetchall()
            finally:
                conn.close()

        out = dict(r)
        try:
            out["plan"] = json.loads(out.get("plan_json") or "[]")
        except Exception:
            out["plan"] = []
        try:
            out["pending_approval"] = (
                json.loads(out["pending_approval_json"]) if out.get("pending_approval_json") else None
            )
        except Exception:
            out["pending_approval"] = None

        steps_fmt = []
        for sr in steps_rows:
            sd = dict(sr)
            try:
                sd["tool_args"] = json.loads(sd.get("tool_args_json") or "{}")
            except Exception:
                sd["tool_args"] = {}
            try:
                sd["artifacts"] = json.loads(sd.get("artifacts_json") or "[]")
            except Exception:
                sd["artifacts"] = []
            sd["requires_approval"] = bool(sd["requires_approval"])
            steps_fmt.append(sd)
        out["steps"] = steps_fmt
        return out

    def list_runs(self, user_id: str = DEFAULT_USER_ID, limit: int = 20) -> List[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT id, goal, status, current_step, max_steps, cancelled, created_at, updated_at FROM agent_runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()
