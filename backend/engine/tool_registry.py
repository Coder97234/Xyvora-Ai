import asyncio
import csv
import inspect
import io
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from database import DEFAULT_USER_ID, get_conn, _db_lock
from security import redact_sensitive


@dataclass
class ToolDefinition:
    name: str
    activity_label: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    permissions: List[str]
    timeout_sec: float
    requires_confirmation: bool
    handler: Callable[..., Any]


class ToolRegistry:
    """
    Central schema-validated Tool Registry & Safe Executor with:
      - Input validation against JSON Schema required fields
      - Per-tool timeout enforcement
      - Infinite-loop / duplicate call prevention
      - Permission & confirmation gate checks
    """

    MAX_CALLS_PER_TURN = 6

    def __init__(
        self,
        math_engine: Any,
        code_engine: Any,
        knowledge_engine: Any,
        memory_engine: Any,
        web_engine: Any,
        coding_engine: Any,
        text_intel: Any,
        provider_manager: Any,
    ) -> None:
        self.math = math_engine
        self.code = code_engine
        self.knowledge = knowledge_engine
        self.memory = memory_engine
        self.web = web_engine
        self.coding = coding_engine
        self.text_intel = text_intel
        self.providers = provider_manager
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_core_tools()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "activity_label": t.activity_label,
                "description": t.description,
                "input_schema": t.input_schema,
                "output_schema": t.output_schema,
                "permissions": t.permissions,
                "timeout_sec": t.timeout_sec,
                "requires_confirmation": t.requires_confirmation,
            }
            for t in self._tools.values()
        ]

    def to_openai_tools_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def _validate_args(self, tool: ToolDefinition, args: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(args, dict):
            return False, "Tool arguments must be a JSON object."
        required = tool.input_schema.get("required") or []
        for req_key in required:
            if req_key not in args or args[req_key] is None or (isinstance(args[req_key], str) and not args[req_key].strip()):
                return False, f"Missing required parameter '{req_key}' for tool '{tool.name}'."
        return True, "OK"

    async def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_id: str = DEFAULT_USER_ID,
        project_id: Optional[str] = None,
        approved: bool = False,
        allowed_permissions: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"ok": False, "tool": name, "error": f"Unknown tool '{name}'."}

        if allowed_permissions is not None:
            missing_perms = [p for p in tool.permissions if p not in allowed_permissions]
            if missing_perms:
                return {
                    "ok": False,
                    "tool": name,
                    "error": f"Permission denied: missing scope(s) {', '.join(missing_perms)}.",
                }

        # Check confirmation requirement (or destructive action inside arguments)
        needs_confirm = tool.requires_confirmation or (
            name == "file_create" and bool(arguments.get("overwrite_destructive"))
        ) or (
            name == "task_manager" and arguments.get("action") == "delete_all"
        )
        if needs_confirm and not approved:
            return {
                "ok": False,
                "requires_confirmation": True,
                "tool": name,
                "activity_label": tool.activity_label,
                "arguments": arguments,
                "error": f"Tool '{name}' requires explicit user approval before executing.",
            }

        valid, reason = self._validate_args(tool, arguments)
        if not valid:
            return {"ok": False, "tool": name, "error": reason}

        t0 = time.perf_counter()
        try:
            call_kwargs = {
                "args": arguments,
                "user_id": user_id,
                "project_id": project_id,
            }
            if inspect.iscoroutinefunction(tool.handler):
                res = await asyncio.wait_for(tool.handler(**call_kwargs), timeout=tool.timeout_sec)
            else:
                res = await asyncio.wait_for(
                    asyncio.to_thread(tool.handler, **call_kwargs), timeout=tool.timeout_sec
                )
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if isinstance(res, dict):
                res.setdefault("ok", True)
                res["tool"] = name
                res["activity_label"] = tool.activity_label
                res["duration_ms"] = ms
                return res
            return {"ok": True, "tool": name, "activity_label": tool.activity_label, "result": res, "duration_ms": ms}
        except asyncio.TimeoutError:
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {
                "ok": False,
                "tool": name,
                "activity_label": tool.activity_label,
                "duration_ms": ms,
                "error": f"Tool '{name}' timed out after {tool.timeout_sec:.1f}s.",
            }
        except Exception as exc:
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            return {
                "ok": False,
                "tool": name,
                "activity_label": tool.activity_label,
                "duration_ms": ms,
                "error": f"Tool '{name}' error: {redact_sensitive(str(exc))[:280]}",
            }

    # ----------------- 11 Registered Production Tools -----------------

    def _register_core_tools(self) -> None:
        # 1. Calculator & Symbolic Mathematics
        self.register(
            ToolDefinition(
                name="calculator_math",
                activity_label="Calculating symbolic math",
                description="Solves arithmetic, percentages, algebraic equations, derivatives, integrals, limits, eigenvalues, and generates interactive function plots.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical query or symbolic expression, e.g. 'Differentiate sin(x)*cos(x)' or 'Solve x^2 - 5x + 6 = 0'.",
                        }
                    },
                    "required": ["expression"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "result": {"type": "string"}, "steps": {"type": "array"}}},
                permissions=["compute:math"],
                timeout_sec=8.0,
                requires_confirmation=False,
                handler=self._handle_math,
            )
        )

        # 2. Hardened Python Execution
        self.register(
            ToolDefinition(
                name="python_sandbox",
                activity_label="Running code in isolated Python sandbox",
                description="Executes Python 3 code inside an OS resource-limited isolated sandbox with math, statistics, numpy, sympy, and emit_chart(title, labels, values, chart_type).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Valid Python 3 source code to run safely in the sandbox.",
                        }
                    },
                    "required": ["code"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "stdout": {"type": "string"}, "stderr": {"type": "string"}}},
                permissions=["sandbox:python"],
                timeout_sec=8.0,
                requires_confirmation=False,
                handler=self._handle_python,
            )
        )

        # 3. Document Search (RAG Knowledge Vault)
        self.register(
            ToolDefinition(
                name="document_search",
                activity_label="Searching Knowledge Vault documents",
                description="Searches user-uploaded documents (PDF, DOCX, XLSX, PPTX, CSV, Markdown, Code) and reference articles with chunk-level section citations.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for semantic & keyword retrieval."},
                        "collection": {"type": "string", "description": "Optional collection filter."},
                    },
                    "required": ["query"],
                },
                output_schema={"type": "object", "properties": {"rag_chunks": {"type": "array"}, "encyclopedia": {"type": "array"}}},
                permissions=["vault:read"],
                timeout_sec=6.0,
                requires_confirmation=False,
                handler=self._handle_doc_search,
            )
        )

        # 4. Memory Retrieval
        self.register(
            ToolDefinition(
                name="memory_retrieve",
                activity_label="Retrieving from Tri-Layer Memory Palace",
                description="Recalls user preferences, facts, relationships, and past episodic interactions from the user's isolated Memory Palace.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Memory lookup query."}
                    },
                    "required": ["query"],
                },
                output_schema={"type": "object", "properties": {"semantic": {"type": "array"}, "episodic": {"type": "array"}}},
                permissions=["memory:read"],
                timeout_sec=5.0,
                requires_confirmation=False,
                handler=self._handle_memory_retrieve,
            )
        )

        # 5. Memory Updates
        self.register(
            ToolDefinition(
                name="memory_update",
                activity_label="Updating Semantic Memory Graph",
                description="Stores, updates, or deletes a user-confirmed fact or preference in the Semantic Knowledge Graph.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["upsert", "forget"], "description": "Whether to upsert or forget a memory."},
                        "subject": {"type": "string", "description": "Entity subject (e.g. 'User' or project name)."},
                        "relation": {"type": "string", "description": "Predicate relation (e.g. 'name', 'prefers', 'location')."},
                        "object": {"type": "string", "description": "Target value or fact."},
                        "category": {"type": "string", "description": "Category: identity, preference, knowledge, project."},
                    },
                    "required": ["operation"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "record": {"type": "object"}}},
                permissions=["memory:write"],
                timeout_sec=5.0,
                requires_confirmation=False,
                handler=self._handle_memory_update,
            )
        )

        # 6. Web Research
        self.register(
            ToolDefinition(
                name="web_research",
                activity_label="Researching the web",
                description="Searches the web and fetches readable webpage content with publication dates and verified source citations.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query or URL to investigate."},
                        "mode": {"type": "string", "enum": ["quick", "deep"], "description": "Quick search or deep multi-page research."},
                    },
                    "required": ["query"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "sources": {"type": "array"}, "citations": {"type": "array"}}},
                permissions=["web:search"],
                timeout_sec=22.0,
                requires_confirmation=False,
                handler=self._handle_web_research,
            )
        )

        # 7. File Reading
        self.register(
            ToolDefinition(
                name="file_read",
                activity_label="Reading workspace file",
                description="Reads a file from the active Project Workspace or retrieves an uploaded Knowledge Vault document by path or title.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path in project or document title."}
                    },
                    "required": ["path"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "content": {"type": "string"}}},
                permissions=["files:read"],
                timeout_sec=5.0,
                requires_confirmation=False,
                handler=self._handle_file_read,
            )
        )

        # 8. File Creation / Modification
        self.register(
            ToolDefinition(
                name="file_create",
                activity_label="Creating / updating project file",
                description="Creates or updates a file inside the user's Project Workspace with automatic unified diff generation and reversible revision history.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path, e.g. 'src/utils.py'."},
                        "content": {"type": "string", "description": "Full text content to write."},
                        "language": {"type": "string", "description": "Programming language identifier."},
                        "summary": {"type": "string", "description": "Short change summary."},
                    },
                    "required": ["path", "content"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "path": {"type": "string"}, "diff": {"type": "string"}}},
                permissions=["files:write"],
                timeout_sec=6.0,
                requires_confirmation=False,
                handler=self._handle_file_create,
            )
        )

        # 9. Quantitative Data Analysis
        self.register(
            ToolDefinition(
                name="data_analysis",
                activity_label="Analyzing dataset & computing statistics",
                description="Analyzes numeric series, CSV data, or label-value pairs: computes mean, median, std dev, quartiles, linear trend, outliers, and generates interactive charts.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data_input": {
                            "type": "string",
                            "description": "Raw CSV text, JSON array, or natural label-value pairs (e.g., 'Jan 120, Feb 165, Mar 210').",
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "area"],
                            "description": "Preferred chart visualization type.",
                        },
                    },
                    "required": ["data_input"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "stats": {"type": "object"}, "artifacts": {"type": "array"}}},
                permissions=["data:analyze"],
                timeout_sec=6.0,
                requires_confirmation=False,
                handler=self._handle_data_analysis,
            )
        )

        # 10. Task Management
        self.register(
            ToolDefinition(
                name="task_manager",
                activity_label="Managing Task Board",
                description="Creates, lists, completes, or deletes persistent tasks and milestones on the user's Task Board.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["create", "list", "complete", "delete"], "description": "Task operation."},
                        "title": {"type": "string", "description": "Task title (for create/complete/delete)."},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "due_note": {"type": "string"},
                    },
                    "required": ["action"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "tasks": {"type": "array"}}},
                permissions=["tasks:manage"],
                timeout_sec=5.0,
                requires_confirmation=False,
                handler=self._handle_task_manager,
            )
        )

        # 11. AI Image Generation
        self.register(
            ToolDefinition(
                name="generate_image",
                activity_label="Generating image via AI provider",
                description="Generates an image from a descriptive prompt using a configured image model (such as OpenAI DALL·E 3 / gpt-image-1).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Detailed visual prompt to generate."}
                    },
                    "required": ["prompt"],
                },
                output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "data_url": {"type": "string"}}},
                permissions=["media:generate"],
                timeout_sec=48.0,
                requires_confirmation=False,
                handler=self._handle_generate_image,
            )
        )

    # ----------------- Tool Handlers -----------------

    def _handle_math(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        return self.math.solve_query(str(args["expression"]))

    def _handle_python(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        extra_files = None
        if project_id:
            pfiles = self.coding.list_project_files(project_id, user_id)
            extra_files = {f["path"]: f["content"] for f in pfiles if f["path"].endswith(".py")}
        res = self.code.execute_python(str(args["code"]), user_id=user_id, extra_files=extra_files)
        exec_artifact = {
            "type": "code_execution",
            "title": "Sandboxed Python Execution",
            "code": str(args["code"]),
            "stdout": res["stdout"],
            "stderr": res["stderr"],
            "ok": res["ok"],
            "duration_ms": res["duration_ms"],
        }
        res["artifacts"] = [exec_artifact] + res.get("artifacts", [])
        return res

    def _handle_doc_search(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        res = self.knowledge.search(
            query=str(args["query"]),
            top_k=4,
            user_id=user_id,
            collection=args.get("collection"),
            project_id=project_id,
        )
        citations = [
            {
                "source_type": c["source_kind"],
                "title": c["title"],
                "section": c["section_label"],
                "snippet": c["content"][:260],
                "score": c["score"],
            }
            for c in res["rag_chunks"]
        ]
        return {"ok": True, **res, "citations": citations}

    def _handle_memory_retrieve(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        recalled = self.memory.recall(
            query=str(args["query"]),
            threshold=0.15,
            top_k=10,
            include_all_user_profile=True,
            user_id=user_id,
            project_id=project_id,
        )
        return {"ok": True, **recalled}

    def _handle_memory_update(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        op = str(args.get("operation", "upsert")).lower()
        if op == "forget":
            q = f"{args.get('subject', '')} {args.get('relation', '')} {args.get('object', '')}".strip()
            removed = self.memory.forget_matching(q, user_id=user_id)
            return {"ok": True, "operation": "forget", "removed_count": len(removed), "removed": removed}

        subj = str(args.get("subject") or "User").strip()
        rel = str(args.get("relation") or "note").strip()
        obj = str(args.get("object") or "").strip()
        if not obj:
            return {"ok": False, "error": "Memory upsert requires a non-empty 'object' value."}
        rec = self.memory.upsert_semantic_memory(
            subject=subj,
            relation=rel,
            obj=obj,
            category=str(args.get("category") or "fact"),
            verified=1,
            source_text="Tool memory_update",
            user_id=user_id,
            project_id=project_id,
        )
        return {"ok": True, "operation": "upsert", "record": rec, "memories_added": [rec]}

    async def _handle_web_research(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        return await self.web.research(
            query=str(args["query"]),
            mode=str(args.get("mode") or "quick"),
            max_results=5,
            follow_links=(args.get("mode") == "deep"),
        )

    def _handle_file_read(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        target = str(args["path"]).strip()
        # 1. Check active or default user project files
        with _db_lock:
            conn = get_conn()
            try:
                if project_id:
                    pf = conn.execute(
                        "SELECT * FROM project_files WHERE project_id = ? AND user_id = ? AND lower(path) = lower(?)",
                        (project_id, user_id, target.lstrip("/")),
                    ).fetchone()
                else:
                    pf = conn.execute(
                        "SELECT * FROM project_files WHERE user_id = ? AND lower(path) = lower(?) ORDER BY updated_at DESC LIMIT 1",
                        (user_id, target.lstrip("/")),
                    ).fetchone()

                if pf:
                    return {
                        "ok": True,
                        "source": "project_file",
                        "path": pf["path"],
                        "language": pf["language"],
                        "version": pf["version"],
                        "content": pf["content"],
                    }

                # 2. Check Knowledge Vault documents by title/filename
                doc = conn.execute(
                    "SELECT id, title, filename, collection, content FROM documents WHERE user_id = ? AND (lower(title) LIKE ? OR lower(filename) LIKE ?) LIMIT 1",
                    (user_id, f"%{target.lower()}%", f"%{target.lower()}%"),
                ).fetchone()
                if doc:
                    return {
                        "ok": True,
                        "source": "knowledge_vault",
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"][:8000],
                    }
            finally:
                conn.close()

        return {"ok": False, "error": f"File or document '{target}' was not found in your workspace."}

    def _handle_file_create(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        # Resolve target project_id or user's primary project
        target_proj = project_id
        if not target_proj:
            with _db_lock:
                conn = get_conn()
                try:
                    row = conn.execute(
                        "SELECT id FROM projects WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
                        (user_id,),
                    ).fetchone()
                    if row:
                        target_proj = row["id"]
                    else:
                        now = time.time()
                        target_proj = str(uuid.uuid4())
                        conn.execute(
                            "INSERT INTO projects (id, user_id, name, description, instructions, created_at, updated_at) VALUES (?, ?, 'Default Coding Project', 'Auto-created workspace', '', ?, ?)",
                            (target_proj, user_id, now, now),
                        )
                        conn.commit()
                finally:
                    conn.close()

        saved = self.coding.save_project_file(
            project_id=target_proj,
            path=str(args["path"]),
            content=str(args["content"]),
            language=str(args.get("language") or ""),
            summary=str(args.get("summary") or "Created/updated via AI tool"),
            user_id=user_id,
        )
        return {
            "ok": True,
            "project_id": target_proj,
            "file_id": saved["id"],
            "path": saved["path"],
            "version": saved["version"],
            "diff": saved.get("diff", ""),
        }

    def _handle_data_analysis(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        raw = str(args["data_input"]).strip()
        pref_chart = str(args.get("chart_type") or "").strip().lower()

        # Try CSV table first if multiline with commas
        if "\n" in raw and "," in raw.splitlines()[0]:
            reader = list(csv.reader(io.StringIO(raw)))
            if len(reader) >= 2 and len(reader[0]) >= 2:
                labels = []
                vals = []
                for r in reader[1:]:
                    if len(r) >= 2:
                        try:
                            vals.append(float(r[1].strip().replace(",", "")))
                            labels.append(r[0].strip() or f"Row {len(labels)+1}")
                        except ValueError:
                            pass
                if len(vals) >= 2:
                    return self._compute_series_analysis(labels, vals, pref_chart or "bar", f"{reader[0][1]} by {reader[0][0]}")

        parsed = self.text_intel.parse_chart_request(f"chart {raw}")
        if parsed:
            labels = [d["label"] for d in parsed["chart_artifact"]["data"]]
            vals = [float(d["value"]) for d in parsed["chart_artifact"]["data"]]
            return self._compute_series_analysis(labels, vals, pref_chart or parsed["chart_artifact"]["chart_type"], "Quantitative Series Analysis")

        nums = [float(n) for n in re.findall(r"[\-+]?\d+(?:\.\d+)?", raw)]
        if len(nums) >= 2:
            labels = [f"#{i+1}" for i in range(len(nums))]
            return self._compute_series_analysis(labels, nums, pref_chart or "line", "Numeric Distribution Analysis")

        return {"ok": False, "error": "Provide at least 2 numeric data points, CSV rows, or label-value pairs."}

    def _compute_series_analysis(
        self, labels: List[str], values: List[float], chart_type: str, title: str
    ) -> Dict[str, Any]:
        arr = np.array(values, dtype=float)
        n = len(arr)
        mean_v = float(np.mean(arr))
        med_v = float(np.median(arr))
        std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        q25, q75 = [float(x) for x in np.percentile(arr, [25, 75])]
        xs = np.arange(n, dtype=float)
        slope = float(np.polyfit(xs, arr, 1)[0]) if n >= 2 else 0.0
        outliers = [
            labels[i]
            for i, v in enumerate(values)
            if std_v > 0 and abs(v - mean_v) / std_v >= 2.0
        ]
        chart_art = {
            "type": "chart",
            "chart_type": chart_type if chart_type in ("bar", "line", "area") else "bar",
            "title": title,
            "data": [{"label": l, "value": round(v, 4)} for l, v in zip(labels[:40], values[:40])],
        }
        stats = {
            "count": n,
            "mean": round(mean_v, 4),
            "median": round(med_v, 4),
            "std_dev": round(std_v, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "iqr_25_75": [round(q25, 4), round(q75, 4)],
            "trend_slope_per_step": round(slope, 4),
            "outliers_2sigma": outliers,
        }
        return {"ok": True, "stats": stats, "artifacts": [chart_art]}

    def _handle_task_manager(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        action = str(args.get("action") or "list").lower()
        title = str(args.get("title") or "").strip()
        prio = str(args.get("priority") or "medium").lower()
        if prio not in ("high", "medium", "low"):
            prio = "medium"
        due = str(args.get("due_note") or "").strip()
        now = time.time()

        with _db_lock:
            conn = get_conn()
            try:
                if action == "create" and title:
                    tid = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO tasks (id, user_id, project_id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, ?, ?, '', ?, 'todo', ?, ?, ?)",
                        (tid, user_id, project_id, title[:180], prio, due[:50], now, now),
                    )
                    conn.commit()
                elif action == "complete" and title:
                    rows = conn.execute(
                        "SELECT id, title FROM tasks WHERE user_id = ? AND status != 'done'",
                        (user_id,),
                    ).fetchall()
                    for r in rows:
                        if title.lower() in r["title"].lower():
                            conn.execute(
                                "UPDATE tasks SET status = 'done', updated_at = ? WHERE id = ? AND user_id = ?",
                                (now, r["id"], user_id),
                            )
                    conn.commit()
                elif action == "delete" and title:
                    conn.execute(
                        "DELETE FROM tasks WHERE user_id = ? AND lower(title) LIKE ?",
                        (user_id, f"%{title.lower()}%"),
                    )
                    conn.commit()

                tasks = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
                        (user_id,),
                    ).fetchall()
                ]
                return {
                    "ok": True,
                    "action": action,
                    "tasks": tasks,
                    "artifacts": [{"type": "task_list", "title": "Active Agent Task Board", "tasks": tasks[:10]}],
                }
            finally:
                conn.close()

    async def _handle_generate_image(self, args: Dict[str, Any], user_id: str, project_id: Optional[str]) -> Dict[str, Any]:
        res = await self.providers.generate_image(user_id=user_id, prompt=str(args["prompt"]))
        if res.get("ok"):
            res["artifacts"] = [
                {
                    "type": "generated_image",
                    "title": "AI-Generated Image",
                    "prompt": res["prompt"],
                    "revised_prompt": res.get("revised_prompt", res["prompt"]),
                    "provider": res["provider"],
                    "model": res["model"],
                    "data_url": res["data_url"],
                    "is_generated": True,
                }
            ]
        return res
