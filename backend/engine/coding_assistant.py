import ast
import difflib
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from database import DEFAULT_USER_ID, get_conn, _db_lock
from engine.code_engine import CodeEngine
from security import check_safety_policy

EXT_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".html": "html",
    ".css": "css",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".luau": "luau",
    ".lua": "luau",
    ".go": "go",
    ".rs": "rust",
    ".sql": "sql",
    ".json": "json",
    ".md": "markdown",
}


def detect_language(path: str, explicit: str = "") -> str:
    if explicit and explicit.strip():
        return explicit.strip().lower()
    ext = os.path.splitext(path.lower())[1]
    return EXT_TO_LANGUAGE.get(ext, "plaintext")


def compute_unified_diff(path: str, old_content: str, new_content: str) -> str:
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="\n",
    )
    return "".join(diff)


class CodingAssistantEngine:
    def __init__(self, code_engine: CodeEngine) -> None:
        self.code_engine = code_engine

    # ----------------- Project File Management (Reviewable & Reversible) -----------------

    def list_project_files(self, project_id: str, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM project_files WHERE project_id = ? AND user_id = ? ORDER BY path ASC",
                    (project_id, user_id),
                ).fetchall()
                out: List[Dict[str, Any]] = []
                for r in rows:
                    item = dict(r)
                    item["revisions"] = [
                        dict(rv)
                        for rv in conn.execute(
                            "SELECT id, version, summary, created_at FROM project_file_revisions WHERE file_id = ? ORDER BY version DESC LIMIT 15",
                            (r["id"],),
                        ).fetchall()
                    ]
                    out.append(item)
                return out
            finally:
                conn.close()

    def get_project_file(
        self, project_id: str, path: str, user_id: str = DEFAULT_USER_ID
    ) -> Optional[Dict[str, Any]]:
        clean_path = path.strip().lstrip("/")
        with _db_lock:
            conn = get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM project_files WHERE project_id = ? AND user_id = ? AND path = ?",
                    (project_id, user_id, clean_path),
                ).fetchone()
                if not row:
                    return None
                revs = [
                    dict(rv)
                    for rv in conn.execute(
                        "SELECT id, version, summary, created_at FROM project_file_revisions WHERE file_id = ? ORDER BY version DESC LIMIT 15",
                        (row["id"],),
                    ).fetchall()
                ]
                out = dict(row)
                out["revisions"] = revs
                return out
            finally:
                conn.close()

    def save_project_file(
        self,
        project_id: str,
        path: str,
        content: str,
        language: str = "",
        summary: str = "Updated file",
        user_id: str = DEFAULT_USER_ID,
    ) -> Dict[str, Any]:
        clean_path = re.sub(r"\\+", "/", path.strip()).lstrip("/")
        if not clean_path or ".." in clean_path.split("/"):
            raise ValueError("Invalid relative file path.")

        lang = detect_language(clean_path, language)
        now = time.time()

        with _db_lock:
            conn = get_conn()
            try:
                proj = conn.execute(
                    "SELECT id FROM projects WHERE id = ? AND user_id = ?",
                    (project_id, user_id),
                ).fetchone()
                if not proj:
                    raise PermissionError("Project not found or access denied.")

                existing = conn.execute(
                    "SELECT * FROM project_files WHERE project_id = ? AND path = ?",
                    (project_id, clean_path),
                ).fetchone()

                if existing:
                    prev_content = existing["content"]
                    new_ver = int(existing["version"]) + 1
                    conn.execute(
                        """
                        INSERT INTO project_file_revisions
                        (id, file_id, project_id, user_id, version, previous_content, new_content, summary, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            existing["id"],
                            project_id,
                            user_id,
                            existing["version"],
                            prev_content,
                            content,
                            summary[:200],
                            now,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE project_files
                        SET content = ?, language = ?, version = ?, updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (content, lang, new_ver, now, existing["id"], user_id),
                    )
                    file_id = existing["id"]
                    diff_text = compute_unified_diff(clean_path, prev_content, content)
                else:
                    file_id = str(uuid.uuid4())
                    conn.execute(
                        """
                        INSERT INTO project_files
                        (id, project_id, user_id, path, language, content, version, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (file_id, project_id, user_id, clean_path, lang, content, now, now),
                    )
                    diff_text = compute_unified_diff(clean_path, "", content)

                conn.execute(
                    """
                    INSERT INTO project_activity (id, project_id, user_id, action, summary, created_at)
                    VALUES (?, ?, ?, 'file_saved', ?, ?)
                    """,
                    (str(uuid.uuid4()), project_id, user_id, f"{summary}: {clean_path}", now),
                )
                conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
                conn.commit()
                saved = conn.execute("SELECT * FROM project_files WHERE id = ?", (file_id,)).fetchone()
                out = dict(saved)
                out["diff"] = diff_text
                return out
            finally:
                conn.close()

    def revert_file_revision(
        self, project_id: str, file_id: str, revision_id: str, user_id: str = DEFAULT_USER_ID
    ) -> Dict[str, Any]:
        with _db_lock:
            conn = get_conn()
            try:
                rev = conn.execute(
                    "SELECT * FROM project_file_revisions WHERE id = ? AND file_id = ? AND project_id = ? AND user_id = ?",
                    (revision_id, file_id, project_id, user_id),
                ).fetchone()
                if not rev:
                    raise ValueError("Revision not found.")
                f_row = conn.execute(
                    "SELECT * FROM project_files WHERE id = ? AND user_id = ?",
                    (file_id, user_id),
                ).fetchone()
                if not f_row:
                    raise ValueError("File not found.")
            finally:
                conn.close()

        return self.save_project_file(
            project_id=project_id,
            path=f_row["path"],
            content=rev["previous_content"],
            language=f_row["language"],
            summary=f"Reverted to v{rev['version']}",
            user_id=user_id,
        )

    def delete_project_file(
        self, project_id: str, file_id: str, user_id: str = DEFAULT_USER_ID
    ) -> bool:
        with _db_lock:
            conn = get_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM project_files WHERE id = ? AND project_id = ? AND user_id = ?",
                    (file_id, project_id, user_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ----------------- Multi-Language Static Diagnostics & Dependency Analysis -----------------

    def extract_dependencies_and_symbols(self, language: str, content: str) -> Dict[str, List[str]]:
        deps: List[str] = []
        symbols: List[str] = []

        if language == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            deps.append(a.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        deps.append(node.module)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(f"def {node.name}()")
                    elif isinstance(node, ast.ClassDef):
                        symbols.append(f"class {node.name}")
            except SyntaxError:
                pass
        elif language in ("javascript", "typescript", "tsx", "jsx"):
            deps.extend(re.findall(r"""from\s+['"]([^'"]+)['"]""", content))
            deps.extend(re.findall(r"""require\(\s*['"]([^'"]+)['"]\s*\)""", content))
            symbols.extend(re.findall(r"\b(?:function|class|interface|type|const)\s+([A-Za-z_][A-Za-z0-9_]*)", content)[:25])
        elif language == "java":
            deps.extend(re.findall(r"^\s*import\s+([A-Za-z0-9_.]+);", content, re.MULTILINE))
            symbols.extend(re.findall(r"\b(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)", content))
        elif language in ("cpp", "c"):
            deps.extend(re.findall(r"^\s*#include\s*[<\"]([^>\"]+)[>\"]", content, re.MULTILINE))
            symbols.extend(re.findall(r"\b(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", content))
        elif language == "csharp":
            deps.extend(re.findall(r"^\s*using\s+([A-Za-z0-9_.]+);", content, re.MULTILINE))
            symbols.extend(re.findall(r"\b(?:class|interface|struct|record)\s+([A-Za-z_][A-Za-z0-9_]*)", content))
        elif language == "luau":
            deps.extend(re.findall(r"require\(([^)]+)\)", content))
            symbols.extend(re.findall(r"\bfunction\s+([A-Za-z0-9_.:]+)", content))

        return {
            "dependencies": sorted(set(deps)),
            "symbols": list(dict.fromkeys(symbols))[:30],
        }

    def diagnose_code(self, path: str, content: str, language: str = "") -> List[Dict[str, Any]]:
        lang = detect_language(path, language)
        issues: List[Dict[str, Any]] = []
        lines = content.splitlines()

        if lang == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        issues.append(
                            {
                                "severity": "warning",
                                "line": node.lineno,
                                "message": "Bare `except:` catches SystemExit/KeyboardInterrupt; prefer `except Exception as exc:`.",
                            }
                        )
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for d in node.args.defaults:
                            if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                                issues.append(
                                    {
                                        "severity": "warning",
                                        "line": node.lineno,
                                        "message": f"Function `{node.name}` uses a mutable default argument; use `None` and initialize inside the body.",
                                    }
                                )
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                        issues.append(
                            {
                                "severity": "error",
                                "line": node.lineno,
                                "message": f"Security risk: `{node.func.id}()` executes arbitrary code.",
                            }
                        )
            except SyntaxError as exc:
                issues.append(
                    {
                        "severity": "error",
                        "line": exc.lineno or 1,
                        "message": f"SyntaxError: {exc.msg}",
                    }
                )
        else:
            # Structural delimiter balance check across JS/TS/TSX/Java/C++/C#/CSS
            if lang in {"javascript", "typescript", "tsx", "jsx", "java", "cpp", "c", "csharp", "css", "json"}:
                if lang == "json":
                    import json as _json
                    try:
                        _json.loads(content)
                    except Exception as exc:
                        issues.append({"severity": "error", "line": 1, "message": f"Invalid JSON: {exc}"})
                else:
                    open_b = content.count("{")
                    close_b = content.count("}")
                    if open_b != close_b:
                        issues.append(
                            {
                                "severity": "error",
                                "line": len(lines),
                                "message": f"Unbalanced curly braces: {open_b} '{{' vs {close_b} '}}'.",
                            }
                        )

            for idx, ln in enumerate(lines, start=1):
                if lang in ("javascript", "typescript", "tsx", "jsx"):
                    if "dangerouslySetInnerHTML" in ln or ".innerHTML" in ln:
                        issues.append(
                            {
                                "severity": "warning",
                                "line": idx,
                                "message": "Potential XSS sink (`innerHTML`); ensure untrusted HTML is sanitized.",
                            }
                        )
                    if re.search(r"[^=!]==[^=]", ln):
                        issues.append(
                            {
                                "severity": "info",
                                "line": idx,
                                "message": "Prefer strict equality `===` over coercive `==` in JS/TS.",
                            }
                        )
                elif lang in ("cpp", "c"):
                    if re.search(r"\b(?:gets|strcpy|sprintf)\s*\(", ln):
                        issues.append(
                            {
                                "severity": "error",
                                "line": idx,
                                "message": "Buffer-overflow hazard: replace `gets/strcpy/sprintf` with bounds-checked `fgets/snprintf/std::string`.",
                            }
                        )
                elif lang == "luau":
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*", ln) and not ln.startswith("export "):
                        issues.append(
                            {
                                "severity": "info",
                                "line": idx,
                                "message": "Luau best practice: declare variables with `local` to avoid global environment lookups.",
                            }
                        )

        return issues

    def analyze_project(self, project_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        files = self.list_project_files(project_id, user_id)
        file_summaries = []
        all_diagnostics = []
        lang_counts: Dict[str, int] = {}

        for f in files:
            lang = f["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            meta = self.extract_dependencies_and_symbols(lang, f["content"])
            diags = self.diagnose_code(f["path"], f["content"], lang)
            for d in diags:
                all_diagnostics.append({**d, "path": f["path"]})
            file_summaries.append(
                {
                    "id": f["id"],
                    "path": f["path"],
                    "language": lang,
                    "lines": len(f["content"].splitlines()),
                    "version": f["version"],
                    "dependencies": meta["dependencies"],
                    "symbols": meta["symbols"],
                    "issue_count": len(diags),
                }
            )

        return {
            "project_id": project_id,
            "file_count": len(files),
            "languages": lang_counts,
            "files": file_summaries,
            "diagnostics": all_diagnostics,
        }

    def run_approved_tests(
        self, project_id: str, user_id: str = DEFAULT_USER_ID
    ) -> Dict[str, Any]:
        """
        Runs all Python unittest suites in the project inside the hardened isolated subprocess sandbox.
        """
        files = self.list_project_files(project_id, user_id)
        py_files = {f["path"]: f["content"] for f in files if f["path"].endswith(".py")}
        if not py_files:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "No Python (.py) files found in this project workspace.",
                "duration_ms": 0.0,
            }

        test_modules = []
        for path in sorted(py_files.keys()):
            base = os.path.basename(path)
            if base.startswith("test_") or base.endswith("_test.py"):
                mod_name = path[:-3].replace("/", ".")
                test_modules.append(mod_name)

        if not test_modules:
            return {
                "ok": False,
                "stdout": "",
                "stderr": "No test files (`test_*.py` or `*_test.py`) found in project. Create a test module to run approved unit tests.",
                "duration_ms": 0.0,
            }

        harness = (
            "import io\nimport unittest\n\n"
            "suite = unittest.TestSuite()\n"
            "loader = unittest.TestLoader()\n"
            + "\n".join(
                f"import {m}\nsuite.addTests(loader.loadTestsFromModule({m}))"
                for m in test_modules
            )
            + "\nstream = io.StringIO()\n"
            "runner = unittest.TextTestRunner(stream=stream, verbosity=2)\n"
            "result = runner.run(suite)\n"
            "print(stream.getvalue())\n"
            "print(f'Summary: ran={result.testsRun}, failures={len(result.failures)}, errors={len(result.errors)}')\n"
            "if not result.wasSuccessful():\n"
            "    raise AssertionError('Unit test suite reported failures.')\n"
        )

        return self.code_engine.execute_python(
            code=harness,
            timeout_sec=8.0,
            user_id=user_id,
            extra_files=py_files,
        )

    async def assist_on_code(
        self,
        action: str,
        prompt: str,
        target_file: Optional[Dict[str, Any]],
        project_files: List[Dict[str, Any]],
        provider_manager: Any,
        user_id: str = DEFAULT_USER_ID,
    ) -> Dict[str, Any]:
        """
        Multi-file project-aware AI coding assistance (explain, debug, review, refactor, generate_patch).
        Uses configured LLM when available, or deterministic structural AST/diagnostic engine locally.
        """
        safety = check_safety_policy(f"{action} {prompt}")
        if safety:
            return {
                "ok": False,
                "explanation": safety["message"],
                "proposed_content": None,
                "diff": "",
                "diagnostics": [],
            }

        target_path = target_file["path"] if target_file else "untitled.py"
        target_lang = target_file["language"] if target_file else "python"
        target_src = target_file["content"] if target_file else ""
        diags = self.diagnose_code(target_path, target_src, target_lang) if target_src else []
        meta = self.extract_dependencies_and_symbols(target_lang, target_src) if target_src else {"dependencies": [], "symbols": []}

        prov = provider_manager.resolve_active_provider(user_id)
        if prov.get("available") and prov.get("provider_id") != "xyvora_local":
            proj_context = "\n\n".join(
                f"--- FILE: {f['path']} ({f['language']}) ---\n{f['content'][:3000]}"
                for f in project_files[:8]
            )
            sys_msg = (
                "You are Xyvora Forge, a senior multi-language software architect supporting Python, JS, TS, React, HTML, CSS, Java, C++, C#, and Luau.\n"
                "Never generate cheats, game exploits, or malware.\n"
                "When asked to refactor, fix bugs, or edit the active file, include the complete updated file inside a single fenced code block so Xyvora can compute a reviewable unified diff."
            )
            user_msg = (
                f"Action: {action}\nTarget File: {target_path} ({target_lang})\nUser Request: {prompt}\n\n"
                f"Project Files Context:\n{proj_context}"
            )
            tokens: List[str] = []
            async for evt in provider_manager.stream_chat_completion(
                prov,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                temperature=0.25,
                timeout_sec=35.0,
            ):
                if evt["type"] == "token":
                    tokens.append(evt["delta"])
                elif evt["type"] == "error":
                    break

            reply = "".join(tokens).strip()
            if reply:
                proposed = None
                diff_str = ""
                if action in ("refactor", "debug", "generate_patch", "edit"):
                    m_code = re.search(r"```[a-zA-Z0-9_+]*\n([\s\S]*?)```", reply)
                    if m_code:
                        proposed = m_code.group(1)
                        diff_str = compute_unified_diff(target_path, target_src, proposed)
                return {
                    "ok": True,
                    "provider": prov["provider_id"],
                    "model": prov["model"],
                    "explanation": reply,
                    "proposed_content": proposed,
                    "diff": diff_str,
                    "diagnostics": diags,
                }

        # Deterministic Local Structural Analysis & Refactoring Fallback
        lines_cnt = len(target_src.splitlines())
        deps_md = ", ".join(f"`{d}`" for d in meta["dependencies"]) or "None"
        syms_md = ", ".join(f"`{s}`" for s in meta["symbols"]) or "Top-level script"
        diag_md = (
            "\n".join(f"- **Line {d['line']} [{d['severity'].upper()}]**: {d['message']}" for d in diags)
            if diags
            else "- Zero static syntax or anti-pattern issues detected."
        )

        proposed_src: Optional[str] = None
        diff_out = ""

        if action in ("refactor", "generate_patch", "debug") and target_src:
            # Apply clean, safe structural improvements (strip trailing whitespace, add module docstring/header if missing, fix `==` in JS/TS or bare `except:` in Python)
            updated_lines = [ln.rstrip() for ln in target_src.splitlines()]
            updated = "\n".join(updated_lines)
            if target_lang == "python":
                updated = re.sub(r"^(\s*)except\s*:", r"\1except Exception:", updated, flags=re.MULTILINE)
                if not updated.lstrip().startswith(('"""', "'''", "#")):
                    updated = f'"""Module `{target_path}` — refactored and validated by Xyvora Coding Studio."""\n' + updated
            elif target_lang in ("typescript", "javascript", "tsx", "jsx"):
                if not updated.lstrip().startswith(("//", "/*")):
                    updated = f"// {target_path} — audited by Xyvora Coding Studio\n" + updated
            if not updated.endswith("\n"):
                updated += "\n"
            if updated != target_src:
                proposed_src = updated
                diff_out = compute_unified_diff(target_path, target_src, proposed_src)

        explanation = (
            f"### Code Intelligence Report: `{target_path}` (`{target_lang}`)\n\n"
            f"- **Lines of Code**: `{lines_cnt}` across `{len(project_files)}` project file(s)\n"
            f"- **Detected Imports / Dependencies**: {deps_md}\n"
            f"- **Exported / Defined Symbols**: {syms_md}\n\n"
            f"#### Static Diagnostics & Security Review\n{diag_md}\n\n"
            + (
                "#### Proposed Patch Ready\nReview the unified diff in the **Diff Viewer** tab before clicking **Apply Patch** (all changes create a reversible snapshot)."
                if proposed_src
                else "*Note: Running in Local Structural Analysis mode. Connect an LLM provider (OpenAI, Anthropic, Gemini, Groq, or Ollama) in Settings for open-ended generative refactoring.*"
            )
        )

        return {
            "ok": True,
            "provider": "xyvora_local",
            "model": "xyvora-cortex-v2",
            "explanation": explanation,
            "proposed_content": proposed_src,
            "diff": diff_out,
            "diagnostics": diags,
        }
