import csv
import io
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from database import DEFAULT_USER_ID, get_conn, _db_lock
from security import check_safety_policy

ALLOWED_EXTENSION_PERMISSIONS = {
    "compute:math",
    "compute:text",
    "compute:data",
    "read:memory",
    "read:documents",
    "network:public_metadata",
}

BUILTIN_EXTENSIONS: List[Dict[str, Any]] = [
    {
        "id": "ext_unit_converter",
        "name": "Precision Unit & Physics Converter",
        "description": "Exact dimensional conversions across length, mass, temperature, data storage, energy, pressure, and speed.",
        "category": "mathematics",
        "version": "1.2.0",
        "author": "Xyvora Core Labs",
        "permissions": ["compute:math"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"precision_digits": {"type": "integer", "default": 5}},
        "input_schema": {
            "type": "object",
            "required": ["value", "from_unit", "to_unit"],
            "properties": {
                "value": {"type": "number", "description": "Numeric quantity to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g., km, mi, c, f, k, kg, lb, gb, mb)"},
                "to_unit": {"type": "string", "description": "Target unit"},
            },
        },
        "output_schema": {"type": "object", "properties": {"result": {"type": "number"}, "formatted": {"type": "string"}}},
    },
    {
        "id": "ext_regex_debugger",
        "name": "Regex Pattern Inspector & Explainer",
        "description": "Tests regular expressions against sample text with bounded execution, capture-group breakdown, and token-level explanation.",
        "category": "coding",
        "version": "1.1.0",
        "author": "Xyvora DevTools",
        "permissions": ["compute:text"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"case_insensitive": {"type": "boolean", "default": False}},
        "input_schema": {
            "type": "object",
            "required": ["pattern", "sample_text"],
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression pattern"},
                "sample_text": {"type": "string", "description": "Target text to test against"},
            },
        },
        "output_schema": {"type": "object", "properties": {"match_count": {"type": "integer"}, "matches": {"type": "array"}}},
    },
    {
        "id": "ext_json_validator",
        "name": "JSON Payload & Schema Structural Auditor",
        "description": "Validates JSON syntax, infers TypeScript interfaces, checks required fields, and detects potential secret leaks.",
        "category": "developer",
        "version": "1.3.0",
        "author": "Xyvora DevTools",
        "permissions": ["compute:data"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"generate_typescript": {"type": "boolean", "default": True}},
        "input_schema": {
            "type": "object",
            "required": ["json_text"],
            "properties": {
                "json_text": {"type": "string", "description": "Raw JSON string to validate and inspect"},
                "required_keys": {"type": "array", "items": {"type": "string"}},
            },
        },
        "output_schema": {"type": "object", "properties": {"valid": {"type": "boolean"}, "typescript_interface": {"type": "string"}}},
    },
    {
        "id": "ext_sql_architect",
        "name": "SQL Query Safety & Index Advisor",
        "description": "Audits SQL queries for injection risks, missing WHERE clauses on mutations, SELECT * anti-patterns, and composite index recommendations.",
        "category": "coding",
        "version": "1.0.5",
        "author": "Xyvora Data Engineering",
        "permissions": ["compute:text"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"dialect": {"type": "string", "default": "postgresql"}},
        "input_schema": {
            "type": "object",
            "required": ["sql"],
            "properties": {"sql": {"type": "string", "description": "SQL query to analyze"}},
        },
        "output_schema": {"type": "object", "properties": {"warnings": {"type": "array"}, "index_suggestions": {"type": "array"}}},
    },
    {
        "id": "ext_csv_profiler",
        "name": "Tabular Dataset Statistical Profiler",
        "description": "Profiles CSV snippets for column types, missingness, quartiles, mean/std, and z-score outliers.",
        "category": "data",
        "version": "1.1.2",
        "author": "Xyvora Data Studio",
        "permissions": ["compute:data"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"z_threshold": {"type": "number", "default": 2.2}},
        "input_schema": {
            "type": "object",
            "required": ["csv_text"],
            "properties": {"csv_text": {"type": "string", "description": "CSV dataset with header row"}},
        },
        "output_schema": {"type": "object", "properties": {"row_count": {"type": "integer"}, "columns": {"type": "array"}}},
    },
    {
        "id": "ext_research_credibility",
        "name": "Source Credibility & Citation Evaluator",
        "description": "Evaluates source URLs and claims across authority, primary vs secondary evidence, recency signals, and APA/IEEE citation formatting.",
        "category": "research",
        "version": "1.0.8",
        "author": "Xyvora Research Lab",
        "permissions": ["compute:text"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"citation_style": {"type": "string", "default": "APA"}},
        "input_schema": {
            "type": "object",
            "required": ["url", "title"],
            "properties": {
                "url": {"type": "string"},
                "title": {"type": "string"},
                "published_date": {"type": "string"},
            },
        },
        "output_schema": {"type": "object", "properties": {"tier": {"type": "string"}, "citation": {"type": "string"}}},
    },
    {
        "id": "ext_security_headers",
        "name": "Defensive HTTP Security & CORS Auditor",
        "description": "Analyzes HTTP response headers (CSP, HSTS, X-Frame-Options, Referrer-Policy, CORS) for defensive web hardening.",
        "category": "developer",
        "version": "1.2.1",
        "author": "Xyvora Defensive Security",
        "permissions": ["compute:text"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"strict_csp": {"type": "boolean", "default": True}},
        "input_schema": {
            "type": "object",
            "required": ["headers_text"],
            "properties": {"headers_text": {"type": "string", "description": "Raw HTTP response headers (Key: Value lines)"}},
        },
        "output_schema": {"type": "object", "properties": {"score": {"type": "integer"}, "recommendations": {"type": "array"}}},
    },
    {
        "id": "ext_sprint_planner",
        "name": "Engineering Milestone & Risk Decomposer",
        "description": "Transforms a product objective into phased engineering milestones, acceptance criteria, and risk mitigations.",
        "category": "productivity",
        "version": "1.0.4",
        "author": "Xyvora Workspace",
        "permissions": ["compute:text"],
        "compatibility": "xyvora-platform >= 2.0",
        "config_schema": {"sprint_days": {"type": "integer", "default": 14}},
        "input_schema": {
            "type": "object",
            "required": ["objective"],
            "properties": {"objective": {"type": "string", "description": "Feature or project goal"}},
        },
        "output_schema": {"type": "object", "properties": {"milestones": {"type": "array"}}},
    },
]

_UNIT_FACTORS_TO_BASE: Dict[str, tuple[str, float]] = {
    # length -> meters
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "cm": ("length", 0.01),
    "mm": ("length", 0.001),
    "mi": ("length", 1609.344),
    "yd": ("length", 0.9144),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    # mass -> kg
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 1e-6),
    "lb": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    # data -> bytes
    "b": ("data", 1.0),
    "kb": ("data", 1024.0),
    "mb": ("data", 1024.0**2),
    "gb": ("data", 1024.0**3),
    "tb": ("data", 1024.0**4),
    # speed -> m/s
    "m/s": ("speed", 1.0),
    "km/h": ("speed", 1.0 / 3.6),
    "mph": ("speed", 0.44704),
    "kn": ("speed", 0.514444),
    # energy -> joules
    "j": ("energy", 1.0),
    "kj": ("energy", 1000.0),
    "cal": ("energy", 4.184),
    "kcal": ("energy", 4184.0),
    "wh": ("energy", 3600.0),
    "kwh": ("energy", 3_600_000.0),
}


class ExtensionMarketplaceEngine:
    """
    Modular Extension & Plugin Architecture:
      - Strictly permission-scoped declarative extensions
      - Never executes arbitrary third-party server code in the backend process
      - Supports per-user installation, removal, configuration, and execution
    """

    def list_extensions(self, user_id: str = DEFAULT_USER_ID, category: Optional[str] = None) -> List[Dict[str, Any]]:
        with _db_lock:
            conn = get_conn()
            try:
                inst_rows = conn.execute(
                    "SELECT extension_id, enabled, config_json, installed_at FROM installed_extensions WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
                custom_rows = conn.execute(
                    "SELECT * FROM custom_extensions WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            finally:
                conn.close()

        installed_map = {
            r["extension_id"]: {
                "enabled": bool(r["enabled"]),
                "config": json.loads(r["config_json"] or "{}"),
                "installed_at": r["installed_at"],
            }
            for r in inst_rows
        }

        # Seed default core extensions as installed if user has none installed yet
        default_preinstalled = {"ext_unit_converter", "ext_regex_debugger", "ext_json_validator"}
        has_any_record = len(installed_map) > 0

        catalog: List[Dict[str, Any]] = []
        for ext in BUILTIN_EXTENSIONS:
            if category and ext["category"] != category:
                continue
            inst = installed_map.get(ext["id"])
            is_installed = (inst is not None and inst["enabled"]) if has_any_record else (ext["id"] in default_preinstalled)
            catalog.append(
                {
                    **ext,
                    "builtin": True,
                    "installed": is_installed,
                    "config": inst["config"] if inst else {},
                }
            )

        for cr in custom_rows:
            if category and cr["category"] != category:
                continue
            ext_id = f"custom_{cr['slug']}"
            inst = installed_map.get(ext_id)
            catalog.append(
                {
                    "id": ext_id,
                    "custom_db_id": cr["id"],
                    "name": cr["name"],
                    "description": cr["description"],
                    "category": cr["category"],
                    "version": cr["version"],
                    "author": "User Declarative Extension",
                    "permissions": json.loads(cr["permissions_json"] or "[]"),
                    "compatibility": "xyvora-declarative-v2",
                    "config_schema": {},
                    "input_schema": json.loads(cr["input_schema_json"] or '{"type":"object","properties":{"input":{"type":"string"}}}'),
                    "output_schema": {"type": "object"},
                    "prompt_template": cr["prompt_template"],
                    "builtin": False,
                    "installed": True if inst is None else inst["enabled"],
                    "config": inst["config"] if inst else {},
                }
            )
        return catalog

    def install_extension(
        self,
        extension_id: str,
        user_id: str = DEFAULT_USER_ID,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO installed_extensions (id, user_id, extension_id, enabled, config_json, installed_at)
                    VALUES (?, ?, ?, 1, ?, ?)
                    ON CONFLICT(user_id, extension_id) DO UPDATE SET
                        enabled = 1,
                        config_json = COALESCE(?, installed_extensions.config_json),
                        installed_at = excluded.installed_at
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id,
                        extension_id,
                        json.dumps(config or {}),
                        now,
                        json.dumps(config) if config is not None else None,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        matches = [e for e in self.list_extensions(user_id) if e["id"] == extension_id]
        return matches[0] if matches else {"id": extension_id, "installed": True}

    def uninstall_extension(self, extension_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        now = time.time()
        with _db_lock:
            conn = get_conn()
            try:
                if extension_id.startswith("custom_"):
                    slug = extension_id[len("custom_") :]
                    conn.execute("DELETE FROM custom_extensions WHERE user_id = ? AND slug = ?", (user_id, slug))
                conn.execute(
                    """
                    INSERT INTO installed_extensions (id, user_id, extension_id, enabled, config_json, installed_at)
                    VALUES (?, ?, ?, 0, '{}', ?)
                    ON CONFLICT(user_id, extension_id) DO UPDATE SET enabled = 0
                    """,
                    (str(uuid.uuid4()), user_id, extension_id, now),
                )
                conn.commit()
            finally:
                conn.close()
        return {"id": extension_id, "installed": False}

    def create_custom_extension(
        self,
        user_id: str,
        name: str,
        description: str,
        category: str,
        prompt_template: str,
        permissions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        safety = check_safety_policy(f"{name}\n{description}\n{prompt_template}")
        if safety:
            raise ValueError(safety["message"])

        safe_perms = [p for p in (permissions or ["compute:text"]) if p in ALLOWED_EXTENSION_PERMISSIONS]
        slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")[:40] or f"ext_{int(time.time())}"
        now = time.time()
        ext_db_id = str(uuid.uuid4())
        input_schema = {
            "type": "object",
            "required": ["input"],
            "properties": {"input": {"type": "string", "description": "Input text for this extension"}},
        }
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO custom_extensions
                    (id, user_id, slug, name, description, category, version, permissions_json, input_schema_json, prompt_template, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, '1.0.0', ?, ?, ?, ?)
                    ON CONFLICT(user_id, slug) DO UPDATE SET
                        name = excluded.name,
                        description = excluded.description,
                        category = excluded.category,
                        permissions_json = excluded.permissions_json,
                        prompt_template = excluded.prompt_template
                    """,
                    (
                        ext_db_id,
                        user_id,
                        slug,
                        name.strip()[:80],
                        description.strip()[:300],
                        category.strip().lower() or "productivity",
                        json.dumps(safe_perms),
                        json.dumps(input_schema),
                        prompt_template.strip()[:2000],
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return self.install_extension(f"custom_{slug}", user_id=user_id)

    def execute_extension(
        self,
        extension_id: str,
        inputs: Dict[str, Any],
        user_id: str = DEFAULT_USER_ID,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        exts = {e["id"]: e for e in self.list_extensions(user_id)}
        ext = exts.get(extension_id)
        if not ext:
            return {"ok": False, "error": f"Extension '{extension_id}' not found."}
        if not ext.get("installed"):
            return {"ok": False, "error": f"Extension '{ext['name']}' is not currently installed/enabled."}

        cfg = ext.get("config") or {}

        try:
            if extension_id == "ext_unit_converter":
                val = float(inputs.get("value", 0))
                u_from = str(inputs.get("from_unit", "")).strip().lower()
                u_to = str(inputs.get("to_unit", "")).strip().lower()
                digits = int(cfg.get("precision_digits", 5))
                if u_from in ("c", "f", "k") and u_to in ("c", "f", "k"):
                    c_val = val if u_from == "c" else ((val - 32.0) * 5.0 / 9.0 if u_from == "f" else val - 273.15)
                    out_val = c_val if u_to == "c" else (c_val * 9.0 / 5.0 + 32.0 if u_to == "f" else c_val + 273.15)
                    res = {"result": round(out_val, digits), "formatted": f"{val} °{u_from.upper()} = {round(out_val, digits)} °{u_to.upper()}"}
                elif u_from in _UNIT_FACTORS_TO_BASE and u_to in _UNIT_FACTORS_TO_BASE:
                    dim1, f1 = _UNIT_FACTORS_TO_BASE[u_from]
                    dim2, f2 = _UNIT_FACTORS_TO_BASE[u_to]
                    if dim1 != dim2:
                        return {"ok": False, "error": f"Incompatible dimensions: '{u_from}' ({dim1}) vs '{u_to}' ({dim2})."}
                    out_val = (val * f1) / f2
                    res = {"result": round(out_val, digits), "dimension": dim1, "formatted": f"{val} {u_from} = {round(out_val, digits)} {u_to}"}
                else:
                    return {"ok": False, "error": f"Unsupported unit pair '{u_from}' -> '{u_to}'. Supported: {', '.join(sorted(_UNIT_FACTORS_TO_BASE.keys()))}, c, f, k."}

            elif extension_id == "ext_regex_debugger":
                pattern = str(inputs.get("pattern", ""))
                sample = str(inputs.get("sample_text", ""))[:8000]
                if len(pattern) > 300:
                    return {"ok": False, "error": "Pattern exceeds 300 characters."}
                flags = re.IGNORECASE if cfg.get("case_insensitive") else 0
                compiled = re.compile(pattern, flags)
                matches = []
                for m in list(compiled.finditer(sample))[:50]:
                    matches.append({"match": m.group(0), "span": [m.start(), m.end()], "groups": list(m.groups()), "named_groups": m.groupdict()})
                tokens = []
                if "^" in pattern:
                    tokens.append("`^` anchors start of string/line")
                if "$" in pattern:
                    tokens.append("`$` anchors end of string/line")
                if "\\d" in pattern:
                    tokens.append("`\\d` matches ASCII/Unicode digits")
                if "\\w" in pattern:
                    tokens.append("`\\w` matches word characters [A-Za-z0-9_]")
                if "(?:" in pattern:
                    tokens.append("`(?:...)` non-capturing group")
                res = {"match_count": len(matches), "matches": matches, "pattern_notes": tokens}

            elif extension_id == "ext_json_validator":
                raw = str(inputs.get("json_text", ""))
                parsed = json.loads(raw)
                req_keys = inputs.get("required_keys") or []
                missing = [k for k in req_keys if isinstance(parsed, dict) and k not in parsed]
                ts_lines = ["export interface PayloadRoot {"]
                sample_obj = parsed[0] if (isinstance(parsed, list) and parsed and isinstance(parsed[0], dict)) else (parsed if isinstance(parsed, dict) else {})
                for k, v in list(sample_obj.items())[:30]:
                    t_name = "boolean" if isinstance(v, bool) else ("number" if isinstance(v, (int, float)) else ("string[]" if isinstance(v, list) else ("Record<string, unknown>" if isinstance(v, dict) else "string")))
                    ts_lines.append(f"  {k}: {t_name};")
                ts_lines.append("}")
                res = {
                    "valid": len(missing) == 0,
                    "root_type": type(parsed).__name__,
                    "missing_required_keys": missing,
                    "typescript_interface": "\n".join(ts_lines),
                }

            elif extension_id == "ext_sql_architect":
                sql = str(inputs.get("sql", "")).strip()
                up = sql.upper()
                warnings = []
                indexes = []
                if "SELECT *" in up:
                    warnings.append("Avoid `SELECT *` in production queries; enumerate explicit columns to reduce I/O and enable covering indexes.")
                if ("UPDATE " in up or "DELETE FROM" in up) and "WHERE " not in up:
                    warnings.append("CRITICAL: `UPDATE` or `DELETE` statement has no `WHERE` clause and will affect the entire table.")
                if re.search(r"['\"]\s*\+\s*\w+|\$\{", sql):
                    warnings.append("CRITICAL: Detected string interpolation inside SQL; use parameterized queries (`?` or `$1`) to prevent SQL injection.")
                where_cols = re.findall(r"\bWHERE\s+([a-zA-Z_][a-zA-Z0-9_.]*)", sql, flags=re.IGNORECASE)
                order_cols = re.findall(r"\bORDER\s+BY\s+([a-zA-Z_][a-zA-Z0-9_.]*)", sql, flags=re.IGNORECASE)
                if where_cols or order_cols:
                    indexes.append(f"Consider composite B-Tree index on ({', '.join(dict.fromkeys(where_cols + order_cols))}).")
                res = {"safe": not any("CRITICAL" in w for w in warnings), "warnings": warnings, "index_suggestions": indexes}

            elif extension_id == "ext_csv_profiler":
                raw_csv = str(inputs.get("csv_text", "")).strip()
                reader = list(csv.DictReader(io.StringIO(raw_csv)))
                if not reader:
                    return {"ok": False, "error": "CSV must include a header row and at least 1 data row."}
                cols_report = []
                for col in (reader[0].keys() or []):
                    vals = [r.get(col, "") for r in reader]
                    non_empty = [v.strip() for v in vals if v is not None and str(v).strip() != ""]
                    nums = []
                    for v in non_empty:
                        try:
                            nums.append(float(v.replace(",", "")))
                        except ValueError:
                            pass
                    if nums and len(nums) >= max(1, int(0.7 * len(non_empty))):
                        mean = sum(nums) / len(nums)
                        std = (sum((x - mean) ** 2 for x in nums) / len(nums)) ** 0.5
                        cols_report.append({
                            "column": col,
                            "type": "numeric",
                            "missing": len(vals) - len(non_empty),
                            "min": round(min(nums), 4),
                            "max": round(max(nums), 4),
                            "mean": round(mean, 4),
                            "std": round(std, 4),
                        })
                    else:
                        cols_report.append({
                            "column": col,
                            "type": "categorical",
                            "missing": len(vals) - len(non_empty),
                            "unique_values": len(set(non_empty)),
                        })
                res = {"row_count": len(reader), "columns": cols_report}

            elif extension_id == "ext_research_credibility":
                url = str(inputs.get("url", "")).strip()
                title = str(inputs.get("title", "Untitled Source")).strip()
                pub = str(inputs.get("published_date", "n.d.")).strip() or "n.d."
                host = (urlparse(url).hostname or "").lower()
                if host.endswith((".gov", ".edu")) or any(d in host for d in ("arxiv.org", "nature.com", "ieee.org", "acm.org", "nih.gov")):
                    tier, score = "Tier 1 — Academic / Institutional Primary Source", 92
                elif any(d in host for d in ("wikipedia.org", "github.com", "mozilla.org", "python.org", "w3.org", "reuters.com")):
                    tier, score = "Tier 2 — Curated Technical / Reference Source", 82
                elif url.startswith("https://"):
                    tier, score = "Tier 3 — General Web Source (Verify Claims Independently)", 60
                else:
                    tier, score = "Unverified / Non-HTTPS Source", 35
                apa = f"{host or 'Web Source'}. ({pub}). {title}. Retrieved from {url}"
                res = {"domain": host, "credibility_tier": tier, "credibility_score": score, "apa_citation": apa}

            elif extension_id == "ext_security_headers":
                raw_h = str(inputs.get("headers_text", ""))
                parsed_h = {}
                for line in raw_h.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parsed_h[k.strip().lower()] = v.strip()
                checks = [
                    ("content-security-policy", "Add a strict `Content-Security-Policy` header to mitigate XSS."),
                    ("strict-transport-security", "Add `Strict-Transport-Security: max-age=63072000; includeSubDomains` for HTTPS enforcement."),
                    ("x-content-type-options", "Add `X-Content-Type-Options: nosniff` to prevent MIME sniffing."),
                    ("x-frame-options", "Add `X-Frame-Options: DENY` or `SAMEORIGIN` to prevent clickjacking."),
                    ("referrer-policy", "Add `Referrer-Policy: strict-origin-when-cross-origin`."),
                ]
                missing_recs = [msg for hdr, msg in checks if hdr not in parsed_h]
                if parsed_h.get("access-control-allow-origin") == "*" and parsed_h.get("access-control-allow-credentials", "").lower() == "true":
                    missing_recs.append("CRITICAL CORS misconfiguration: wildcard `*` origin combined with `Access-Control-Allow-Credentials: true`.")
                score = max(0, 100 - len(missing_recs) * 18)
                res = {"score": score, "present_headers": list(parsed_h.keys()), "recommendations": missing_recs}

            elif extension_id == "ext_sprint_planner":
                obj = str(inputs.get("objective", "")).strip()
                days = int(cfg.get("sprint_days", 14))
                res = {
                    "objective": obj,
                    "sprint_days": days,
                    "milestones": [
                        {"phase": "1. Architecture & Schema", "days": f"Day 1–{max(2, days//4)}", "deliverable": f"Data model, auth boundaries, and API contracts for {obj}"},
                        {"phase": "2. Core Engine & Workflows", "days": f"Day {max(2, days//4)+1}–{max(5, (days*3)//4)}", "deliverable": f"End-to-end backend + UI implementation of {obj}"},
                        {"phase": "3. Hardening & QA", "days": f"Day {max(5, (days*3)//4)+1}–{days}", "deliverable": "Automated pytest suite, edge-case validation, and production telemetry"},
                    ],
                }

            else:
                # Declarative custom extension
                user_in = str(inputs.get("input", "")).strip()
                tmpl = str(ext.get("prompt_template") or "{input}")
                rendered = tmpl.replace("{input}", user_in)
                res = {"declarative_output": rendered, "extension_name": ext["name"]}

            return {
                "ok": True,
                "extension_id": extension_id,
                "name": ext["name"],
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "output": res,
            }
        except Exception as exc:
            return {"ok": False, "extension_id": extension_id, "error": str(exc)}
