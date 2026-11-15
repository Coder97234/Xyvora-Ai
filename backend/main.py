import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import (
    DEFAULT_USER_ID,
    ensure_user_initialized,
    export_user_workspace,
    get_admin_analytics,
    get_conn,
    get_storage_diagnostics,
    get_user_settings,
    get_user_usage_summary,
    init_db,
    log_security_event,
    record_usage_event,
    update_user_settings,
    upsert_provider_config,
    _db_lock,
)
from security import (
    check_safety_policy,
    generate_session_token,
    get_allowed_origins,
    hash_password,
    hash_session_token,
    rate_limiter,
    validate_email,
    validate_password_strength,
    verify_password,
)
from engine.cortex import PERSONA_PROFILES, XyvoraCortex
from engine.multimodal import MAX_UPLOAD_BYTES, inspect_image_bytes, parse_document_bytes

init_db()
cortex = XyvoraCortex()

app = FastAPI(title="Xyvora AI Intelligence Platform", version="2.0.0")


@app.middleware("http")
async def normalize_vercel_api_prefix(request: Request, call_next):
    path = request.scope.get("path", "")
    if path in ("/api/index.py", "/api/index"):
        matched = request.headers.get("x-matched-path") or request.url.path
        if matched and matched not in ("/api/index.py", "/api/index"):
            request.scope["path"] = matched if matched.startswith("/api") else f"/api{matched}"
    elif path and not path.startswith("/api/") and path != "/api":
        if os.environ.get("VERCEL"):
            request.scope["path"] = f"/api{path}" if path.startswith("/") else f"/api/{path}"
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=r"^https://([a-zA-Z0-9\-]+\.)*(e2b\.app|vercel\.app|xyvora\.net)$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Requested-With"],
)


# ----------------- Auth & Rate Limit Dependencies -----------------

SESSION_TTL_SEC = 30 * 86400  # 30 days


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def _rate_check(request: Request, scope: str, max_req: int, window_sec: float = 60.0) -> None:
    ip = _client_ip(request)
    allowed, retry_after = rate_limiter.check(f"{scope}:{ip}", max_req, window_sec)
    if not allowed:
        log_security_event("rate_limit_exceeded", f"scope={scope}", ip_address=ip, severity="warn", resource=scope)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {scope}. Please retry in {retry_after}s.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def _user_public_dict(row: Any) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "username": row["username"],
        "display_name": row["display_name"],
        "role": row["role"],
        "is_guest": bool(row["is_guest"]),
        "avatar_url": row["avatar_url"] if "avatar_url" in row.keys() else "",
        "bio": row["bio"] if "bio" in row.keys() else "",
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    xyvora_session: Optional[str] = Cookie(default=None),
) -> Dict[str, Any]:
    raw_token = ""
    explicit_bearer = False
    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization[7:].strip()
        explicit_bearer = True
    elif xyvora_session:
        raw_token = xyvora_session.strip()

    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            if raw_token:
                thash = hash_session_token(raw_token)
                sess = conn.execute(
                    "SELECT user_id, expires_at FROM auth_sessions WHERE token_hash = ?",
                    (thash,),
                ).fetchone()
                if sess and sess["expires_at"] > now:
                    u_row = conn.execute("SELECT * FROM users WHERE id = ?", (sess["user_id"],)).fetchone()
                    if u_row:
                        return _user_public_dict(u_row)
                if explicit_bearer:
                    raise HTTPException(status_code=401, detail="Invalid or expired authentication token.")

            # Default local workspace user when unauthenticated (preserves zero-friction local mode)
            default_row = conn.execute("SELECT * FROM users WHERE id = ?", (DEFAULT_USER_ID,)).fetchone()
            if not default_row:
                init_db()
                default_row = conn.execute("SELECT * FROM users WHERE id = ?", (DEFAULT_USER_ID,)).fetchone()
            return _user_public_dict(default_row)
        finally:
            conn.close()


def require_admin_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("role") not in ("admin", "owner"):
        log_security_event("admin_access_denied", "Non-admin attempted admin endpoint", user_id=user["id"], severity="warn", resource="/api/admin")
        raise HTTPException(status_code=403, detail="Admin privileges are required to view platform analytics.")
    return user


# ----------------- Pydantic Schemas -----------------

class RegisterReq(BaseModel):
    email: str
    username: str
    password: str
    display_name: Optional[str] = ""


class LoginReq(BaseModel):
    identifier: str  # email or username
    password: str


class ProfilePatchReq(BaseModel):
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


class PasswordChangeReq(BaseModel):
    current_password: str
    new_password: str


class ProviderUpdateReq(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    enabled: Optional[bool] = None


class CreateProjectReq(BaseModel):
    name: str
    description: Optional[str] = ""
    instructions: Optional[str] = ""
    color: Optional[str] = "#22d3ee"


class PatchProjectReq(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    color: Optional[str] = None


class SaveProjectFileReq(BaseModel):
    path: str
    content: str
    language: Optional[str] = ""
    summary: Optional[str] = "Updated in Coding Studio"


class RevertFileReq(BaseModel):
    revision_id: str


class CodingAssistReq(BaseModel):
    project_id: Optional[str] = None
    action: str = "review"  # explain, debug, review, refactor, generate_patch
    prompt: str = ""
    path: Optional[str] = "main.py"
    language: Optional[str] = "python"
    content: Optional[str] = ""


class CreateSessionReq(BaseModel):
    title: Optional[str] = "New Neural Session"
    persona: Optional[str] = None
    project_id: Optional[str] = None


class PatchSessionReq(BaseModel):
    title: Optional[str] = None
    pinned: Optional[int] = None
    project_id: Optional[str] = None


class ChatStreamReq(BaseModel):
    session_id: str
    message: str
    project_id: Optional[str] = None
    attachment_ids: Optional[List[str]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    client_memories: Optional[List[Dict[str, Any]]] = None


class EditMessageReq(BaseModel):
    content: str


class ClientVaultSyncReq(BaseModel):
    memories: Optional[List[Dict[str, Any]]] = None
    skills: Optional[List[Dict[str, Any]]] = None
    settings: Optional[Dict[str, Any]] = None


class FeedbackReq(BaseModel):
    feedback: int


class MemoryRecallReq(BaseModel):
    query: str
    threshold: Optional[float] = 0.18
    project_id: Optional[str] = None


class CreateMemoryReq(BaseModel):
    subject: str
    relation: str
    object: str
    category: Optional[str] = "fact"
    pinned: Optional[int] = 0
    confidence: Optional[float] = 0.94
    ttl_days: Optional[int] = 0
    project_id: Optional[str] = None


class PatchMemoryReq(BaseModel):
    subject: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    pinned: Optional[int] = None
    importance: Optional[float] = None
    verified: Optional[int] = None


class ResearchQueryReq(BaseModel):
    query: str
    mode: Optional[str] = "quick"
    max_results: Optional[int] = 5
    follow_links: Optional[bool] = False
    project_id: Optional[str] = None


class StartAgentReq(BaseModel):
    goal: str
    session_id: Optional[str] = None
    project_id: Optional[str] = None


class AgentApproveReq(BaseModel):
    approved: bool


class CustomPersonaReq(BaseModel):
    name: str
    description: Optional[str] = ""
    system_instructions: str
    tone: Optional[str] = "balanced"
    creativity: Optional[float] = 0.65
    verbosity: Optional[str] = "balanced"
    domain_focus: Optional[str] = ""


class CustomExtensionReq(BaseModel):
    name: str
    description: str
    category: Optional[str] = "productivity"
    prompt_template: str
    permissions: Optional[List[str]] = Field(default_factory=lambda: ["compute:text"])


class RunExtensionReq(BaseModel):
    inputs: Dict[str, Any]


class NeuralPredictReq(BaseModel):
    text: str


class TrainingSampleReq(BaseModel):
    text: str
    intent: str


class DatasetImportReq(BaseModel):
    samples: List[Dict[str, str]]
    replace_existing: Optional[bool] = False


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
    collection: Optional[str] = "general"
    project_id: Optional[str] = None


class DocSearchReq(BaseModel):
    query: str
    collection: Optional[str] = None
    project_id: Optional[str] = None
    top_k: Optional[int] = 5


class CreateTaskReq(BaseModel):
    title: str
    description: Optional[str] = ""
    priority: Optional[str] = "medium"
    due_note: Optional[str] = ""
    project_id: Optional[str] = None


class PatchTaskReq(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


class ImageGenReq(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"


class VoiceTTSReq(BaseModel):
    text: str
    voice: Optional[str] = "alloy"


# ----------------- Health & System Telemetry (NEVER leaks API keys) -----------------

@app.get("/api/health")
def get_health(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            mem_count = conn.execute("SELECT COUNT(*) AS c FROM semantic_memories WHERE user_id = ?", (uid,)).fetchone()["c"]
            epi_count = conn.execute("SELECT COUNT(*) AS c FROM episodic_memories WHERE user_id = ?", (uid,)).fetchone()["c"]
            doc_count = conn.execute("SELECT COUNT(*) AS c FROM documents WHERE user_id = ?", (uid,)).fetchone()["c"]
            task_open = conn.execute("SELECT COUNT(*) AS c FROM tasks WHERE user_id = ? AND status != 'done'", (uid,)).fetchone()["c"]
            skill_count = conn.execute("SELECT COUNT(*) AS c FROM custom_skills WHERE user_id = ?", (uid,)).fetchone()["c"]
            proj_count = conn.execute("SELECT COUNT(*) AS c FROM projects WHERE user_id = ?", (uid,)).fetchone()["c"]
        finally:
            conn.close()

    n_stats = cortex.neural.get_stats()
    active_prov = cortex.providers.resolve_active_provider(uid)
    return {
        "status": "online",
        "engine": "Xyvora Intelligence Platform v2.0",
        "active_provider": active_prov["provider_id"],
        "active_model": active_prov["model"],
        "semantic_memories": mem_count,
        "episodic_memories": epi_count,
        "documents": doc_count,
        "open_tasks": task_open,
        "custom_skills": skill_count,
        "projects": proj_count,
        "neural_accuracy": n_stats["training_accuracy"],
        "neural_cv_accuracy": n_stats.get("cv_accuracy", n_stats["training_accuracy"]),
        "neural_params": n_stats["parameter_count"],
        "storage": get_storage_diagnostics(),
        "user": user,
        "settings": get_user_settings(uid),
    }


# ----------------- Authentication & User Account System -----------------

def _set_auth_cookie(response: Response, token: str) -> None:
    is_https = bool(os.environ.get("VERCEL"))
    response.set_cookie(
        key="xyvora_session",
        value=token,
        max_age=SESSION_TTL_SEC,
        httponly=True,
        samesite="lax",
        secure=is_https,
        path="/",
    )


@app.post("/api/auth/register")
def register_account(req: RegisterReq, request: Request, response: Response) -> Dict[str, Any]:
    _rate_check(request, "auth_register", max_req=10, window_sec=60.0)
    email = req.email.strip().lower()
    username = re.sub(r"[^a-zA-Z0-9_\-]", "", req.username.strip().lower())
    if not validate_email(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if len(username) < 3 or len(username) > 32:
        raise HTTPException(status_code=400, detail="Username must be 3–32 alphanumeric characters.")
    pw_ok, pw_msg = validate_password_strength(req.password)
    if not pw_ok:
        raise HTTPException(status_code=400, detail=pw_msg)

    user_id = f"usr_{uuid.uuid4().hex[:16]}"
    now = time.time()
    display_name = (req.display_name or username).strip()[:60]
    pw_hash = hash_password(req.password)
    raw_token = generate_session_token()
    tok_hash = hash_session_token(raw_token)

    with _db_lock:
        conn = get_conn()
        try:
            dup = conn.execute(
                "SELECT id FROM users WHERE lower(email) = ? OR lower(username) = ?",
                (email, username),
            ).fetchone()
            if dup:
                raise HTTPException(status_code=409, detail="An account with that email or username already exists.")

            # First human registrant becomes admin; subsequent registrants are regular 'user' accounts
            human_admins = conn.execute("SELECT COUNT(*) AS c FROM users WHERE is_guest = 0 AND role = 'admin'").fetchone()["c"]
            assigned_role = "admin" if human_admins == 0 else "user"

            conn.execute(
                """
                INSERT INTO users (id, email, username, display_name, password_hash, role, is_guest, avatar_url, bio, email_verified, created_at, last_login_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, '', '', 1, ?, ?)
                """,
                (user_id, email, username, display_name, pw_hash, assigned_role, now, now),
            )
            conn.execute(
                """
                INSERT INTO auth_sessions (token_hash, user_id, ip_address, user_agent, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tok_hash, user_id, _client_ip(request), (request.headers.get("user-agent") or "")[:200], now, now + SESSION_TTL_SEC),
            )
            conn.commit()
        finally:
            conn.close()

    ensure_user_initialized(user_id)
    log_security_event("user_registered", f"username={username} role={assigned_role}", user_id=user_id, ip_address=_client_ip(request), resource="/api/auth/register")
    _set_auth_cookie(response, raw_token)

    with _db_lock:
        conn = get_conn()
        try:
            u_row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            pub_user = _user_public_dict(u_row)
        finally:
            conn.close()

    return {"token": raw_token, "user": pub_user, "settings": get_user_settings(user_id)}


@app.post("/api/auth/login")
def login_account(req: LoginReq, request: Request, response: Response) -> Dict[str, Any]:
    _rate_check(request, "auth_login", max_req=12, window_sec=60.0)
    ident = req.identifier.strip().lower()
    now = time.time()

    with _db_lock:
        conn = get_conn()
        try:
            u_row = conn.execute(
                "SELECT * FROM users WHERE (lower(email) = ? OR lower(username) = ?) AND is_guest = 0",
                (ident, ident),
            ).fetchone()
            if not u_row or not verify_password(req.password, u_row["password_hash"]):
                log_security_event("auth_login_failed", f"identifier={ident[:40]}", ip_address=_client_ip(request), severity="warn", resource="/api/auth/login")
                raise HTTPException(status_code=401, detail="Invalid email/username or password.")

            raw_token = generate_session_token()
            tok_hash = hash_session_token(raw_token)
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, u_row["id"]))
            conn.execute(
                """
                INSERT INTO auth_sessions (token_hash, user_id, ip_address, user_agent, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tok_hash, u_row["id"], _client_ip(request), (request.headers.get("user-agent") or "")[:200], now, now + SESSION_TTL_SEC),
            )
            conn.commit()
            pub_user = _user_public_dict(u_row)
        finally:
            conn.close()

    ensure_user_initialized(pub_user["id"])
    log_security_event("user_logged_in", f"username={pub_user['username']}", user_id=pub_user["id"], ip_address=_client_ip(request), resource="/api/auth/login")
    _set_auth_cookie(response, raw_token)
    return {"token": raw_token, "user": pub_user, "settings": get_user_settings(pub_user["id"])}


@app.post("/api/auth/logout")
def logout_account(
    response: Response,
    authorization: Optional[str] = Header(default=None),
    xyvora_session: Optional[str] = Cookie(default=None),
) -> Dict[str, Any]:
    raw_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization[7:].strip()
    elif xyvora_session:
        raw_token = xyvora_session.strip()

    if raw_token:
        thash = hash_session_token(raw_token)
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (thash,))
                conn.commit()
            finally:
                conn.close()
    response.delete_cookie("xyvora_session", path="/")
    return {"logged_out": True}


@app.get("/api/auth/me")
def get_me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return {
        "user": user,
        "usage": get_user_usage_summary(user["id"]),
        "settings": get_user_settings(user["id"]),
    }


@app.patch("/api/auth/profile")
def update_profile(req: ProfilePatchReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
            d_name = req.display_name.strip()[:60] if req.display_name is not None else row["display_name"]
            av_url = req.avatar_url.strip()[:400] if req.avatar_url is not None else (row["avatar_url"] or "")
            bio = req.bio.strip()[:400] if req.bio is not None else (row["bio"] or "")
            conn.execute(
                "UPDATE users SET display_name = ?, avatar_url = ?, bio = ? WHERE id = ?",
                (d_name or row["username"], av_url, bio, uid),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
            return {"user": _user_public_dict(updated)}
        finally:
            conn.close()


@app.post("/api/auth/change-password")
def change_password(req: PasswordChangeReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("is_guest"):
        raise HTTPException(status_code=400, detail="Create a registered account first before changing a password.")
    pw_ok, pw_msg = validate_password_strength(req.new_password)
    if not pw_ok:
        raise HTTPException(status_code=400, detail=pw_msg)
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (uid,)).fetchone()
            if not row or not verify_password(req.current_password, row["password_hash"]):
                raise HTTPException(status_code=401, detail="Current password is incorrect.")
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(req.new_password), uid))
            conn.commit()
        finally:
            conn.close()
    log_security_event("password_changed", "User updated password", user_id=uid, resource="/api/auth/change-password")
    return {"ok": True}


@app.post("/api/auth/refresh")
def refresh_session_token(
    request: Request,
    response: Response,
    authorization: Optional[str] = Header(default=None),
    xyvora_session: Optional[str] = Cookie(default=None),
) -> Dict[str, Any]:
    raw_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization[7:].strip()
    elif xyvora_session:
        raw_token = xyvora_session.strip()
    if not raw_token:
        raise HTTPException(status_code=401, detail="Active session token required to refresh.")

    old_hash = hash_session_token(raw_token)
    now = time.time()
    new_raw = generate_session_token()
    new_hash = hash_session_token(new_raw)

    with _db_lock:
        conn = get_conn()
        try:
            sess = conn.execute(
                "SELECT * FROM auth_sessions WHERE token_hash = ? AND expires_at > ?",
                (old_hash, now),
            ).fetchone()
            if not sess:
                raise HTTPException(status_code=401, detail="Session expired or invalid.")
            uid = sess["user_id"]
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (old_hash,))
            conn.execute(
                """
                INSERT INTO auth_sessions (token_hash, user_id, ip_address, user_agent, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_hash, uid, _client_ip(request), (request.headers.get("user-agent") or "")[:200], now, now + SESSION_TTL_SEC),
            )
            u_row = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
            conn.commit()
            pub_user = _user_public_dict(u_row)
        finally:
            conn.close()

    _set_auth_cookie(response, new_raw)
    return {"token": new_raw, "user": pub_user}


@app.post("/api/auth/verify-email/request")
def request_email_verification(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user.get("is_guest"):
        raise HTTPException(status_code=400, detail="Register an account first.")
    raw_code = f"xyv-verify-{uuid.uuid4().hex[:16]}"
    tok_hash = hash_session_token(raw_code)
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, purpose, created_at, expires_at, used) VALUES (?, ?, 'verify_email', ?, ?, 0)",
                (tok_hash, user["id"], now, now + 900.0),
            )
            conn.commit()
        finally:
            conn.close()
    smtp_configured = bool(os.environ.get("SMTP_HOST"))
    return {
        "ok": True,
        "delivery_mode": "smtp" if smtp_configured else "self_hosted_dev_token",
        "verification_token": None if smtp_configured else raw_code,
        "expires_in_sec": 900,
    }


@app.post("/api/auth/verify-email/confirm")
def confirm_email_verification(payload: Dict[str, Any]) -> Dict[str, Any]:
    token = str(payload.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Verification token is required.")
    tok_hash = hash_session_token(token)
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM auth_tokens WHERE token_hash = ? AND purpose = 'verify_email' AND used = 0 AND expires_at > ?",
                (tok_hash, now),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
            conn.execute("UPDATE auth_tokens SET used = 1 WHERE token_hash = ?", (tok_hash,))
            conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (row["user_id"],))
            conn.commit()
        finally:
            conn.close()
    return {"verified": True}


@app.post("/api/auth/password-reset/request")
def request_password_reset(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
    _rate_check(request, "pw_reset", max_req=8, window_sec=60.0)
    email = str(payload.get("email") or "").strip().lower()
    if not validate_email(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    now = time.time()
    raw_code = f"xyv-reset-{uuid.uuid4().hex[:20]}"
    tok_hash = hash_session_token(raw_code)
    issued_dev_token = None
    smtp_configured = bool(os.environ.get("SMTP_HOST"))

    with _db_lock:
        conn = get_conn()
        try:
            u_row = conn.execute("SELECT id FROM users WHERE lower(email) = ? AND is_guest = 0", (email,)).fetchone()
            if u_row:
                conn.execute(
                    "INSERT INTO auth_tokens (token_hash, user_id, purpose, created_at, expires_at, used) VALUES (?, ?, 'password_reset', ?, ?, 0)",
                    (tok_hash, u_row["id"], now, now + 900.0),
                )
                conn.commit()
                if not smtp_configured:
                    issued_dev_token = raw_code
        finally:
            conn.close()

    return {
        "ok": True,
        "delivery_mode": "smtp" if smtp_configured else "self_hosted_dev_token",
        "reset_token": issued_dev_token,
        "message": "If an account matches that email, a 15-minute password reset token has been generated.",
    }


@app.post("/api/auth/password-reset/confirm")
def confirm_password_reset(payload: Dict[str, Any]) -> Dict[str, Any]:
    token = str(payload.get("token") or "").strip()
    new_pw = str(payload.get("new_password") or "")
    pw_ok, pw_msg = validate_password_strength(new_pw)
    if not pw_ok:
        raise HTTPException(status_code=400, detail=pw_msg)
    tok_hash = hash_session_token(token)
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM auth_tokens WHERE token_hash = ? AND purpose = 'password_reset' AND used = 0 AND expires_at > ?",
                (tok_hash, now),
            ).fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid or expired password reset token.")
            conn.execute("UPDATE auth_tokens SET used = 1 WHERE token_hash = ?", (tok_hash,))
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new_pw), row["user_id"]))
            conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (row["user_id"],))
            conn.commit()
        finally:
            conn.close()
    log_security_event("password_reset_completed", "Password reset via token", resource="/api/auth/password-reset/confirm")
    return {"reset": True}


@app.delete("/api/auth/account")
def delete_user_account(response: Response, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    if uid == DEFAULT_USER_ID or user.get("is_guest"):
        raise HTTPException(status_code=400, detail="The built-in local guest workspace cannot be deleted.")
    with _db_lock:
        conn = get_conn()
        try:
            for tbl in ("messages", "sessions", "semantic_memories", "episodic_memories", "documents", "tasks", "projects", "custom_skills", "custom_personalities", "provider_configs", "auth_sessions"):
                conn.execute(f"DELETE FROM {tbl} WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM users WHERE id = ?", (uid,))
            conn.commit()
        finally:
            conn.close()
    response.delete_cookie("xyvora_session", path="/")
    log_security_event("account_deleted", "User deleted account and owned records", user_id=uid, resource="/api/auth/account")
    return {"deleted": True}


@app.get("/api/account/export")
def export_account_bundle(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return export_user_workspace(user["id"])


@app.post("/api/account/import")
def import_account_bundle(bundle: Dict[str, Any], user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    if isinstance(bundle.get("settings"), dict):
        update_user_settings(uid, bundle["settings"])
    imported_memories = 0
    for m in (bundle.get("memories") or [])[:200]:
        if isinstance(m, dict) and m.get("subject") and m.get("relation") and m.get("object"):
            cortex.memory.upsert_semantic_memory(
                subject=str(m["subject"]),
                relation=str(m["relation"]),
                obj=str(m["object"]),
                category=str(m.get("category") or "fact"),
                confidence=float(m.get("confidence", 0.9)),
                pinned=int(m.get("pinned", 0)),
                user_id=uid,
            )
            imported_memories += 1
    return {"imported": True, "memories_imported": imported_memories}


# ----------------- Multi-Provider LLM Configuration & Health -----------------

@app.get("/api/providers")
def list_providers(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    active = dict(cortex.providers.resolve_active_provider(uid))
    active.pop("api_key", None)
    return {
        "providers": cortex.providers.list_providers_status(uid),
        "active": active,
    }


@app.put("/api/providers/{provider_id}")
def update_provider(
    provider_id: str,
    req: ProviderUpdateReq,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    uid = user["id"]
    pub = upsert_provider_config(
        user_id=uid,
        provider_id=provider_id,
        api_key=req.api_key,
        base_url=req.base_url,
        default_model=req.default_model,
        enabled=req.enabled,
    )
    log_security_event("provider_config_updated", f"provider={provider_id}", user_id=uid, resource="/api/providers")
    return {"provider": pub, "providers": cortex.providers.list_providers_status(uid)}


@app.post("/api/providers/{provider_id}/test")
async def test_provider_connection(
    provider_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    return await cortex.providers.check_provider_health(user["id"], provider_id)


# ----------------- Projects & Coding Studio Workspace -----------------

@app.get("/api/projects")
def list_projects(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT p.*,
                       (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id = p.id) AS file_count,
                       (SELECT COUNT(*) FROM sessions s WHERE s.project_id = p.id AND s.user_id = p.user_id) AS session_count,
                       (SELECT COUNT(*) FROM tasks t WHERE t.project_id = p.id AND t.user_id = p.user_id AND t.status != 'done') AS open_task_count
                FROM projects p
                WHERE p.user_id = ?
                ORDER BY p.updated_at DESC
                """,
                (uid,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.post("/api/projects")
def create_project(req: CreateProjectReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Project name is required.")
    uid = user["id"]
    pid = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO projects (id, user_id, name, description, instructions, color, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pid, uid, req.name.strip()[:80], (req.description or "")[:300], (req.instructions or "")[:1500], req.color or "#22d3ee", now, now),
            )
            conn.execute(
                "INSERT INTO project_activity (id, project_id, user_id, action, summary, created_at) VALUES (?, ?, ?, 'created', ?, ?)",
                (str(uuid.uuid4()), pid, uid, f"Created project '{req.name.strip()[:60]}'", now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (pid,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.patch("/api/projects/{project_id}")
def patch_project(project_id: str, req: PatchProjectReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, uid)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Project not found")
            name = req.name.strip()[:80] if req.name is not None else existing["name"]
            desc = req.description[:300] if req.description is not None else existing["description"]
            inst = req.instructions[:1500] if req.instructions is not None else existing["instructions"]
            col = req.color if req.color is not None else existing["color"]
            conn.execute(
                "UPDATE projects SET name = ?, description = ?, instructions = ?, color = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (name, desc, inst, col, now, project_id, uid),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("UPDATE sessions SET project_id = NULL WHERE project_id = ? AND user_id = ?", (project_id, uid))
            conn.execute("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, uid))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.get("/api/projects/{project_id}/detail")
def get_project_detail(project_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            proj = conn.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, uid)).fetchone()
            if not proj:
                raise HTTPException(status_code=404, detail="Project not found")
            files = cortex.coding.list_project_files(project_id, user_id=uid)
            activity = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM project_activity WHERE project_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT 25",
                    (project_id, uid),
                ).fetchall()
            ]
            sessions = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM sessions WHERE project_id = ? AND user_id = ? ORDER BY updated_at DESC",
                    (project_id, uid),
                ).fetchall()
            ]
            tasks = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM tasks WHERE project_id = ? AND user_id = ? ORDER BY created_at DESC",
                    (project_id, uid),
                ).fetchall()
            ]
            docs = cortex.knowledge.list_documents(user_id=uid, project_id=project_id)
            return {
                "project": dict(proj),
                "files": files,
                "activity": activity,
                "sessions": sessions,
                "tasks": tasks,
                "documents": docs,
            }
        finally:
            conn.close()


@app.get("/api/projects/{project_id}/files")
def list_project_files(project_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return cortex.coding.list_project_files(project_id, user_id=user["id"])


@app.put("/api/projects/{project_id}/files")
def save_project_file(project_id: str, req: SaveProjectFileReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return cortex.coding.save_project_file(
            project_id=project_id,
            path=req.path,
            content=req.content,
            language=req.language or "",
            summary=req.summary or "Updated file",
            user_id=user["id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/projects/{project_id}/files/{file_id}")
def delete_project_file(project_id: str, file_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    ok = cortex.coding.delete_project_file(project_id, file_id, user_id=user["id"])
    return {"deleted": ok}


@app.post("/api/projects/{project_id}/files/{file_id}/revert")
def revert_project_file(
    project_id: str,
    file_id: str,
    req: RevertFileReq,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    try:
        return cortex.coding.revert_file_revision(project_id, file_id, req.revision_id, user_id=user["id"])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/projects/{project_id}/analyze")
def analyze_project_workspace(project_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return cortex.coding.analyze_project(project_id, user_id=user["id"])


@app.post("/api/projects/{project_id}/run-tests")
def run_project_tests(project_id: str, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "project_tests", max_req=20, window_sec=60.0)
    return cortex.coding.run_approved_tests(project_id, user_id=user["id"])


@app.post("/api/coding/assist")
async def coding_ai_assist(req: CodingAssistReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "coding_assist", max_req=25, window_sec=60.0)
    uid = user["id"]
    proj_files = cortex.coding.list_project_files(req.project_id, user_id=uid) if req.project_id else []
    target_file = {"path": req.path or "main.py", "language": req.language or "python", "content": req.content or ""}
    return await cortex.coding.assist_on_code(
        action=req.action,
        prompt=req.prompt,
        target_file=target_file,
        project_files=proj_files,
        provider_manager=cortex.providers,
        user_id=uid,
    )


# ----------------- Sessions, Search, Attachments & Streaming Chat -----------------

@app.get("/api/sessions")
def list_sessions(
    project_id: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            params: List[Any] = [uid]
            where = ["s.user_id = ?"]
            if project_id:
                where.append("s.project_id = ?")
                params.append(project_id)
            if q and q.strip():
                like_q = f"%{q.strip()}%"
                where.append("(s.title LIKE ? OR EXISTS (SELECT 1 FROM messages m2 WHERE m2.session_id = s.id AND m2.content LIKE ?))")
                params.extend([like_q, like_q])
            sql = f"""
                SELECT s.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
                FROM sessions s
                WHERE {' AND '.join(where)}
                ORDER BY s.pinned DESC, s.updated_at DESC
            """
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.post("/api/sessions")
def create_session(req: CreateSessionReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    now = time.time()
    sess_id = str(uuid.uuid4())
    settings = get_user_settings(uid)
    persona = req.persona or settings.get("persona", "nexus")

    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, user_id, project_id, title, persona, pinned, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (sess_id, uid, req.project_id, req.title or "New Neural Session", persona, now, now),
            )
            name_row = conn.execute(
                "SELECT object FROM semantic_memories WHERE user_id = ? AND lower(subject)='user' AND lower(relation)='name' AND verified = 1 LIMIT 1",
                (uid,),
            ).fetchone()
            user_name = name_row["object"] if name_row else ""
            greeting = (
                f"Welcome back{f', **{user_name}**' if user_name else ''}! This is a fresh conversation thread, "
                f"and your personal **Semantic Knowledge Graph** retains everything you've saved across sessions.\n\n"
                f"What would you like to explore, build, research, or analyze next?"
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
                (id, session_id, user_id, role, content, intent, confidence, thought_trace, artifacts, memories_added, suggestions, created_at)
                VALUES (?, ?, ?, 'assistant', ?, 'welcome', 1.0, '[]', '[]', '[]', ?, ?)
                """,
                (str(uuid.uuid4()), sess_id, uid, greeting, json.dumps(suggestions), now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM sessions WHERE id = ? AND user_id = ?", (sess_id, uid)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.patch("/api/sessions/{session_id}")
def patch_session(session_id: str, req: PatchSessionReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute("SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Session not found")
            title = req.title.strip()[:80] if req.title is not None else existing["title"]
            pinned = int(req.pinned) if req.pinned is not None else existing["pinned"]
            proj = req.project_id if req.project_id is not None else existing["project_id"]
            conn.execute(
                "UPDATE sessions SET title = ?, pinned = ?, project_id = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, pinned, proj if proj != "" else None, now, session_id, uid),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ? AND user_id = ?", (session_id, uid))
            cur = conn.execute("DELETE FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid))
            conn.commit()
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found")
        finally:
            conn.close()
    return {"deleted": True}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            sess = conn.execute("SELECT id FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid)).fetchone()
            if not sess:
                raise HTTPException(status_code=404, detail="Session not found or access denied.")
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
                (session_id, uid),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                for field in ("thought_trace", "tool_calls", "citations", "attachments", "artifacts", "memories_added", "suggestions"):
                    try:
                        d[field] = json.loads(d.get(field) or "[]")
                    except Exception:
                        d[field] = []
                out.append(d)
            return out
        finally:
            conn.close()


@app.post("/api/sessions/{session_id}/summarize")
def summarize_session(session_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return cortex.memory.summarize_and_compress_session(session_id, user_id=user["id"])


@app.post("/api/attachments")
async def upload_chat_attachment(
    request: Request,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    project_id: Optional[str] = Form(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    _rate_check(request, "upload_attachment", max_req=25, window_sec=60.0)
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    fname = (file.filename or "attachment").strip()
    mime = (file.content_type or "application/octet-stream").lower()
    uid = user["id"]
    att_id = str(uuid.uuid4())
    now = time.time()

    is_image = mime.startswith("image/") or fname.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"))
    try:
        if is_image:
            info = inspect_image_bytes(fname, raw, mime)
            kind = "image"
            width = info.get("width")
            height = info.get("height")
            extracted_text = info.get("ocr_text", "")
            data_url = info.get("data_url", "")
            meta = {"format": info.get("format"), "ocr_engine": info.get("ocr_engine")}
        else:
            parsed = parse_document_bytes(fname, raw, mime)
            kind = "document"
            width = None
            height = None
            extracted_text = parsed.get("text", "")
            data_url = ""
            meta = parsed.get("metadata", {})
            mime = parsed.get("mime_type", mime)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO attachments
                (id, user_id, session_id, project_id, filename, mime_type, kind, size_bytes, width, height, extracted_text, data_url, is_generated, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (att_id, uid, session_id, project_id, fname, mime, kind, len(raw), width, height, extracted_text[:40000], data_url, json.dumps(meta), now),
            )
            conn.commit()
        finally:
            conn.close()

    return {
        "id": att_id,
        "filename": fname,
        "mime_type": mime,
        "kind": kind,
        "size_bytes": len(raw),
        "width": width,
        "height": height,
        "data_url": data_url if kind == "image" else "",
        "extracted_preview": extracted_text[:300],
        "is_generated": False,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)):
    _rate_check(request, "chat_stream", max_req=40, window_sec=60.0)
    if not req.message.strip() and not req.attachment_ids:
        raise HTTPException(status_code=400, detail="Message or attachment is required.")
    if len(req.message) > 32000:
        raise HTTPException(status_code=400, detail="Message exceeds 32,000 character maximum.")

    uid = user["id"]
    # Verify session is not owned by a different user!
    with _db_lock:
        conn = get_conn()
        try:
            existing_sess = conn.execute("SELECT user_id FROM sessions WHERE id = ?", (req.session_id,)).fetchone()
            if existing_sess and existing_sess["user_id"] != uid:
                raise HTTPException(status_code=403, detail="Access denied to another user's session.")
        finally:
            conn.close()

    async def event_generator():
        async for evt in cortex.process_message_stream(
            session_id=req.session_id,
            user_text=req.message,
            user_id=uid,
            project_id=req.project_id,
            attachment_ids=req.attachment_ids,
            requested_provider=req.provider,
            requested_model=req.model,
            client_memories=req.client_memories,
        ):
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


@app.post("/api/sessions/{session_id}/regenerate")
async def regenerate_session_response(
    session_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    _rate_check(request, "chat_regenerate", max_req=30, window_sec=60.0)
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            sess = conn.execute("SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid)).fetchone()
            if not sess:
                raise HTTPException(status_code=404, detail="Session not found")
            last_user = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND user_id = ? AND role = 'user' ORDER BY created_at DESC LIMIT 1",
                (session_id, uid),
            ).fetchone()
            if not last_user:
                raise HTTPException(status_code=400, detail="No user message to regenerate.")
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND user_id = ? AND created_at > ?",
                (session_id, uid, last_user["created_at"]),
            )
            conn.commit()
            user_text = last_user["content"]
            atts = json.loads(last_user["attachments"] or "[]")
            att_ids = [a["id"] for a in atts if isinstance(a, dict) and a.get("id")]
        finally:
            conn.close()

    async def event_generator():
        async for evt in cortex.process_message_stream(
            session_id=session_id,
            user_text=user_text,
            user_id=uid,
            project_id=sess["project_id"],
            attachment_ids=att_ids,
            skip_saving_user_message=True,
        ):
            yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.put("/api/messages/{message_id}")
async def edit_and_resend_message(
    message_id: str,
    req: EditMessageReq,
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
):
    _rate_check(request, "chat_edit", max_req=30, window_sec=60.0)
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Edited message cannot be empty.")
    uid = user["id"]
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            msg = conn.execute("SELECT * FROM messages WHERE id = ? AND user_id = ?", (message_id, uid)).fetchone()
            if not msg or msg["role"] != "user":
                raise HTTPException(status_code=404, detail="User message not found")
            sess_id = msg["session_id"]
            conn.execute("UPDATE messages SET content = ?, edited_at = ? WHERE id = ? AND user_id = ?", (req.content.strip(), now, message_id, uid))
            conn.execute("DELETE FROM messages WHERE session_id = ? AND user_id = ? AND created_at > ?", (sess_id, uid, msg["created_at"]))
            conn.commit()
            atts = json.loads(msg["attachments"] or "[]")
            att_ids = [a["id"] for a in atts if isinstance(a, dict) and a.get("id")]
        finally:
            conn.close()

    async def event_generator():
        async for evt in cortex.process_message_stream(
            session_id=sess_id,
            user_text=req.content.strip(),
            user_id=uid,
            attachment_ids=att_ids,
            skip_saving_user_message=True,
        ):
            yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/sync")
def sync_client_vault(req: ClientVaultSyncReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    if req.settings:
        update_user_settings(uid, req.settings)
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
                    user_id=uid,
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
                        "SELECT id FROM custom_skills WHERE user_id = ? AND lower(trigger_phrase) = lower(?)",
                        (uid, trig),
                    ).fetchone()
                    if not exists:
                        conn.execute(
                            "INSERT INTO custom_skills (id, user_id, trigger_phrase, intent_label, response_template, created_at) VALUES (?, ?, ?, 'custom_skill', ?, ?)",
                            (str(uuid.uuid4()), uid, trig, tmpl, now),
                        )
                        retrain_needed = True
                conn.commit()
            finally:
                conn.close()
        if retrain_needed:
            cortex.neural.train()

    all_mems = [m for m in cortex.memory.get_knowledge_graph(user_id=uid)["memories"] if m["category"] != "system"]
    return {"synced": True, "memories": all_mems, "settings": get_user_settings(uid)}


@app.post("/api/messages/{message_id}/feedback")
def set_message_feedback(message_id: str, req: FeedbackReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("UPDATE messages SET feedback = ? WHERE id = ? AND user_id = ?", (req.feedback, message_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    return {"ok": True, "feedback": req.feedback}


# ----------------- Web Research Engine & History -----------------

def _credibility_badge(url: str) -> Dict[str, Any]:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith((".gov", ".edu")) or any(d in host for d in ("arxiv.org", "nature.com", "ieee.org", "acm.org", "nih.gov")):
        return {"tier": "high", "label": "Institutional / Academic"}
    if any(d in host for d in ("wikipedia.org", "github.com", "python.org", "mozilla.org", "w3.org", "reuters.com", "apnews.com")):
        return {"tier": "high", "label": "Verified Reference"}
    if url.startswith("https://"):
        return {"tier": "medium", "label": "Web Source"}
    return {"tier": "low", "label": "Unverified"}


@app.get("/api/research/status")
def get_research_status() -> Dict[str, Any]:
    return cortex.web.get_provider_status()


@app.post("/api/research/query")
async def run_web_research(req: ResearchQueryReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "web_research", max_req=20, window_sec=60.0)
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Research query cannot be empty.")
    uid = user["id"]
    t0 = time.perf_counter()
    res = await cortex.web.research(
        query=req.query.strip(),
        mode=req.mode or "quick",
        max_results=max(1, min(10, req.max_results or 5)),
        follow_links=bool(req.follow_links) or req.mode == "deep",
    )
    enriched_sources = []
    for s in res.get("sources", []):
        cred = _credibility_badge(s.get("url", ""))
        enriched_sources.append({**s, "credibility": cred})
    res["sources"] = enriched_sources

    clean_q = req.query.strip().rstrip("?.!")
    follow_ups = [
        f"Compare primary architectures and tradeoffs for {clean_q[:50]}",
        f"What are the main empirical benchmarks and limitations of {clean_q[:50]}?",
        f"Summarize recent 2025–2026 developments in {clean_q[:50]}",
    ]
    res["follow_up_questions"] = follow_ups

    rep_id = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO research_reports (id, user_id, project_id, query, mode, summary, sources_json, follow_up_json, diagnostics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rep_id,
                    uid,
                    req.project_id,
                    req.query.strip(),
                    req.mode or "quick",
                    res.get("report_markdown") or res.get("summary") or "",
                    json.dumps(enriched_sources),
                    json.dumps(follow_ups),
                    json.dumps(res.get("diagnostics") or {}),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    record_usage_event(uid, "/api/research/query", "", "", "web_research", 0, (time.perf_counter() - t0) * 1000, "ok")
    res["report_id"] = rep_id
    return res


@app.get("/api/research/history")
def list_research_history(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM research_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT 25",
                (uid,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["sources"] = json.loads(d.pop("sources_json", "[]") or "[]")
                d["follow_up_questions"] = json.loads(d.pop("follow_up_json", "[]") or "[]")
                d["diagnostics"] = json.loads(d.pop("diagnostics_json", "{}") or "{}")
                out.append(d)
            return out
        finally:
            conn.close()


@app.delete("/api/research/history/{report_id}")
def delete_research_report(report_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM research_reports WHERE id = ? AND user_id = ?", (report_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


# ----------------- Autonomous Multi-Step Agents & Tool Registry -----------------

@app.get("/api/tools")
def list_registered_tools() -> List[Dict[str, Any]]:
    return cortex.tools.list_tools()


@app.post("/api/agents/runs")
async def start_agent_run(req: StartAgentReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "agent_run", max_req=15, window_sec=60.0)
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Agent goal cannot be empty.")
    try:
        return await cortex.agents.start_run(
            goal=req.goal.strip(),
            user_id=user["id"],
            session_id=req.session_id,
            project_id=req.project_id,
            auto_execute=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/agents/runs")
def list_agent_runs(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return cortex.agents.list_runs(user_id=user["id"])


@app.get("/api/agents/runs/{run_id}")
def get_agent_run(run_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    run = cortex.agents.get_run(run_id, user_id=user["id"])
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run


@app.post("/api/agents/runs/{run_id}/approve")
async def approve_agent_step(run_id: str, req: AgentApproveReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    res = await cortex.agents.resolve_approval(run_id, approved=req.approved, user_id=user["id"])
    if not res:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return res


@app.post("/api/agents/runs/{run_id}/cancel")
def cancel_agent_run(run_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    res = cortex.agents.cancel_run(run_id, user_id=user["id"])
    if not res:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return res


# ----------------- Smart Memory Palace & Approval Controls -----------------

@app.get("/api/memory/graph")
def get_memory_graph(
    project_id: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    return cortex.memory.get_knowledge_graph(user_id=user["id"], project_id=project_id)


@app.post("/api/memory/recall")
def test_memory_recall(req: MemoryRecallReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return cortex.memory.recall(req.query, threshold=req.threshold or 0.15, top_k=10, user_id=user["id"], project_id=req.project_id)


@app.post("/api/memory/facts")
def add_memory_fact(req: CreateMemoryReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    exp = (time.time() + req.ttl_days * 86400) if (req.ttl_days and req.ttl_days > 0) else None
    return cortex.memory.upsert_semantic_memory(
        subject=req.subject,
        relation=req.relation,
        obj=req.object,
        category=req.category or "fact",
        confidence=req.confidence if req.confidence is not None else 0.94,
        pinned=req.pinned or 0,
        verified=1,
        expires_at=exp,
        source_text="Manual Memory Palace entry",
        user_id=user["id"],
        project_id=req.project_id,
    )


@app.patch("/api/memory/facts/{memory_id}")
def patch_memory_fact(memory_id: str, req: PatchMemoryReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute("SELECT * FROM semantic_memories WHERE id = ? AND user_id = ?", (memory_id, uid)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Memory node not found")
            subj = req.subject if req.subject is not None else existing["subject"]
            rel = req.relation if req.relation is not None else existing["relation"]
            obj = req.object if req.object is not None else existing["object"]
            pin = req.pinned if req.pinned is not None else existing["pinned"]
            imp = req.importance if req.importance is not None else existing["importance"]
            ver = req.verified if req.verified is not None else existing["verified"]
            conn.execute(
                """
                UPDATE semantic_memories
                SET subject = ?, relation = ?, object = ?, pinned = ?, importance = ?, verified = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (subj, rel, obj, pin, imp, ver, now, memory_id, uid),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (memory_id,)).fetchone()
            return dict(updated)
        finally:
            conn.close()


@app.post("/api/memory/facts/{memory_id}/approve")
def approve_memory_fact(memory_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            cur = conn.execute("UPDATE semantic_memories SET verified = 1, updated_at = ? WHERE id = ? AND user_id = ?", (now, memory_id, uid))
            conn.commit()
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Memory node not found")
            row = conn.execute("SELECT * FROM semantic_memories WHERE id = ?", (memory_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/memory/facts/{memory_id}")
def delete_memory_fact(memory_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM semantic_memories WHERE id = ? AND user_id = ?", (memory_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.post("/api/memory/reset")
def reset_user_memories(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM semantic_memories WHERE user_id = ? AND category != 'system'", (uid,))
            conn.execute("DELETE FROM episodic_memories WHERE user_id = ?", (uid,))
            conn.commit()
        finally:
            conn.close()
    return {"reset": True}


# ----------------- Personalities & Extension Marketplace -----------------

@app.get("/api/personalities")
def list_personalities(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    builtins = [
        {"slug": slug, "name": info["name"], "builtin": True, "closing_style": info["closing_style"]}
        for slug, info in PERSONA_PROFILES.items()
    ]
    with _db_lock:
        conn = get_conn()
        try:
            customs = [
                {**dict(r), "builtin": False}
                for r in conn.execute("SELECT * FROM custom_personalities WHERE user_id = ? ORDER BY created_at DESC", (user["id"],)).fetchall()
            ]
        finally:
            conn.close()
    return {"builtin": builtins, "custom": customs}


@app.post("/api/personalities")
def create_custom_personality(req: CustomPersonaReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    safety = check_safety_policy(f"{req.name}\n{req.system_instructions}\n{req.domain_focus}")
    if safety:
        raise HTTPException(status_code=400, detail=safety["message"])
    if not req.name.strip() or not req.system_instructions.strip():
        raise HTTPException(status_code=400, detail="Name and system instructions are required.")
    uid = user["id"]
    slug = "custom_" + (re.sub(r"[^a-z0-9_]+", "_", req.name.strip().lower()).strip("_")[:32] or uuid.uuid4().hex[:8])
    pid = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO custom_personalities
                (id, user_id, slug, name, description, system_instructions, tone, creativity, verbosity, domain_focus, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, slug) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    system_instructions = excluded.system_instructions,
                    tone = excluded.tone,
                    creativity = excluded.creativity,
                    verbosity = excluded.verbosity,
                    domain_focus = excluded.domain_focus
                """,
                (
                    pid,
                    uid,
                    slug,
                    req.name.strip()[:60],
                    (req.description or "")[:200],
                    req.system_instructions.strip()[:2000],
                    req.tone or "balanced",
                    float(req.creativity if req.creativity is not None else 0.65),
                    req.verbosity or "balanced",
                    (req.domain_focus or "")[:300],
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM custom_personalities WHERE user_id = ? AND slug = ?", (uid, slug)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/personalities/{persona_id}")
def delete_custom_personality(persona_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM custom_personalities WHERE (id = ? OR slug = ?) AND user_id = ?", (persona_id, persona_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.get("/api/extensions")
def list_extensions(category: Optional[str] = Query(default=None), user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return cortex.extensions.list_extensions(user_id=user["id"], category=category)


@app.post("/api/extensions/{extension_id}/install")
def install_extension(extension_id: str, config: Optional[Dict[str, Any]] = None, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return cortex.extensions.install_extension(extension_id, user_id=user["id"], config=config)


@app.post("/api/extensions/{extension_id}/uninstall")
def uninstall_extension(extension_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return cortex.extensions.uninstall_extension(extension_id, user_id=user["id"])


@app.post("/api/extensions/custom")
def create_custom_extension(req: CustomExtensionReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return cortex.extensions.create_custom_extension(
            user_id=user["id"],
            name=req.name,
            description=req.description,
            category=req.category or "productivity",
            prompt_template=req.prompt_template,
            permissions=req.permissions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/extensions/{extension_id}/run")
def run_extension(extension_id: str, req: RunExtensionReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    res = cortex.extensions.execute_extension(extension_id, req.inputs, user_id=user["id"])
    record_usage_event(user["id"], f"/api/extensions/{extension_id}/run", "", "", extension_id, 0, res.get("duration_ms", 0), "ok" if res.get("ok") else "error")
    return res


# ----------------- Neural Intent Classifier Training Studio -----------------

@app.get("/api/neural/stats")
def get_neural_stats(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    stats = cortex.neural.get_stats()
    with _db_lock:
        conn = get_conn()
        try:
            samples = [dict(r) for r in conn.execute("SELECT * FROM training_samples WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (uid,)).fetchall()]
            skills = [dict(r) for r in conn.execute("SELECT * FROM custom_skills WHERE user_id = ? ORDER BY created_at DESC LIMIT 60", (uid,)).fetchall()]
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


@app.get("/api/neural/dataset/validate")
def validate_neural_dataset() -> Dict[str, Any]:
    return cortex.neural.validate_dataset()


@app.get("/api/neural/dataset/export")
def export_neural_dataset() -> Dict[str, Any]:
    return cortex.neural.export_dataset()


@app.post("/api/neural/dataset/import")
def import_neural_dataset(req: DatasetImportReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return cortex.neural.import_dataset(req.samples, user_id=user["id"], replace_existing=bool(req.replace_existing))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/neural/samples")
def add_training_sample(req: TrainingSampleReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not req.text.strip() or not req.intent.strip():
        raise HTTPException(status_code=400, detail="Text and intent are required.")
    sample_id = str(uuid.uuid4())
    now = time.time()
    clean_intent = req.intent.strip().lower().replace(" ", "_")
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO training_samples (id, user_id, text, intent, source, created_at) VALUES (?, ?, ?, ?, 'user', ?)",
                (sample_id, user["id"], req.text.strip(), clean_intent, now),
            )
            conn.commit()
        finally:
            conn.close()
    return {"id": sample_id, "text": req.text.strip(), "intent": clean_intent, "created_at": now}


@app.delete("/api/neural/samples/{sample_id}")
def delete_training_sample(sample_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM training_samples WHERE id = ? AND user_id = ?", (sample_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


@app.post("/api/neural/skills")
def add_custom_skill(req: CustomSkillReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not req.trigger_phrase.strip() or not req.response_template.strip():
        raise HTTPException(status_code=400, detail="Trigger phrase and response template are required.")
    safety = check_safety_policy(f"{req.trigger_phrase}\n{req.response_template}")
    if safety:
        raise HTTPException(status_code=400, detail=safety["message"])
    skill_id = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO custom_skills (id, user_id, trigger_phrase, intent_label, response_template, created_at) VALUES (?, ?, ?, 'custom_skill', ?, ?)",
                (skill_id, user["id"], req.trigger_phrase.strip(), req.response_template.strip(), now),
            )
            conn.commit()
        finally:
            conn.close()
    cortex.neural.train(user_id=user["id"])
    return {"id": skill_id, "trigger_phrase": req.trigger_phrase.strip(), "response_template": req.response_template.strip()}


@app.delete("/api/neural/skills/{skill_id}")
def delete_custom_skill(skill_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM custom_skills WHERE id = ? AND user_id = ?", (skill_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    cortex.neural.train(user_id=user["id"])
    return {"deleted": True}


@app.post("/api/neural/retrain")
def retrain_neural_cortex(req: RetrainReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "neural_retrain", max_req=10, window_sec=60.0)
    h1 = max(32, min(256, req.hidden_1 or 128))
    h2 = max(16, min(192, req.hidden_2 or 64))
    iters = max(30, min(200, req.max_iter or 95))
    return cortex.neural.train(hidden_layers=(h1, h2), max_iter=iters, user_id=user["id"])


# ----------------- Sandbox, Math, Knowledge Vault, Voice, Images & Tasks -----------------

@app.post("/api/tools/python")
def run_python_sandbox(req: PythonExecReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "python_sandbox", max_req=25, window_sec=60.0)
    res = cortex.code.execute_python(req.code, user_id=user["id"])
    record_usage_event(user["id"], "/api/tools/python", "", "", "python_sandbox", 0, res.get("duration_ms", 0), "ok" if res.get("ok") else "error")
    return res


@app.get("/api/sandbox/history")
def list_sandbox_history(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    with _db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM sandbox_audit_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT 30",
                (user["id"],),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.post("/api/tools/math")
def run_symbolic_math(req: MathQueryReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = cortex.math.solve_query(req.query)
    record_usage_event(user["id"], "/api/tools/math", "", "", "calculator_math", 0, (time.perf_counter() - t0) * 1000, "ok" if res.get("ok") else "error")
    return res


@app.get("/api/documents")
def list_docs(
    collection: Optional[str] = Query(default=None),
    project_id: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    return cortex.knowledge.list_documents(user_id=user["id"], collection=collection, project_id=project_id, search_filter=q)


@app.get("/api/documents/{doc_id}")
def get_doc_detail(doc_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    detail = cortex.knowledge.get_document_detail(doc_id, user_id=user["id"])
    if not detail:
        raise HTTPException(status_code=404, detail="Document not found")
    return detail


@app.post("/api/documents")
def create_doc(req: CreateDocReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return cortex.knowledge.ingest_document(
            title=req.title,
            content=req.content,
            source_type=req.source_type or "note",
            user_id=user["id"],
            project_id=req.project_id,
            collection=req.collection or "general",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/documents/upload")
async def upload_doc_file(
    request: Request,
    file: UploadFile = File(...),
    collection: Optional[str] = Form(default="general"),
    project_id: Optional[str] = Form(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    _rate_check(request, "doc_upload", max_req=20, window_sec=60.0)
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    try:
        return cortex.knowledge.ingest_uploaded_file(
            filename=file.filename or "Uploaded Document",
            raw_bytes=raw_bytes,
            mime_type=file.content_type or "",
            user_id=user["id"],
            project_id=project_id,
            collection=collection or "general",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/documents/{doc_id}/reindex")
def reindex_doc(doc_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    res = cortex.knowledge.reindex_document(doc_id, user_id=user["id"])
    if not res:
        raise HTTPException(status_code=404, detail="Document not found")
    return res


@app.post("/api/documents/search")
def search_docs(req: DocSearchReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return cortex.knowledge.search(
        query=req.query,
        top_k=req.top_k or 5,
        user_id=user["id"],
        collection=req.collection,
        project_id=req.project_id,
    )


@app.delete("/api/documents/{doc_id}")
def delete_doc(doc_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    ok = cortex.knowledge.delete_document(doc_id, user_id=user["id"])
    return {"deleted": ok}


@app.post("/api/images/generate")
async def generate_image_route(req: ImageGenReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "image_gen", max_req=10, window_sec=60.0)
    safety = check_safety_policy(req.prompt)
    if safety:
        raise HTTPException(status_code=400, detail=safety["message"])
    return await cortex.providers.generate_image(user["id"], req.prompt, size=req.size or "1024x1024")


@app.post("/api/voice/tts")
async def neural_tts_route(req: VoiceTTSReq, request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    _rate_check(request, "voice_tts", max_req=20, window_sec=60.0)
    return await cortex.providers.synthesize_speech_neural(user["id"], req.text[:2000], voice=req.voice or "alloy")


@app.get("/api/tasks")
def list_tasks(
    project_id: Optional[str] = Query(default=None),
    user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    uid = user["id"]
    with _db_lock:
        conn = get_conn()
        try:
            if project_id:
                rows = conn.execute("SELECT * FROM tasks WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC", (uid, project_id)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at DESC", (uid,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.post("/api/tasks")
def create_task(req: CreateTaskReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Task title required")
    uid = user["id"]
    tid = str(uuid.uuid4())
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute(
                "INSERT INTO tasks (id, user_id, project_id, title, description, priority, status, due_note, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'todo', ?, ?, ?)",
                (tid, uid, req.project_id, req.title.strip(), req.description or "", req.priority or "medium", req.due_note or "", now, now),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.patch("/api/tasks/{task_id}")
def patch_task(task_id: str, req: PatchTaskReq, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    uid = user["id"]
    now = time.time()
    with _db_lock:
        conn = get_conn()
        try:
            existing = conn.execute("SELECT * FROM tasks WHERE id = ? AND user_id = ?", (task_id, uid)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Task not found")
            title = req.title if req.title is not None else existing["title"]
            status = req.status if req.status is not None else existing["status"]
            priority = req.priority if req.priority is not None else existing["priority"]
            conn.execute(
                "UPDATE tasks SET title = ?, status = ?, priority = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (title, status, priority, now, task_id, uid),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            return dict(row)
        finally:
            conn.close()


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    with _db_lock:
        conn = get_conn()
        try:
            conn.execute("DELETE FROM tasks WHERE id = ? AND user_id = ?", (task_id, user["id"]))
            conn.commit()
        finally:
            conn.close()
    return {"deleted": True}


# ----------------- Cognitive Settings & Role-Protected Admin Analytics -----------------

@app.get("/api/settings")
def read_settings(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return get_user_settings(user["id"])


@app.patch("/api/settings")
def patch_settings_route(patch: Dict[str, Any], user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return update_user_settings(user["id"], patch)


@app.get("/api/admin/analytics")
def admin_analytics_dashboard(_admin: Dict[str, Any] = Depends(require_admin_user)) -> Dict[str, Any]:
    return get_admin_analytics()


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
