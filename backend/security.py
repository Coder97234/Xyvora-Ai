import base64
import hashlib
import hmac
import ipaddress
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Structured logger with automatic secret redaction
_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{12,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9._\-]{12,})", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"]?)([^'\"\s,&]+)", re.IGNORECASE),
]


def redact_sensitive(text: str) -> str:
    if not text:
        return text
    out = str(text)
    out = _SECRET_PATTERNS[0].sub("sk-[REDACTED]", out)
    out = _SECRET_PATTERNS[1].sub("Bearer [REDACTED]", out)
    out = _SECRET_PATTERNS[2].sub(r"\1[REDACTED]", out)
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive(record.msg)
        if record.args:
            record.args = tuple(
                redact_sensitive(str(a)) if isinstance(a, str) else a for a in record.args
            )
        return True


def get_logger(name: str = "xyvora") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handler.addFilter(RedactingFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("xyvora.security")


# ----------------- Password Hashing (PBKDF2-HMAC-SHA256) -----------------

PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${dk_b64}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iter_s, salt_b64, dk_b64 = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iter_s)
        salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
        expected = base64.urlsafe_b64decode(dk_b64.encode("ascii"))
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False


def validate_password_strength(password: str) -> Tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 256:
        return False, "Password must not exceed 256 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password must contain at least one letter and one number."
    return True, "OK"


def validate_email(email: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", email.strip()))


# ----------------- Session Token Management -----------------

def generate_session_token() -> str:
    return secrets.token_urlsafe(36)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ----------------- Server-Side Secret Vault (Authenticated Encryption) -----------------

_cached_master_key: Optional[bytes] = None
_key_lock = threading.Lock()


def _get_master_key(data_dir: str) -> bytes:
    global _cached_master_key
    with _key_lock:
        if _cached_master_key is not None:
            return _cached_master_key
        env_secret = os.environ.get("XYVORA_SECRET_KEY", "").strip()
        if env_secret:
            _cached_master_key = hashlib.sha256(env_secret.encode("utf-8")).digest()
            return _cached_master_key

        key_file = os.path.join(data_dir, ".xyvora_master_key")
        try:
            if os.path.isfile(key_file):
                with open(key_file, "rb") as f:
                    raw = f.read().strip()
                if len(raw) >= 32:
                    _cached_master_key = hashlib.sha256(raw).digest()
                    return _cached_master_key
            raw_new = secrets.token_urlsafe(48).encode("utf-8")
            os.makedirs(data_dir, exist_ok=True)
            fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(raw_new)
            _cached_master_key = hashlib.sha256(raw_new).digest()
            return _cached_master_key
        except Exception:
            _cached_master_key = hashlib.sha256(b"xyvora-ephemeral-fallback-key").digest()
            return _cached_master_key


def _keystream(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(
            enc_key, nonce + counter.to_bytes(4, "big"), hashlib.sha256
        ).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def encrypt_secret(plaintext: str, data_dir: str) -> str:
    if not plaintext:
        return ""
    master = _get_master_key(data_dir)
    enc_key = hmac.new(master, b"xyvora-enc-v1", hashlib.sha256).digest()
    mac_key = hmac.new(master, b"xyvora-mac-v1", hashlib.sha256).digest()
    nonce = secrets.token_bytes(16)
    pt_bytes = plaintext.encode("utf-8")
    stream = _keystream(enc_key, nonce, len(pt_bytes))
    ciphertext = bytes(a ^ b for a, b in zip(pt_bytes, stream))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    blob = nonce + tag + ciphertext
    return "enc_v1$" + base64.urlsafe_b64encode(blob).decode("ascii")


def decrypt_secret(encoded: str, data_dir: str) -> str:
    if not encoded:
        return ""
    if not encoded.startswith("enc_v1$"):
        return ""
    try:
        blob = base64.urlsafe_b64decode(encoded[7:].encode("ascii"))
        if len(blob) < 48:
            return ""
        nonce = blob[:16]
        tag = blob[16:48]
        ciphertext = blob[48:]
        master = _get_master_key(data_dir)
        enc_key = hmac.new(master, b"xyvora-enc-v1", hashlib.sha256).digest()
        mac_key = hmac.new(master, b"xyvora-mac-v1", hashlib.sha256).digest()
        expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            return ""
        stream = _keystream(enc_key, nonce, len(ciphertext))
        pt_bytes = bytes(a ^ b for a, b in zip(ciphertext, stream))
        return pt_bytes.decode("utf-8")
    except Exception:
        return ""


def mask_secret(secret: str) -> str:
    if not secret:
        return ""
    s = secret.strip()
    if len(s) <= 8:
        return "••••••••"
    return f"{s[:3]}••••••{s[-4:]}"


# ----------------- CORS Origin Allowlist -----------------

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://xyvora.net",
    "https://www.xyvora.net",
    "https://xyvora-ai.vercel.app",
]


def get_allowed_origins() -> List[str]:
    raw = os.environ.get("XYVORA_ALLOWED_ORIGINS", "").strip()
    if raw:
        custom = [o.strip().rstrip("/") for o in raw.split(",") if o.strip() and o.strip() != "*"]
        if custom:
            return custom
    return list(DEFAULT_ALLOWED_ORIGINS)


def is_origin_allowed(origin: Optional[str]) -> bool:
    if not origin:
        return True
    clean = origin.strip().rstrip("/")
    allowed = get_allowed_origins()
    if clean in allowed:
        return True
    # Allow official preview proxy subdomains safely over HTTPS
    try:
        parsed = urlparse(clean)
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "https" and (
            host.endswith(".e2b.app") or host.endswith(".vercel.app") or host == "xyvora.net" or host.endswith(".xyvora.net")
        ):
            return True
        if parsed.scheme == "http" and host in ("localhost", "127.0.0.1"):
            return True
    except Exception:
        return False
    return False


# ----------------- Sliding-Window Rate Limiter -----------------

class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = {}

    def check(self, key: str, max_requests: int, window_sec: float) -> Tuple[bool, float]:
        now = time.monotonic()
        cutoff = now - window_sec
        with self._lock:
            dq = self._buckets.get(key)
            if dq is None:
                dq = deque()
                self._buckets[key] = dq
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= max_requests:
                retry_after = max(0.5, window_sec - (now - dq[0]))
                return False, round(retry_after, 1)
            dq.append(now)
            # Periodic cleanup of stale buckets
            if len(self._buckets) > 4000:
                stale = [k for k, v in self._buckets.items() if not v or v[-1] < cutoff]
                for k in stale[:1000]:
                    self._buckets.pop(k, None)
            return True, 0.0


rate_limiter = SlidingWindowRateLimiter()


# ----------------- SSRF Protection for Outbound Web Fetching -----------------

def validate_external_url(url: str, allow_local_ollama: bool = False) -> Tuple[bool, str]:
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False, "Malformed URL."

    if parsed.scheme not in ("http", "https"):
        return False, "Only http:// and https:// URLs are permitted."

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False, "URL is missing a hostname."

    if allow_local_ollama and hostname in ("localhost", "127.0.0.1") and parsed.port in (11434, 1234, 8080):
        return True, "OK"

    if hostname in ("localhost", "0.0.0.0", "metadata.google.internal") or hostname.endswith(".local") or hostname.endswith(".internal"):
        return False, f"Access to internal/local host '{hostname}' is blocked by SSRF policy."

    # Check IP literals or DNS resolution if available
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_multicast:
            return False, f"Access to private/reserved IP '{hostname}' is blocked by SSRF policy."
    except ValueError:
        # Hostname is a domain; resolve if DNS available to block DNS rebinding to private IPs
        try:
            infos = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
            for info in infos:
                addr = info[4][0]
                ip_obj = ipaddress.ip_address(addr)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
                    return False, f"Domain '{hostname}' resolved to restricted internal address '{addr}'."
        except socket.gaierror:
            # Let the HTTP client report exact DNS/network reachability error
            pass

    return True, "OK"


# ----------------- Responsible AI & Security Guardrail -----------------

_UNSAFE_PATTERNS: List[Tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\b(?:aimbot|wallhack|esp\s+script|roblox\s+exploit|script\s+executor|luau\s+exploit|bypass\s+anticheat|anti-cheat\s+bypass|game\s+cheat|auto-?aim\s+hack|speedhack|memory\s+injector)\b",
            re.IGNORECASE,
        ),
        "game_exploit",
        "I cannot create game cheats, unauthorized script executors, aimbots, or anti-cheat bypasses. "
        "If you are developing your own game (in Luau/Roblox, Unity/C#, or C++), I'd be happy to help you build legitimate gameplay mechanics, server-authoritative anti-cheat validation, raycasting physics, or debugging tools!",
    ),
    (
        re.compile(
            r"\b(?:write\s+(?:a\s+)?(?:ransomware|keylogger|trojan|rootkit|credential\s+stealer|info\s*stealer|wiper\s+malware)|steal\s+(?:browser\s+)?(?:passwords|cookies|credentials|session\s+tokens)|phishing\s+(?:kit|page\s+to\s+steal))\b",
            re.IGNORECASE,
        ),
        "malware_or_credential_theft",
        "I cannot generate malware, ransomware, keyloggers, phishing kits, or credential-theft tools. "
        "I can, however, explain how these threats operate conceptually and help you implement defensive protections—such as PBKDF2/Argon2 password hashing, HttpOnly session hardening, CSP headers, anomaly detection, and incident response.",
    ),
    (
        re.compile(
            r"\b(?:bypass\s+(?:2fa|mfa|authentication|firewall|waf)|sql\s+injection\s+payload\s+to\s+dump|exploit\s+payload\s+to\s+hack)\b",
            re.IGNORECASE,
        ),
        "security_bypass",
        "I cannot provide instructions or payloads for bypassing authentication, MFA, or security controls on systems without authorization. "
        "I can help you audit your own code for vulnerabilities (like parameterized SQL queries, CSRF protection, rate limiting, and RBAC) and write automated security tests!",
    ),
]


def check_safety_policy(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    for pattern, category, message in _UNSAFE_PATTERNS:
        if pattern.search(text):
            return {"blocked": "true", "category": category, "message": message}
    return None
