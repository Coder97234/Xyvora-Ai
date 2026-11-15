import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from database import DEFAULT_USER_ID, get_conn, _db_lock

ALLOWED_IMPORTS = {
    "math",
    "cmath",
    "random",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "operator",
    "decimal",
    "fractions",
    "datetime",
    "time",
    "json",
    "csv",
    "io",
    "re",
    "heapq",
    "bisect",
    "string",
    "typing",
    "dataclasses",
    "enum",
    "abc",
    "copy",
    "pprint",
    "unittest",
    "numpy",
    "sympy",
}

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "ssl",
    "http",
    "urllib",
    "httpx",
    "requests",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "telnetlib",
    "asyncio",
    "multiprocessing",
    "threading",
    "concurrent",
    "ctypes",
    "cffi",
    "mmap",
    "fcntl",
    "pty",
    "resource",
    "signal",
    "shutil",
    "pathlib",
    "glob",
    "tempfile",
    "sqlite3",
    "pickle",
    "marshal",
    "shelve",
    "importlib",
    "builtins",
    "inspect",
    "gc",
    "code",
    "codeop",
    "webbrowser",
}

FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}

MAX_STDOUT_BYTES = 16_384
MAX_STDERR_BYTES = 8_192
MAX_CODE_CHARS = 32_000


def validate_python_ast(
    code: str, extra_allowed_modules: Optional[set[str]] = None
) -> Tuple[bool, str]:
    if not code or not code.strip():
        return False, "Code cannot be empty."
    if len(code) > MAX_CODE_CHARS:
        return False, f"Code exceeds maximum allowed size ({MAX_CODE_CHARS:,} characters)."

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError on line {exc.lineno}: {exc.msg}"

    allowed = ALLOWED_IMPORTS | (extra_allowed_modules or set())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES or root not in allowed:
                    return (
                        False,
                        f"Execution denied: import '{root}' is restricted by sandbox security policy. "
                        f"Allowed standard/scientific modules: {', '.join(sorted(ALLOWED_IMPORTS))}.",
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 1:
                return False, "Execution denied: parent relative imports outside workspace are forbidden."
            root = (node.module or "").split(".")[0]
            if root and (root in BLOCKED_MODULES or root not in allowed):
                return (
                    False,
                    f"Execution denied: import from '{root}' is restricted by sandbox security policy.",
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return (
                    False,
                    f"Execution denied: built-in '{node.func.id}()' is restricted inside the isolated sandbox.",
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                if node.attr not in {"__init__", "__repr__", "__str__", "__len__", "__iter__", "__next__", "__eq__", "__lt__", "__name__"}:
                    return (
                        False,
                        f"Execution denied: dunder attribute '{node.attr}' access is blocked in the sandbox.",
                    )

    return True, "OK"


# Runtime harness enforcing OS rlimits, network socket blocking, safe builtins, and guarded __import__
SANDBOX_RUNNER_SCRIPT = r'''import builtins as _builtins
import json as _json
import os as _os
import sys as _sys

# 1. Apply OS kernel resource limits (CPU, Virtual Memory, File Size, Descriptors, Core Dumps)
try:
    import resource as _resource
    _cpu_sec = int(_os.environ.get("_XY_CPU_LIMIT", "5"))
    _mem_mb = int(_os.environ.get("_XY_MEM_MB", "512"))
    _resource.setrlimit(_resource.RLIMIT_CPU, (_cpu_sec, _cpu_sec + 1))
    _resource.setrlimit(_resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
    _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
    try:
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass
    try:
        # Virtual address space cap (allows numpy/sympy shared libs while stopping RAM bombs)
        _mem_bytes = _mem_mb * 1024 * 1024
        _resource.setrlimit(_resource.RLIMIT_AS, (_mem_bytes, _mem_bytes))
    except Exception:
        pass
except Exception:
    pass

# 2. Pre-import numpy/sympy/math safely before locking down sys.modules and network/filesystem
_ALLOWED = set(_json.loads(_os.environ.get("_XY_ALLOWED_MODS", "[]")))
_CODE_FILE = _os.environ.get("_XY_USER_CODE_PATH", "")
with _builtins.open(_CODE_FILE, "r", encoding="utf-8") as _f:
    _USER_SOURCE = _f.read()

# Scrub internal env vars immediately
for _k in list(_os.environ.keys()):
    if _k not in ("PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
        _os.environ.pop(_k, None)

# 3. Hard-block network sockets & dangerous OS modules at runtime
class _BlockedSocketModule:
    def __getattr__(self, name):
        raise PermissionError("Network access is strictly disabled inside the Xyvora sandbox.")

for _net_mod in ("socket", "_socket", "ssl", "_ssl", "http", "urllib", "httpx", "requests", "ftplib", "smtplib"):
    _sys.modules[_net_mod] = _BlockedSocketModule()

_REAL_IMPORT = _builtins.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED:
        raise ImportError(f"Module '{root}' is blocked by Xyvora runtime sandbox policy.")
    return _REAL_IMPORT(name, globals, locals, fromlist, level)

_XYVORA_CHARTS = []

def emit_chart(title, labels, values, chart_type="bar"):
    _XYVORA_CHARTS.append({
        "type": "chart",
        "chart_type": str(chart_type),
        "title": str(title)[:120],
        "data": [{"label": str(l)[:40], "value": round(float(v), 4)} for l, v in list(zip(labels, values))[:60]]
    })

_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "classmethod", "complex", "dict", "divmod", "enumerate", "Exception",
    "ArithmeticError", "AssertionError", "AttributeError", "EOFError", "FloatingPointError",
    "GeneratorExit", "ImportError", "IndexError", "KeyError", "LookupError", "MemoryError",
    "NameError", "NotImplementedError", "OSError", "OverflowError", "PermissionError",
    "RecursionError", "RuntimeError", "StopIteration", "SyntaxError", "TypeError",
    "UnboundLocalError", "UnicodeError", "ValueError", "ZeroDivisionError",
    "filter", "float", "format", "frozenset", "hash", "hex", "id", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "oct",
    "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "zip",
    "True", "False", "None", "__build_class__",
]

_SAFE_BUILTINS = {k: getattr(_builtins, k) for k in _SAFE_BUILTIN_NAMES if hasattr(_builtins, k)}
_SAFE_BUILTINS["__import__"] = _guarded_import

_SANDBOX_GLOBALS = {
    "__builtins__": _SAFE_BUILTINS,
    "__name__": "__main__",
    "emit_chart": emit_chart,
}

_compiled = _builtins.compile(_USER_SOURCE, "<xyvora_sandbox>", "exec")
_builtins.exec(_compiled, _SANDBOX_GLOBALS, _SANDBOX_GLOBALS)

if _XYVORA_CHARTS:
    _builtins.print("\n__XYVORA_ARTIFACTS__:" + _json.dumps(_XYVORA_CHARTS))
'''


class CodeEngine:
    def _audit_execution(
        self,
        user_id: str,
        code: str,
        ok: bool,
        denied_reason: str,
        duration_ms: float,
        stdout_bytes: int,
    ) -> None:
        code_hash = hashlib.sha256(code.encode("utf-8", errors="ignore")).hexdigest()
        with _db_lock:
            conn = get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO sandbox_audit_logs
                    (id, user_id, code_sha256, ok, denied_reason, duration_ms, stdout_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        user_id or DEFAULT_USER_ID,
                        code_hash,
                        1 if ok else 0,
                        denied_reason[:300],
                        duration_ms,
                        stdout_bytes,
                        time.time(),
                    ),
                )
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    def execute_python(
        self,
        code: str,
        timeout_sec: float = 6.0,
        user_id: str = DEFAULT_USER_ID,
        extra_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        extra_allowed: set[str] = set()
        if extra_files:
            for rel_path in extra_files.keys():
                if rel_path.endswith(".py"):
                    first_seg = rel_path.replace("\\", "/").lstrip("/").split("/")[0]
                    mod_root = first_seg[:-3] if first_seg.endswith(".py") else first_seg
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", mod_root):
                        extra_allowed.add(mod_root)
            for rel_path, file_src in extra_files.items():
                if rel_path.endswith(".py"):
                    ok_f, reason_f = validate_python_ast(file_src, extra_allowed_modules=extra_allowed | {"unittest"})
                    if not ok_f:
                        self._audit_execution(user_id, code, False, f"Project file '{rel_path}': {reason_f}", 0.0, 0)
                        return {
                            "ok": False,
                            "stdout": "",
                            "stderr": f"Project file '{rel_path}' failed security check: {reason_f}",
                            "duration_ms": 0.0,
                            "artifacts": [],
                        }

        ok, reason = validate_python_ast(code, extra_allowed_modules=extra_allowed)
        if not ok:
            self._audit_execution(user_id, code, False, reason, 0.0, 0)
            return {
                "ok": False,
                "stdout": "",
                "stderr": reason,
                "duration_ms": 0.0,
                "artifacts": [],
            }

        sandbox_dir = tempfile.mkdtemp(prefix="xyvora_sbx_")
        os.chmod(sandbox_dir, 0o700)
        proc: Optional[subprocess.Popen[bytes]] = None
        t0 = time.perf_counter()

        try:
            # Write any approved workspace Python modules into isolated sandbox_dir
            if extra_files:
                for rel_path, file_src in extra_files.items():
                    if not rel_path.endswith(".py"):
                        continue
                    norm = os.path.normpath(rel_path).lstrip(os.sep)
                    if norm.startswith("..") or os.path.isabs(norm):
                        continue
                    full_dest = os.path.join(sandbox_dir, norm)
                    os.makedirs(os.path.dirname(full_dest), exist_ok=True)
                    # Ensure package __init__.py if nested
                    parent = os.path.dirname(full_dest)
                    if parent != sandbox_dir:
                        init_py = os.path.join(parent, "__init__.py")
                        if not os.path.exists(init_py):
                            with open(init_py, "w", encoding="utf-8") as inf:
                                inf.write("")
                    with open(full_dest, "w", encoding="utf-8") as f_out:
                        f_out.write(file_src)

            user_code_path = os.path.join(sandbox_dir, "_user_entry.py")
            runner_path = os.path.join(sandbox_dir, "_runner.py")
            with open(user_code_path, "w", encoding="utf-8") as uf:
                uf.write(code)
            with open(runner_path, "w", encoding="utf-8") as rf:
                rf.write(SANDBOX_RUNNER_SCRIPT)

            py_paths = [sandbox_dir] + [p for p in sys.path if p]
            clean_env = {
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": os.pathsep.join(py_paths),
                "PYTHONDONTWRITEBYTECODE": "1",
                "_XY_CPU_LIMIT": str(max(2, int(timeout_sec))),
                "_XY_MEM_MB": "768",
                "_XY_ALLOWED_MODS": json.dumps(sorted(ALLOWED_IMPORTS | extra_allowed)),
                "_XY_USER_CODE_PATH": user_code_path,
            }

            proc = subprocess.Popen(
                [sys.executable, "-B", runner_path],
                cwd=sandbox_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=clean_env,
                start_new_session=True,
            )

            try:
                raw_out_bytes, raw_err_bytes = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.communicate()
                elapsed_ms = round(timeout_sec * 1000.0, 1)
                msg = f"Execution timed out after {timeout_sec:.1f}s (CPU / infinite-loop protection triggered; process group terminated)."
                self._audit_execution(user_id, code, False, "timeout", elapsed_ms, 0)
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": msg,
                    "duration_ms": elapsed_ms,
                    "artifacts": [],
                }

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            out_truncated = len(raw_out_bytes) > MAX_STDOUT_BYTES
            err_truncated = len(raw_err_bytes) > MAX_STDERR_BYTES

            raw_out = raw_out_bytes[:MAX_STDOUT_BYTES + 4096].decode("utf-8", errors="replace")
            raw_err = raw_err_bytes[:MAX_STDERR_BYTES].decode("utf-8", errors="replace").strip()

            artifacts: List[Dict[str, Any]] = []
            if "__XYVORA_ARTIFACTS__:" in raw_out:
                parts = raw_out.split("__XYVORA_ARTIFACTS__:", 1)
                clean_out = parts[0].rstrip()
                try:
                    artifacts = json.loads(parts[1].strip())
                except Exception:
                    artifacts = []
            else:
                clean_out = raw_out[:MAX_STDOUT_BYTES].rstrip()

            if out_truncated and "__XYVORA_ARTIFACTS__:" not in raw_out:
                clean_out = clean_out[:MAX_STDOUT_BYTES] + "\n...[stdout truncated at 16 KB sandbox limit]"
            if err_truncated:
                raw_err = raw_err + "\n...[stderr truncated at 8 KB sandbox limit]"

            # Friendly diagnostic for negative signal return codes (e.g. SIGXCPU / SIGKILL from resource limit)
            if proc.returncode < 0 and not raw_err:
                sig_num = -proc.returncode
                raw_err = f"Process terminated by OS signal {sig_num} (sandbox CPU or memory limit exceeded)."

            ok_flag = proc.returncode == 0
            self._audit_execution(
                user_id,
                code,
                ok_flag,
                "" if ok_flag else raw_err[:180],
                elapsed_ms,
                len(raw_out_bytes),
            )
            return {
                "ok": ok_flag,
                "stdout": clean_out,
                "stderr": raw_err,
                "duration_ms": elapsed_ms,
                "artifacts": artifacts,
            }
        finally:
            if proc and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def synthesize_code(self, prompt: str) -> Dict[str, Any]:
        p = prompt.lower()
        nums = [int(n) for n in re.findall(r"\b\d+\b", prompt) if 2 <= int(n) <= 200000]

        if "dice" in p or "die" in p:
            trials = nums[0] if nums else 1000
            code = f"""import random
from collections import Counter

random.seed(42)
trials = {trials}
rolls = [random.randint(1, 6) + random.randint(1, 6) for _ in range(trials)]
counts = Counter(rolls)

labels = [str(total) for total in range(2, 13)]
frequencies = [counts[total] for total in range(2, 13)]

print(f"Simulated {{trials}} two-dice rolls:")
for total in range(2, 13):
    pct = (counts[total] / trials) * 100
    print(f"  Sum {{total:2d}}: {{counts[total]:4d}} hits ({{pct:5.2f}}%)")

emit_chart(f"Two-Dice Sum Distribution (n={{trials}})", labels, frequencies, "bar")
"""
            return {"language": "python", "title": f"Monte Carlo Two-Dice Simulation (n={trials})", "code": code, "executable": True}

        if "monte carlo" in p or ("pi" in p and "simulate" in p):
            trials = nums[0] if nums else 25000
            code = f"""import random
import math

random.seed(7)
samples = {trials}
inside = 0
checkpoints = []
labels = []

step = max(1, samples // 10)
for i in range(1, samples + 1):
    x, y = random.random(), random.random()
    if x * x + y * y <= 1.0:
        inside += 1
    if i % step == 0:
        est = 4.0 * inside / i
        labels.append(f"{{i//1000}}k" if i >= 1000 else str(i))
        checkpoints.append(est)

pi_est = 4.0 * inside / samples
err = abs(math.pi - pi_est)
print(f"Samples: {{samples:,}}")
print(f"Estimated Pi : {{pi_est:.6f}}")
print(f"Actual Pi    : {{math.pi:.6f}}")
print(f"Absolute Err : {{err:.6f}}")

emit_chart("Monte Carlo Convergence Toward Pi", labels, checkpoints, "line")
"""
            return {"language": "python", "title": "Monte Carlo Pi Estimation", "code": code, "executable": True}

        if "compound interest" in p or "investment" in p:
            years = nums[0] if (nums and nums[0] <= 60) else 20
            code = f"""principal = 10000.0
annual_rate = 0.08
monthly_contrib = 300.0
years = {years}

labels = []
balances = []
balance = principal

for yr in range(1, years + 1):
    for _ in range(12):
        balance = balance * (1 + annual_rate / 12) + monthly_contrib
    labels.append(f"Yr {{yr}}")
    balances.append(round(balance, 2))

total_invested = principal + monthly_contrib * 12 * years
print(f"Initial Principal : ${{principal:,.2f}}")
print(f"Total Contributed : ${{total_invested:,.2f}}")
print(f"Final Balance     : ${{balance:,.2f}}")
print(f"Compound Growth   : ${{balance - total_invested:,.2f}}")

emit_chart(f"Compound Growth Over {{years}} Years (8% APY)", labels, balances, "area")
"""
            return {"language": "python", "title": "Compound Interest & Portfolio Growth Simulator", "code": code, "executable": True}

        if "random walk" in p or "coin" in p:
            steps = nums[0] if (nums and nums[0] <= 500) else 60
            code = f"""import random

random.seed(21)
steps = {steps}
position = 0
labels = []
positions = []

for step in range(1, steps + 1):
    position += random.choice([-1, 1])
    labels.append(str(step))
    positions.append(position)

print(f"Completed {{steps}}-step 1D symmetric random walk.")
print(f"Peak Displacement : {{max(positions)}}")
print(f"Min Displacement  : {{min(positions)}}")
print(f"Final Position    : {{position}}")

emit_chart(f"1D Random Walk Trajectory ({{steps}} Steps)", labels, positions, "line")
"""
            return {"language": "python", "title": "1D Symmetric Random Walk Simulation", "code": code, "executable": True}

        if "prime" in p:
            limit = nums[0] if (nums and nums[0] <= 50000) else 200
            code = f"""def sieve_of_eratosthenes(limit: int) -> list[int]:
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False
    return [n for n, flag in enumerate(is_prime) if flag]

primes = sieve_of_eratosthenes({limit})
twin_primes = [(primes[i], primes[i + 1]) for i in range(len(primes) - 1) if primes[i + 1] - primes[i] == 2]

print(f"Found {{len(primes)}} primes <= {limit}:")
print(primes[:30], "..." if len(primes) > 30 else "")
print(f"Found {{len(twin_primes)}} twin prime pairs <= {limit}:")
print(twin_primes[:12])
"""
            return {"language": "python", "title": "Sieve of Eratosthenes & Twin Prime Finder", "code": code, "executable": True}

        if "fibonacci" in p:
            n_count = nums[0] if (nums and nums[0] <= 80) else 20
            code = f"""from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

seq = [fib(i) for i in range({n_count})]
print(f"First {n_count} Fibonacci numbers:")
print(seq)
emit_chart("Fibonacci Growth Curve", [str(i) for i in range(min(15, {n_count}))], seq[:15], "line")
"""
            return {"language": "python", "title": "Memoized Fibonacci Sequence", "code": code, "executable": True}

        if "lru" in p and "cache" in p:
            code = """from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: OrderedDict[str, int] = OrderedDict()

    def get(self, key: str) -> int:
        if key not in self.store:
            return -1
        self.store.move_to_end(key)
        return self.store[key]

    def put(self, key: str, value: int) -> None:
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            evicted = self.store.popitem(last=False)
            print(f"Evicted least-recently-used entry: {evicted}")

cache = LRUCache(capacity=3)
cache.put("alpha", 10)
cache.put("beta", 20)
cache.put("gamma", 30)
print("get('alpha') ->", cache.get("alpha"))
cache.put("delta", 40)
print("get('beta')  ->", cache.get("beta"))
print("Active keys  ->", list(cache.store.keys()))
"""
            return {"language": "python", "title": "O(1) LRU Cache Implementation", "code": code, "executable": True}

        if "binary search" in p or "merge sort" in p or "sort" in p:
            code = """def merge_sort(items: list[int]) -> list[int]:
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = merge_sort(items[:mid])
    right = merge_sort(items[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def binary_search(sorted_arr: list[int], target: int) -> int:
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == target:
            return mid
        elif sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

sample = [42, 7, 19, 88, 3, 25, 61, 14]
ordered = merge_sort(sample)
idx = binary_search(ordered, 25)
print("Original :", sample)
print("Sorted   :", ordered)
print("Index of 25:", idx)
"""
            return {"language": "python", "title": "Merge Sort O(n log n) + Binary Search O(log n)", "code": code, "executable": True}

        if "bfs" in p or "breadth first" in p or "graph" in p or "dijkstra" in p:
            code = """from collections import deque

def shortest_path_bfs(graph: dict[str, list[str]], start: str, goal: str) -> list[str]:
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []

network = {
    "Earth": ["Moon", "MarsRelay"],
    "Moon": ["MarsRelay", "VenusProbe"],
    "MarsRelay": ["EuropaBase", "CeresHub"],
    "CeresHub": ["TitanStation"],
    "EuropaBase": ["TitanStation"],
}

route = shortest_path_bfs(network, "Earth", "TitanStation")
print("Optimal multi-hop route:", " -> ".join(route))
"""
            return {"language": "python", "title": "Graph Shortest-Path BFS Solver", "code": code, "executable": True}

        if "luau" in p or "roblox" in p:
            code = """--!strict
-- Legitimate Luau Cooldown & Server-Validated Action Manager
export type CooldownRegistry = {
    durations: { [string]: number },
    lastTriggered: { [Player]: { [string]: number } },
}

local CooldownService = {}
CooldownService.__index = CooldownService

function CooldownService.new(defaultDurations: { [string]: number })
    local self = setmetatable({}, CooldownService)
    self.durations = defaultDurations
    self.lastTriggered = {}
    return self
end

function CooldownService:CanPerform(playerKey: string, actionName: string, nowClock: number): (boolean, number)
    local duration = self.durations[actionName] or 1.0
    local playerMap = self.lastTriggered[playerKey]
    if not playerMap then
        playerMap = {}
        self.lastTriggered[playerKey] = playerMap
    end
    local prev = playerMap[actionName]
    if prev and (nowClock - prev) < duration then
        return false, duration - (nowClock - prev)
    end
    playerMap[actionName] = nowClock
    return true, 0
end

return CooldownService"""
            return {"language": "luau", "title": "Typed Luau Server-Authoritative Cooldown Module", "code": code, "executable": False}

        if "react" in p or "tsx" in p:
            code = """import React, { useMemo, useState } from "react";

interface MetricItem {
  id: string;
  label: string;
  value: number;
}

export const LiveMetricsPanel: React.FC<{ items: MetricItem[] }> = ({ items }) => {
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () => items.filter((item) => item.label.toLowerCase().includes(query.toLowerCase())),
    [items, query]
  );

  return (
    <section className="p-4 rounded-2xl bg-slate-900 border border-slate-800 space-y-3">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Filter metrics..."
        className="w-full rounded-lg bg-slate-950 border border-slate-700 px-3 py-1.5 text-sm text-white"
      />
      <ul className="divide-y divide-slate-800">
        {filtered.map((m) => (
          <li key={m.id} className="py-2 flex justify-between text-sm">
            <span className="text-slate-300">{m.label}</span>
            <strong className="font-mono text-cyan-300">{m.value}</strong>
          </li>
        ))}
      </ul>
    </section>
  );
};"""
            return {"language": "tsx", "title": "React + TypeScript Filtered Metrics Component", "code": code, "executable": False}

        if "typescript" in p or "javascript" in p or "fetch" in p:
            code = """export interface RetryOptions {
  retries?: number;
  baseDelayMs?: number;
  timeoutMs?: number;
}

export async function fetchWithRetry<T>(
  url: string,
  init: RequestInit = {},
  options: RetryOptions = {}
): Promise<T> {
  const { retries = 3, baseDelayMs = 300, timeoutMs = 8000 } = options;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...init, signal: controller.signal });
      clearTimeout(timer);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return (await response.json()) as T;
    } catch (err) {
      clearTimeout(timer);
      if (attempt === retries) throw err;
      const backoff = baseDelayMs * Math.pow(2, attempt);
      await new Promise((r) => setTimeout(r, backoff));
    }
  }
  throw new Error("Unreachable");
}"""
            return {"language": "typescript", "title": "TypeScript Exponential-Backoff Fetch Wrapper", "code": code, "executable": False}

        if "java" in p and "javascript" not in p:
            code = """import java.util.LinkedHashMap;
import java.util.Map;

public final class BoundedLruCache<K, V> extends LinkedHashMap<K, V> {
    private final int capacity;

    public BoundedLruCache(int capacity) {
        super(capacity, 0.75f, true);
        if (capacity <= 0) throw new IllegalArgumentException("Capacity must be positive");
        this.capacity = capacity;
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        return size() > capacity;
    }
}"""
            return {"language": "java", "title": "Java Generic Thread-Ready LRU Cache", "code": code, "executable": False}

        if "c++" in p or "cpp" in p:
            code = """#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

double computeMovingAverage(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    const double total = std::accumulate(xs.begin(), xs.end(), 0.0);
    return total / static_cast<double>(xs.size());
}

int main() {
    std::vector<double> samples{12.4, 15.8, 19.2, 21.0, 18.6};
    std::cout << "Mean: " << computeMovingAverage(samples) << "\\n";
    return 0;
}"""
            return {"language": "cpp", "title": "Modern C++20 Signal Average Routine", "code": code, "executable": False}

        if "c#" in p or "csharp" in p:
            code = """using System;
using System.Collections.Generic;
using System.Linq;

public static class TelemetryStats
{
    public static (double Mean, double Max) Summarize(IReadOnlyList<double> values)
    {
        if (values == null || values.Count == 0)
            throw new ArgumentException("Collection cannot be empty.", nameof(values));
        return (values.Average(), values.Max());
    }
}"""
            return {"language": "csharp", "title": "C# Strongly-Typed Telemetry Summary", "code": code, "executable": False}

        if "sql" in p:
            code = """WITH customer_totals AS (
    SELECT
        c.customer_id,
        c.full_name,
        COUNT(o.order_id) AS order_count,
        SUM(o.total_amount) AS lifetime_revenue,
        AVG(o.total_amount) AS avg_order_value
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
      AND o.created_at >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY c.customer_id, c.full_name
)
SELECT
    customer_id,
    full_name,
    order_count,
    ROUND(lifetime_revenue, 2) AS lifetime_revenue,
    ROUND(avg_order_value, 2) AS avg_order_value,
    RANK() OVER (ORDER BY lifetime_revenue DESC) AS revenue_rank
FROM customer_totals
ORDER BY lifetime_revenue DESC
LIMIT 5;"""
            return {"language": "sql", "title": "Analytical Top-Customers SQL CTE & Window Query", "code": code, "executable": False}

        code = """import statistics

dataset = [14.2, 18.7, 22.1, 19.5, 27.4, 31.0, 29.8, 35.6]
mean_val = statistics.mean(dataset)
stdev_val = statistics.stdev(dataset)
z_scores = [round((x - mean_val) / stdev_val, 3) for x in dataset]

print(f"Samples  : {dataset}")
print(f"Mean     : {mean_val:.2f}")
print(f"Std Dev  : {stdev_val:.2f}")
print(f"Z-Scores : {z_scores}")

emit_chart("Sample Progression", [f"T{i+1}" for i in range(len(dataset))], dataset, "line")
"""
        return {"language": "python", "title": "Python Analytical Pipeline", "code": code, "executable": True}
